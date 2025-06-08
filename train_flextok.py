import os
import glob
import random
from dataclasses import dataclass
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import get_cosine_schedule_with_warmup

from models.tokenizer import ImageTokenizer, ImageTokenizerConfig


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@dataclass
class TrainingConfig:
    run_name: str
    vae_path: str
    project_name: str = "flextok"

    num_epochs: int = 10
    train_batch_size: int = 8
    val_batch_size: int = 8
    val_interval: int = 500
    seed: int = 42
    gradient_accumulation_steps: int = 1

    vq_loss_weight: float = 0.1

    image_size: int = 512

    model_dim: int = 256
    num_heads: int = 8
    num_registers: int = 256
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2

    rope_freq_base: float = 10000.0
    patch_size: int = 2
    quant_vocab_size: int = 1024
    quant_dim: int = 256
    vae_downscaling_factor: int = 8

    sample_steps: int = 50
    use_compile: bool = True


class ImageDataset(Dataset):
    def __init__(self, hf_dataset, indices, image_size: int = 512):
        self.dataset = hf_dataset
        self.indices = indices
        self.transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        item = self.dataset[real_idx]
        image = item["jpg"]
        return self.transform(image)


def load_data():
    data_files = glob.glob("data/datasets--BLIP3o--BLIP3o-Pretrain-JourneyDB/snapshots/*/JourneyDB_*.tar")
    if not data_files:
        print("Downloading dataset...")
        snapshot_download(repo_id="BLIP3o/BLIP3o-Pretrain-JourneyDB", repo_type="dataset", allow_patterns=[f"JourneyDB_00{i}.tar" for i in range(1, 5)], cache_dir="data/")

    data_files = glob.glob("data/datasets--BLIP3o--BLIP3o-Pretrain-JourneyDB/snapshots/*/JourneyDB_*.tar")

    dataset = load_dataset("webdataset", data_files=data_files, cache_dir="data", split="train", streaming=False)
    return dataset


def plot_reconstruction_grid(model: ImageTokenizer, val_images: torch.Tensor, num_tokens_list: list[int]) -> None:
    model.eval()

    fig, axes = plt.subplots(val_images.size(0), 10, figsize=(30, 3 * val_images.size(0)))

    with torch.no_grad():
        vae_latents = model.vae_encode(val_images)

        for i in range(val_images.size(0)):
            orig_for_plot = (val_images[i].cpu().permute(1, 2, 0) + 1) / 2
            axes[i, 0].imshow(orig_for_plot.clamp(0, 1))
            axes[i, 0].set_title("Original")
            axes[i, 0].axis("off")

            for j, num_tokens in enumerate(num_tokens_list):
                try:
                    reconstruction = model.sample(vae_latents[i : i + 1], sample_steps=20, num_tokens_to_use=num_tokens)
                    recon_img = model.vae_decode(reconstruction)
                    recon_for_plot = (recon_img[0].cpu().permute(1, 2, 0) + 1) / 2
                    axes[i, j + 1].imshow(recon_for_plot.clamp(0, 1))
                    axes[i, j + 1].set_title(f"{num_tokens}")
                except:
                    axes[i, j + 1].text(0.5, 0.5, "Error", ha="center", va="center", transform=axes[i, j + 1].transAxes)
                axes[i, j + 1].axis("off")


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name", type=str, required=True, help="Name for this training run")
    parser.add_argument("--vae_path", type=str, default="stabilityai/sd-vae-ft-mse", help="VAE model path")
    parser.add_argument("--project_name", type=str, default="flextok", help="W&B project name")

    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=8, help="Validation batch size")
    parser.add_argument("--val_interval", type=int, default=500, help="Validation interval (steps)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")

    parser.add_argument("--vq_loss_weight", type=float, default=0.1, help="VQ loss weight")

    parser.add_argument("--image_size", type=int, default=512, help="Image resize resolution")

    parser.add_argument("--model_dim", type=int, default=256, help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_registers", type=int, default=256, help="Number of register tokens")
    parser.add_argument("--num_encoder_layers", type=int, default=2, help="Number of encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, default=2, help="Number of decoder layers")

    parser.add_argument("--rope_freq_base", type=float, default=10000.0, help="RoPE frequency base")
    parser.add_argument("--patch_size", type=int, default=2, help="Patch size for tokenization")
    parser.add_argument("--quant_vocab_size", type=int, default=1024, help="Quantization vocabulary size")
    parser.add_argument("--quant_dim", type=int, default=256, help="Quantization dimension")
    parser.add_argument("--vae_downscaling_factor", type=int, default=8, help="VAE downscaling factor")

    parser.add_argument("--sample_steps", type=int, default=20, help="Number of sampling steps for reconstruction")
    parser.add_argument("--use_compile", action="store_true", default=True, help="Use torch.compile")
    parser.add_argument("--no_compile", dest="use_compile", action="store_false", help="Disable torch.compile")

    parsed_args = parser.parse_args()

    return TrainingConfig(
        run_name=parsed_args.run_name,
        vae_path=parsed_args.vae_path,
        project_name=parsed_args.project_name,
        num_epochs=parsed_args.num_epochs,
        train_batch_size=parsed_args.train_batch_size,
        val_batch_size=parsed_args.val_batch_size,
        val_interval=parsed_args.val_interval,
        seed=parsed_args.seed,
        gradient_accumulation_steps=parsed_args.gradient_accumulation_steps,
        vq_loss_weight=parsed_args.vq_loss_weight,
        image_size=parsed_args.image_size,
        model_dim=parsed_args.model_dim,
        num_heads=parsed_args.num_heads,
        num_registers=parsed_args.num_registers,
        num_encoder_layers=parsed_args.num_encoder_layers,
        num_decoder_layers=parsed_args.num_decoder_layers,
        rope_freq_base=parsed_args.rope_freq_base,
        patch_size=parsed_args.patch_size,
        quant_vocab_size=parsed_args.quant_vocab_size,
        quant_dim=parsed_args.quant_dim,
        vae_downscaling_factor=parsed_args.vae_downscaling_factor,
        sample_steps=parsed_args.sample_steps,
        use_compile=parsed_args.use_compile,
    )


def train(args: TrainingConfig) -> None:
    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{ddp_local_rank}")
    is_master_process = ddp_rank == 0
    print(f"DDP rank: {ddp_rank}, world size: {ddp_world_size}, local rank: {ddp_local_rank}")

    global_step = 0

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    config = ImageTokenizerConfig(
        vae_path=args.vae_path,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        rope_freq_base=args.rope_freq_base,
        height=args.image_size // args.vae_downscaling_factor,  # VAE latent size
        width=args.image_size // args.vae_downscaling_factor,
        patch_size=args.patch_size,
        quant_vocab_size=args.quant_vocab_size,
        quant_dim=args.quant_dim,
        num_registers=args.num_registers,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
    )

    model = ImageTokenizer(config).to(device)
    if args.use_compile:
        model = torch.compile(model)
    if is_master_process:
        print("Number of parameters: ", sum(p.numel() for p in model.parameters()) / 1e6, "M")
        wandb.init(project=args.project_name, name=args.run_name)
        wandb.config.update({"model_config": config.__dict__, "args": args.__dict__})

    model = DDP(model, device_ids=[ddp_local_rank])

    dataset = load_data()

    total_size = len(dataset)
    train_size = int(total_size * 0.98)
    indices = list(range(total_size))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = ImageDataset(dataset, train_indices, image_size=args.image_size)
    val_dataset = ImageDataset(dataset, val_indices, image_size=args.image_size)

    train_sampler = DistributedSampler(train_dataset, num_replicas=ddp_world_size, rank=ddp_rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=ddp_world_size, rank=ddp_rank)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, sampler=train_sampler, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, sampler=val_sampler, shuffle=False, num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=args.num_epochs * len(train_loader))

    num_tokens_options = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    assert args.num_registers in num_tokens_options, f"Number of registers must be one of {num_tokens_options}"
    losses = []

    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)
        progress_bar = tqdm(train_loader, disable=not is_master_process)

        for i, images in enumerate(progress_bar):
            images = images.to(device)
            batch_size = images.shape[0]

            with ctx:
                with torch.no_grad():
                    vae_latents = model.module.vae_encode(images)

                t = torch.rand(batch_size, device=device)
                z1 = torch.randn_like(vae_latents)

                t_expanded = t.view(batch_size, 1, 1, 1)
                zt = (1 - t_expanded) * vae_latents + t_expanded * z1

                num_tokens_to_use = random.choice(num_tokens_options)

                velocity_pred, vq_loss, idx_Br = model(vae_latents, t, zt, num_tokens_to_use)

                target_velocity = z1 - vae_latents

                velocity_loss = F.mse_loss(velocity_pred, target_velocity)
                total_loss = velocity_loss + args.vq_loss_weight * vq_loss
                total_loss = total_loss / args.gradient_accumulation_steps

            total_loss.backward()

            if (i + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            if is_master_process:
                progress_bar.set_description(f"Epoch {epoch}")
                progress_bar.set_postfix(loss=total_loss.item(), velocity_loss=velocity_loss.item(), vq_loss=vq_loss.item(), tokens=num_tokens_to_use)
                losses.append(total_loss.item())
                wandb.log(
                    {
                        "loss": total_loss.item(),
                        "velocity_loss": velocity_loss.item(),
                        "vq_loss": vq_loss.item(),
                        "num_tokens": num_tokens_to_use,
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )

            with torch.no_grad():
                if i % args.val_interval == 0 and i != 0:
                    if is_master_process:
                        os.makedirs(f"images/val_images/flextok", exist_ok=True)

                    val_progress_bar = tqdm(val_loader, desc="Validating", leave=False, disable=not is_master_process)
                    val_losses = []
                    val_velocity_losses = []
                    val_vq_losses = []
                    all_val_indices = []

                    for j, val_images in enumerate(val_progress_bar):
                        val_images = val_images.to(device)
                        batch_size = val_images.shape[0]

                        with ctx:
                            val_vae_latents = model.module.vae_encode(val_images)

                            val_t = torch.rand(batch_size, device=device)
                            val_z1 = torch.randn_like(val_vae_latents)

                            val_t_expanded = val_t.view(batch_size, 1, 1, 1)
                            val_zt = (1 - val_t_expanded) * val_vae_latents + val_t_expanded * val_z1

                            val_num_tokens = random.choice(num_tokens_options)

                            val_velocity_pred, val_vq_loss, val_idx_Br = model(val_vae_latents, val_t, val_zt, val_num_tokens)

                            val_target_velocity = val_z1 - val_vae_latents

                            val_velocity_loss = F.mse_loss(val_velocity_pred, val_target_velocity)
                            val_total_loss = val_velocity_loss + args.vq_loss_weight * val_vq_loss

                        val_losses.append(val_total_loss.item())
                        val_velocity_losses.append(val_velocity_loss.item())
                        val_vq_losses.append(val_vq_loss.item())

                        # Collect token indices for codebook usage analysis
                        all_val_indices.append(val_idx_Br.flatten())

                        if j == 0 and is_master_process:
                            plot_reconstruction_grid(model.module, val_images[:8], num_tokens_options)
                            plt.savefig(f"images/val_images/flextok/epoch_{epoch}_iter_{i}.png")
                            plt.close()

                    # Calculate codebook usage metrics
                    all_indices = torch.cat(all_val_indices, dim=0)
                    unique_codes = torch.unique(all_indices)
                    codebook_usage = len(unique_codes) / config.quant_vocab_size
                    code_histogram = torch.bincount(all_indices.cpu(), minlength=config.quant_vocab_size)
                    code_entropy = -(code_histogram.float() / len(all_indices)) * torch.log2(code_histogram.float() / len(all_indices) + 1e-10)
                    code_entropy = code_entropy.sum()

                    val_loss_tensor = torch.tensor(np.mean(val_losses)).to(device)
                    val_velocity_loss_tensor = torch.tensor(np.mean(val_velocity_losses)).to(device)
                    val_vq_loss_tensor = torch.tensor(np.mean(val_vq_losses)).to(device)
                    codebook_usage_tensor = torch.tensor(codebook_usage).to(device)
                    code_entropy_tensor = torch.tensor(code_entropy).to(device)

                    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(val_velocity_loss_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(val_vq_loss_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(codebook_usage_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(code_entropy_tensor, op=dist.ReduceOp.SUM)

                    val_loss_avg = val_loss_tensor.item() / dist.get_world_size()
                    val_velocity_loss_avg = val_velocity_loss_tensor.item() / dist.get_world_size()
                    val_vq_loss_avg = val_vq_loss_tensor.item() / dist.get_world_size()
                    codebook_usage_avg = codebook_usage_tensor.item() / dist.get_world_size()
                    code_entropy_avg = code_entropy_tensor.item() / dist.get_world_size()

                    dist.barrier()

                    if is_master_process:
                        print(f"Validation loss: {val_loss_avg:.4f}")
                        print(f"Validation velocity loss: {val_velocity_loss_avg:.4f}")
                        print(f"Validation VQ loss: {val_vq_loss_avg:.4f}")
                        print(f"Codebook usage: {codebook_usage_avg:.2%}")
                        print(f"Code entropy: {code_entropy_avg:.2f}")

                        wandb.log(
                            {
                                "val_loss": val_loss_avg,
                                "val_velocity_loss": val_velocity_loss_avg,
                                "val_vq_loss": val_vq_loss_avg,
                                "codebook_usage": codebook_usage_avg,
                                "code_entropy": code_entropy_avg,
                                "code_histogram": wandb.Histogram(code_histogram.numpy()),
                                "reconstruction_grid": wandb.Image(f"images/val_images/flextok/epoch_{epoch}_iter_{i}.png"),
                            }
                        )

        if is_master_process:
            model_state = model.module.state_dict()
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                },
                f"checkpoint_epoch_{epoch}.pt",
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    train(args)
