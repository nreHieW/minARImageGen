import os
import glob
import random
from dataclasses import dataclass
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from datasets import load_dataset
from huggingface_hub import snapshot_download

from models.tokenizer import ImageTokenizer, ImageTokenizerConfig
from models.tokenizer import LastLayer
from torch.optim.lr_scheduler import ReduceLROnPlateau


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@dataclass
class TrainingConfig:
    run_name: str
    vae_path: str
    project_name: str = "flextok-overfit"

    num_epochs: int = 1000
    batch_size: int = 8
    plot_interval: int = 50
    seed: int = 42

    vq_loss_weight: float = 0.1
    image_size: int = 512

    # Learning rate scheduler parameters
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 10
    lr_scheduler_min_lr: float = 1e-7

    model_dim: int = 256
    num_heads: int = 8
    num_registers: int = 256
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2

    rope_freq_base: float = 10000.0
    patch_size: int = 16
    quant_vocab_size: int = 8192
    quant_dim: int = 4

    sample_steps: int = 20
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
        snapshot_download(repo_id="BLIP3o/BLIP3o-Pretrain-JourneyDB", repo_type="dataset", allow_patterns=[f"JourneyDB_00{i}.tar" for i in range(1, 2)], cache_dir="data/")

    data_files = glob.glob("data/datasets--BLIP3o--BLIP3o-Pretrain-JourneyDB/snapshots/*/JourneyDB_*.tar")
    print(f"Found {len(data_files)} data files")

    dataset = load_dataset("webdataset", data_files=data_files, cache_dir="data", split="train", streaming=False)
    return dataset


def plot_reconstruction_grid(model: ImageTokenizer, images: torch.Tensor, epoch: int, step: int, num_tokens_list: list[int]) -> None:
    model.eval()

    fig, axes = plt.subplots(min(4, images.size(0)), 10, figsize=(30, 3 * min(4, images.size(0))))
    if images.size(0) == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for i in range(min(4, images.size(0))):
            # Original image
            orig_for_plot = (images[i].cpu().permute(1, 2, 0) + 1) / 2
            axes[i, 0].imshow(orig_for_plot.clamp(0, 1))
            axes[i, 0].set_title("Original")
            axes[i, 0].axis("off")

            # Reconstructions with different token counts
            for j, num_tokens in enumerate(num_tokens_list):
                try:
                    reconstruction = model.sample(images[i : i + 1], sample_steps=20, num_tokens_to_use=num_tokens)
                    recon_for_plot = (reconstruction[0].cpu().permute(1, 2, 0) + 1) / 2
                    axes[i, j + 1].imshow(recon_for_plot.clamp(0, 1))
                    axes[i, j + 1].set_title(f"{num_tokens}")
                except Exception as e:
                    axes[i, j + 1].text(0.5, 0.5, "Error", ha="center", va="center", transform=axes[i, j + 1].transAxes)
                axes[i, j + 1].axis("off")

    os.makedirs("images/overfit", exist_ok=True)
    plt.savefig(f"images/overfit/epoch_{epoch}_step_{step}.png", dpi=150, bbox_inches="tight")
    plt.close()
    model.train()


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name", type=str, required=True, help="Name for this training run")
    parser.add_argument("--vae_path", type=str, required=True, help="VAE model path")
    parser.add_argument("--project_name", type=str, default="flextok-overfit", help="W&B project name")

    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--plot_interval", type=int, default=50, help="Plot interval (epochs)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--vq_loss_weight", type=float, default=0.1, help="VQ loss weight")
    parser.add_argument("--image_size", type=int, default=512, help="Image resize resolution")

    parser.add_argument("--lr_scheduler_factor", type=float, default=0.5, help="Learning rate scheduler factor")
    parser.add_argument("--lr_scheduler_patience", type=int, default=10, help="Learning rate scheduler patience")
    parser.add_argument("--lr_scheduler_min_lr", type=float, default=1e-7, help="Learning rate scheduler minimum learning rate")

    parser.add_argument("--model_dim", type=int, default=256, help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_registers", type=int, default=256, help="Number of register tokens")
    parser.add_argument("--num_encoder_layers", type=int, default=2, help="Number of encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, default=2, help="Number of decoder layers")

    parser.add_argument("--rope_freq_base", type=float, default=10000.0, help="RoPE frequency base")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size for tokenization")
    parser.add_argument("--quant_vocab_size", type=int, default=8192, help="Quantization vocabulary size")
    parser.add_argument("--quant_dim", type=int, default=4, help="Quantization dimension")

    parser.add_argument("--sample_steps", type=int, default=20, help="Number of sampling steps for reconstruction")
    parser.add_argument("--use_compile", action="store_true", default=True, help="Use torch.compile")
    parser.add_argument("--no_compile", dest="use_compile", action="store_false", help="Disable torch.compile")

    parsed_args = parser.parse_args()

    return TrainingConfig(
        run_name=parsed_args.run_name,
        vae_path=parsed_args.vae_path,
        project_name=parsed_args.project_name,
        num_epochs=parsed_args.num_epochs,
        batch_size=parsed_args.batch_size,
        plot_interval=parsed_args.plot_interval,
        seed=parsed_args.seed,
        vq_loss_weight=parsed_args.vq_loss_weight,
        image_size=parsed_args.image_size,
        lr_scheduler_factor=parsed_args.lr_scheduler_factor,
        lr_scheduler_patience=parsed_args.lr_scheduler_patience,
        lr_scheduler_min_lr=parsed_args.lr_scheduler_min_lr,
        model_dim=parsed_args.model_dim,
        num_heads=parsed_args.num_heads,
        num_registers=parsed_args.num_registers,
        num_encoder_layers=parsed_args.num_encoder_layers,
        num_decoder_layers=parsed_args.num_decoder_layers,
        rope_freq_base=parsed_args.rope_freq_base,
        patch_size=parsed_args.patch_size,
        quant_vocab_size=parsed_args.quant_vocab_size,
        quant_dim=parsed_args.quant_dim,
        sample_steps=parsed_args.sample_steps,
        use_compile=parsed_args.use_compile,
    )


def train(args: TrainingConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create config for raw image processing (no VAE)
    config = ImageTokenizerConfig(
        vae_path=args.vae_path,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        rope_freq_base=args.rope_freq_base,
        height=args.image_size,
        width=args.image_size,
        patch_size=args.patch_size,
        quant_vocab_size=args.quant_vocab_size,
        quant_dim=args.quant_dim,
        num_registers=args.num_registers,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
    )

    model = ImageTokenizer(config).to(device)

    # Override for raw image processing (no VAE)
    print("Configuring model for raw image processing (no VAE)")
    model.latent_dim = 3
    model.init_proj = torch.nn.Linear(3 * config.patch_size * config.patch_size, config.model_dim).to(device)
    model.last_layer = LastLayer(hidden_size=config.model_dim, patch_size=config.patch_size, out_channels=3).to(device)

    if args.use_compile:
        model = torch.compile(model)

    print("Number of parameters: ", sum(p.numel() for p in model.parameters()) / 1e6, "M")

    wandb.init(project=args.project_name, name=args.run_name)
    wandb.config.update({"model_config": config.__dict__, "args": args.__dict__})

    # Load dataset and get one batch
    dataset = load_data()
    train_dataset = ImageDataset(dataset, list(range(min(10, len(dataset)))), image_size=args.image_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Get one batch to overfit on
    single_batch = next(iter(train_loader)).to(device)
    print(f"Overfitting on batch with shape: {single_batch.shape}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_scheduler_factor, patience=args.lr_scheduler_patience, min_lr=args.lr_scheduler_min_lr, verbose=True)

    num_tokens_options = [256]  # Fixed for simplicity
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    losses = []

    for epoch in range(args.num_epochs):
        model.train()

        # Use the same batch every time
        images = single_batch
        batch_size = images.shape[0]

        with ctx:
            # Use raw images directly (no VAE encoding)
            vae_latents = images

            t = torch.rand(batch_size, device=device)
            z1 = torch.randn_like(vae_latents)

            t_expanded = t.view(batch_size, 1, 1, 1)
            zt = (1 - t_expanded) * vae_latents + t_expanded * z1

            num_tokens_to_use = random.choice(num_tokens_options)
            velocity_pred, vq_loss, _ = model(vae_latents, t, zt, num_tokens_to_use)

            target_velocity = z1 - vae_latents
            velocity_loss = F.mse_loss(velocity_pred, target_velocity)
            total_loss = velocity_loss + args.vq_loss_weight * vq_loss

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(total_loss.item())

        # Logging
        if epoch % 10 == 0:
            print(f"Epoch {epoch:4d} | Loss: {total_loss.item():.6f} | Velocity: {velocity_loss.item():.6f} | VQ: {vq_loss.item():.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        wandb.log(
            {
                "loss": total_loss.item(),
                "velocity_loss": velocity_loss.item(),
                "vq_loss": vq_loss.item(),
                "num_tokens": num_tokens_to_use,
                "lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
            }
        )

        # Plot reconstructions
        if epoch % args.plot_interval == 0:
            plot_reconstruction_grid(model, images[:4], epoch, 0, num_tokens_options)
            wandb.log(
                {
                    "reconstruction_grid": wandb.Image(f"images/overfit/epoch_{epoch}_step_0.png"),
                }
            )

        # Update scheduler every 100 epochs
        if epoch % 100 == 0 and epoch > 0:
            recent_loss = np.mean(losses[-100:])
            scheduler.step(recent_loss)

    print("Training completed!")

    # Save final model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "final_loss": total_loss.item(),
        },
        f"overfit_model_{args.run_name}.pt",
    )


if __name__ == "__main__":
    args = parse_args()
    train(args)
