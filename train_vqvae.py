import os
import random
from dataclasses import dataclass
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid
from tqdm import tqdm

from models.vqvae import VQVAE, VQVAEConfig
from utils.evaluate.fid import calculate_fid_given_paths
from utils.gan import NLayerDiscriminator, calculate_adaptive_weight, PatchDiscriminator
from utils.imagenet_dataset import get_imagenet_dataloader
from utils.losses import LPIPS
from transformers import get_cosine_schedule_with_warmup


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@dataclass
class TrainConfig:
    num_epochs: int = 2
    train_batch_size: int = 16
    val_batch_size: int = 128
    val_interval: int = 1000
    seed: int = 42
    gan_start_iter: int = 1000
    gradient_accumulation_steps: int = 4

    recon_loss_weight: float = 1.0
    perceptual_loss_weight: float = 1.0
    q_loss_weight: float = 0.5
    gan_loss_weight: float = 1


def plot_images(pred, original=None):
    n = pred.size(0)
    pred = pred * 0.5 + 0.5
    pred = pred.clamp(0, 1)
    img = pred.cpu().detach()

    if original is not None:
        original = original * 0.5 + 0.5
        original = original.clamp(0, 1)
        original = original.cpu().detach()
        img = torch.cat([original, img], dim=0)

    img_grid = make_grid(img, nrow=n)
    img_grid = img_grid.permute(1, 2, 0).numpy()
    img_grid = (img_grid * 255).astype("uint8")
    plt.figure(figsize=(16, 16))
    plt.imshow(img_grid, cmap=None)
    plt.axis("off")


def train(run_name: str):
    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{ddp_local_rank}")
    is_master_process = ddp_rank == 0
    print(f"DDP rank: {ddp_rank}, world size: {ddp_world_size}, local rank: {ddp_local_rank}")

    args = TrainConfig()
    global_step = 0  # Add global step counter

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if is_master_process:
        if os.path.exists("images/val_images/generated"):
            for img in os.listdir("images/val_images/generated"):
                os.remove(os.path.join("images/val_images/generated", img))
        if os.path.exists("images/val_images/original"):
            for img in os.listdir("images/val_images/original"):
                os.remove(os.path.join("images/val_images/original", img))

    vqvae_config = VQVAEConfig(
        in_channels=3,
        dim=128,
        ch_mult=[1, 1, 2, 2, 4],
        num_res_blocks=2,
        z_channels=256,
        codebook_dim=8,
        out_ch=3,
        vocab_size=16384,
        patch_sizes=[1, 2, 3, 4, 5, 6, 8, 10, 13, 16],
    )
    vqvae = VQVAE(vqvae_config).to(device)
    if is_master_process:
        print("Number of parameters: ", sum(p.numel() for p in vqvae.parameters()) / 1e6, "M")
        wandb.init(project="var", name=run_name)
        wandb.config.update({"vqvae_config": vqvae_config.__dict__, "train_config": args.__dict__})

    vqvae = DDP(vqvae, device_ids=[ddp_local_rank])
    # lpips_model = lpips.LPIPS(net="vgg").to(device).eval()
    lpips_model = LPIPS().to(device).eval()
    for param in lpips_model.parameters():
        param.requires_grad = False

    gan = NLayerDiscriminator().to(device)
    # gan = PatchDiscriminator().to(device)
    gan = DDP(gan, device_ids=[ddp_local_rank])
    train_loader, val_loader = get_imagenet_dataloader(ddp_rank, ddp_world_size, train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size, num_workers=16)

    # optimizer_vqvae = torch.optim.AdamW(
    #     [{"params": [p for n, p in vqvae.named_parameters() if "codebook" not in n], "lr": 3e-5}, {"params": [p for n, p in vqvae.named_parameters() if "codebook" in n], "lr": 1e-6}]
    # )
    optimizer_vqvae = torch.optim.AdamW(vqvae.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.05)
    optimizer_gan = torch.optim.AdamW(gan.parameters(), lr=1e-4)
    scheduler_vqvae = get_cosine_schedule_with_warmup(optimizer_vqvae, num_warmup_steps=800, num_training_steps=args.num_epochs * len(train_loader))

    losses = []

    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    for epoch in range(args.num_epochs):
        progress_bar = tqdm(train_loader, disable=not is_master_process)
        for i, batch in enumerate(progress_bar):
            image, _label = batch
            image = image.to(device)

            with ctx:
                xhat, r_maps, idxs, scales, q_loss = vqvae(image)

                real_pred = gan(image)
                fake_pred = gan(xhat.detach())

            optimizer_gan.zero_grad()

            with ctx:
                # real_loss = F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred))
                # fake_loss = F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred))
                # gan_discriminator_loss = (real_loss + fake_loss) / 2
                d_loss_real = torch.mean(F.relu(1.0 - real_pred))
                d_loss_fake = torch.mean(F.relu(1.0 + fake_pred))
                gan_discriminator_loss = (d_loss_real + d_loss_fake) / 2
                if global_step < args.gan_start_iter:
                    gan_discriminator_loss *= 0
                gan_discriminator_loss = gan_discriminator_loss / args.gradient_accumulation_steps

            gan_discriminator_loss.backward()

            if (i + 1) % args.gradient_accumulation_steps == 0:
                optimizer_gan.step()
                optimizer_gan.zero_grad()

            with ctx:
                recon_loss = F.mse_loss(xhat, image)
                perceptual_loss = lpips_model(xhat, image).mean()
                fake_pred = gan(xhat)
                real_pred = gan(image).detach()
                gan_loss = (real_pred - fake_pred - 0.1).relu().mean()
                if global_step < args.gan_start_iter:
                    gan_loss *= 0

                adaptive_weight = calculate_adaptive_weight(vqvae.module.decoder.conv_out.weight, perceptual_loss, gan_loss)
                loss = args.recon_loss_weight * recon_loss + args.perceptual_loss_weight * perceptual_loss + args.q_loss_weight * q_loss + args.gan_loss_weight * adaptive_weight * gan_loss
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            if (i + 1) % args.gradient_accumulation_steps == 0:
                optimizer_vqvae.step()
                scheduler_vqvae.step()
                optimizer_vqvae.zero_grad()

            global_step += 1

            if is_master_process:
                progress_bar.set_description(f"Epoch {epoch}")
                progress_bar.set_postfix(
                    loss=loss.item(), loss_recon=recon_loss.item(), loss_perceptual=perceptual_loss.item(), loss_q=q_loss.item(), loss_gan=gan_loss.item(), loss_gan_disc=gan_discriminator_loss.item()
                )
                losses.append(loss.item())
                wandb.log(
                    {
                        "loss": loss.item(),
                        "loss_recon": recon_loss.item(),
                        "loss_perceptual": perceptual_loss.item(),
                        "loss_q": q_loss.item(),
                        "loss_gan": gan_loss.item(),
                        "loss_gan_disc": gan_discriminator_loss.item(),
                        "lr_vqvae": optimizer_vqvae.param_groups[0]["lr"],
                    }
                )

            with torch.no_grad():
                if i % args.val_interval == 0 and i != 0:
                    if is_master_process:
                        os.makedirs(f"images/val_images/generated", exist_ok=True)
                        os.makedirs(f"images/val_images/original", exist_ok=True)

                    val_progress_bar = tqdm(val_loader, desc="Validating", leave=False)
                    val_recon_losses = []
                    val_perceptual_losses = []
                    val_gan_losses = []

                    for j, val_batch in enumerate(val_progress_bar):
                        val_image, _val_label = val_batch
                        val_image = val_image.to(device)

                        with ctx:
                            val_xhat, _, idx_R_BL, _, _ = vqvae(val_image)
                            val_recon_loss = F.mse_loss(val_xhat, val_image)
                            val_perceptual_loss = lpips_model(val_xhat, val_image).mean()
                            fake_preds = gan(val_xhat)
                            real_preds = gan(val_image)
                        idxs = [x.flatten() for x in idx_R_BL]
                        idxs = torch.cat(idxs, dim=0)

                        unique_codes = torch.unique(idxs)
                        codebook_usage = len(unique_codes) / vqvae.module.quantizer.vocab_size
                        code_histogram = torch.bincount(idxs.cpu(), minlength=vqvae.module.quantizer.vocab_size)
                        code_entropy = -(code_histogram.float() / len(idxs)) * torch.log2(code_histogram.float() / len(idxs) + 1e-10)
                        code_entropy = code_entropy.sum()

                        val_recon_losses.append(val_recon_loss.item())
                        val_perceptual_losses.append(val_perceptual_loss.item())

                        fake_correct = (fake_preds < 0).float().mean()
                        real_correct = (real_preds > 0).float().mean()
                        val_gan_acc = (fake_correct + real_correct) / 2
                        val_gan_losses.append(val_gan_acc.item())

                        for b in range(val_image.size(0)):
                            gen_path = f"images/val_images/generated/epoch_{epoch}_iter_{i}_val_{j}_batch_{b}_{ddp_local_rank}.png"
                            orig_path = f"images/val_images/original/epoch_{epoch}_iter_{i}_val_{j}_batch_{b}_{ddp_local_rank}.png"
                            torchvision.utils.save_image(val_xhat[b], gen_path)
                            torchvision.utils.save_image(val_image[b], orig_path)

                    val_recon_loss_tensor = torch.tensor(np.mean(val_recon_losses)).to(device)
                    val_perceptual_loss_tensor = torch.tensor(np.mean(val_perceptual_losses)).to(device)
                    val_gan_acc_tensor = torch.tensor(np.mean(val_gan_losses)).to(device)

                    dist.all_reduce(val_recon_loss_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(val_perceptual_loss_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(val_gan_acc_tensor, op=dist.ReduceOp.SUM)

                    val_recon_loss_avg = val_recon_loss_tensor.item() / dist.get_world_size()
                    val_perceptual_loss_avg = val_perceptual_loss_tensor.item() / dist.get_world_size()
                    val_gan_acc_avg = val_gan_acc_tensor.item() / dist.get_world_size()

                    dist.barrier()

                    if is_master_process:
                        fid_score = calculate_fid_given_paths(
                            ["images/val_images/generated", "images/val_images/original"],
                            args.val_batch_size,
                            device,
                        )
                        print(f"FID score: {fid_score}")
                        print(f"Validation recon loss: {val_recon_loss_avg}")
                        print(f"Validation perceptual loss: {val_perceptual_loss_avg}")
                        print(f"Validation gan accuracy: {val_gan_acc_avg}")
                        print(f"Codebook usage: {codebook_usage:.2%}")
                        print(f"Code entropy: {code_entropy:.2f}")

                        wandb.log(
                            {
                                "fid_score": fid_score,
                                "val_recon_loss": val_recon_loss_avg,
                                "val_perceptual_loss": val_perceptual_loss_avg,
                                "val_gan_accuracy": val_gan_acc_avg,
                                "codebook_usage": codebook_usage,
                                "code_entropy": code_entropy,
                                "code_histogram": wandb.Histogram(code_histogram.numpy()),
                            }
                        )

                        for path in [f"images/val_images/generated", f"images/val_images/original"]:
                            for img in os.listdir(path):
                                os.remove(os.path.join(path, img))

                        plot_images(xhat, image)
                        plt.savefig(f"images/epoch_{epoch}_iter_{i}.png")
                        plt.close()
                        wandb.log({"image": wandb.Image(f"images/epoch_{epoch}_iter_{i}.png")})

    dist.destroy_process_group()


if __name__ == "__main__":
    #  uv run torchrun --nproc-per-node 2 train_vqvae.py --run_name vqvae_1
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str)
    args = parser.parse_args()
    train(args.run_name)
