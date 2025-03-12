import os
import random
from dataclasses import dataclass

import lpips
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
from utils.gan import NLayerDiscriminator, calculate_adaptive_weight
from utils.imagenet_dataset import get_imagenet_dataloader


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@dataclass
class TrainConfig:
    train_batch_size: int = 16
    val_batch_size: int = 128
    val_interval: int = 1000
    seed: int = 42

    recon_loss_weight: float = 1.0
    perceptual_loss_weight: float = 1.0
    q_loss_weight: float = 0.1
    gan_loss_weight: float = 1


class GradNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(weight)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        weight = ctx.saved_tensors[0]

        # grad_output_norm = torch.linalg.vector_norm(
        #     grad_output, dim=list(range(1, len(grad_output.shape))), keepdim=True
        # ).mean()
        grad_output_norm = torch.norm(grad_output).mean().item()
        # nccl over all nodes
        grad_output_norm = avg_scalar_over_nodes(grad_output_norm, device=grad_output.device)

        grad_output_normalized = weight * grad_output / (grad_output_norm + 1e-8)

        return grad_output_normalized, None


@torch.no_grad()
def avg_scalar_over_nodes(value: float, device):
    value = torch.tensor(value, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.AVG)
    return value.item()


def gradnorm(x, weight=1.0):
    weight = torch.tensor(weight, device=x.device)
    return GradNormFunction.apply(x, weight)


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


def train():
    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{ddp_local_rank}")
    is_master_process = ddp_rank == 0
    print(f"DDP rank: {ddp_rank}, world size: {ddp_world_size}, local rank: {ddp_local_rank}")

    args = TrainConfig()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    vqvae_config = VQVAEConfig(
        resolution=256,
        in_channels=3,
        dim=128,
        ch_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        z_channels=64,
        out_ch=3,
        vocab_size=8192,
        patch_sizes=[1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 32],
    )
    vqvae = VQVAE(vqvae_config).to(device)
    if is_master_process:
        print("Number of parameters: ", sum(p.numel() for p in vqvae.parameters()) / 1e6, "M")
        wandb.init(project="var")

    vqvae = DDP(vqvae, device_ids=[ddp_local_rank])
    lpips_model = lpips.LPIPS(net="vgg").to(device).eval()
    for param in lpips_model.parameters():
        param.requires_grad = False

    gan = NLayerDiscriminator().to(device)
    gan = DDP(gan, device_ids=[ddp_local_rank])
    train_loader, val_loader = get_imagenet_dataloader(ddp_rank, ddp_world_size, train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size, num_workers=16)

    optimizer_vqvae = torch.optim.AdamW(vqvae.parameters(), lr=1e-5)
    optimizer_gan = torch.optim.AdamW(gan.parameters(), lr=1e-5)

    losses = []
    for epoch in range(5):
        progress_bar = tqdm(train_loader, disable=not is_master_process)
        for i, batch in enumerate(progress_bar):
            image, _label = batch
            image = image.to(device)
            xhat, r_maps, idxs, scales, q_loss = vqvae(image)

            real_pred = gan(image)
            fake_pred = gan(xhat.detach())

            optimizer_gan.zero_grad()
            real_loss = F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred))
            fake_loss = F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred))
            gan_discriminator_loss = (real_loss + fake_loss) / 2
            gan_discriminator_loss.backward()
            optimizer_gan.step()

            optimizer_vqvae.zero_grad()

            recon_loss = F.mse_loss(xhat, image)
            perceptual_loss = lpips_model(xhat, image).mean()
            fake_pred = gan(gradnorm(xhat, weight=0.001))
            gan_loss = F.binary_cross_entropy_with_logits(fake_pred, torch.ones_like(fake_pred))

            adaptive_weight = calculate_adaptive_weight(vqvae.module.decoder.conv_out.weight, perceptual_loss, gan_loss)
            loss = args.recon_loss_weight * recon_loss + args.perceptual_loss_weight * perceptual_loss + args.q_loss_weight * q_loss + args.gan_loss_weight * adaptive_weight * gan_loss
            loss.backward()
            optimizer_vqvae.step()

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
                        val_xhat, _, _, _, _ = vqvae(val_image)

                        val_recon_loss = F.mse_loss(val_xhat, val_image)
                        val_perceptual_loss = lpips_model(val_xhat, val_image).mean()
                        val_recon_losses.append(val_recon_loss.item())
                        val_perceptual_losses.append(val_perceptual_loss.item())

                        fake_preds = gan(val_xhat)
                        real_preds = gan(val_image)
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

                        wandb.log({"fid_score": fid_score, "val_recon_loss": val_recon_loss_avg, "val_perceptual_loss": val_perceptual_loss_avg, "val_gan_accuracy": val_gan_acc_avg})

                        for path in [f"images/val_images/generated", f"images/val_images/original"]:
                            for img in os.listdir(path):
                                os.remove(os.path.join(path, img))

                        plot_images(xhat, image)
                        plt.savefig(f"images/epoch_{epoch}_iter_{i}.png")
                        plt.close()
                        wandb.log({"image": wandb.Image(f"images/epoch_{epoch}_iter_{i}.png")})

    dist.destroy_process_group()


if __name__ == "__main__":
    #  uv run torchrun --nproc-per-node 2 train_vqvae.py
    train()
