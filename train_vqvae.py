import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid
from tqdm import tqdm

from models.vqvae import VQVAE, VQVAEConfig
from utils.imagenet_dataset import get_imagenet_dataloader

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


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
    plt.figure(figsize=(12, 12))
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

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

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
    vqvae = DDP(vqvae, device_ids=[ddp_local_rank])
    # TODO: torch.compile(vqvae)

    train_loader, val_loader = get_imagenet_dataloader(ddp_rank, ddp_world_size, batch_size=16, num_workers=16)
    optimizer = torch.optim.AdamW(vqvae.parameters(), lr=1e-4)

    losses = []
    for epoch in range(5):
        progress_bar = tqdm(train_loader, disable=not is_master_process)
        for i, batch in enumerate(progress_bar):
            image, _label = batch
            image = image.to(device)
            xhat, r_maps, idxs, scales, q_loss = vqvae(image)
            recon_loss = F.mse_loss(xhat, image)
            loss = recon_loss + q_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if is_master_process:
                progress_bar.set_description(f"Epoch {epoch}")
                progress_bar.set_postfix(loss=loss.item(), recon_loss=recon_loss.item(), q_loss=q_loss.item())
                losses.append(loss.item())

                if i % 500 == 0:
                    plot_images(xhat, image)
                    plt.savefig(f"images/epoch_{epoch}_iter_{i}.png")

    dist.destroy_process_group()

    if is_master_process:
        plt.plot(losses)
        plt.savefig("losses.png")


if __name__ == "__main__":
    train()
