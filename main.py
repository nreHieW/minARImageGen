import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import gc
import wandb
import datetime

from vqvae import VQVAE
from var import VAR

VQVAE_DIM = 64
VOCAB_SIZE = 32
PATCH_SIZES = [1, 2, 3, 4, 8]
VAR_DIM = 128
N_HEADS = 4
N_LAYERS = 6


def get_data(batch_size=1024):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Pad(2),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader


if __name__ == "__main__":
    curr_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run = wandb.init(
        project="minvar",
        name=f"minvar_{curr_time}",
        config={
            "VQVAE_DIM": VQVAE_DIM,
            "VOCAB_SIZE": VOCAB_SIZE,
            "PATCH_SIZES": PATCH_SIZES,
            "VAR_DIM": VAR_DIM,
            "N_HEADS": N_HEADS,
            "N_LAYERS": N_LAYERS,
        },
    )
    print("=" * 10 + "Training VQVAE" + "=" * 10)
    vq_model = VQVAE(VQVAE_DIM, VOCAB_SIZE, PATCH_SIZES, num_channels=1)
    optimizer = torch.optim.AdamW(vq_model.parameters(), lr=5e-4)

    train_loader, test_loader = get_data(batch_size=8192)
    vq_model = vq_model.to("cuda")
    for epoch in range(75):
        for i, (x, c) in enumerate(train_loader):
            x, c = x.cuda(), c.cuda()
            optimizer.zero_grad()
            xhat, r_maps, idxs, scales, q_loss = vq_model(x)
            loss = F.mse_loss(xhat, x) + q_loss
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}")
                wandb.log({"vqvae_train_loss": loss.item()})

        if epoch % 5 == 0:
            with torch.no_grad():
                total_loss = 0
                for i, (x, c) in enumerate(test_loader):
                    x, c = x.cuda(), c.cuda()
                    xhat, r_maps, idxs, scales, q_loss = vq_model(x)
                    loss = F.mse_loss(xhat, x) + q_loss
                    total_loss += loss.item()

                print(f"Epoch: {epoch}, Test Loss: {total_loss / len(test_loader)}")
                wandb.log({"vqvae_test_loss": total_loss / len(test_loader)})

                x = x[:10, :].cuda()
                x_hat = vq_model(x)[0]

                x = x.cpu()
                x_hat = x_hat * 0.5 + 0.5
                x_hat = x_hat.clamp(0, 1)
                x_hat = x_hat.cpu().detach().numpy()
                fig, axs = plt.subplots(2, 10, figsize=(10, 2))
                for i in range(10):
                    axs[0, i].imshow(x[i, 0], cmap="gray")
                    axs[1, i].imshow(x_hat[i, 0], cmap="gray")

                fig.savefig(f"vqvae_{epoch}.png")
                plt.close(fig)
                wandb.log({"vqvae_images": wandb.Image(f"vqvae_{epoch}.png")})

    torch.save(vq_model.state_dict(), "vqvae.pth")
    wandb.save("vqvae.pth")
    del vq_model, optimizer, x, x_hat, train_loader, test_loader
    gc.collect()
    torch.cuda.empty_cache()

    print("=" * 10 + "Training VAR" + "=" * 10)
    vqvae = VQVAE(VQVAE_DIM, VOCAB_SIZE, PATCH_SIZES, num_channels=1)
    vqvae.load_state_dict(torch.load("vqvae.pth"))
    vqvae = vqvae.to("cuda")
    vqvae.eval()

    for param in vqvae.parameters():
        param.requires_grad = False

    var_model = VAR(vqvae, dim=VAR_DIM, n_heads=N_HEADS, n_layers=N_LAYERS, patch_sizes=PATCH_SIZES, n_classes=10)
    optimizer = torch.optim.AdamW(var_model.parameters(), lr=3e-4)

    train_loader, test_loader = get_data(batch_size=1024)
    var_model = var_model.to("cuda")

    for epoch in range(100):
        for i, (x, c) in enumerate(train_loader):
            x, c = x.cuda(), c.cuda()
            optimizer.zero_grad()

            _, _, idxs_R_BL, scales_BlC, _ = vqvae(x)
            idx_BL = torch.cat(idxs_R_BL, dim=1)
            scales_BlC = scales_BlC.cuda()
            logits_BLV = var_model(scales_BlC, cond=c)
            loss = F.cross_entropy(logits_BLV.view(-1, logits_BLV.size(-1)), idx_BL.view(-1))

            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}")
                wandb.log({"var_train_loss": loss.item()})

        if epoch % 5 == 0:
            with torch.no_grad():
                cond = torch.randint(0, 10, (10,)).cuda()
                out_B3HW = var_model.generate(cond, 6)
                out_B3HW = out_B3HW * 0.5 + 0.5
                out_B3HW = out_B3HW.clamp(0, 1)
                out_B3HW = out_B3HW.cpu().detach().numpy()

                fig, axs = plt.subplots(1, 10, figsize=(10, 1))
                for i in range(10):
                    axs[i].imshow(out_B3HW[i, 0], cmap="gray")

                fig.savefig(f"var_{epoch}.png")
                plt.close(fig)
                wandb.log({"var_images": wandb.Image(f"var_{epoch}.png")})

    torch.save(var_model.state_dict(), "var.pth")
    wandb.save("var.pth")
