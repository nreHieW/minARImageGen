import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import gc
import wandb
import datetime

from modelling.vqvae import VQVAE
from modelling.var import VAR


model_params = {
    "mnist": {
        "VQVAE_DIM": 64,
        "VOCAB_SIZE": 32,
        "PATCH_SIZES": [1, 2, 3, 4, 8],
        "VAR_DIM": 64,
        "N_HEADS": 4,
        "N_LAYERS": 6,
        "channels": 1,
    },
    "cifar": {
        "VQVAE_DIM": 128,
        "VOCAB_SIZE": 512,
        "PATCH_SIZES": [1, 2, 3, 4, 6, 8],
        "VAR_DIM": 256,
        "N_HEADS": 8,
        "N_LAYERS": 11,
        "channels": 3,
    },
}

training_params = {
    "mnist": {
        "VQVAE": {
            "batch_size": 2048,
            "lr": 3e-4,
            "epochs": 40,
        },
        "VAR": {
            "batch_size": 1024,
            "lr": 1e-3,
            "epochs": 100,
        },
    },
    "cifar": {
        "VQVAE": {
            "batch_size": 2048,
            "lr": 3e-4,
            "epochs": 60,
        },
        "VAR": {
            "batch_size": 512,
            "lr": 2e-3,
            "epochs": 100,
        },
    },
}


def get_data(batch_size=1024, use_cifar=False):
    if use_cifar:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    else:
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
    plt.imshow(img_grid)
    plt.axis("off")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cifar", action="store_true")
    args = parser.parse_args()
    use_cifar = args.cifar

    model_params = model_params["cifar"] if use_cifar else model_params["mnist"]
    training_params = training_params["cifar"] if use_cifar else training_params["mnist"]

    curr_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run = wandb.init(
        project="minvar",
        name=f"minvar_{curr_time}" + ("_cifar" if use_cifar else "mnist"),
        config=model_params,
    )
    print("=" * 10 + "Training VQVAE" + "=" * 10)
    vq_model = VQVAE(model_params["VQVAE_DIM"], model_params["VOCAB_SIZE"], model_params["PATCH_SIZES"], num_channels=model_params["channels"])
    optimizer = torch.optim.AdamW(vq_model.parameters(), lr=training_params["VQVAE"]["lr"])

    train_loader, test_loader = get_data(batch_size=training_params["VQVAE"]["batch_size"], use_cifar=use_cifar)
    vq_model = vq_model.to("cuda")
    for epoch in range(training_params["VQVAE"]["epochs"]):
        epoch_loss = 0
        epoch_recon_loss = 0
        for i, (x, c) in enumerate(train_loader):
            x, c = x.cuda(), c.cuda()
            optimizer.zero_grad()
            xhat, r_maps, idxs, scales, q_loss = vq_model(x)
            recon_loss = F.mse_loss(xhat, x)
            loss = recon_loss + q_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()

        epoch_loss /= len(train_loader)
        epoch_recon_loss /= len(train_loader)
        print(f"Epoch: {epoch}, Loss: {epoch_loss}, Recon Loss: {epoch_recon_loss}")
        wandb.log({"vqvae_train_loss": epoch_loss, "vqvae_recon_loss": epoch_recon_loss})

        if epoch % 5 == 0:
            with torch.no_grad():
                total_loss = 0
                total_recon_loss = 0
                for i, (x, c) in enumerate(test_loader):
                    x, c = x.cuda(), c.cuda()
                    xhat, r_maps, idxs, scales, q_loss = vq_model(x)
                    recon_loss = F.mse_loss(xhat, x)
                    loss = recon_loss + q_loss
                    total_loss += loss.item()
                    total_recon_loss += recon_loss.item()

                total_loss /= len(test_loader)
                total_recon_loss /= len(test_loader)

                print(f"Epoch: {epoch}, Test Loss: {total_loss}, Test Recon Loss: {total_recon_loss}")
                wandb.log({"vqvae_test_loss": total_loss, "vqvae_test_recon_loss": total_recon_loss})

                x = x[:10, :].cuda()
                x_hat = vq_model(x)[0]

                plot_images(pred=x_hat, original=x)
                plt.savefig(f"vqvae_{epoch}.png")
                plt.close()
                wandb.log({"vqvae_images": wandb.Image(f"vqvae_{epoch}.png")})

    torch.save(vq_model.state_dict(), "vqvae.pth")
    wandb.save("vqvae.pth")
    del vq_model, optimizer, x, x_hat, train_loader, test_loader
    gc.collect()
    torch.cuda.empty_cache()

    print("=" * 10 + "Training VAR" + "=" * 10)
    vqvae = VQVAE(model_params["VQVAE_DIM"], model_params["VOCAB_SIZE"], model_params["PATCH_SIZES"], num_channels=model_params["channels"])
    vqvae.load_state_dict(torch.load("vqvae.pth"))
    vqvae = vqvae.to("cuda")
    vqvae.eval()

    for param in vqvae.parameters():
        param.requires_grad = False

    var_model = VAR(vqvae=vqvae, dim=model_params["VAR_DIM"], n_heads=model_params["N_HEADS"], n_layers=model_params["N_LAYERS"], patch_sizes=model_params["PATCH_SIZES"], n_classes=10)
    optimizer = torch.optim.AdamW(var_model.parameters(), lr=training_params["VAR"]["lr"])

    print(f"VQVAE Parameters: {sum(p.numel() for p in vqvae.parameters())/1e6:.2f}M")
    print(f"VAR Parameters: {sum(p.numel() for p in var_model.parameters())/1e6:.2f}M")

    train_loader, test_loader = get_data(batch_size=training_params["VAR"]["batch_size"], use_cifar=use_cifar)
    var_model = var_model.to("cuda")
    for epoch in range(training_params["VAR"]["epochs"]):
        epoch_loss = 0
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

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        print(f"Epoch: {epoch}, Loss: {epoch_loss}")
        wandb.log({"var_train_loss": epoch_loss})

        if epoch % 5 == 0:
            with torch.no_grad():

                cond = torch.arange(10).cuda()
                out_B3HW = var_model.generate(cond, 0)
                plot_images(pred=out_B3HW)

                plt.savefig(f"var_{epoch}.png")
                plt.close()
                wandb.log({"var_images": wandb.Image(f"var_{epoch}.png")})

    torch.save(var_model.state_dict(), "var.pth")
    wandb.save("var.pth")
