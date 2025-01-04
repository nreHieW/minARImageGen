# Encoder and Decoder from https://github.com/karpathy/deep-vector-quantization/blob/main/dvq/model/deepmind_enc_dec.py
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResBlock(nn.Module):
    def __init__(self, input_channels, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, input_channels, 1),
        )

    def forward(self, x_BCHW):
        out = self.conv(x_BCHW)
        out += x_BCHW
        out = F.relu(out)
        return out


class Encoder(nn.Module):

    def __init__(self, input_channels=3, n_hid=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, n_hid, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_hid, 2 * n_hid, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * n_hid, 2 * n_hid, 3, padding=1),
            nn.ReLU(),
            ResBlock(2 * n_hid, 2 * n_hid // 4),
            ResBlock(2 * n_hid, 2 * n_hid // 4),
        )

        self.output_channels = 2 * n_hid
        self.output_stide = 4

    def forward(self, x_BCHW):
        return self.net(x_BCHW)


class Decoder(nn.Module):

    def __init__(self, n_init=32, n_hid=64, output_channels=3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(n_init, 2 * n_hid, 3, padding=1),
            nn.ReLU(),
            ResBlock(2 * n_hid, 2 * n_hid // 4),
            ResBlock(2 * n_hid, 2 * n_hid // 4),
            nn.ConvTranspose2d(2 * n_hid, n_hid, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(n_hid, output_channels, 4, stride=2, padding=1),
        )

    def forward(self, x_BCHW):
        return self.net(x_BCHW)


class Phi(nn.Module):
    def __init__(self, dim, residual_ratio: float = 0.5):
        super().__init__()
        self.residual_ratio = residual_ratio
        self.conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1)

    def forward(self, h_BChw):
        return (1 - self.residual_ratio) * h_BChw + self.residual_ratio * self.conv(h_BChw)


class PhiPartiallyShared(nn.Module):
    def __init__(self, dim, residual_ratio: float = 0.5, num_phi: int = 4):
        super().__init__()
        self.phis = nn.ModuleList([Phi(dim, residual_ratio) for _ in range(num_phi)])
        self.num_phi = num_phi
        if self.num_phi == 4:
            self.ticks = np.linspace(1 / 3 / self.num_phi, 1 - 1 / 3 / self.num_phi, self.num_phi)
        else:
            self.ticks = np.linspace(1 / 2 / self.num_phi, 1 - 1 / 2 / self.num_phi, self.num_phi)

    def forward(self, x: torch.Tensor, idx_ratio: float) -> Phi:
        return self.phis[np.argmin(np.abs(self.ticks - idx_ratio)).item()](x)


class VectorQuantizer(nn.Module):
    def __init__(self, vocab_size: int, dim: int, patch_sizes: List[int], residual_ratio: float = 0.5, num_phi: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.resolutions = patch_sizes
        self.phis = PhiPartiallyShared(dim, residual_ratio, num_phi)
        self.codebook = nn.Embedding(self.vocab_size, dim)
        self.codebook.weight.data.uniform_(-1 / self.vocab_size, 1 / self.vocab_size)

    def forward(self, f_BCHW: torch.Tensor):
        r_R_BChw, idx_R_BL, zqs_post_conv_R_BCHW = self.encode(f_BCHW)
        f_hat_BCHW, scales_BLC, loss = self.decode(f_BCHW, zqs_post_conv_R_BCHW)
        return f_hat_BCHW, r_R_BChw, idx_R_BL, scales_BLC, loss

    def encode(self, f_BCHW: torch.Tensor):
        B, C, H, W = f_BCHW.shape
        r_R_BChw = []
        idx_R_BL = []
        zqs_post_conv_R_BCHW = []
        for resolution_idx, resolution_k in enumerate(self.resolutions):
            r_BChw = F.interpolate(f_BCHW, (resolution_k, resolution_k), mode="area")
            r_flattened_NC = r_BChw.permute(0, 2, 3, 1).reshape(-1, self.dim)
            dist = r_flattened_NC.pow(2).sum(1, keepdim=True) + self.codebook.weight.data.pow(2).sum(1) - 2 * r_flattened_NC @ self.codebook.weight.data.T

            idx_Bhw = torch.argmin(dist, dim=1).view(B, resolution_k, resolution_k)
            idx_R_BL.append(idx_Bhw.reshape(B, -1))
            r_R_BChw.append(r_BChw)

            zq_BChw = self.codebook(idx_Bhw).permute(0, 3, 1, 2)
            zq_BCHW = F.interpolate(zq_BChw, size=(H, W), mode="bicubic")
            phi_idx = resolution_idx / (len(self.resolutions) - 1)
            zq_BCHW = self.phis(zq_BCHW, phi_idx)
            zqs_post_conv_R_BCHW.append(zq_BCHW)

            f_BCHW = f_BCHW - zq_BCHW

        return r_R_BChw, idx_R_BL, zqs_post_conv_R_BCHW

    def decode(self, f_BCHW: torch.Tensor, zqs_post_conv_R_BCHW: torch.Tensor):
        f_hat_BCHW = torch.zeros_like(f_BCHW)
        loss = 0
        scales = []  # this is for the teacher forcing input so doesnt include the first scale
        for resolution_idx, resolution_k in enumerate(self.resolutions):
            zq_BCHW = zqs_post_conv_R_BCHW[resolution_idx]
            f_hat_BCHW = f_hat_BCHW + zq_BCHW
            if resolution_idx < len(self.resolutions) - 1:
                next_size = self.resolutions[resolution_idx + 1]
                scales.append(F.interpolate(f_hat_BCHW, (next_size, next_size), mode="area").flatten(-2).transpose(1, 2))

            commitment_loss = torch.mean((f_hat_BCHW.detach() - f_BCHW) ** 2)
            codebook_loss = torch.mean((f_hat_BCHW - f_BCHW.detach()) ** 2)
            loss += codebook_loss + 0.25 * commitment_loss

        loss /= len(self.resolutions)
        f_hat_BCHW = f_BCHW + (f_hat_BCHW - f_BCHW).detach()
        return f_hat_BCHW, torch.cat(scales, dim=1), loss

    def get_next_autoregressive_input(self, idx: int, f_hat_BCHW: torch.Tensor, h_BChw: torch.Tensor):
        final_patch_size = self.resolutions[-1]
        h_BCHW = F.interpolate(h_BChw, (final_patch_size, final_patch_size), mode="bicubic")
        h_BCHW = self.phis(h_BCHW, idx / (len(self.resolutions) - 1))
        f_hat_BCHW = f_hat_BCHW + h_BCHW
        return f_hat_BCHW


class VQVAE(nn.Module):
    def __init__(self, dim: int, vocab_size: int, patch_sizes: list, num_channels: int = 3):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.patch_sizes = patch_sizes
        self.encoder = Encoder(num_channels, n_hid=dim)
        self.decoder = Decoder(n_init=self.encoder.output_channels, n_hid=dim, output_channels=num_channels)
        self.quantizer = VectorQuantizer(vocab_size=vocab_size, dim=self.encoder.output_channels, patch_sizes=patch_sizes)
        self.latent_channels = self.encoder.output_channels

    def forward(self, x):
        f = self.encoder(x)
        fhat, r_maps, idxs, scales, loss = self.quantizer(f)
        x_hat = self.decoder(fhat)
        return x_hat, r_maps, idxs, scales, loss

    def get_nearest_embedding(self, idxs):
        return self.quantizer.codebook(idxs)

    def get_next_autoregressive_input(self, idx, f_hat_BCHW, h_BChw):
        return self.quantizer.get_next_autoregressive_input(idx, f_hat_BCHW, h_BChw)

    def to_img(self, f_hat_BCHW):
        return self.decoder(f_hat_BCHW).clamp(-1, 1)

    def img_to_indices(self, x):
        f = self.encoder(x)
        fhat, r_maps, idxs, scales, loss = self.quantizer(f)
        return idxs


if __name__ == "__main__":
    patch_sizes = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 32]
    model = VQVAE(64, 10, patch_sizes, num_channels=1)
    image = torch.randn((1, 1, 32, 32), requires_grad=True)
    xhat, r_maps, idxs, scales, loss = model(image)
    assert xhat.shape == image.shape, f"Expected shape {image.shape} but got {xhat.shape}"
    assert len(r_maps) == len(idxs) == len(patch_sizes)
    loss = loss + F.mse_loss(xhat, torch.randn_like(xhat))
    loss.backward()
    assert image.grad is not None
    for param in model.parameters():
        assert param.grad is not None
    print("Success")
