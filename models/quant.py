import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List


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
        self.codebook.weight.data = F.normalize(self.codebook.weight.data, dim=1)

    def forward(self, f_BCHW: torch.Tensor):
        r_R_BChw, idx_R_BL, zqs_post_conv_R_BCHW = self.encode(f_BCHW)
        f_hat_BCHW, scales_BLC, loss = self.decode(f_BCHW, zqs_post_conv_R_BCHW)
        return f_hat_BCHW, r_R_BChw, idx_R_BL, scales_BLC, loss

    def encode(self, f_BCHW: torch.Tensor):
        B, C, H, W = f_BCHW.shape
        r_R_BChw = []
        idx_R_BL = []
        zqs_post_conv_R_BCHW = []
        codebook = F.normalize(self.codebook.weight, dim=1)
        for resolution_idx, resolution_k in enumerate(self.resolutions):
            r_BChw = F.interpolate(f_BCHW, (resolution_k, resolution_k), mode="area")
            r_flattened_NC = r_BChw.permute(0, 2, 3, 1).reshape(-1, self.dim).contiguous()
            # dist = r_flattened_NC.pow(2).sum(1, keepdim=True) + self.codebook.weight.data.pow(2).sum(1) - 2 * r_flattened_NC @ self.codebook.weight.data.T
            r_flattened_NC = F.normalize(r_flattened_NC, dim=1)
            dist = r_flattened_NC @ codebook.T
            idx_Bhw = torch.argmax(dist, dim=1).view(B, resolution_k, resolution_k)

            # idx_Bhw = torch.argmin(dist, dim=1).view(B, resolution_k, resolution_k)
            idx_R_BL.append(idx_Bhw.reshape(B, -1))
            r_R_BChw.append(r_BChw)

            # zq_BChw = self.codebook(idx_Bhw).permute(0, 3, 1, 2).contiguous()
            zq_BChw = codebook[idx_Bhw].reshape(B, resolution_k, resolution_k, self.dim).permute(0, 3, 1, 2).contiguous()
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
                scales.append(F.interpolate(f_hat_BCHW, (next_size, next_size), mode="area").flatten(-2).transpose(1, 2).contiguous())

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
