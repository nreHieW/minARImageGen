# References: https://github.com/black-forest-labs/flux/blob/main/src/flux/modules/autoencoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass
from .layers import AttnBlock4D, swish
from .quant import VectorQuantizer


@dataclass
class VQVAEConfig:
    in_channels: int
    dim: int
    ch_mult: list[int]
    num_res_blocks: int
    z_channels: int
    codebook_dim: int
    out_ch: int
    vocab_size: int
    patch_sizes: list[int]


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor):
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn.append(AttnBlock4D(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
            self.down.append(down)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock4D(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, z_channels, kernel_size=3, stride=1, padding=1)
        self.encoder_output_dim = block_in

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        z_channels: int,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        block_in = ch * ch_mult[self.num_resolutions - 1]

        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock4D(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn.append(AttnBlock4D(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
            self.up.insert(0, up)  # prepend to get consistent order

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class VQVAE(nn.Module):
    def __init__(self, config: VQVAEConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(in_channels=config.in_channels, ch=config.dim, ch_mult=config.ch_mult, num_res_blocks=config.num_res_blocks, z_channels=config.z_channels)
        self.decoder = Decoder(
            ch=config.dim,
            out_ch=config.out_ch,
            ch_mult=config.ch_mult,
            num_res_blocks=config.num_res_blocks,
            in_channels=config.in_channels,
            z_channels=config.z_channels,
        )
        self.quantizer = VectorQuantizer(vocab_size=config.vocab_size, dim=config.codebook_dim, patch_sizes=config.patch_sizes)
        self.quant_conv = nn.Conv2d(config.z_channels, config.codebook_dim, kernel_size=1, stride=1, padding=0)
        self.post_quant_conv = nn.Conv2d(config.codebook_dim, config.z_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        f = self.encoder(x)
        f = self.quant_conv(f)
        fhat, r_maps, idxs, scales, loss = self.quantizer(f)
        fhat = self.post_quant_conv(fhat)
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
    config = VQVAEConfig(
        in_channels=3,
        dim=128,
        ch_mult=[1, 1, 2, 2, 4],
        num_res_blocks=2,
        z_channels=256,
        codebook_dim=8,
        out_ch=3,
        vocab_size=16384,
        patch_sizes=patch_sizes,
    )

    model = VQVAE(config)
    print(model)
    image = torch.randn((1, 3, 256, 256), requires_grad=True)
    xhat, r_maps, idxs, scales, loss = model(image)
    total_tokens = sum(len(idx) for idx in idxs)
    assert xhat.shape == image.shape, f"Expected shape {image.shape} but got {xhat.shape}"
    assert len(r_maps) == len(idxs) == len(patch_sizes)
    loss = loss + F.mse_loss(xhat, torch.randn_like(xhat))
    loss.backward()
    assert image.grad is not None
    for n, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {n} has no gradient"
    print("Success")
    print("Number of parameters:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
