import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def compute_freqs(theta: float, d_k: int, max_seq_len: int, device=None) -> torch.Tensor:
    thetas = theta ** -((2 * torch.arange(0, d_k // 2, device=device)) / d_k)
    pos_ids = torch.arange(0, max_seq_len, device=device)
    angles = torch.einsum("h,c->ch", thetas, pos_ids)
    return torch.stack([torch.sin(angles), torch.cos(angles)])


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    sin, cos = freqs[0], freqs[1]

    x_reshaped = rearrange(x, "... (half_d even_odd) -> even_odd ... half_d", even_odd=2)
    x_even_orig = x_reshaped[0]
    x_odd_orig = x_reshaped[1]

    x_even = x_even_orig * cos - x_odd_orig * sin
    x_odd = x_odd_orig * cos + x_even_orig * sin
    return rearrange([x_even, x_odd], "n ... d_half -> ... (d_half n)")


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.d_model = d_model
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt((1 / self.d_model * x.pow(2).sum(-1, keepdim=True) + self.eps))
        return ((x / rms) * self.weight).to(in_dtype)


class AttnBlock4D(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x_BCHW: Tensor) -> Tensor:
        x_BCHW = self.norm(x_BCHW)
        q_BCHW = self.q(x_BCHW)
        k_BCHW = self.k(x_BCHW)
        v_BCHW = self.v(x_BCHW)

        B, C, H, W = x_BCHW.shape
        q_B1HWC = rearrange(q_BCHW, "b c h w -> b 1 (h w) c").contiguous()
        k_B1HWC = rearrange(k_BCHW, "b c h w -> b 1 (h w) c").contiguous()
        v_B1HWC = rearrange(v_BCHW, "b c h w -> b 1 (h w) c").contiguous()
        h_B1HWC = F.scaled_dot_product_attention(q_B1HWC, k_B1HWC, v_B1HWC)
        h_BCHW = rearrange(h_B1HWC, "b 1 (h w) c -> b c h w", h=H, w=W, c=C, b=B).contiguous()
        return x_BCHW + self.proj_out(h_BCHW)


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.wqkv = nn.Linear(dim, dim * 3, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        self.q_norm = RMSNorm(dim)
        self.k_norm = RMSNorm(dim)

    def forward(
        self,
        x_BLD: torch.Tensor,
        freqs: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, _ = x_BLD.shape
        dtype = x_BLD.dtype

        qkv_BLD = self.wqkv(x_BLD)
        xq_BLD, xk_BLD, xv_BLD = qkv_BLD.chunk(3, dim=-1)

        xq_BLD = self.q_norm(xq_BLD)
        xk_BLD = self.k_norm(xk_BLD)

        xq_BLD = apply_rope(xq_BLD, freqs).to(dtype)
        xk_BLD = apply_rope(xk_BLD, freqs).to(dtype)

        xq_BHLK = rearrange(xq_BLD, "b l (h d) -> b h l d", h=self.n_heads)
        xk_BHLK = rearrange(xk_BLD, "b l (h d) -> b h l d", h=self.n_heads)
        xv_BHLK = rearrange(xv_BLD, "b l (h d) -> b h l d", h=self.n_heads)

        out_BHLK = F.scaled_dot_product_attention(xq_BHLK, xk_BHLK, xv_BHLK, attn_mask=attn_mask)
        out_BLD = rearrange(out_BHLK, "b h l d -> b l (h d)")
        return self.wo(out_BLD)


class FeedForwardGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim * 4
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x_BLD: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x_BLD)) * self.w3(x_BLD))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.ff1 = nn.Linear(dim, hidden_dim)
        self.act = nn.SiLU()
        self.ff2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        out = self.ff1(x)
        out = self.act(out)
        out = self.drop(out)
        out = self.ff2(out)
        out = self.drop(out)
        return out


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    B, C, H, W = x.shape
    return rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)


def unpatchify(x: torch.Tensor, height: int, width: int, patch_size: int) -> torch.Tensor:
    B, L, D = x.shape
    patch_height = height // patch_size
    patch_width = width // patch_size

    return rearrange(x, "b (h w) (c p1 p2) -> b c (h p1) (w p2)", h=patch_height, w=patch_width, p1=patch_size, p2=patch_size)
