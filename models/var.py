# Layers based on https://github.com/cloneofsimo/minRF/blob/main/dit.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from modelling.vqvae import VQVAE


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    _freqs_cis = freqs_cis[: x.shape[1]]
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return _freqs_cis.view(*shape)


def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis_xq = reshape_for_broadcast(freqs_cis, xq_)
    freqs_cis_xk = reshape_for_broadcast(freqs_cis, xk_)

    xq_out = torch.view_as_real(xq_ * freqs_cis_xq).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis_xk).flatten(3)
    return xq_out, xk_out


def sample(logits: torch.Tensor, temperature: float, top_p: float):
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1)

    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)

    probs = torch.softmax(logits_flat / temperature, dim=-1)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > top_p
    probs_sort[mask] = 0.0
    probs_sort /= probs_sort.sum(dim=-1, keepdim=True)

    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    next_token = next_token.reshape(batch_size, seq_len)
    return next_token


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim * 4
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x_BLD: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x_BLD)) * self.w3(x_BLD))


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        self.q_norm = nn.LayerNorm(dim)
        self.k_norm = nn.LayerNorm(dim)

    def forward(self, x_BLD: torch.Tensor, attn_mask: torch.Tensor, freq_cis: torch.Tensor) -> torch.Tensor:
        B, L, _ = x_BLD.shape
        dtype = x_BLD.dtype

        xq_BLD = self.wq(x_BLD)
        xk_BLD = self.wk(x_BLD)
        xv_BLD = self.wv(x_BLD)

        xq_BLD = self.q_norm(xq_BLD)
        xk_BLD = self.k_norm(xk_BLD)

        xq_BLD, xk_BLD = apply_rotary_emb(xq_BLD, xk_BLD, freq_cis)
        xq_BLD = xq_BLD.to(dtype)
        xk_BLD = xk_BLD.to(dtype)

        xq_BHLK = xq_BLD.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # (bs, num_heads, L, head_dim)
        xk_BHLK = xk_BLD.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        xv_BHLK = xv_BLD.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        out_BHLK = F.scaled_dot_product_attention(xq_BHLK, xk_BHLK, xv_BHLK, attn_mask=attn_mask).transpose(1, 2).reshape(B, L, self.dim)
        return self.wo(out_BHLK)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = Attention(dim, n_heads)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, dim * 4)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6, bias=True),
        )

    def forward(self, x_BLD: torch.Tensor, cond_BD: torch.Tensor, attn_mask: torch.Tensor, freq_cis: torch.Tensor) -> torch.Tensor:
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaLN(cond_BD).chunk(6, dim=1)

        attn_input_BLD = modulate(self.attention_norm(x_BLD), beta1, gamma1)
        attn_output_BLD = self.attention(attn_input_BLD, attn_mask, freq_cis) * alpha1.unsqueeze(1)
        x_BLD = x_BLD + attn_output_BLD

        ffn_input_BLD = modulate(self.ffn_norm(x_BLD), beta2, gamma2)
        ffn_output_BLD = self.ffn(ffn_input_BLD) * alpha2.unsqueeze(1)
        x_BLD = x_BLD + ffn_output_BLD

        return x_BLD


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.embedding = nn.Embedding(num_classes + int(dropout_prob > 0), hidden_size)

    def forward(self, labels, train=True):
        if self.dropout_prob > 0 and train:
            drop_mask = torch.rand_like(labels.float()) < self.dropout_prob
            drop_mask = drop_mask.to(labels.device)
            labels = torch.where(drop_mask, self.num_classes, labels)

        return self.embedding(labels)


class FinalLayer(nn.Module):
    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 2, bias=True),
        )
        self.fc = nn.Linear(dim, vocab_size)

    def forward(self, x_BLC: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.adaLN(x_BLC).chunk(2, dim=2)
        x_BLC = self.layer_norm(x_BLC)
        x_BLC = x_BLC * (1 + gamma) + beta
        return self.fc(x_BLC)


class VAR(nn.Module):
    def __init__(self, vqvae: VQVAE, dim: int, n_heads: int, n_layers: int, patch_sizes: tuple, n_classes: int, cls_dropout: float = 0.1):
        super().__init__()
        self.vqvae = vqvae
        self.dim = dim
        self.max_len = sum(p**2 for p in patch_sizes)
        self.patch_sizes = patch_sizes
        self.final_patch_size = patch_sizes[-1]
        self.latent_dim = vqvae.latent_channels
        self.idxs_L = torch.cat([torch.full((patch * patch,), i) for i, patch in enumerate(patch_sizes)]).view(1, self.max_len)
        self.attn_mask = torch.where(self.idxs_L.unsqueeze(-1) >= self.idxs_L.unsqueeze(-2), 0.0, -torch.inf)

        self.class_embedding = LabelEmbedder(n_classes, dim, cls_dropout)
        self.stage_idx_embedding = nn.Embedding(len(patch_sizes), dim)

        self.in_proj = nn.Linear(self.latent_dim, dim)
        self.freqs_cis = precompute_freqs_cis(self.dim, self.max_len)

        self.layers = nn.ModuleList([TransformerBlock(dim, n_heads) for _ in range(n_layers)])
        self.vocab_size = vqvae.vocab_size
        self.final_layer = FinalLayer(dim, self.vocab_size)
        self.num_classes = n_classes

    def predict_logits(self, x_BlD, cond_BD: torch.Tensor) -> torch.Tensor:
        attn_mask = self.attn_mask.to(x_BlD.device)[:, : x_BlD.shape[1], : x_BlD.shape[1]]
        for layer in self.layers:
            x_BlD = layer(x_BlD, cond_BD, attn_mask, self.freqs_cis.to(x_BlD.device))
        return self.final_layer(x_BlD)

    def forward(self, x_BlC: torch.Tensor, cond: torch.LongTensor) -> torch.Tensor:
        B, _, _ = x_BlC.shape  # for training, l = L - (patch_size[0]) = L - 1
        sos = cond_BD = self.class_embedding(cond)
        sos = sos.unsqueeze(1).expand(B, 1, self.dim).to(x_BlC.dtype)
        x_BLD = torch.cat([sos, self.in_proj(x_BlC)], dim=1) + self.stage_idx_embedding(self.idxs_L.expand(B, -1).to(x_BlC.device))
        logits_BLC = self.predict_logits(x_BLD, cond_BD)
        return logits_BLC

    @torch.no_grad()
    def generate(self, cond: torch.LongTensor, cfg_scale: float, temperature: float = 0.1, top_p: float = 0.35) -> torch.Tensor:
        bs = cond.shape[0]
        B = bs * 2  # for classifier free guidance
        out_bCHW = torch.zeros(bs, self.latent_dim, self.final_patch_size, self.final_patch_size).to(cond.device)
        sos = cond_bD = self.class_embedding(cond, train=False)
        cond_BD = torch.cat([cond_bD, torch.full_like(cond_bD, fill_value=self.num_classes)], dim=0)
        sos_B1D = sos.unsqueeze(1).repeat(2, 1, 1).to(cond.device)
        stage_embedding = self.stage_idx_embedding(self.idxs_L.expand(B, -1).to(cond.device))

        all_scales = [sos_B1D]
        curr_start = 0
        for idx, patch_size in enumerate(self.patch_sizes):
            curr_end = curr_start + patch_size**2
            stage_ratio = idx / (len(self.patch_sizes) - 1)

            x_BlD = torch.cat(all_scales, dim=1)
            x_BlD = x_BlD + stage_embedding[:, : x_BlD.shape[1]]
            logits_BlV = self.predict_logits(x_BlD, cond_BD)[:, curr_start:curr_end]

            cfg = cfg_scale * stage_ratio
            # original paper uses logits_BlV = (1 + cfg) * logits_BlV[:bs] - cfg * logits_BlV[bs:]
            # cond_out_blV = logits_BlV[:bs]
            # uncond_out_blV = logits_BlV[bs:]
            # logits_blV = uncond_out_blV + cfg * (cond_out_blV - uncond_out_blV)
            # logits_blV = cond_out_blV
            logits_blV = (1 + cfg) * logits_BlV[:bs] - cfg * logits_BlV[bs:]

            # idx_bl = torch.argmax(logits_blV, dim=-1)
            idx_bl = sample(logits_blV, temperature, top_p)
            idx_bhw = idx_bl.view(bs, patch_size, patch_size)

            zq_bChw = self.vqvae.get_nearest_embedding(idx_bhw).permute(0, 3, 1, 2)
            zq_bCHW = F.interpolate(zq_bChw, (self.final_patch_size, self.final_patch_size), mode="bicubic")
            h_bCHW = self.vqvae.quantizer.phis(zq_bCHW, stage_ratio)
            out_bCHW = out_bCHW + h_bCHW

            if idx != len(self.patch_sizes) - 1:
                next_bCHW = F.interpolate(out_bCHW, (self.patch_sizes[idx + 1], self.patch_sizes[idx + 1]), mode="area")
                next_blC = next_bCHW.flatten(-2).transpose(1, 2)
                next_BlD = self.in_proj(next_blC).repeat(2, 1, 1)
                all_scales.append(next_BlD)

            curr_start = curr_end

        return self.vqvae.to_img(out_bCHW)


if __name__ == "__main__":
    with torch.no_grad():
        patch_sizes = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 32]
        max_len = sum(p**2 for p in patch_sizes)
        model = VQVAE(64, 7, patch_sizes)
        var = VAR(model, 64, 8, 3, patch_sizes, 10)
        x = torch.randn(1, max_len - 1, var.latent_dim)
        cond = torch.randint(0, 10, (1,))
        out = var(x, cond)
        assert out.shape == (1, max_len, var.vocab_size)
        out = var.generate(cond, 6)
        print(out.shape)
        print("Success")
