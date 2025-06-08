import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from diffusers import AutoencoderKL
from layers import Attention, FeedForward, patchify, unpatchify, compute_freqs, modulate
from quant import SimpleVectorQuantizer


@dataclass
class ImageTokenizerConfig:
    vae_path: str
    model_dim: int = 256
    num_heads: int = 12
    rope_freq_base: float = 10000.0

    height: int = 224  # height of vae latent
    width: int = 224  # width of vae latent
    patch_size: int = 16
    quant_vocab_size: int = 1024
    quant_dim: int = 256
    num_registers: int = 256

    num_encoder_layers: int = 1
    num_decoder_layers: int = 1


class LastLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, 1024), 2 * hidden_size, bias=True),
        )
        # init zero
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half) / half).to(t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(dtype=next(self.parameters()).dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class EncoderBlock(nn.Module):
    def __init__(
        self,
        dim,
        mlp_dim,
        heads: int = 8,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, n_heads=heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = FeedForward(dim=dim, hidden_dim=mlp_dim)

    def forward(self, x, freqs, attn_mask=None):
        x = x + self.attn(self.norm1(x), freqs, attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        dim,
        mlp_dim,
        heads: int = 8,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, n_heads=heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = FeedForward(dim=dim, hidden_dim=mlp_dim)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6, bias=True),
        )

    def forward(self, x, freqs, t, attn_mask=None):
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaLN(t).chunk(6, dim=1)

        attn_input = modulate(self.norm1(x), beta1, gamma1)
        attn_output = self.attn(attn_input, freqs, attn_mask) * alpha1.unsqueeze(1)
        x = x + attn_output

        ffn_input = modulate(self.norm2(x), beta2, gamma2)
        ffn_output = self.mlp(ffn_input) * alpha2.unsqueeze(1)
        x = x + ffn_output

        return x


class Encoder(nn.Module):
    def __init__(self, config: ImageTokenizerConfig):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(dim=config.model_dim, mlp_dim=config.model_dim * 4, heads=config.num_heads) for _ in range(config.num_encoder_layers)])

    def forward(self, x, freqs, attn_mask=None):
        for layer in self.layers:
            x = layer(x, freqs, attn_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, config: ImageTokenizerConfig):
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(dim=config.model_dim, mlp_dim=config.model_dim * 4, heads=config.num_heads) for _ in range(config.num_decoder_layers)])

    def forward(self, x, freqs, t, attn_mask=None):
        for layer in self.layers:
            x = layer(x, freqs, t, attn_mask)
        return x


class ImageTokenizer(nn.Module):
    def __init__(self, config: ImageTokenizerConfig):
        super().__init__()
        self.config = config
        self.vae = AutoencoderKL.from_pretrained(config.vae_path, subfolder="vae")
        self._disable_vae()

        self.latent_dim = self.vae.config.latent_dim
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.vocab_embedding = nn.Embedding(config.quant_vocab_size, config.model_dim)
        self.init_proj = nn.Linear(self.latent_dim * config.patch_size * config.patch_size, config.model_dim)

        self.num_patches = (config.height // config.patch_size) * (config.width // config.patch_size)
        self.seq_len = self.num_patches + config.num_registers
        self.num_registers = config.num_registers

        # bidirectional for patches, causal for registers
        self.total_seq_len = self.num_patches + self.num_registers
        self.causal_attn_mask = torch.ones(self.total_seq_len, self.total_seq_len, dtype=torch.bool)
        causal_mask = torch.triu(torch.ones(self.num_registers, self.num_registers), diagonal=1).bool()
        self.causal_attn_mask[self.num_patches :, self.num_patches :] = ~causal_mask

        self.freqs = compute_freqs(theta=config.rope_freq_base, d_k=config.model_dim, max_seq_len=self.total_seq_len)

        self.quantizer = SimpleVectorQuantizer(vocab_size=config.quant_vocab_size, dim=config.quant_dim)
        self.pre_quant_proj = nn.Linear(config.model_dim, config.quant_dim)

        self.encoder_registers = nn.Parameter(torch.randn(config.num_registers, config.model_dim), requires_grad=True)
        self.mask_token = nn.Parameter(torch.randn(config.model_dim), requires_grad=True)
        self.timestep_embedder = TimestepEmbedder(hidden_size=config.model_dim)
        self.last_layer = LastLayer(hidden_size=config.model_dim, patch_size=config.patch_size, out_channels=self.latent_dim)

    def _disable_vae(self):
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

    def encode(self, x_BCHW, num_tokens_to_use: int | None = None):
        x_BLd = patchify(x_BCHW, self.config.patch_size)
        x_BLD = self.init_proj(x_BLd)
        x_BlrD = torch.cat([x_BLD, self.encoder_registers.unsqueeze(0).repeat(x_BLD.shape[0], 1, 1)], dim=1)
        x_BlrD = self.encoder(x_BlrD, self.freqs, self.causal_attn_mask)
        x_BrD = x_BlrD[:, -self.num_registers :]

        x_BrQ = self.pre_quant_proj(x_BrD)
        f_hat_BrD, idx_Br, vq_loss = self.quantizer(x_BrQ)

        if num_tokens_to_use is not None:
            idx_Br = idx_Br[:, :num_tokens_to_use]

        return idx_Br, vq_loss

    def decode(self, idx_Br, noised_latent_BCHW, t: torch.Tensor, num_tokens_to_use: int | None = None):
        noised_latent_BLd = patchify(noised_latent_BCHW, self.config.patch_size)
        noised_latent_BLD = self.init_proj(noised_latent_BLd)
        B, L, _ = noised_latent_BLD.shape

        x_BrD = self.vocab_embedding(idx_Br)

        if num_tokens_to_use is not None:
            num_mask = self.num_registers - num_tokens_to_use
            mask_tokens = self.mask_token.unsqueeze(0).repeat(B, num_mask, 1)
            x_BrD = torch.cat([x_BrD, mask_tokens, noised_latent_BLD], dim=1)
        else:
            x_BrD = torch.cat([x_BrD, noised_latent_BLD], dim=1)

        t_emb = self.timestep_embedder(t)
        x_BrD = self.decoder(x_BrD, self.freqs, t_emb, self.causal_attn_mask)
        x_BrD = x_BrD[:, -L:]
        x_BChw = self.last_layer(x_BrD, t_emb)
        x_BChw = unpatchify(x_BChw, self.config.height, self.config.width, self.config.patch_size)

        return x_BChw

    def sample(self, image_latents_BCHW, sample_steps=50, num_tokens_to_use=256):
        B, C, H, W = image_latents_BCHW.shape
        assert num_tokens_to_use <= self.num_registers
        idx_Br, _ = self.encode(image_latents_BCHW, num_tokens_to_use)
        z = torch.randn_like(image_latents_BCHW)

        dt = 1.0 / sample_steps

        for i in range(sample_steps, 0, -1):
            t = torch.tensor([i / sample_steps] * B, device=z.device)
            velocity = self.decode(idx_Br, z, t, num_tokens_to_use)
            z = z - dt * velocity

        return z

    # only used for training
    def forward(
        self,
        x_BCHW,  # vae latent
        t: torch.Tensor,
        noised_latent_BCHW: torch.Tensor,
        num_tokens_to_use: int | None = None,
    ):
        idx_Br, vq_loss = self.encode(x_BCHW, num_tokens_to_use)
        x_BChw = self.decode(idx_Br, noised_latent_BCHW, t, num_tokens_to_use)

        return x_BChw, vq_loss

    def vae_encode(self, x_BCHW: torch.Tensor):
        x_BCHW = x_BCHW.to(self.vae.device)
        return self.vae.encode(x_BCHW).latent_dist.sample()

    def vae_decode(self, x_BCHW: torch.Tensor):
        x_BCHW = x_BCHW.to(self.vae.device)
        return self.vae.decode(x_BCHW).sample


if __name__ == "__main__":
    dummy_config = ImageTokenizerConfig(
        vae_path="models/vae",
        model_dim=32,
        num_heads=2,
        rope_freq_base=10000.0,
        height=32,
        width=32,
        patch_size=8,
        quant_vocab_size=256,
        quant_dim=2,
        num_registers=32,
        num_encoder_layers=1,
        num_decoder_layers=1,
    )
    model = ImageTokenizer(dummy_config)
    x = torch.randn(1, 16, 32, 32)
    t = torch.randn((1,))
    out, vq_loss = model(x, t, torch.randn_like(x), num_tokens_to_use=1)
    assert out.shape == x.shape
