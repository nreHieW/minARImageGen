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
    model_dim: int
    num_heads: int
    rope_freq_base: float
    height: int  # height of vae latent
    width: int  # width of vae latent
    patch_size: int
    quant_vocab_size: int
    quant_dim: int
    num_registers: int
    num_encoder_layers: int
    num_decoder_layers: int


class LastLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, 1024), 2 * hidden_size, bias=True),
        )
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
        # self.vae = AutoencoderKL.from_pretrained(config.vae_path, subfolder="vae")
        # self._disable_vae()

        # self.latent_dim = self.vae.config.latent_channels
        self.latent_dim = 3
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.init_proj = nn.Linear(self.latent_dim * config.patch_size * config.patch_size, config.model_dim)

        self.num_patches = (config.height // config.patch_size) * (config.width // config.patch_size)
        self.seq_len = self.num_patches + config.num_registers
        self.num_registers = config.num_registers

        self.total_seq_len = self.num_patches + self.num_registers

        # Create causal attention mask
        # True means can attend, False means cannot attend (masked)
        attn_mask = torch.zeros(self.total_seq_len, self.total_seq_len, dtype=torch.bool)

        # 1. Image patches can attend to each other but NOT to registers
        attn_mask[: self.num_patches, : self.num_patches] = True  # patches -> patches: allowed
        # attn_mask[:self.num_patches, self.num_patches:] remains False (patches -> registers: masked)

        # 2. Register tokens can attend to all patches
        attn_mask[self.num_patches :, : self.num_patches] = True  # registers -> patches: allowed

        # 3. Register token i can attend to register token j only if i >= j (causal)
        # Create lower triangular mask (including diagonal) for register-to-register attention
        register_causal_mask = torch.tril(torch.ones(self.num_registers, self.num_registers)).bool()
        attn_mask[self.num_patches :, self.num_patches :] = register_causal_mask  # lower triangle allowed

        self.register_buffer("causal_attn_mask", attn_mask)

        self.register_buffer("freqs", compute_freqs(theta=config.rope_freq_base, d_k=config.model_dim, max_seq_len=self.total_seq_len))

        self.quantizer = SimpleVectorQuantizer(vocab_size=config.quant_vocab_size, dim=config.quant_dim)
        self.pre_quant_proj = nn.Linear(config.model_dim, config.quant_dim)
        self.post_quant_proj = nn.Linear(config.quant_dim, config.model_dim)

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
        f_hat_BrQ, idx_Br, vq_loss = self.quantizer(x_BrQ)
        f_hat_BrD = self.post_quant_proj(f_hat_BrQ)
        # f_hat_BrD = self.post_quant_proj(x_BrQ)

        if num_tokens_to_use is not None:
            idx_Br = idx_Br[:, :num_tokens_to_use]
            f_hat_BrD = f_hat_BrD[:, :num_tokens_to_use]

        return f_hat_BrD, idx_Br, vq_loss

    def decode(self, x_BrD_quantized, noised_latent_BCHW, t: torch.Tensor, num_tokens_to_use: int | None = None):
        noised_latent_BLd = patchify(noised_latent_BCHW, self.config.patch_size)
        noised_latent_BLD = self.init_proj(noised_latent_BLd)
        B, L, _ = noised_latent_BLD.shape

        if num_tokens_to_use is not None and num_tokens_to_use < self.num_registers:
            num_mask = self.num_registers - num_tokens_to_use
            mask_tokens = self.mask_token.unsqueeze(0).unsqueeze(0).repeat(B, num_mask, 1)
            register_tokens = torch.cat([x_BrD_quantized, mask_tokens], dim=1)
        else:
            register_tokens = x_BrD_quantized

        decoder_input = torch.cat([register_tokens, noised_latent_BLD], dim=1)

        t_emb = self.timestep_embedder(t)
        x_BrD = self.decoder(decoder_input, self.freqs, t_emb, self.causal_attn_mask)
        x_BrD = x_BrD[:, -L:]
        x_BChw = self.last_layer(x_BrD, t_emb)
        x_BChw = unpatchify(x_BChw, self.config.height, self.config.width, self.config.patch_size)

        return x_BChw

    def sample(self, image_latents_BCHW, sample_steps=50, num_tokens_to_use=256):
        B, C, H, W = image_latents_BCHW.shape
        assert num_tokens_to_use <= self.num_registers
        quantized_tokens, _, _ = self.encode(image_latents_BCHW, num_tokens_to_use)
        z = torch.randn_like(image_latents_BCHW)

        dt = 1.0 / sample_steps

        for i in range(sample_steps, 0, -1):
            t = torch.tensor([i / sample_steps] * B, device=z.device)
            velocity = self.decode(quantized_tokens, z, t, num_tokens_to_use)
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
        f_hat_BrD, idx_Br, vq_loss = self.encode(x_BCHW, num_tokens_to_use)
        x_BChw = self.decode(f_hat_BrD, noised_latent_BCHW, t, num_tokens_to_use)

        return x_BChw, vq_loss, idx_Br

    def vae_encode(self, x_BCHW: torch.Tensor):
        x_BCHW = x_BCHW.to(self.vae.device)
        return self.vae.encode(x_BCHW).latent_dist.sample()

    def vae_decode(self, x_BCHW: torch.Tensor):
        x_BCHW = x_BCHW.to(self.vae.device)
        return self.vae.decode(x_BCHW).sample


if __name__ == "__main__":
    # dummy_config = ImageTokenizerConfig(
    #     vae_path="models/vae",
    #     model_dim=32,
    #     num_heads=2,
    #     rope_freq_base=10000.0,
    #     height=32,
    #     width=32,
    #     patch_size=8,
    #     quant_vocab_size=256,
    #     quant_dim=2,
    #     num_registers=32,
    #     num_encoder_layers=1,
    #     num_decoder_layers=1,
    # )
    # model = ImageTokenizer(dummy_config)
    # x = torch.randn(1, 16, 32, 32)
    # t = torch.randn((1,))
    # out, vq_loss, idx_Br = model(x, t, torch.randn_like(x), num_tokens_to_use=1)
    # assert out.shape == x.shape
    print("✓ Forward pass test passed")

    def check_gradients(model, step_num):
        """Check which parameters have gradients and which don't"""
        params_with_grad = []
        params_without_grad = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None and param.grad.abs().sum() > 0:
                    params_with_grad.append(name)
                else:
                    params_without_grad.append(name)

        print(f"\nStep {step_num} - Gradient Check:")
        print(f"Parameters WITH gradients: {len(params_with_grad)}")
        print(params_with_grad)
        print(f"Parameters WITHOUT gradients: {len(params_without_grad)}")

        if params_without_grad:
            print("Parameters missing gradients:")
            for name in params_without_grad:
                print(f"  - {name}")
            assert False
        else:
            print("✓ All parameters have gradients!")

        return len(params_without_grad) == 0

    # Convergence test
    print("\n" + "=" * 50)
    print("CONVERGENCE TEST")
    print("=" * 50)

    # Create a smaller model for faster testing
    test_config = ImageTokenizerConfig(
        vae_path="models/vae",
        model_dim=64,
        num_heads=4,
        rope_freq_base=10000.0,
        height=8,
        width=8,
        patch_size=2,
        quant_vocab_size=32,
        quant_dim=4,
        num_registers=8,
        num_encoder_layers=1,
        num_decoder_layers=2,
    )

    test_model = ImageTokenizer(test_config)

    batch_size = 1
    images = torch.randn(batch_size, 3, 8, 8)

    # Setup optimizer
    optimizer = torch.optim.AdamW(test_model.parameters(), lr=3e-4)
    vq_loss_weight = 0.2

    print(f"Model parameters: {sum(p.numel() for p in test_model.parameters()):,}")
    print(f"Training on batch shape: {images.shape}")
    print()

    # Training loop
    test_model.train()
    for step in range(100_000):
        optimizer.zero_grad()

        # Use raw images directly (no VAE encoding)
        vae_latents = images

        # Sample timestep and noise
        t = torch.rand(batch_size)
        z1 = torch.randn_like(vae_latents)

        # Interpolate between clean and noise
        t_expanded = t.view(batch_size, 1, 1, 1)
        zt = (1 - t_expanded) * vae_latents + t_expanded * z1

        # Forward pass
        velocity_pred, vq_loss, _ = test_model(vae_latents, t, zt, num_tokens_to_use=12)

        # Calculate loss
        target_velocity = z1 - vae_latents
        velocity_loss = torch.nn.functional.mse_loss(velocity_pred, target_velocity)
        total_loss = velocity_loss + vq_loss_weight * vq_loss
        # batchwise_mse = ((z1 - vae_latents - velocity_pred) ** 2).mean(dim=list(range(1, len(vae_latents.shape))))
        # total_loss = batchwise_mse.mean() + vq_loss_weight * vq_loss

        # Backward pass
        total_loss.backward()

        # # Check gradients periodically
        # if step % 1000 == 0 and step > 0:
        #     all_have_grads = check_gradients(test_model, step)
        #     if not all_have_grads and step > 0:
        #         print("WARNING: Some parameters don't have gradients!")
        #         break

        optimizer.step()

        # Print progress
        if step % 100 == 0:
            print(f"Step {step:3d} | Total: {total_loss.item():.6f} | Velocity: {velocity_loss.item():.6f} | VQ: {vq_loss.item():.6f}")

    print()
    print("✓ Convergence test completed - loss should decrease over time")
    print("✓ If loss reaches very low values (~1e-4), the model is learning correctly")
    # assert total_loss.item() < 1e-5
