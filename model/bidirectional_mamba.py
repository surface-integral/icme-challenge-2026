"""
model/bidirectional_mamba.py

Bidirectional Mamba Flow Network — the CORE GENERATIVE MODEL trained from scratch.

Unlike the autoregressive MambaLM, this model operates non-causally:
  - It receives the ENTIRE noisy latent x_t as input (batch, T, C)
  - At each flow step t ∈ [0, 1], it predicts the velocity field v_θ(x_t, t, text)
  - Non-causality is achieved by running two Mamba scans (forward + backward)
    and merging their outputs — "Bidirectional Mamba"

Architecture per block:
  ┌──────────────────────────────────────────────────────┐
  │  x  →  Norm  →  BidirMambaSSM  →  + residual        │
  │     →  Norm  →  CrossAttn(T5)   →  + residual        │
  │     →  Norm  →  AdaLN-FFN(t)   →  + residual        │
  └──────────────────────────────────────────────────────┘

BidirMambaSSM runs:
  forward_out  = MambaSSM(x,  direction=forward)
  backward_out = MambaSSM(x_flipped, direction=forward).flip(dim=1)
  merged = linear(concat(forward_out, backward_out))

Timestep conditioning uses AdaLayerNorm (DiT-style):
  Given t embedding τ, compute (scale, shift) → modulate normed activations.

Parameter count (default config):
  d_model=512, n_layers=24 → ~120M parameters (well within 500M cap)
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from einops import rearrange


# ── Config ─────────────────────────────────────────────────────────────────

@dataclass
class BidirMambaConfig:
    latent_dim:         int   = 128     # DCAE latent dimension after reshaping (input/output)
    d_model:            int   = 512     # Core model dimension
    n_layers:           int   = 24
    d_state:            int   = 16
    d_conv:             int   = 4
    expand:             int   = 2
    dt_rank:            str   = "auto"
    dt_min:             float = 0.001
    dt_max:             float = 0.1
    bias:               bool  = False
    conv_bias:          bool  = True
    conditioning_dim:   int   = 768     # T5-base
    conditioning_heads: int   = 8
    time_embed_dim:     int   = 256
    use_ada_ln:         bool  = True
    dropout:            float = 0.1

    def __post_init__(self):
        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)
        self.d_inner = self.expand * self.d_model


# ── Timestep embedding ──────────────────────────────────────────────────────

class TimestepEmbedding(nn.Module):
    """
    Sinusoidal positional encoding → 2-layer MLP → time_embed_dim vector.
    Identical to the approach used in DDPM and DiT.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        half = embed_dim // 2
        self.register_buffer(
            "freqs",
            torch.exp(-math.log(10000) * torch.arange(half) / (half - 1)).float(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) float in [0, 1]
        Returns:
            emb: (B, embed_dim)
        """
        args = t[:, None].float() * self.freqs[None, :]   # (B, half)
        emb  = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, embed_dim)
        return self.mlp(emb)


# ── Adaptive LayerNorm (DiT-style) ─────────────────────────────────────────

class AdaLayerNorm(nn.Module):
    """
    LayerNorm with scale and shift modulated by a conditioning vector (time emb).
    γ, β = linear(cond).chunk(2)
    out  = γ * LayerNorm(x) + β
    """

    def __init__(self, d_model: int, cond_dim: int):
        super().__init__()
        self.norm  = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj  = nn.Linear(cond_dim, 2 * d_model, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, L, d_model)
            cond: (B, cond_dim)
        """
        gamma, beta = self.proj(cond).unsqueeze(1).chunk(2, dim=-1)  # each (B, 1, d_model)
        return (1 + gamma) * self.norm(x) + beta

# ── Bidirectional Mamba SSM ─────────────────────────────────────────────────

class BidirMambaSSM(nn.Module):
    """
    Runs two independent Mamba scans (forward + backward) and merges.

    forward:  MambaSSM(x)
    backward: MambaSSM(x.flip(1)).flip(1)
    out:      linear(concat(forward, backward, dim=-1))

    Each direction has its own independent SSM parameters, as in Audio Mamba
    (Erol et al. 2024) and Vision Mamba (Zhu et al. 2024).
    """

    def __init__(self, cfg: BidirMambaConfig):
        super().__init__()
        self.fwd_ssm  = Mamba(d_model=cfg.d_model,
                              d_state = cfg.d_state,
                              d_conv=cfg.d_conv,
                              expand=cfg.expand)
        self.bwd_ssm  = Mamba(d_model=cfg.d_model,
                              d_state = cfg.d_state,
                              d_conv=cfg.d_conv,
                              expand=cfg.expand)
        # Merge forward + backward outputs (both are d_model) → d_model
        self.merge    = nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model)  →  (B, L, d_model)"""
        fwd = self.fwd_ssm(x)
        bwd = self.bwd_ssm(x.flip(1)).flip(1)
        return self.merge(torch.cat([fwd, bwd], dim=-1))


# ── Cross-attention for text conditioning ──────────────────────────────────

class CrossAttention(nn.Module):
    """Standard multi-head cross-attention: query from hidden, key/value from T5."""

    def __init__(self, cfg: BidirMambaConfig):
        super().__init__()
        self.n_heads  = cfg.conditioning_heads
        self.head_dim = cfg.d_model // cfg.conditioning_heads
        self.scale    = self.head_dim ** -0.5

        self.q    = nn.Linear(cfg.d_model,         cfg.d_model, bias=False)
        self.k    = nn.Linear(cfg.conditioning_dim, cfg.d_model, bias=False)
        self.v    = nn.Linear(cfg.conditioning_dim, cfg.d_model, bias=False)
        self.out  = nn.Linear(cfg.d_model,         cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor,
                ctx_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = x.shape
        S = ctx.shape[1]
        H, D = self.n_heads, self.head_dim

        def heads(t, seq): return t.view(B, seq, H, D).transpose(1, 2)

        Q = heads(self.q(x),   L)
        K = heads(self.k(ctx), S)
        V = heads(self.v(ctx), S)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        if ctx_mask is not None:
            attn = attn.masked_fill(~ctx_mask[:, None, None, :], float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, L, H * D)
        return self.out(out)


# ── Bidirectional Mamba Block ───────────────────────────────────────────────

class BidirMambaBlock(nn.Module):
    """
    One full bidirectional Mamba block with text and timestep conditioning.

      Residual 1: AdaLN(x, t) → BidirMambaSSM
      Residual 2: LayerNorm(x) → CrossAttn(T5 context)
      Residual 3: AdaLN(x, t) → FFN
    """

    def __init__(self, cfg: BidirMambaConfig):
        super().__init__()
        self.norm1    = AdaLayerNorm(cfg.d_model, cfg.time_embed_dim) if cfg.use_ada_ln \
                        else nn.LayerNorm(cfg.d_model)
        self.bidir    = BidirMambaSSM(cfg)

        self.norm2    = nn.LayerNorm(cfg.d_model)
        self.cross    = CrossAttention(cfg)

        self.norm3    = AdaLayerNorm(cfg.d_model, cfg.time_embed_dim) if cfg.use_ada_ln \
                        else nn.LayerNorm(cfg.d_model)
        self.ffn      = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )
        self.use_ada_ln = cfg.use_ada_ln

    def _norm1(self, x, t_emb):
        return self.norm1(x, t_emb) if self.use_ada_ln else self.norm1(x)

    def _norm3(self, x, t_emb):
        return self.norm3(x, t_emb) if self.use_ada_ln else self.norm3(x)

    def forward(
        self,
        x:        torch.Tensor,
        t_emb:    torch.Tensor,
        ctx:      Optional[torch.Tensor]      = None,
        ctx_mask: Optional[torch.Tensor]      = None,
    ) -> torch.Tensor:
        x = x + self.bidir(self._norm1(x, t_emb))
        if ctx is not None:
            x = x + self.cross(self.norm2(x), ctx, ctx_mask)
        x = x + self.ffn(self._norm3(x, t_emb))
        return x


# ── Input / Output projections ──────────────────────────────────────────────

class LatentPatchEmbed(nn.Module):
    """
    Projects latent frames (B, T, C_lat) → (B, T, d_model).
    A lightweight 1D conv patchification is used so local temporal context
    is established before the global Mamba scan.
    """

    def __init__(self, latent_dim: int, d_model: int, patch_size: int = 1):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(latent_dim, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C_lat) → (B, T', d_model)"""
        x = rearrange(x, "b t c -> b c t")
        x = self.proj(x)
        return rearrange(x, "b d t -> b t d")


# ── Full Flow Network ───────────────────────────────────────────────────────

class BidirMambaFlowNet(nn.Module):
    """
    The core generative model — trained from scratch.

    Input:
        x_t:          (B, T, C_lat)  — noisy latent at time t
        t:            (B,)           — flow time in [0, 1]
        ctx:          (B, S, cond_dim) — T5 text features
        ctx_mask:     (B, S) bool

    Output:
        velocity: (B, T, C_lat) — predicted vector field v_θ(x_t, t, text)
    """

    def __init__(self, cfg: BidirMambaConfig):
        super().__init__()
        self.cfg = cfg

        # Input projection: latent_dim → d_model
        self.input_proj = LatentPatchEmbed(cfg.latent_dim, cfg.d_model)

        # Timestep embedding
        self.time_embed = TimestepEmbedding(cfg.time_embed_dim)

        # Mamba blocks
        self.blocks = nn.ModuleList([BidirMambaBlock(cfg) for _ in range(cfg.n_layers)])

        # Output projection: d_model → latent_dim
        self.norm_out = nn.LayerNorm(cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.latent_dim, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm) and m.elementwise_affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x_t:      torch.Tensor,
        t:        torch.Tensor,
        ctx:      Optional[torch.Tensor] = None,
        ctx_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        h = self.input_proj(x_t)           # (B, T, d_model)
        t_emb = self.time_embed(t)          # (B, time_embed_dim)

        for block in self.blocks:
            h = block(h, t_emb, ctx, ctx_mask)

        h = self.norm_out(h)               # (B, T, d_model)
        v = self.out_proj(h)               # (B, T, C_lat)
        return v

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    cfg   = BidirMambaConfig(d_model=256, n_layers=4, latent_dim=128)
    model = BidirMambaFlowNet(cfg)
    n     = model.count_parameters()
    print(f"BidirMambaFlowNet  |  d_model={cfg.d_model}  n_layers={cfg.n_layers}  params={n:,}")

    B, T, C = 2, 323, 8
    x_t      = torch.randn(B, T, C)
    t        = torch.rand(B)
    ctx      = torch.randn(B, 32, 768)
    ctx_mask = torch.ones(B, 32, dtype=torch.bool)

    v = model(x_t, t, ctx, ctx_mask)
    assert v.shape == (B, T, C), f"Shape mismatch: {v.shape}"
    print(f"Velocity shape: {v.shape}  ✓")
