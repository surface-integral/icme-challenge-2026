"""
model/flow_generate.py

Inference for the Bidirectional Mamba + DCAE flow matching model.

Generation algorithm (Euler ODE integration):
  1. Sample z ~ N(0, I)   in latent space   (B, T, C)
  2. For each step i in [0, n_steps):
       t   = i / n_steps
       t1  = (i + 1) / n_steps
       v   = cfg_guided_velocity(z, t, text_ctx)
       z   = z + (t1 - t) * v              # Euler step
  3. Decode z via MusicDCAE → waveform

CFG guidance:
  v_guided = v_uncond + coeff * (v_cond - v_uncond)

Usage:
    from model.flow_generate import generate_music_flow
    wav = generate_music_flow(
        prompt      = "An energetic rock track with electric guitar.",
        model       = model,
        conditioner = conditioner,
        dcae        = dcae_wrapper,
        duration_sec= 10.0,
        n_steps     = 50,
        cfg_coeff   = 3.5,
    )

CLI:
    python model/flow_generate.py \
        --prompt "A mellow jazz piano piece" \
        --checkpoint checkpoints_flow/final.pt \
        --output output.wav \
        --duration 10 \
        --n_steps 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.bidirectional_mamba import BidirMambaConfig, BidirMambaFlowNet
from model.conditioning import T5TextConditioner
from model.dcae_wrapper import MusicDCAEWrapper


@torch.inference_mode()
def generate_music_flow(
    prompt:       str,
    model:        BidirMambaFlowNet,
    conditioner:  T5TextConditioner,
    dcae:         MusicDCAEWrapper,
    duration_sec: float = 10.0,
    n_steps:      int   = 50,
    cfg_coeff:    float = 3.5,
    latent_fps:   float = 10.77,
    latent_dim:   int = 128,
    device:       str   = "cuda",
    seed:         int   = 0,
) -> np.ndarray | torch.Tensor:
    """
    Full text-to-music generation via flow matching + DCAE decode.

    Returns:
        waveform: numpy array, mono, 48kHz
    """
    model.eval()
    dev = torch.device(device)

    torch.manual_seed(seed)

    # Latent shape for the requested duration
    T = int(duration_sec * latent_fps)
    C = latent_dim

    # ── 1. Encode text prompt ──────────────────────────────────────────────
    ctx,      ctx_mask  = conditioner.encode_text([prompt], device=dev)    # (1, S, d)
    null_ctx, null_mask = conditioner.get_null_conditioning(1, dev)        # (1, 1, d)

    # ── 2. Initialise with Gaussian noise ──────────────────────────────────
    z = torch.randn(1, T, C, device=dev)

    # ── 3. Euler integration ───────────────────────────────────────────────
    dt = 1.0 / n_steps

    for i in range(n_steps):
        t_scalar = i / n_steps
        t_batch  = torch.full((1,), t_scalar, device=dev, dtype=torch.float32)

        # Conditional velocity
        v_cond   = model(z, t_batch, ctx,      ctx_mask)    # (1, T, C)
        # Unconditional velocity
        v_uncond = model(z, t_batch, null_ctx, null_mask)   # (1, T, C)

        # CFG
        v = v_uncond + cfg_coeff * (v_cond - v_uncond)

        # Euler step
        z = z + dt * v

    # z is now approximately x_0 (clean latent)
    clean_latent = z[0]   # (T, C)

    # ── 4. Decode with MusicDCAE ───────────────────────────────────────────
    # DCAE expects (C, T)
    wav = dcae.decode(clean_latent.permute(1, 0).cpu())   # numpy (n_samples,)

    return wav


def save_wav(wav: torch.Tensor, path: str, sample_rate: int = 44100):
    torchaudio.save(path, wav, sample_rate)
    print(f"Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt",      required=True)
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--output",      default="output_flow.wav")
    parser.add_argument("--duration",    type=float, default=10.0)
    parser.add_argument("--n_steps",     type=int,   default=50)
    parser.add_argument("--cfg_coeff",   type=float, default=3.5)
    parser.add_argument("--seed",        type=int,   default=0)
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dcae_model",  default="ACE-Step/ACE-Step-v1-3.5B")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    cfg   = BidirMambaConfig(**ckpt["model_config"])
    model = BidirMambaFlowNet(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(args.device).eval()

    conditioner = T5TextConditioner(d_model=cfg.conditioning_dim)
    conditioner.load_state_dict(ckpt.get("cond_state", {}), strict=False)
    conditioner.to(args.device).eval()

    dcae = MusicDCAEWrapper(model_id=args.dcae_model, device=args.device)

    print(f"Prompt: {args.prompt}")
    wav = generate_music_flow(
        prompt       = args.prompt,
        model        = model,
        conditioner  = conditioner,
        dcae         = dcae,
        duration_sec = args.duration,
        n_steps      = args.n_steps,
        cfg_coeff    = args.cfg_coeff,
        device       = args.device,
        seed         = args.seed,
    )
    save_wav(wav, args.output)
