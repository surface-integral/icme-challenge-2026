# ATTM-Mamba v2: Bidirectional Mamba + Flow Matching + MusicDCAE

The second architecture for the ICME 2026 ATTM Grand Challenge (Efficiency Track),
incorporating Dr. Dubnov's suggestion of Mamba + flow matching with DCAE.

## Architecture Overview

```
Text Prompt
    │
    ▼
[T5 Text Encoder]           ← auxiliary, frozen
    │  (context)
    ▼
┌────────────────────────────────────────────────┐
│  Bidirectional Mamba Flow Network  (CORE MODEL) │
│  ─ trained from scratch                         │
│                                                  │
│  Input: noisy latent x_t  (B, T, C=8)          │
│         timestep t ∈ [0,1]                       │
│         T5 context                               │
│                                                  │
│  Per block:                                      │
│    AdaLN(t) → BidirMambaSSM ─── + residual      │
│    LayerNorm → CrossAttn(T5) ── + residual       │
│    AdaLN(t) → FFN ──────────── + residual        │
│                                                  │
│  BidirMambaSSM:                                  │
│    forward_out  = MambaSSM(x)                    │
│    backward_out = MambaSSM(x.flip).flip          │
│    merged       = linear(cat(fwd, bwd))          │
│                                                  │
│  Output: velocity field v_θ(x_t, t, text)       │
└────────────────────────────────────────────────┘
    │
    │  Euler ODE integration:
    │  z_{t+dt} = z_t + dt * v_θ(z_t, t, text)
    ▼
[MusicDCAE Decoder]         ← auxiliary, frozen (ACE-Step/ACE-Step-v1-3.5B)
    │
    ▼
Audio Waveform (48kHz)
```

## Key Differences from v1 (Autoregressive)

| | v1 Autoregressive | v2 Flow Matching |
|---|---|---|
| Generation paradigm | Token-by-token LM | ODE integration (50 steps) |
| Mamba direction | Causal (unidirectional) | Bidirectional |
| Audio representation | EnCodec discrete tokens | DCAE continuous latents |
| Latent space | 4 codebooks × vocab(2048) | 8 continuous channels |
| Conditioning | Cross-attention per block | Cross-attention + AdaLN(t) |
| Training loss | Cross-entropy over tokens | MSE on vector field (OT-CFM) |
| Parameter count | ~300M | ~120M (smaller, more efficient) |

## Pipeline

```
Step 1: bash scripts/1_download_dataset.sh          # same as v1
Step 2: bash scripts/2_preprocess.sh                # same as v1 (vocal separation)
Step 3: python scripts/3_encode_latents.py \       # NEW: DCAE instead of EnCodec
            --audio_dir ../mtg_jamendo_separated \
            --output_dir ../mtg_latents \
            --stats     # first run: estimate mean/std, update config
Step 4: python scripts/4_caption_audio.py           # same as v1 (or use provided captions)
Step 5: python training/train_flow.py \
            --config configs/flow_default.yaml
Step 6: python model/flow_generate.py \
            --prompt "A mellow jazz piano piece" \
            --checkpoint checkpoints_flow/final.pt \
            --output output.wav
```

## Flow Matching Training Objective

OT-CFM (Optimal Transport Conditional Flow Matching):

```
  Sample: x_0 ~ data,  z ~ N(0, I),  t ~ LogitNormal(0, 1)
  Interpolate: x_t = (1 - t) * z + t * x_0
  Target velocity: v* = x_0 - z
  Loss: E[ || v_θ(x_t, t, text) - v* ||² ]
```

The straight-line paths of OT-CFM make it easier to learn than DDPM/score
matching, and require fewer integration steps at inference.

## Parameter Budget

| Component | Params | Role |
|---|---|---|
| BidirMambaFlowNet | ~120M | **Core model — trained from scratch** |
| T5-base encoder | ~220M | Auxiliary (text), frozen |
| MusicDCAE f8c8 | ~314MB | Auxiliary (audio), frozen |

Core model: 120M << 500M efficiency track limit ✓

## DCAE Attribution

The MusicDCAE (`ACE-Step/ACE-Step-v1-3.5B / music_dcae_f8c8`) is released
under Apache 2.0 by ACE Studio & StepFun. It is used here as an auxiliary
audio encoder/decoder and its weights are not trained.

Reference: Gong et al., "ACE-Step: A Step Towards Music Generation
Foundation Model", 2025.
