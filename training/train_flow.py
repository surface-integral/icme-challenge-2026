"""
training/train_flow.py

Flow Matching training loop for the Bidirectional Mamba + DCAE pipeline.

Loss:
    Optimal Transport Conditional Flow Matching (OT-CFM).
    Given clean latent x_0 ~ p_data and noise z ~ N(0, I):

      x_t = (1 - t) * z + t * x_0          # linear interpolation (straight path)
      v*  = x_0 - z                          # target velocity (constant along path)

    The model v_θ is trained to minimise:
      L = E_{t,x_0,z} [ || v_θ(x_t, t, c) - v* ||^2 ]

    masked to only count valid (non-padded) frames.

CFG training:
    With probability cfg_dropout_prob, replace the text context with the
    learnable null embedding in the T5TextConditioner.

Usage:
    python training/train_flow.py --config configs/flow_default.yaml [--resume PATH]
"""

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm
from torch.amp.autocast_mode import autocast

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.flow_dataset import build_flow_dataloader
from model.bidirectional_mamba import BidirMambaConfig, BidirMambaFlowNet
from model.conditioning import T5TextConditioner   # reused unchanged from v1


# ── Time sampling ──────────────────────────────────────────────────────────

def sample_times(
    batch_size: int,
    device:     torch.device,
    method:     str   = "logit_normal",
    mean:       float = 0.0,
    std:        float = 1.0,
) -> torch.Tensor:
    """
    Sample flow times t ∈ (0, 1) for a batch.

    "uniform":      t ~ Uniform(0, 1)
    "logit_normal": t ~ sigmoid(N(mean, std))  — concentrates on midpoint,
                    where the denoising task is hardest.
    """
    if method == "logit_normal":
        u = torch.randn(batch_size, device=device) * std + mean
        return torch.sigmoid(u)
    else:
        return torch.rand(batch_size, device=device)


# ── Flow matching loss ─────────────────────────────────────────────────────

def flow_matching_loss(
    model:    BidirMambaFlowNet,
    x_0:      torch.Tensor,       # (B, T, C) clean latents
    mask:     torch.Tensor,       # (B, T) bool — True = valid frame
    ctx:      torch.Tensor,       # (B, S, d_cond) text features
    ctx_mask: torch.Tensor,       # (B, S) bool
    cfg:      object,
) -> torch.Tensor:
    """
    OT-CFM loss with masking over padded frames.
    """
    B, T, C = x_0.shape
    device  = x_0.device

    # Sample noise and times
    z   = torch.randn_like(x_0)
    t   = sample_times(
        B, device,
        method = cfg.flow.time_sampling,
        mean   = cfg.flow.logit_normal_mean,
        std    = cfg.flow.logit_normal_std,
    )

    # Linear interpolation: x_t = (1 - t) * z + t * x_0
    t_expand = t[:, None, None]             # (B, 1, 1) for broadcasting
    x_t      = (1.0 - t_expand) * z + t_expand * x_0

    # Target velocity: x_0 - z  (constant along straight path)
    v_target = x_0 - z                     # (B, T, C)

    # Forward pass
    v_pred = model(x_t, t, ctx, ctx_mask)  # (B, T, C)

    # MSE loss, masked to valid frames only
    loss = F.mse_loss(v_pred, v_target, reduction="none")   # (B, T, C)
    loss = loss.mean(dim=-1)                                  # (B, T)
    loss = loss * mask.float()                               # zero out padding
    loss = loss.sum() / mask.float().sum().clamp(min=1.0)

    return loss


# ── LR schedule ───────────────────────────────────────────────────────────

def get_lr(step, warmup, max_steps, lr, lr_min_ratio):
    lr_min = lr * lr_min_ratio
    if step < warmup:
        return lr * step / max(1, warmup)
    prog  = (step - warmup) / max(1, max_steps - warmup)
    return lr_min + (lr - lr_min) * 0.5 * (1.0 + math.cos(math.pi * prog))


# ── Checkpoint helpers ─────────────────────────────────────────────────────

def save_checkpoint(path, step, model, conditioner, optimizer, scaler, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step":         step,
        "model_state":  model.state_dict(),
        "cond_state":   {k: v for k, v in conditioner.state_dict().items()
                         if "encoder" not in k},
        "optimizer":    optimizer.state_dict(),
        "scaler":       scaler.state_dict() if scaler else None,
        "model_config": {
            "latent_channels":    cfg.model.latent_channels,
            "d_model":            cfg.model.d_model,
            "n_layers":           cfg.model.n_layers,
            "d_state":            cfg.model.d_state,
            "d_conv":             cfg.model.d_conv,
            "expand":             cfg.model.expand,
            "dt_rank":            cfg.model.dt_rank,
            "conditioning_dim":   cfg.model.conditioning_dim,
            "conditioning_heads": cfg.model.conditioning_heads,
            "time_embed_dim":     cfg.model.time_embed_dim,
            "use_ada_ln":         cfg.model.use_ada_ln,
            "dropout":            cfg.model.dropout,
        },
    }
    torch.save(ckpt, path)
    print(f"  [ckpt] Saved → {path}")


def prune_checkpoints(ckpt_dir, keep_n):
    ckpts = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    for old in ckpts[:-keep_n]:
        old.unlink()


# ── Main ───────────────────────────────────────────────────────────────────

def train(cfg, resume_from: Optional[str] = None, use_default_captions: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.training.seed)

    print(f"\n{'='*60}")
    print(f" ATTM-Mamba v2  |  Bidirectional Mamba + Flow Matching")
    print(f" Device: {device}  |  Mixed precision: {cfg.training.mixed_precision}")
    print(f"{'='*60}\n")

    # ── DataLoaders ──────────────────────────────────────────────────────
    max_frames = int(cfg.audio.max_duration_sec * cfg.audio.latent_fps) + 5

    train_loader = build_flow_dataloader(
        latent_dir  = cfg.paths.latent_dir,
        caption_dir = cfg.paths.caption_dir,
        use_default_captions = use_default_captions,
        split       = "train",
        batch_size  = cfg.training.batch_size,
        num_workers = cfg.training.num_workers,
        max_frames  = max_frames,
    )
    val_loader = build_flow_dataloader(
        latent_dir  = cfg.paths.latent_dir,
        caption_dir = cfg.paths.caption_dir,
        use_default_captions = use_default_captions,
        split       = "val",
        batch_size  = cfg.training.batch_size,
        num_workers = cfg.training.num_workers,
        max_frames  = max_frames,
    )

    # ── Core model (trained from scratch) ────────────────────────────────
    model_cfg = BidirMambaConfig(
        latent_channels    = cfg.model.latent_channels,
        d_model            = cfg.model.d_model,
        n_layers           = cfg.model.n_layers,
        d_state            = cfg.model.d_state,
        d_conv             = cfg.model.d_conv,
        expand             = cfg.model.expand,
        dt_rank            = cfg.model.dt_rank,
        conditioning_dim   = cfg.model.conditioning_dim,
        conditioning_heads = cfg.model.conditioning_heads,
        time_embed_dim     = cfg.model.time_embed_dim,
        use_ada_ln         = cfg.model.use_ada_ln,
        dropout            = cfg.model.dropout,
    )
    model = BidirMambaFlowNet(model_cfg).to(device)
    n_params = model.count_parameters()
    print(f"[Model] BidirMambaFlowNet  |  {n_params / 1e6:.1f}M parameters")
    assert n_params <= 500e6, f"Core model exceeds 500M cap! ({n_params / 1e6:.1f}M)"

    # ── Text conditioner (auxiliary — T5, frozen) ─────────────────────────
    conditioner = T5TextConditioner(
        model_name = cfg.text_encoder.model_name,
        d_model    = cfg.model.conditioning_dim,
        max_length = cfg.text_encoder.max_text_len,
    ).to(device)

    # ── Optimiser ─────────────────────────────────────────────────────────
    trainable = list(model.parameters()) + [
        p for p in conditioner.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        betas=(0.9, 0.95),
    )

    # ── Mixed precision ───────────────────────────────────────────────────
    use_amp   = cfg.training.mixed_precision in ("fp16", "bf16")
    amp_dtype = torch.bfloat16 if cfg.training.mixed_precision == "bf16" else torch.float16
    scaler    = GradScaler() if cfg.training.mixed_precision == "fp16" else None

    # ── W&B ───────────────────────────────────────────────────────────────
    try:
        import wandb
        wandb.init(project="attm-mamba-flow", config=OmegaConf.to_container(cfg))
        use_wandb = True
    except Exception:
        use_wandb = False

    # ── Resume ────────────────────────────────────────────────────────────
    start_step = 0
    if resume_from:
        print(f"[Resume] {resume_from}")
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        conditioner.load_state_dict(ckpt.get("cond_state", {}), strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        if scaler and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt["step"] + 1
        print(f"[Resume] Step {start_step}")

    # ── Training loop ─────────────────────────────────────────────────────
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    ckpt_dir     = Path(project_root / cfg.paths.checkpoint_dir)
    model.train()
    conditioner.train()

    step         = start_step
    running_loss = 0.0
    t0           = time.time()
    train_iter   = iter(train_loader)

    while step < cfg.training.max_steps:

        lr = get_lr(step, cfg.training.warmup_steps, cfg.training.max_steps,
                    cfg.training.learning_rate, cfg.training.lr_min_ratio)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(cfg.training.gradient_accumulation_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            x_0      = batch["latents"].to(device, non_blocking=True)   # (B, T, C)
            mask     = batch["mask"].to(device, non_blocking=True)       # (B, T)
            texts    = batch["captions"]

            # Encode text (frozen T5)
            with torch.no_grad():
                ctx, ctx_mask = conditioner(
                    texts,
                    cfg_dropout_prob=cfg.flow.cfg_dropout_prob,
                    device=device,
                )

            with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                loss = flow_matching_loss(model, x_0, mask, ctx, ctx_mask, cfg)
                loss = loss / cfg.training.gradient_accumulation_steps

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss += loss.item()

        if scaler:
            scaler.unscale_(optimizer)

        grad_norm = nn.utils.clip_grad_norm_(trainable, cfg.training.grad_clip)

        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        running_loss += accum_loss
        step += 1

        # ── Logging ──
        if step % cfg.training.log_every == 0:
            avg  = running_loss / cfg.training.log_every
            sps  = cfg.training.log_every / (time.time() - t0)
            print(f"step {step:>7d} | loss {avg:.5f} | lr {lr:.2e} | "
                  f"gnorm {grad_norm:.3f} | {sps:.1f} steps/s")
            if use_wandb:
                wandb.log({"train/loss": avg, "lr": lr, "grad_norm": grad_norm}, step=step)
            running_loss = 0.0
            t0 = time.time()

        # ── Validation ──
        if step % cfg.training.eval_every == 0:
            val_loss = evaluate(model, conditioner, val_loader, device, use_amp, amp_dtype, cfg)
            print(f"  [val] step {step:>7d} | val_loss {val_loss:.5f}")
            if use_wandb:
                wandb.log({"val/loss": val_loss}, step=step)
            model.train()
            conditioner.train()

        # ── Checkpoint ──
        if step % cfg.training.save_every == 0:
            save_checkpoint(ckpt_dir / f"step_{step:08d}.pt", step,
                            model, conditioner, optimizer, scaler, cfg)
            prune_checkpoints(ckpt_dir, cfg.training.keep_last_n_checkpoints)

    print("\n[Training complete]")
    save_checkpoint(ckpt_dir / "final.pt", step, model, conditioner, optimizer, scaler, cfg)


@torch.no_grad()
def evaluate(model, conditioner, val_loader, device, use_amp, amp_dtype, cfg,
             max_batches=50):
    model.eval()
    conditioner.eval()
    total, n = 0.0, 0

    max_frames = int(cfg.audio.max_duration_sec * cfg.audio.latent_fps) + 5

    for batch in tqdm(val_loader, desc="Val", leave=False, total=max_batches):
        if n >= max_batches:
            break
        x_0   = batch["latents"].to(device)
        mask  = batch["mask"].to(device)
        texts = batch["captions"]

        ctx, ctx_mask = conditioner.encode_text(texts, device)

        with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            loss = flow_matching_loss(model, x_0, mask, ctx, ctx_mask, cfg)

        total += loss.item()
        n     += 1

    return total / max(n, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/flow_default.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--use_default_captions", default=True, action="store_true",
                        help="Whether to use competition-provided captions instead of generating your own with scripts/4_caption_audio.py")
    args = parser.parse_args()
    cfg  = OmegaConf.load(args.config)
    train(cfg, resume_from=args.resume, use_default_captions=args.use_default_captions)
