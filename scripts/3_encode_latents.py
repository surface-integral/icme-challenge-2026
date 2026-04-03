#!/usr/bin/env python3
"""
scripts/3b_encode_latents.py

Pre-computes MusicDCAE latents for every audio file and saves them as .pt files.
This replaces scripts/3_tokenize_audio.py in the flow-matching pipeline.

The MusicDCAE (ACE-Step/ACE-Step-v1-3.5B / music_dcae_f8c8) is an auxiliary
component with pre-trained weights permitted by the challenge rules.

Latent shape per 30s clip:
  (8, ~323)  — 8 channels, ~10.77 Hz temporal resolution

Usage:
    python scripts/3b_encode_latents.py \
        --audio_dir  ../mtg_jamendo_separated \
        --output_dir ../mtg_latents \
        [--model_id  ACE-Step/ACE-Step-v1-3.5B] \
        [--stats]        # estimate mean/std before encoding
        [--device cuda]

    # After running with --stats, update latent_mean / latent_std in
    # configs/flow_default.yaml, then re-run without --stats.
"""

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Pre-compute MusicDCAE latents for ATTM flow training")
    p.add_argument("--audio_dir",   required=True)
    p.add_argument("--output_dir",  required=True)
    p.add_argument("--model_id",    default="ACE-Step/ACE-Step-v1-3.5B")
    p.add_argument("--subfolder",   default="music_dcae_f8c8")
    p.add_argument("--max_duration",type=float, default=30.0)
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--overwrite",   action="store_true")
    p.add_argument("--stats",       action="store_true",
                   help="Run a statistics pass first to estimate latent mean/std")
    p.add_argument("--latent_mean", type=float, default=0.0)
    p.add_argument("--latent_std",  type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    audio_dir  = Path(args.audio_dir)
    output_dir = Path(args.output_dir)

    if not audio_dir.exists():
        sys.exit(f"[ERROR] audio_dir does not exist: {audio_dir}")

    extensions = {".mp3", ".wav", ".flac", ".ogg"}
    audio_files = sorted([p for p in audio_dir.rglob("*") if p.suffix.lower() in extensions])
    if not audio_files:
        sys.exit(f"[ERROR] No audio files found in {audio_dir}")

    print(f"Found {len(audio_files):,} audio files.")

    # ── Load MusicDCAE wrapper ──────────────────────────────────────────────
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from model.dcae_wrapper import MusicDCAEWrapper

    dcae = MusicDCAEWrapper(
        model_id    = args.model_id,
        subfolder   = args.subfolder,
        device      = args.device,
        latent_mean = args.latent_mean,
        latent_std  = args.latent_std,
    )

    # ── Optional statistics pass ────────────────────────────────────────────
    if args.stats:
        print("\n[Stats pass] Estimating latent mean/std from first 500 files...")
        audio_paths_str = [str(f) for f in audio_files[:500]]
        mean, std = dcae.estimate_stats(audio_paths_str)
        print(f"\nRe-run with: --latent_mean {mean:.4f} --latent_std {std:.4f}")
        return

    # ── Filter already-encoded ──────────────────────────────────────────────
    if not args.overwrite:
        todo = [
            af for af in audio_files
            if not (output_dir / af.relative_to(audio_dir)).with_suffix(".pt").exists()
        ]
        skipped = len(audio_files) - len(todo)
        if skipped:
            print(f"Skipping {skipped:,} already-encoded files (use --overwrite to redo).")
        audio_files = todo

    if not audio_files:
        print("Nothing to do.")
        return

    # ── Encode loop ─────────────────────────────────────────────────────────
    errors    = 0
    processed = 0

    for af in tqdm(audio_files, unit="file", desc="Encoding latents"):
        try:
            latents = dcae.encode(str(af))   # (C, T) normalised float32
        except Exception as e:
            print(f"  [WARN] Failed to encode {af.name}: {e}", file=sys.stderr)
            errors += 1
            continue

        rel   = af.relative_to(audio_dir)
        out_f = (output_dir / rel).with_suffix(".pt")
        out_f.parent.mkdir(parents=True, exist_ok=True)
        torch.save(latents.cpu().half(), out_f)   # save as fp16 to halve disk usage
        processed += 1

    print(f"\n{'='*60}")
    print(f" Latent encoding complete!")
    print(f"   Processed : {processed:,}")
    print(f"   Errors    : {errors:,}")
    print(f"   Output    : {output_dir}")
    print(f"\n Next step: if you want to generate captions, run:")
    print(f"              python scripts/4_caption_audio.py \\")
    print(f"              --audio_dir {args.audio_dir} \\")
    print(f"              --output_dir ../mtg_captions")
    print("\n Otherwise to use competition-provided captions, skip to training with:")
    print(f"\n Next step: python training/train_flow.py --config configs/flow_default.yaml --use_default_captions True")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
