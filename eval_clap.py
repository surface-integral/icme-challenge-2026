#!/usr/bin/env python3
"""
eval_clap.py

Computes CLAP score between generated audio files and their text prompts.
Uses the exact checkpoint specified by the ATTM challenge: 
  music_audioset_epoch_15_esc_90.14.pt

Usage:
    python eval_clap.py \
        --audio_dir  ./generated_audio \
        --prompt_file ./test_prompts.txt \
        --ckpt_path  ./music_audioset_epoch_15_esc_90.14.pt

Assumes audio files are named to match prompt order, e.g.:
    0000.wav, 0001.wav, ... or prompt_0.wav, prompt_1.wav, ...
Or pass a JSON file mapping filename → prompt.
"""

import argparse
import json
import os
import csv
from pathlib import Path

import numpy as np
import torch
import laion_clap


def load_prompts(prompt_file: str) -> list[str]:
    """Load prompts from a .csv file"""
    path = Path(prompt_file)
    prompts = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append(row['prompt'])
    return prompts

def load_audio_paths(audio_dir: str) -> list[Path]:
    """Return sorted list of .wav/.flac/.mp3 files in audio_dir."""
    extensions = {".wav", ".flac", ".mp3", ".ogg"}
    paths = sorted([
        p for p in Path(audio_dir).iterdir()
        if p.suffix.lower() in extensions
    ], key=lambda f: int(f.name.split('/')[-1].split('.')[0]))  # Sort by numeric filename
    return paths


def compute_clap_score(
    audio_paths: list[Path],
    prompts:     list[str],
    ckpt_path:   str,
    device:      str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size:  int = 16,
) -> dict:
    """
    Compute per-sample and mean CLAP cosine similarity scores.

    Returns a dict with:
        scores      — list of per-sample cosine similarities
        mean        — mean CLAP score across all samples
        std         — standard deviation
        per_sample  — list of (audio_filename, prompt, score) tuples
    """
    assert len(audio_paths) == len(prompts), (
        f"Mismatch: {len(audio_paths)} audio files but {len(prompts)} prompts."
    )

    print(f"Loading CLAP model from {ckpt_path} ...")
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    model.load_ckpt(ckpt_path)
    model.eval()
    model = model.to(device)
    print(f"CLAP model loaded on {device}.\n")

    all_scores = []
    per_sample = []

    # Process in batches
    for i in range(0, len(audio_paths), batch_size):
        batch_paths   = audio_paths[i : i + batch_size]
        batch_prompts = prompts[i : i + batch_size]

        # ── Audio embeddings ──────────────────────────────────────────────
        audio_embed = model.get_audio_embedding_from_filelist(
            x=[str(p) for p in batch_paths],
            use_tensor=True,
        )                                      # (B, embed_dim)

        # ── Text embeddings ───────────────────────────────────────────────
        text_embed = model.get_text_embedding(
            batch_prompts,
            use_tensor=True,
        )                                      # (B, embed_dim)

        # ── Cosine similarity (per sample, not cross-matrix) ──────────────
        audio_norm = torch.nn.functional.normalize(audio_embed, dim=-1)
        text_norm  = torch.nn.functional.normalize(text_embed,  dim=-1)
        scores     = (audio_norm * text_norm).sum(dim=-1)   # (B,) dot product

        for path, prompt, score in zip(batch_paths, batch_prompts, scores):
            s = score.item()
            all_scores.append(s)
            per_sample.append((path.name, prompt, s))
            print(f"  [{path.name}]  score={s:.4f}  |  {prompt[:70]}")

    mean_score = float(np.mean(all_scores))
    std_score  = float(np.std(all_scores))

    print(f"\n{'='*60}")
    print(f" CLAP Score Results")
    print(f" Samples evaluated : {len(all_scores)}")
    print(f" Mean CLAP score   : {mean_score:.4f}")
    print(f" Std               : {std_score:.4f}")
    print(f" Min               : {min(all_scores):.4f}")
    print(f" Max               : {max(all_scores):.4f}")
    print(f"{'='*60}\n")

    return {
        "mean":       mean_score,
        "std":        std_score,
        "scores":     all_scores,
        "per_sample": per_sample,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir",   required=True,
                        help="Directory of generated .wav files")
    parser.add_argument("--prompt_file", required=True,
                        help=".txt (one prompt per line) or .json file")
    parser.add_argument("--ckpt_path",   required=True,
                        help="Path to music_audioset_epoch_15_esc_90.14.pt")
    parser.add_argument("--output_json", default="clap_results.json",
                        help="Where to save per-sample results")
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    audio_paths = load_audio_paths(args.audio_dir)
    prompts     = load_prompts(args.prompt_file)

    print(f"Found {len(audio_paths)} audio files and {len(prompts)} prompts.\n")

    results = compute_clap_score(
        audio_paths = audio_paths,
        prompts     = prompts,
        ckpt_path   = args.ckpt_path,
        device      = args.device,
        batch_size  = args.batch_size,
    )

    # Save detailed results
    output = {
        "mean_clap_score": results["mean"],
        "std":             results["std"],
        "per_sample": [
            {"file": name, "prompt": prompt, "score": score}
            for name, prompt, score in results["per_sample"]
        ],
    }
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Per-sample results saved to {args.output_json}")


if __name__ == "__main__":
    main()