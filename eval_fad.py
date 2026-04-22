#!/usr/bin/env python3
"""
eval_fad.py

Computes Fréchet Audio Distance (FAD) between generated audio and a reference
set, using the CLAP-Laion-Music checkpoint as the embedder — matching the
exact evaluation setup of the ATTM Grand Challenge.

FAD is computed as:
    FAD = ||μ_gen - μ_ref||² + Tr(Σ_gen + Σ_ref - 2*(Σ_gen @ Σ_ref)^(1/2))

Lower is better. A FAD of 0 would mean the generated distribution is
identical to the reference distribution in embedding space.

Usage:
    # First pass: embed and cache the reference set (Jamendo)
    python eval_fad.py \
        --reference_dir ../mtg_jamendo_separated \
        --generated_dir ./generated_audio \
        --ckpt_path     ./music_audioset_epoch_15_esc_90.14.pt \
        --cache_path    ./jamendo_clap_stats.npz

    # Subsequent runs reuse the cached reference stats (much faster)
    python eval_fad.py \
        --generated_dir ./generated_audio \
        --ckpt_path     ./music_audioset_epoch_15_esc_90.14.pt \
        --cache_path    ./jamendo_clap_stats.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import laion_clap
from scipy import linalg
from tqdm import tqdm


# ── Embedding extraction ───────────────────────────────────────────────────

def get_audio_embeddings(
    model:      laion_clap.CLAP_Module,
    audio_dir:  str,
    batch_size: int = 16,
    max_files:  int = None,
    device:     str = "cuda",
) -> np.ndarray:
    """
    Extract L2-normalised CLAP audio embeddings for all files in audio_dir.
    Returns array of shape (N, embed_dim).
    """
    extensions = {".wav", ".flac", ".mp3", ".ogg"}
    paths = sorted([
        p for p in Path(audio_dir).rglob("*")
        if p.suffix.lower() in extensions
    ])

    if max_files is not None:
        paths = paths[:max_files]

    if not paths:
        raise FileNotFoundError(f"No audio files found in {audio_dir}")

    print(f"  Embedding {len(paths):,} files from {audio_dir} ...")
    all_embeddings = []

    for i in tqdm(range(0, len(paths), batch_size), desc="  Embedding"):
        batch = [str(p) for p in paths[i : i + batch_size]]
        with torch.inference_mode():
            emb = model.get_audio_embedding_from_filelist(
                x=batch, use_tensor=True
            )                                          # (B, embed_dim)
            emb = F.normalize(emb, dim=-1)            # L2 normalise
        all_embeddings.append(emb.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)      # (N, embed_dim)


# ── Gaussian statistics ────────────────────────────────────────────────────

def compute_gaussian_stats(embeddings: np.ndarray):
    """Compute mean vector and covariance matrix of an embedding set."""
    mu  = np.mean(embeddings, axis=0)
    cov = np.cov(embeddings, rowvar=False)
    return mu, cov


# ── FAD ───────────────────────────────────────────────────────────────────

def frechet_distance(
    mu1: np.ndarray, cov1: np.ndarray,
    mu2: np.ndarray, cov2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Fréchet distance between two multivariate Gaussians.
    Numerically stabilised with epsilon on the diagonal.
    """
    diff = mu1 - mu2
    mean_term = diff @ diff

    # Matrix square root via eigendecomposition (more stable than sqrtm
    # for high-dimensional embeddings)
    covmean, _ = linalg.sqrtm(cov1 @ cov2, disp=False)

    # Handle numerical issues that produce small imaginary components
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            print("  [WARN] sqrtm produced large imaginary components — "
                  "increasing epsilon.")
        covmean = covmean.real

    # Numerical stabilisation: add eps to diagonal if cov is near-singular
    offset = np.eye(cov1.shape[0]) * eps
    if np.isnan(covmean).any():
        print("  [WARN] sqrtm produced NaN — applying diagonal offset.")
        covmean, _ = linalg.sqrtm((cov1 + offset) @ (cov2 + offset), disp=False)
        covmean = covmean.real

    trace_term = np.trace(cov1) + np.trace(cov2) - 2 * np.trace(covmean)
    fad = float(mean_term + trace_term)
    return fad


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_dir",  required=True,
                        help="Directory of generated audio files")
    parser.add_argument("--reference_dir",  default=None,
                        help="Directory of reference audio (e.g. Jamendo). "
                             "Required if --cache_path does not exist yet.")
    parser.add_argument("--ckpt_path",      required=True,
                        help="Path to music_audioset_epoch_15_esc_90.14.pt")
    parser.add_argument("--cache_path",     default="./reference_clap_stats.npz",
                        help="Where to cache reference embeddings/stats. "
                             "If the file exists, reference_dir is not needed.")
    parser.add_argument("--max_ref_files",  type=int, default=5000,
                        help="Max reference files to embed (default 5000). "
                             "More files = more stable FAD estimate.")
    parser.add_argument("--batch_size",     type=int, default=16)
    parser.add_argument("--output_json",    default="fad_results.json")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # ── Load CLAP model ────────────────────────────────────────────────────
    print(f"Loading CLAP model from {args.ckpt_path} ...")
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    model.load_ckpt(args.ckpt_path)
    model.eval().to(args.device)
    print("CLAP model loaded.\n")

    # ── Reference statistics (cached or computed) ──────────────────────────
    cache = Path(args.cache_path)

    if cache.exists():
        print(f"Loading cached reference stats from {cache} ...")
        data    = np.load(cache)
        mu_ref  = data["mu"]
        cov_ref = data["cov"]
        n_ref   = int(data["n_files"])
        print(f"  Loaded stats for {n_ref:,} reference files.\n")

    else:
        if args.reference_dir is None:
            raise ValueError(
                "--reference_dir is required when cache does not exist yet."
            )
        print("Computing reference embeddings (this runs once and is cached)...")
        ref_embeddings = get_audio_embeddings(
            model      = model,
            audio_dir  = args.reference_dir,
            batch_size = args.batch_size,
            max_files  = args.max_ref_files,
            device     = args.device,
        )
        mu_ref, cov_ref = compute_gaussian_stats(ref_embeddings)
        np.savez(cache, mu=mu_ref, cov=cov_ref, n_files=len(ref_embeddings))
        print(f"  Reference stats cached to {cache}  "
              f"({len(ref_embeddings):,} files, embed_dim={ref_embeddings.shape[1]})\n")
        n_ref = len(ref_embeddings)

    # ── Generated audio embeddings ─────────────────────────────────────────
    print("Computing generated audio embeddings ...")
    gen_embeddings = get_audio_embeddings(
        model      = model,
        audio_dir  = args.generated_dir,
        batch_size = args.batch_size,
        device     = args.device,
    )
    mu_gen, cov_gen = compute_gaussian_stats(gen_embeddings)
    print(f"  Generated: {len(gen_embeddings)} files, "
          f"embed_dim={gen_embeddings.shape[1]}\n")

    # ── FAD ────────────────────────────────────────────────────────────────
    print("Computing FAD ...")
    fad = frechet_distance(mu_gen, cov_gen, mu_ref, cov_ref)

    print(f"\n{'='*60}")
    print(f" Fréchet Audio Distance (FAD)")
    print(f" Reference : {args.reference_dir or cache}  ({n_ref:,} files)")
    print(f" Generated : {args.generated_dir}  ({len(gen_embeddings)} files)")
    print(f" FAD       : {fad:.4f}   (lower is better)")
    print(f"{'='*60}\n")

    results = {
        "fad":            fad,
        "n_generated":    len(gen_embeddings),
        "n_reference":    n_ref,
        "reference_dir":  str(args.reference_dir or cache),
        "generated_dir":  str(args.generated_dir),
    }
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()