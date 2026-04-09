"""
data/flow_dataset.py

Dataset for the DCAE + flow matching pipeline.

Loads pre-computed DCAE latent tensors (.pt files, shape (C, T) float16)
paired with caption .txt files.

Unlike the autoregressive dataset, there is no codebook delay pattern —
latents are continuous tensors that the flow model denoises directly.
"""

import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


class FlowDataset(Dataset):
    """
    Paired (DCAE latent, caption) dataset for flow matching training.

    Args:
        latent_dir:   Root of pre-computed .pt latent files (shape: (C, T) fp16)
        caption_dir:  Root of caption .txt files
        max_frames:   Maximum number of latent frames (trim longer clips)
        split:        "train" | "val" | None
        val_fraction: Fraction held out for validation
        seed:         Random seed for split
    """

    def __init__(
        self,
        latent_dir:   str,
        caption_dir:  str,
        max_frames:   int   = 330,      # ~30s at 10.77 Hz
        split:        Optional[str] = None,
        val_fraction: float = 0.01,
        seed:         int   = 42,
        use_default_captions: bool = True,
    ):
        self.latent_dir  = Path(latent_dir)
        self.caption_dir = Path(caption_dir)
        self.max_frames  = max_frames
        self.use_default_captions = use_default_captions

        # ── Pair up files ──
        latent_files = sorted(self.latent_dir.rglob("*.pt"))
        self.pairs: List[Tuple[Path, Path | str]] = []
        
        if not use_default_captions:
            for lf in latent_files:
                rel = lf.relative_to(self.latent_dir)
                cf  = (self.caption_dir / rel).with_suffix(".txt")
                if cf.exists():
                    self.pairs.append((lf, cf))
        else:
            import json
            file_location = Path(__file__).resolve().parent
            default_captions_dir = file_location.parent / "default_captions"
            default_captions_dir = default_captions_dir.resolve()
            caption_and_path = json.load(open(default_captions_dir / "jamendo_qwen.json", 'r'))
            path_to_caption = {item['path']: item['caption'] for item in caption_and_path}
            
            for lf in latent_files:
                rel = lf.relative_to(self.latent_dir)
                audio_rel = rel.with_suffix(".mp3").as_posix()
                caption = path_to_caption.get(audio_rel, None)
                
                if caption is None:
                    print(f"  [WARN] No caption found for {audio_rel}, skipping.")
                if caption is not None:
                    self.pairs.append((lf, caption))

        if not self.pairs:
            raise FileNotFoundError(
                f"No (latent, caption) pairs found.\n"
                f"  latent_dir:  {self.latent_dir}\n"
                f"  caption_dir: {self.caption_dir}\n"
                "Run scripts/3b_encode_latents.py and scripts/4_caption_audio.py first."
            )

        # ── Train / val split ──
        rng = random.Random(seed)
        shuffled = list(self.pairs)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_fraction))

        if split == "val":
            self.pairs = shuffled[:n_val]
        elif split == "train":
            self.pairs = shuffled[n_val:]
        else:
            self.pairs = shuffled

        print(
            f"[FlowDataset] split={split or 'all'}  "
            f"pairs={len(self.pairs):,}  max_frames={max_frames}"
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        latent_path, caption_path = self.pairs[idx]

        # ── Load latent ──
        latent = torch.load(latent_path, weights_only=True).float()  # (C, T)

        # Random temporal crop if longer than max_frames
        C, T = latent.shape
        if T > self.max_frames:
            start = random.randint(0, T - self.max_frames)
            latent = latent[:, start: start + self.max_frames]

        # Rearrange to (T, C) for sequence-first processing
        latent = latent.permute(1, 0)   # (T, C)

        # ── Load caption ──
        if isinstance(caption_path, str):
            caption = caption_path  # already a caption string from default captions
        else:
            caption = caption_path.read_text(encoding="utf-8").strip()

        return {
            "latent":  latent,    # (T, C) float32
            "caption": caption,
        }


def flow_collate_fn(batch: List[dict]) -> dict:
    """
    Pad variable-length latent sequences to the same T in a batch.
    Padding is with zeros (noise will be mixed in during training anyway).
    """
    latents  = [item["latent"]  for item in batch]
    captions = [item["caption"] for item in batch]
    lengths  = [lat.shape[0]    for lat in latents]

    max_T = max(lengths)
    C     = latents[0].shape[1]

    padded = torch.zeros(len(batch), max_T, C)
    mask   = torch.zeros(len(batch), max_T, dtype=torch.bool)

    for i, (lat, ln) in enumerate(zip(latents, lengths)):
        padded[i, :ln] = lat
        mask[i,   :ln] = True

    return {
        "latents":  padded,                               # (B, T_max, C)
        "lengths":  torch.tensor(lengths, dtype=torch.long),
        "mask":     mask,                                 # (B, T_max) True = real frame
        "captions": captions,
    }


def build_flow_dataloader(
    latent_dir:   str,
    caption_dir:  str,
    use_default_captions: bool,
    split:        str,
    batch_size:   int,
    num_workers:  int   = 4,
    max_frames:   int   = 330,
    val_fraction: float = 0.01,
    seed:         int   = 42,
) -> DataLoader:

    dataset = FlowDataset(
        latent_dir   = latent_dir,
        caption_dir  = caption_dir,
        max_frames   = max_frames,
        split        = split,
        val_fraction = val_fraction,
        seed         = seed,
        use_default_captions = use_default_captions,
    )
    is_train = (split == "train")

    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = is_train,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = is_train,
        collate_fn  = flow_collate_fn,
        persistent_workers = (num_workers > 0),
    )
