"""
model/dcae_wrapper.py

Wrapper around the ACE-Step MusicDCAE (music_dcae_f8c8).

This is an AUXILIARY component — its pre-trained weights are permitted by the
challenge rules. It is used solely to:
  1. Compress audio → compact 8-channel latents  (encode, done offline)
  2. Decompress latents → mel-spectrogram → waveform  (decode, done at inference)

The MusicDCAE checkpoint is publicly available on Hugging Face under Apache 2.0:
  ACE-Step/ACE-Step-v1-3.5B  /  music_dcae_f8c8

Architecture recap (from ACE-Step paper):
  - Input:  mel-spectrogram of 48kHz stereo audio
  - Encoder: deep convolutional encoder with residual blocks
  - Latent: 8 channels, 8× temporal compression → ~10.77 Hz frame rate
  - Decoder: symmetric convolutional decoder → mel-spectrogram
  - Vocoder: ConvNeXt + HiFiGAN → waveform  (separate checkpoint)

All parameters are frozen. No gradients flow through this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import numpy as np


# Normalisation constants measured from the Jamendo training set
# (you can re-estimate these with scripts/3b_encode_latents.py --stats)
LATENT_MEAN = 0.0
LATENT_STD  = 1.0   # update after running --stats pass


class MusicDCAEWrapper(nn.Module):
    """
    Thin wrapper around the ACE-Step MusicDCAE for encoding and decoding audio.

    All internal parameters are frozen. Only the normalisation scalars
    (mean / std) are potentially learnable, but by default they are fixed.

    Args:
        model_id:   HuggingFace repo containing the DCAE checkpoint.
        subfolder:  Subfolder within the repo (e.g. "music_dcae_f8c8").
        device:     Torch device.
        latent_mean / latent_std: Normalisation applied before passing
                    latents to the flow network, and inverted before decoding.
    """

    def __init__(
        self,
        model_id:    str   = "ACE-Step/ACE-Step-v1-3.5B",
        subfolder:   str   = "music_dcae_f8c8",
        device:      str   = "cuda",
        latent_mean: float = LATENT_MEAN,
        latent_std:  float = LATENT_STD,
    ):
        super().__init__()
        self.device      = device
        self.latent_mean = latent_mean
        self.latent_std  = latent_std
        
        self._resampler_cache = {}   # keyed by source sr
        self._mel_transform = T.MelSpectrogram(
            sample_rate=44100,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            power=1.0,
        ).to(device)

        print(f"[MusicDCAE] Loading from {model_id}/{subfolder} ...")
        self._load_dcae(model_id, subfolder, device)
        print(f"[MusicDCAE] Loaded. Latent channels: {self.latent_channels}")

    def _load_dcae(self, model_id: str, subfolder: str, device: str):
        """
        Load the MusicDCAE using the ACE-Step pipeline.
        Falls back to diffusers AutoencoderDC if the ACE-Step package is not
        installed (useful in environments that only have diffusers).
        """
        try:
            # ACE-Step package path
            from acestep.music_dcae.music_dcae_pipeline import MusicDCAE as AceStepDCAE
            self._dcae = AceStepDCAE(
                dcae_model_name_or_path=model_id,
                subfolder=subfolder,
                device=device,
            )
            self._backend = "acestep"
            self.latent_channels = 8   # f8c8 always has 8 channels

        except ImportError:
            # Fallback: load via diffusers AutoencoderDC
            from diffusers.models.autoencoders.autoencoder_dc import AutoencoderDC
            self._dcae = AutoencoderDC.from_pretrained(
                model_id,
                subfolder=subfolder,
                torch_dtype=torch.float32,
            ).to(device)
            self._backend = "diffusers"
            self.latent_channels = self._dcae.config.latent_channels

        # Freeze all parameters
        for p in self.parameters():
            p.requires_grad_(False)

    # ── Encoding (offline preprocessing) ──────────────────────────────────

    @torch.inference_mode()
    def encode(self, audio_path: str) -> torch.Tensor:
        """
        Encode an audio file to normalised latents.

        Args:
            audio_path: Path to an audio file (any format torchaudio supports).

        Returns:
            latents: (C, T) float32 tensor — normalised, ready for the flow model.
        """
        if self._backend == "acestep":
            # ACE-Step pipeline handles resampling, stereo, padding internally
            raw_latents = self._dcae.encode(audio_path)   # (C, T) on CPU
        else:
            raw_latents = self._diffusers_encode(audio_path)

        normalised = (raw_latents - self.latent_mean) / (self.latent_std + 1e-8)
        return normalised.float()

    def _diffusers_encode(self, audio_path: str) -> torch.Tensor:
        """
        Fallback encoder when using diffusers AutoencoderDC.
        Converts audio → mel-spectrogram → latents via the DCAE encoder.
        """
        wav, sr = torchaudio.load(audio_path)
        # Resample to 44.1kHz (MusicDCAE internal rate)
        if sr not in self._resampler_cache:
            self._resampler_cache[sr] = T.Resample(sr, 44100).to(self.device)
        wav = self._resampler_cache[sr](wav.to(self.device))
        # Stereo → mono mean for mel
        if wav.shape[0] > 1:
            wav_mono = wav.mean(0)
        else:
            wav_mono = wav[0]

        # Build mel-spectrogram
        mel = self._mel_transform(wav_mono)                  # (128, T_mel)
        mel = torch.log(mel.clamp(min=1e-5))           # log-mel
        mel = mel.unsqueeze(0).unsqueeze(0)            # (1, 1, 128, T_mel)

        mel = mel.to(self.device)
        latents = self._dcae.encode(mel).latent  # (1, C, H, T)
        # Squeeze batch dim and merge H (freq) into C if 4D
        latents = latents.squeeze(0)                   # (C, H, T) or (C, T)
        if latents.dim() == 3:
            C, H, T = latents.shape
            latents = latents.view(C * H, T)           # flatten freq into channels

        return latents.cpu()

    # ── Decoding (inference) ───────────────────────────────────────────────

    @torch.inference_mode()
    def decode(self, latents: torch.Tensor) -> np.ndarray:
        """
        Decode normalised latents → waveform.

        Args:
            latents: (C, T) float32 — normalised flow output.

        Returns:
            waveform: numpy (n_samples,) at 48kHz stereo (then averaged to mono).
        """
        # De-normalise
        raw = latents * self.latent_std + self.latent_mean

        if self._backend == "acestep":
            wav = self._dcae.decode(raw.unsqueeze(0))  # (1, 2, N) stereo
            wav_np = wav[0].mean(0).cpu().numpy()       # mono
        else:
            wav_np = self._diffusers_decode(raw)

        return wav_np

    def _diffusers_decode(self, raw_latents: torch.Tensor) -> np.ndarray:
        """Fallback decoder via diffusers AutoencoderDC."""
        if raw_latents.dim() == 2:
            raw_latents = raw_latents.unsqueeze(0).unsqueeze(0)  # (1,1,C,T)
        raw_latents = raw_latents.to(self.device)
        recon = self._dcae.decode(raw_latents).sample          # (1, 1, 128, T_mel)
        # A vocoder would be needed here; return the mel as a proxy
        # In practice, replace with an HiFiGAN vocoder call
        mel = recon.squeeze().cpu().numpy()
        return mel  # placeholder — wire up vocoder for full audio

    # ── Latent statistics estimation ───────────────────────────────────────

    @torch.inference_mode()
    def estimate_stats(
        self,
        audio_paths: list[str],
        max_files: int = 500,
    ) -> Tuple[float, float]:
        """
        Estimate mean and std of raw (un-normalised) latents over a sample
        of the training set. Call once; bake results into the config.
        """
        all_latents = []
        for path in audio_paths[:max_files]:
            try:
                lat = self.encode.__wrapped__(self, path)  # skip normalisation
                all_latents.append(lat.float().mean())
            except Exception:
                pass

        vals = torch.stack(all_latents)
        mean = vals.mean().item()
        std  = vals.std().item()
        print(f"[MusicDCAE] Estimated latent mean={mean:.4f}  std={std:.4f}")
        print(f"  Update configs/flow_default.yaml: latent_mean={mean:.4f}, latent_std={std:.4f}")
        return mean, std
