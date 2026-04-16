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
  - Input:  mel-spectrogram of 44.1kHz stereo audio
  - Encoder: deep convolutional encoder with residual blocks
  - Latent: 8 channels, 8× temporal compression → ~10.77 Hz frame rate
  - Decoder: symmetric convolutional decoder → mel-spectrogram
  - Vocoder: ConvNeXt + HiFiGAN → waveform  (separate checkpoint)

All parameters are frozen. No gradients flow through this module.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms
import torch.nn.functional as F
import numpy as np
from huggingface_hub import snapshot_download


# Normalisation constants measured from the Jamendo training set
# (you can re-estimate these with scripts/3_encode_latents.py --stats)
LATENT_MEAN = 0.0527
LATENT_STD  = 0.8134   # update after running --stats pass


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
        self._mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=44100,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            power=1.0,
        ).to(device)

        print(f"[MusicDCAE] Loading autoencoder from {model_id}/{subfolder} ...")
        self._load_dcae(model_id, subfolder, device)
        print(f"[MusicDCAE] Autoencoder loaded with {self._backend}. Latent channels: {self.latent_channels}")

    def _load_dcae(self, model_id: str, subfolder: str, device: str):
        """
        Load the MusicDCAE using the ACE-Step pipeline.
        Falls back to diffusers AutoencoderDC if the ACE-Step package is not
        installed (useful in environments that only have diffusers).
        """
        # ACE-Step package path
        try:
            from acestep.music_dcae.music_dcae_pipeline import MusicDCAE as AceStepDCAE
            repo_id = "ACE-Step/ACE-Step-v1-3.5B"
            local_repo_path = snapshot_download(repo_id=repo_id)
                
            dcae_local_path = os.path.join(local_repo_path, "music_dcae_f8c8")
            vocoder_local_path = os.path.join(local_repo_path, "music_vocoder")
                
            self._dcae = AceStepDCAE(
                source_sample_rate=44100,
                dcae_checkpoint_path=dcae_local_path,
                vocoder_checkpoint_path=vocoder_local_path
                ).to('cuda')
            self._backend = "acestep"
            self.latent_channels = 8   # f8c8 always has 8 channels

        except Exception:
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
            audio, sr = torchaudio.load(audio_path)
            audios = audio.unsqueeze(0).to(self.device)
            
            raw_latents, length = self._dcae.encode(audios, sr=44100)
            raw_latents = raw_latents.squeeze(0)
            if raw_latents.dim() == 3:
                C, H, T = raw_latents.shape
                raw_latents = raw_latents.reshape(C * H, T)  # flatten freq into channels
            raw_latents = raw_latents.cpu()
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
            self._resampler_cache[sr] = torchaudio.transforms.Resample(sr, 44100).to(self.device)
        wav = self._resampler_cache[sr](wav.to(self.device))

        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2]  # take first 2 channels if more than stereo

        mel_L = self._mel_transform(wav[0])   # (128, T_mel)
        mel_R = self._mel_transform(wav[1])   # (128, T_mel)

        mel_L = torch.log(mel_L.clamp(min=1e-5))
        mel_R = torch.log(mel_R.clamp(min=1e-5))

        # Stack to stereo: (1, 2, 128, T_mel)
        mel = torch.stack([mel_L, mel_R], dim=0)   # (2, 128, T_mel)
        
        temporal_downscale_factor = 8
        T_mel = mel.size(-1)
        remainder = T_mel % temporal_downscale_factor
        if remainder > 0:
            pad_len = temporal_downscale_factor - remainder
            # Pad the last dimension (time) by pad_len on the right side
            mel = F.pad(mel, (0, pad_len))
            
        mel = mel.unsqueeze(0)                      # (1, 2, 128, T_mel)

        mel = mel.to(self.device)
        latents = self._dcae.encode(mel).latent  # (1, C, H, T)
        
        # Squeeze batch dim and merge H (freq) into C if 4D
        latents = latents.squeeze(0)                   # (C, H, T) or (C, T)
        if latents.dim() == 3:
            C, H, T = latents.shape
            latents = latents.reshape(C * H, T)           # flatten freq into channels

        return latents.cpu()

    # ── Decoding (inference) ───────────────────────────────────────────────

    @torch.inference_mode()
    def decode(self, latents: torch.Tensor) -> np.ndarray:
        """
        Decode normalised latents → waveform.

        Args:
            latents: (C, T) float32 — normalised flow output.

        Returns:
            waveform: numpy (n_samples,) at 44.1kHz stereo.
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
        running_sum = 0.0
        running_sq_sum = 0.0
        total_elements = 0

        for path in audio_paths[:max_files]:
            try:
                if self._backend == "acestep":
                    audio, sr = torchaudio.load(path)
                    audios = audio.unsqueeze(0).to(self.device)
            
                    raw_latents, length = self._dcae.encode(audios, sr=44100)
                    raw_latents = raw_latents.squeeze(0)
                    if raw_latents.dim() == 3:
                        C, H, T = raw_latents.shape
                        raw_latents = raw_latents.reshape(C * H, T)  # flatten freq into channels
                    raw_latents = raw_latents.cpu()
                else:
                    raw_latents = self._diffusers_encode(path)
                
                lat = raw_latents.float()

            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue
            
            running_sum += lat.sum().item()
            running_sq_sum += (lat ** 2).sum().item()
            total_elements += lat.numel()

        if total_elements == 0:
            raise ValueError("No latents were successfully processed.")

        mean = running_sum / total_elements
        variance = (running_sq_sum / total_elements) - (mean ** 2)
        
        # Guard against floating-point inaccuracies making variance slightly negative
        variance = max(0.0, variance) 
        std = variance ** 0.5

        print(f"[MusicDCAE] Estimated latent mean={mean:.4f}  std={std:.4f}")
        print(f"  Update configs/flow_default.yaml: latent_mean={mean:.4f}, latent_std={std:.4f}")
        return mean, std
