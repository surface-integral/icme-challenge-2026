"""
model/conditioning.py

Text conditioning via a frozen T5 encoder (auxiliary pre-trained component).

The T5 encoder is permitted under the challenge rules because it falls under
"auxiliary components" (text encoders). Its weights are frozen — only the
Mamba LM is trained.

For classifier-free guidance (CFG), we support:
  - Conditional forward: pass text features normally
  - Unconditional forward: replace text features with a learned null embedding

At inference, CFG mixes conditional and unconditional logits:
    logits = logits_uncond + cfg_coeff × (logits_cond - logits_uncond)
"""

import torch
import torch.nn as nn
from transformers import T5EncoderModel, AutoTokenizer
from typing import Optional, Tuple


class T5TextConditioner(nn.Module):
    """
    Wraps a frozen T5 encoder and projects its output to the Mamba d_model
    dimension (if needed — when d_model ≠ T5 hidden size).

    The null embedding for classifier-free guidance is a learnable parameter
    of the same shape as a single T5 token sequence.
    """

    def __init__(
        self,
        model_name: str  = "google/flan-t5-base",
        d_model: int     = 1024,
        max_length: int  = 128,
        null_seq_len: int = 1,        # Length of learnable null sequence
    ):
        super().__init__()
        self.max_length = max_length
        self.d_model    = d_model

        # ── Load frozen T5 encoder ────────────────────────────────────────
        print(f"[T5TextConditioner] Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder   = T5EncoderModel.from_pretrained(model_name)

        # Freeze all T5 parameters
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        t5_dim = self.encoder.config.hidden_size  # 768 for T5-base

        # Optional projection when T5 dim ≠ d_model
        if t5_dim != d_model:
            self.proj = nn.Linear(t5_dim, d_model, bias=False)
        else:
            self.proj = nn.Identity()

        # Learnable null / unconditional embedding for CFG
        # Shape: (null_seq_len, d_model) — broadcast over batch dimension
        self.null_embed = nn.Parameter(
            torch.randn(null_seq_len, d_model) * 0.02
        )

        print(f"[T5TextConditioner] T5 hidden={t5_dim}, d_model={d_model}, "
              f"projection={'Linear' if t5_dim != d_model else 'none'}")

    @property
    def device(self):
        return next(self.encoder.parameters()).device

    @torch.no_grad()
    def encode_text(
        self,
        texts: list[str],
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenise and encode a list of text prompts with the frozen T5 encoder.

        Returns:
            features:  (B, S, d_model)  — T5 features projected to d_model
            attn_mask: (B, S) bool      — True for real tokens
        """
        device = device or self.device

        tokenised = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(device)

        encoder_out = self.encoder(
            input_ids      = tokenised["input_ids"],
            attention_mask = tokenised["attention_mask"],
        )

        features = encoder_out.last_hidden_state.float()   # (B, S, t5_dim)
        features = self.proj(features)                      # (B, S, d_model)

        attn_mask = tokenised["attention_mask"].bool()      # (B, S)

        return features, attn_mask

    def get_null_conditioning(self, batch_size: int, device: torch.device):
        """
        Returns the unconditional (null) text features for CFG.

        Returns:
            null_feat:  (B, 1, d_model)
            null_mask:  (B, 1) bool, all True
        """
        null_feat = self.null_embed.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        null_mask = torch.ones(batch_size, null_feat.shape[1], dtype=torch.bool, device=device)
        return null_feat, null_mask

    def forward(
        self,
        texts: list[str],
        cfg_dropout_prob: float = 0.0,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode texts and optionally replace some with the null embedding
        for classifier-free guidance training.

        Args:
            texts:            List of B text prompts
            cfg_dropout_prob: Probability of replacing a batch item with null
            device:           Target device

        Returns:
            context:      (B, S, d_model)
            context_mask: (B, S) bool
        """
        device = device or self.device
        B = len(texts)

        context, context_mask = self.encode_text(texts, device)

        # CFG training: randomly replace some samples with null conditioning
        if cfg_dropout_prob > 0.0 and self.training:
            null_feat, null_mask = self.get_null_conditioning(B, device)

            drop_mask = torch.rand(B, device=device) < cfg_dropout_prob  # (B,)

            for b in range(B):
                if drop_mask[b]:
                    # Replace with null embedding (pad to same S length)
                    S = context.shape[1]
                    context[b]      = null_feat[b].expand(S, -1)
                    context_mask[b] = False
                    context_mask[b, :null_feat.shape[1]] = True

        return context, context_mask
