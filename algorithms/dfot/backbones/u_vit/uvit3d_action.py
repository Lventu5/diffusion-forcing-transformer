"""
UViT3DAction — UViT3D backbone with structured action conditioning.

Overrides `_build_external_cond_embedding` to replace the generic MLP-based
RandomDropoutCondEmbedding with ActionCondEmbedding, which uses:
  - A learnable nn.Embedding for action type indices (0–5)
  - Sinusoidal Fourier features of the S³ spherical position representation

Expected external_cond format (matches NPZ `actions` shape (T, 3)):
  dim 0 — action type index as float32
  dim 1 — x_end / 1024
  dim 2 — y_end / 768
"""

from __future__ import annotations

from .u_vit3d import UViT3D
from ..modules.action_embedding import ActionCondEmbedding


class UViT3DAction(UViT3D):
    """UViT3D with learnable action-type embedding + S³ sinusoidal position encoding."""

    N_ACTION_TYPES: int = 6   # 0=click … 4=system, 5=no_action
    TYPE_EMB_DIM: int = 64    # action type embedding dimension
    N_FOURIER_FREQ: int = 8   # Fourier frequencies → 4 × 2 × 8 = 64 pos dims

    def _build_external_cond_embedding(self):
        return ActionCondEmbedding(
            n_action_types=self.N_ACTION_TYPES,
            type_emb_dim=self.TYPE_EMB_DIM,
            n_freqs=self.N_FOURIER_FREQ,
            emb_dim=self.external_cond_emb_dim,   # 1024 for u_vit3d default config
            dropout_prob=self.cfg.get("external_cond_dropout", 0.0),
        )
