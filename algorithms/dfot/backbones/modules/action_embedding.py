"""
Action conditioning embedding for file-explorer DFoT training.

Replaces the flat RandomDropoutCondEmbedding with a structured embedding that
uses a learnable lookup for discrete action types and sinusoidal Fourier features
on a spherical (S³) representation of 2D screen coordinates.

Input convention (matches NPZ `actions` array, shape (T, 3)):
  dim 0 — action type index as float32 (0=click … 4=system, 5=no_action)
  dim 1 — x_end / 1024   (normalised to [0, 1])
  dim 2 — y_end / 768    (normalised to [0, 1])
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .embeddings import RandomEmbeddingDropout


class ActionPositionEncoding(nn.Module):
    """
    Encode a 2-D screen coordinate (x, y) ∈ [0, 1]² as sinusoidal Fourier
    features of its S³ spherical representation.

    Given (x, y):
      1. Polar decomposition: r = ||(x, y)||, θ = atan2(y, x)
      2. S³ lift: v = (cos θ, sin θ, cos(π·r/r_max), sin(π·r/r_max)) / √2
         — direction encoded in the first pair, magnitude in the second pair
      3. Fourier features: for each of the 4 components v_i and K frequencies:
           [sin(2^k · π · v_i),  cos(2^k · π · v_i)]  k = 0 … K-1
         Output dimension: 4 × 2K = 8K

    The S³ representation is smooth and periodic; the (0, 0) degenerate case
    maps cleanly to (1, 0, 1, 0) / √2 without special handling.
    """

    _R_MAX: float = math.sqrt(2.0)  # maximum possible r for (x,y) ∈ [0,1]²

    def __init__(self, n_freqs: int = 8):
        super().__init__()
        self.n_freqs = n_freqs
        # log-spaced frequencies: 2^0, 2^1, …, 2^(K-1), multiplied by π
        freqs = torch.tensor(
            [2**k * math.pi for k in range(n_freqs)], dtype=torch.float32
        )
        self.register_buffer("freqs", freqs)  # (K,)

    @property
    def out_dim(self) -> int:
        return 4 * 2 * self.n_freqs  # 4 S³ dims × sin+cos × K freqs

    def _xy_to_s3(self, xy: Tensor) -> Tensor:
        """Map (x, y) ∈ [0, 1]² to a unit vector on S³.

        Args:
            xy: (..., 2)
        Returns:
            v: (..., 4)  with ||v||₂ = 1
        """
        x = xy[..., 0]
        y = xy[..., 1]
        r = torch.sqrt(x * x + y * y).clamp(max=self._R_MAX)
        theta = torch.atan2(y, x)
        angle_r = math.pi * r / self._R_MAX
        v = torch.stack(
            [
                torch.cos(theta),
                torch.sin(theta),
                torch.cos(angle_r),
                torch.sin(angle_r),
            ],
            dim=-1,
        )  # (..., 4)  norm = sqrt(2) before division
        return v / math.sqrt(2.0)  # unit vector on S³

    def forward(self, xy: Tensor) -> Tensor:
        """
        Args:
            xy: (..., 2)  normalised screen coordinates
        Returns:
            features: (..., 8K)
        """
        v = self._xy_to_s3(xy)                          # (..., 4)
        angles = v.unsqueeze(-1) * self.freqs            # (..., 4, K)
        features = torch.cat(
            [angles.sin(), angles.cos()], dim=-1
        )                                                # (..., 4, 2K)
        return features.flatten(-2)                      # (..., 8K)


class ActionCondEmbedding(nn.Module):
    """
    Structured action conditioning embedding combining:
      - Learnable action-type embedding (nn.Embedding)
      - Sinusoidal Fourier features of the S³ position representation
      - Random dropout (classifier-free guidance, same interface as
        RandomDropoutCondEmbedding)

    Args:
        n_action_types: Number of action type indices (default 6: 0–4 + no_action)
        type_emb_dim:   Embedding dimension for action type lookup (default 64)
        n_freqs:        Number of Fourier frequencies for position (default 8 → 64-dim)
        emb_dim:        Output embedding dimension (should match backbone's
                        external_cond_emb_dim, e.g. 1024 for UVit3D)
        dropout_prob:   Probability of zeroing an entire sequence during training
                        (classifier-free guidance; 0 = disabled)
    """

    def __init__(
        self,
        n_action_types: int = 6,
        type_emb_dim: int = 64,
        n_freqs: int = 8,
        emb_dim: int = 1024,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.type_embedding = nn.Embedding(n_action_types, type_emb_dim)
        self.pos_encoding = ActionPositionEncoding(n_freqs=n_freqs)
        in_dim = type_emb_dim + self.pos_encoding.out_dim  # 64 + 64 = 128
        self.proj = nn.Linear(in_dim, emb_dim)
        self.dropout = RandomEmbeddingDropout(p=dropout_prob)

    def forward(self, cond: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            cond: (B, T, 3)  — [action_type_float, x/1024, y/768]
            mask: (B,)       — optional inference-time CFG mask (batch items to zero)
        Returns:
            emb: (B, T, emb_dim)
        """
        action_type = cond[..., 0].long()   # (B, T)
        xy = cond[..., 1:]                  # (B, T, 2)

        type_emb = self.type_embedding(action_type)   # (B, T, type_emb_dim)
        pos_emb = self.pos_encoding(xy)               # (B, T, 8*n_freqs)

        emb = self.proj(torch.cat([type_emb, pos_emb], dim=-1))  # (B, T, emb_dim)
        return self.dropout(emb, mask)
