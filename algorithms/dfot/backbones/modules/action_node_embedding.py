"""
ActionNodeCondEmbedding — action + node-structure conditioning with independent dropout.

Two completely separate paths, each projecting to ``emb_dim``, summed before
returning.  Independent ``RandomEmbeddingDropout`` on each path so that during
training the model learns each signal robustly on its own.

Input convention:
  cond[..., :3]          — action: [action_type_float, x/1024, y/768]
  cond[..., 3:3+node_d]  — node embedding (e.g. 768-dim from all-mpnet-base-v2)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .action_embedding import ActionCondEmbedding
from .embeddings import RandomEmbeddingDropout


class ActionNodeCondEmbedding(nn.Module):
    """
    Conditioning embedding that jointly encodes:
      - The action taken at each timestep (type + 2-D screen position)
      - The precomputed sentence-embedding of the *next* node's structure

    Design
    ------
    Both signals are projected to ``emb_dim`` by independent modules and
    summed.  Crucially, each path has its *own* ``RandomEmbeddingDropout`` so
    that during training they are dropped independently.  This forces the model
    to be useful conditioned on either signal alone, not just on both together.

    The action path re-uses ``ActionCondEmbedding`` without modification —
    identical to the existing ``UViT3DAction`` baseline.  This means the action
    path can be initialised from an action-only checkpoint if desired.

    The node path is a single linear projection with a non-linearity.  The
    sentence-transformer already produces a compact, meaningful representation
    so a deep MLP adds unnecessary capacity.

    Args:
        node_emb_dim:        Dimension of the input node embedding (e.g. 768).
        n_action_types:      Discrete action type indices (default 6).
        type_emb_dim:        Learnable type embedding dimension (default 64).
        n_freqs:             Fourier frequencies for S³ position (default 8).
        emb_dim:             Output dim — must match backbone's external_cond_emb_dim.
        action_dropout_prob: Per-frame action dropout probability (default 0.1).
        node_dropout_prob:   Per-frame node dropout probability (default 0.1).
    """

    ACTION_DIMS: int = 3   # [action_type, x/1024, y/768]

    def __init__(
        self,
        node_emb_dim: int,
        n_action_types: int = 6,
        type_emb_dim: int = 64,
        n_freqs: int = 8,
        emb_dim: int = 1024,
        action_dropout_prob: float = 0.1,
        node_dropout_prob: float = 0.1,
    ):
        super().__init__()

        # ── Action path: identical to ActionCondEmbedding in UViT3DAction ────
        # dropout_prob=0 here so we own the dropout below
        self.action_embedding = ActionCondEmbedding(
            n_action_types = n_action_types,
            type_emb_dim   = type_emb_dim,
            n_freqs        = n_freqs,
            emb_dim        = emb_dim,
            dropout_prob   = 0.0,   # handled separately
        )
        self.action_dropout = RandomEmbeddingDropout(p=action_dropout_prob)

        # ── Node path ─────────────────────────────────────────────────────────
        # Single linear + non-linearity: the sentence-transformer already
        # produces a compact representation, no need for a deep MLP.
        self.node_proj = nn.Sequential(
            nn.Linear(node_emb_dim, emb_dim),
            nn.SiLU(),
        )
        self.node_dropout = RandomEmbeddingDropout(p=node_dropout_prob)

    def forward(self, cond: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            cond:  (B, T, 3 + node_emb_dim)
            mask:  (B,) optional inference-time CFG mask; when set, both signals
                   are zeroed together (standard CFG behaviour).
        Returns:
            emb:   (B, T, emb_dim)
        """
        action_cond = cond[..., : self.ACTION_DIMS]    # (B, T, 3)
        node_emb    = cond[..., self.ACTION_DIMS :]     # (B, T, node_emb_dim)

        # Independent dropout during training; shared mask at inference
        action_out = self.action_dropout(
            self.action_embedding(action_cond),
            mask,
        )                                               # (B, T, emb_dim)
        node_out = self.node_dropout(
            self.node_proj(node_emb.float()),
            mask,
        )                                               # (B, T, emb_dim)

        return action_out + node_out
