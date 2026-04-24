"""Synthetic black/white binary conditioning dataset — cross-attention smoke test.

Each sample is either all-black (label=0) or all-white (label=1) video frames.
The binary label enters cross-attention as a scalar context token (the last dim
of external_cond, after 3 zero-filled action dims).

Expected outcome after ~500 steps:
  - loss drops to near zero
  - model generates black frames when conditioned on 0, white when conditioned on 1

If this fails, the cross-attention pathway has a bug independent of real data.
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset


class SyntheticBinaryDataset(Dataset):
    """All-black or all-white frames with binary cross-attention conditioning."""

    def __init__(self, cfg: DictConfig, split: str = "training", **kwargs) -> None:
        self.n_samples: int = getattr(cfg, "n_samples", 2000)
        C, H, W = cfg.observation_shape
        self.T = cfg.max_frames
        self.C, self.H, self.W = int(C), int(H), int(W)
        self.ext_dim: int = int(cfg.external_cond_dim)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        label = float(idx % 2)  # 0 = black, 1 = white, strictly alternating

        # Video: (T, C, H, W) in [0, 1]; normalized to [-1, 1] by on_after_batch_transfer
        videos = torch.full(
            (self.T, self.C, self.H, self.W), label, dtype=torch.float32
        )

        # Cond: (T, ext_dim)
        #   dims 0..2  : zero action placeholders
        #   dim  -1    : binary label fed as cross-attention context token
        conds = torch.zeros(self.T, self.ext_dim, dtype=torch.float32)
        conds[:, -1] = label

        return {"videos": videos, "conds": conds}
