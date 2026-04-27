"""Synthetic binary conditioning dataset — per-frame cross-attention sanity test.

Each sample contains a binary condition per frame (0/1). Frame pixels directly
follow that per-frame condition: 0 -> black frame, 1 -> white frame.

The key property is that conditions are not constant over the clip, so the model
cannot solve the task by treating the whole trajectory as a single class.
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset


class SyntheticBinaryDataset(Dataset):
    """Black/white frames with per-frame binary cross-attention conditioning."""

    def __init__(self, cfg: DictConfig, split: str = "training", **kwargs) -> None:
        self.n_samples: int = getattr(cfg, "n_samples", 200000)
        C, H, W = cfg.observation_shape
        self.T = cfg.max_frames
        self.C, self.H, self.W = int(C), int(H), int(W)
        self.ext_dim: int = int(cfg.external_cond_dim)
        split_offsets = {"training": 0, "validation": 10_000_000, "test": 20_000_000}
        self.seed_offset = split_offsets.get(split, 30_000_000)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        # Deterministic per-sample RNG gives stable labels while changing over time.
        generator = torch.Generator().manual_seed(self.seed_offset + int(idx))
        labels = torch.randint(0, 2, (self.T,), generator=generator, dtype=torch.long)
        if self.T > 1 and bool((labels == labels[0]).all()):
            # Guarantee at least one transition so conditions are not clip-constant.
            labels[-1] = 1 - labels[0]

        # Video: (T, C, H, W) in [0, 1]; normalized to [-1, 1] by on_after_batch_transfer.
        labels_f = labels.to(torch.float32)
        videos = labels_f[:, None, None, None].expand(self.T, self.C, self.H, self.W)

        # Cond: (T, ext_dim)
        #   dims 0..2  : zero action placeholders
        #   dim  -1    : per-frame binary label fed as cross-attention context token
        conds = torch.zeros(self.T, self.ext_dim, dtype=torch.float32)
        conds[:, -1] = labels_f

        return {"videos": videos, "conds": conds}
