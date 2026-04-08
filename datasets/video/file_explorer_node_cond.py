"""
File-explorer dataset conditioned on both actions and precomputed
node-structure embeddings.

Extends FileExplorerAdvancedVideoDataset so that `load_cond` returns:
    cond[t] = [action_3dim ‖ node_emb_D]   shape: (T, 3 + D)

The node embeddings must be precomputed first:
    python scripts/precompute_node_embeddings.py \\
        --processed-dir data/processed/<dataset> \\
        --raw-data-dir  /work/$USER/data/dataset_compact

Each trajectory's node embeddings are stored as a sibling .pt file:
    <split>/<stem>_node_emb.pt    shape (T_total, D)

At training time only the requested [start_frame:end_frame] slice is loaded.
If the _node_emb.pt file is missing the dataset raises a clear error at startup.

To train WITHOUT node conditioning, use FileExplorerAdvancedVideoDataset
(dataset=file_explorer).  To train WITH node conditioning, use this class
(dataset=file_explorer_node_cond).  They are fully independent — no weight
sharing or special handling is needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig

from .file_explorer import (
    FileExplorerAdvancedVideoDataset,
    FileExplorerSimpleVideoDataset,
)
from .base_video import SPLIT

_NODE_EMB_SUFFIX = "_node_emb.pt"


def _node_emb_path(mp4_path: Path) -> Path:
    return mp4_path.with_name(mp4_path.stem + _NODE_EMB_SUFFIX)


class FileExplorerNodeCondAdvancedVideoDataset(FileExplorerAdvancedVideoDataset):
    """
    File-explorer advanced dataset that appends precomputed node-structure
    embeddings to the action conditioning vector.

    Expected dataset config additions:
        node_emb_dim: 768          # matches the sentence-transformer model used
        external_cond_dim: 771     # = 3 (action) + 768 (node_emb_dim)
    """

    def __init__(
        self,
        cfg: DictConfig,
        split: SPLIT = "training",
        current_epoch: Optional[int] = None,
    ):
        self.node_emb_dim: int = cfg.node_emb_dim
        super().__init__(cfg, split, current_epoch)

    def on_before_prepare_clips(self) -> None:
        """Verify that every trajectory has a companion _node_emb.pt file."""
        super().on_before_prepare_clips()
        missing: list[str] = []
        for vm in self.metadata:
            mp4_path = Path(vm["video_paths"])
            if not _node_emb_path(mp4_path).exists():
                missing.append(str(mp4_path))
        if missing:
            n = len(missing)
            examples = "\n  ".join(missing[:5])
            raise FileNotFoundError(
                f"{n} trajectory/ies are missing _node_emb.pt companion files.\n"
                f"First {min(n, 5)}:\n  {examples}\n\n"
                "Run scripts/precompute_node_embeddings.py first."
            )

    def load_cond(
        self,
        video_metadata: Dict[str, Any],
        start_frame: int,
        end_frame: int,
    ) -> torch.Tensor:
        """
        Return [action ‖ node_emb] concatenated along the feature dimension.

        Shape: (T, 3 + node_emb_dim)
        """
        # Action slice — shape (T, 3)
        action = super().load_cond(video_metadata, start_frame, end_frame)

        # Node embedding slice — shape (T, node_emb_dim)
        mp4_path = Path(video_metadata["video_paths"])
        emb_file = _node_emb_path(mp4_path)
        node_emb = torch.load(emb_file, weights_only=True)[start_frame:end_frame]

        # Sanity check (should not happen if on_before_prepare_clips ran)
        if node_emb.shape[0] != action.shape[0]:
            raise RuntimeError(
                f"Length mismatch for {mp4_path.name}: "
                f"action={action.shape[0]}, node_emb={node_emb.shape[0]}"
            )

        return torch.cat([action, node_emb], dim=-1)   # (T, 3 + node_emb_dim)


class FileExplorerNodeCondSimpleVideoDataset(FileExplorerSimpleVideoDataset):
    """
    Simple (full-video) variant for latent pre-processing.
    Identical to FileExplorerSimpleVideoDataset; included for symmetry so
    that the experiment registry can resolve the dataset class by name.
    """
