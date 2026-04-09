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
from utils.distributed_utils import rank_zero_print
from utils.print_utils import cyan

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
        """
        Drop trajectories that are missing a companion _node_emb.pt file.

        A small number of raw trajectories may be missing the upstream
        node-action-map artifacts, which means precomputation cannot emit an
        embedding file for them. Excluding those samples keeps training from
        crashing on startup while still surfacing the data issue clearly.
        """
        super().on_before_prepare_clips()
        missing: list[str] = []
        kept: list[Dict[str, Any]] = []
        for vm in self.metadata:
            mp4_path = Path(vm["video_paths"])
            if _node_emb_path(mp4_path).exists():
                kept.append(vm)
            else:
                missing.append(str(mp4_path))
        if missing:
            n = len(missing)
            examples = "\n  ".join(missing[:5])
            rank_zero_print(
                cyan(
                    f"Excluding {n} trajectory/ies that are missing _node_emb.pt "
                    f"companion files.\nFirst {min(n, 5)}:\n  {examples}"
                )
            )
            self.metadata = kept

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
        node_emb = torch.load(emb_file, weights_only=True)

        # Legacy support: older precompute runs emitted one embedding per
        # transition instead of one per frame, missing only the initial state.
        # Duplicate the first embedding so frames 1..T stay aligned and the
        # initial frame gets the best available approximation.
        if node_emb.shape[0] + 1 == self.video_length(video_metadata):
            node_emb = torch.cat([node_emb[:1], node_emb], dim=0)

        node_emb = node_emb[start_frame:end_frame]

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
