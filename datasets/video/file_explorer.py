"""
DFoT-compatible dataset for file-explorer (ui_sim) trajectories.

Trajectories are stored as pairs of files produced by
scripts/convert_npz_to_video.py:

  <stem>.mp4   — H.264 lossless video (CRF=0, all-keyframe); random seeking
                 touches only the requested frames (no full-file decompression,
                 unlike the old .npz format).
  <stem>.json  — flat (T, 3) float32 action array stored as a JSON list.

Class hierarchy mirrors the existing Minecraft/RealEstate10K datasets:
  FileExplorerBaseVideoDataset     — shared overrides (transform, splits)
  FileExplorerAdvancedVideoDataset — action-conditioned training/eval
  FileExplorerSimpleVideoDataset   — full-video loading for latent pre-processing
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from omegaconf import DictConfig

from .base_video import (
    BaseAdvancedVideoDataset,
    BaseSimpleVideoDataset,
    BaseVideoDataset,
    SPLIT,
)
from .utils import VideoTransform

# Native screen resolution of all recorded trajectories.
_SCREEN_H: int = 768
_SCREEN_W: int = 1024


class FileExplorerBaseVideoDataset(BaseVideoDataset):
    """
    Shared base for file-explorer video datasets.

    The heavy-lifting (build_metadata, load_metadata, video_length, load_video)
    is handled by BaseVideoDataset, which scans for .mp4 files and uses PyAV
    for frame-accurate seeking.  This class only needs to:
      - declare the available splits, and
      - override build_transform for the non-square native resolution.
    """

    _ALL_SPLITS = ["training", "validation"]

    def download_dataset(self) -> None:
        """No-op: dataset is built locally via scripts/build_dfot_dataset.py."""

    def build_transform(self):
        """
        Resize the native 768×1024 frames to 192×256, preserving aspect ratio.

        For resolution=256 on a 1024×768 source:
          scale = 256 / 1024 = 0.25  →  output: H=192, W=256  (no crop needed)
        """
        scale = self.resolution / max(_SCREEN_H, _SCREEN_W)
        frame_h = round(_SCREEN_H * scale)   # 192 at resolution=256
        frame_w = round(_SCREEN_W * scale)   # 256 at resolution=256
        return VideoTransform((frame_h, frame_w))


class FileExplorerAdvancedVideoDataset(
    FileExplorerBaseVideoDataset, BaseAdvancedVideoDataset
):
    """
    File-explorer advanced video dataset with 3-dim action conditioning.

    Actions are loaded from a <stem>.json file that lives beside the .mp4.
    The JSON contains a (T, 3) float32 array encoded as a list of lists.
    """

    def __init__(
        self,
        cfg: DictConfig,
        split: SPLIT = "training",
        current_epoch: Optional[int] = None,
    ):
        BaseAdvancedVideoDataset.__init__(self, cfg, split, current_epoch)

    def load_cond(
        self,
        video_metadata: Dict[str, Any],
        start_frame: int,
        end_frame: int,
    ) -> torch.Tensor:
        """
        Return the action slice as a (T, 3) float32 tensor.

        Reads only the JSON sidecar — no video decompression needed.
        """
        json_path = Path(video_metadata["video_paths"]).with_suffix(".json")
        with open(json_path) as f:
            actions = json.load(f)
        arr = np.array(actions[start_frame:end_frame], dtype=np.float32)
        return torch.from_numpy(arr)


class FileExplorerSimpleVideoDataset(
    FileExplorerBaseVideoDataset, BaseSimpleVideoDataset
):
    """
    File-explorer simple video dataset for latent pre-processing.

    Loads full trajectories together with the target latent save path.
    """

    def __init__(self, cfg: DictConfig, split: SPLIT = "training"):
        BaseSimpleVideoDataset.__init__(self, cfg, split)
