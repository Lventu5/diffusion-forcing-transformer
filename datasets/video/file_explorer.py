"""
DFoT-compatible dataset for file-explorer (ui_sim) trajectories.

Trajectories are stored as .npz files built by scripts/build_dfot_dataset.py:
  frames  (T, 768, 1024, 3)  uint8
  actions (T, 9)             float32

Class hierarchy mirrors the existing Minecraft/RealEstate10K datasets:
  FileExplorerBaseVideoDataset     — shared I/O overrides
  FileExplorerAdvancedVideoDataset — action-conditioned training/eval
  FileExplorerSimpleVideoDataset   — full-video loading for latent pre-processing
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
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

    Data lives locally as .npz files; there is no remote download step.
    Metadata stores (video_paths, lengths) rather than (video_pts, video_fps)
    because the data is not encoded as video files.
    """

    _ALL_SPLITS = ["training", "validation"]

    # ── Abstract method implementations ──────────────────────────────────────

    def download_dataset(self) -> None:
        """No-op: dataset is built locally via scripts/build_dfot_dataset.py."""

    def build_metadata(self, split: SPLIT) -> None:
        """
        Scan .npz files in save_dir/split and cache their paths and lengths.

        Saves a .pt file with keys:
          video_paths : List[Path]
          lengths     : List[int]   (number of frames / actions per trajectory)
        """
        split_dir = self.save_dir / split
        if not split_dir.exists():
            # Split not yet built; save empty placeholder.
            torch.save(
                {"video_paths": [], "lengths": []},
                self.metadata_dir / f"{split}.pt",
            )
            return

        video_paths = sorted(split_dir.glob("*.npz"), key=str)
        lengths: List[int] = []
        for p in video_paths:
            try:
                data = np.load(p)
                lengths.append(int(data["frames"].shape[0]))
            except Exception:
                lengths.append(0)

        torch.save(
            {"video_paths": video_paths, "lengths": lengths},
            self.metadata_dir / f"{split}.pt",
        )

    def load_metadata(self) -> List[Dict[str, Any]]:
        metadata_path = self.metadata_dir / f"{self.split}.pt"
        if not metadata_path.exists():
            self.metadata_dir.mkdir(exist_ok=True, parents=True)
            self.build_metadata(self.split)
        saved = torch.load(metadata_path, weights_only=False)
        return [
            {
                "video_paths": Path(saved["video_paths"][i]),
                "length": int(saved["lengths"][i]),
            }
            for i in range(len(saved["video_paths"]))
        ]

    def video_length(self, video_metadata: Dict[str, Any]) -> int:
        return int(video_metadata["length"])

    # ── Video loading ─────────────────────────────────────────────────────────

    def load_video(
        self,
        video_metadata: Dict[str, Any],
        start_frame: int,
        end_frame: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Load frames from an .npz file.

        Returns:
            Tensor of shape (T, C, H, W), float32 in [0, 1].
        """
        frames: np.ndarray = np.load(video_metadata["video_paths"])["frames"]
        frames = frames[start_frame:end_frame]          # (T, H, W, C) uint8
        tensor = torch.from_numpy(frames.astype(np.float32) / 255.0)
        return tensor.permute(0, 3, 1, 2).contiguous()  # (T, C, H, W)

    # ── Transform ─────────────────────────────────────────────────────────────

    def build_transform(self):
        """
        Resize so the longer side equals self.resolution (no crop), then
        zero-pad the shorter side to produce a square frame required by DiT3D.

        For resolution=256 on 1024×768 input:
          - resize → 256×192  (W×H, aspect-ratio preserving, no crop)
          - pad height 192 → 256 (32 px top, 32 px bottom)
          - output: 256×256
        """
        scale = self.resolution / max(_SCREEN_H, _SCREEN_W)
        frame_h = round(_SCREEN_H * scale)   # 192
        frame_w = round(_SCREEN_W * scale)   # 256
        resize_fn = VideoTransform((frame_h, frame_w))
        pad_top = (frame_w - frame_h) // 2        # 32
        pad_bottom = frame_w - frame_h - pad_top  # 32

        def _resize_and_pad(images: torch.Tensor) -> torch.Tensor:
            images = resize_fn(images)                          # (..., C, 192, 256)
            return F.pad(images, (0, 0, pad_top, pad_bottom))  # (..., C, 256, 256)

        return _resize_and_pad


class FileExplorerAdvancedVideoDataset(
    FileExplorerBaseVideoDataset, BaseAdvancedVideoDataset
):
    """
    File-explorer advanced video dataset with 9-dim action conditioning.

    The action vector is already in the correct float32 format inside the .npz,
    so load_cond simply returns the pre-encoded slice as a tensor.
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
        Return the pre-encoded action slice as a (T, 9) float32 tensor.
        """
        actions: np.ndarray = np.load(video_metadata["video_paths"])["actions"]
        actions = actions[start_frame:end_frame]  # (T, 9) float32
        return torch.from_numpy(actions).float()


class FileExplorerSimpleVideoDataset(
    FileExplorerBaseVideoDataset, BaseSimpleVideoDataset
):
    """
    File-explorer simple video dataset for future latent pre-processing.

    Loads full trajectories together with the target latent save path.
    """

    def __init__(self, cfg: DictConfig, split: SPLIT = "training"):
        BaseSimpleVideoDataset.__init__(self, cfg, split)
