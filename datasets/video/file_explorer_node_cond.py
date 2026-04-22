"""
File-explorer dataset conditioned on both actions and precomputed
node-structure embeddings.

Extends FileExplorerAdvancedVideoDataset so that ``load_cond`` returns:

    cond[t] = [action_3dim ‖ node_emb_D]   shape: (T, 3 + D)

The node embeddings must be precomputed first::

    python scripts/precompute_node_embeddings.py \\
        --processed-dir data/processed/<dataset> \\
        --raw-data-dir  /work/$USER/data/dataset_compact

Each trajectory's node embeddings are stored as a sibling ``.pt`` file::

    <split>/<stem>_node_emb.pt    shape (T_total, D)

At training time only the requested ``[start_frame:end_frame]`` slice is
loaded.  If the ``_node_emb.pt`` file is missing the dataset raises a clear
error at startup.

To train WITHOUT node conditioning, use ``FileExplorerAdvancedVideoDataset``
(``dataset=file_explorer``).  To train WITH node conditioning, use this class
(``dataset=file_explorer_node_cond``).  They are fully independent — no weight
sharing or special handling is needed.

Performance
-----------
A naive ``torch.load`` of the full ``(T_total, D)`` embedding file on every
``__getitem__`` call causes a progressive training slowdown.  Workers read more
data than needed, the OS page-cache fills up, and GPU utilisation collapses as
DDP ranks stall at the synchronisation barrier waiting for the slowest loader.

Two storage modes are available, selected by ``dataset.node_emb_preload``:

Pre-load mode  (``node_emb_preload: true``, recommended for small datasets)
    All embedding files are loaded into RAM once at dataset init, before any
    worker process is spawned.  On Linux, DataLoader workers are forked from
    the parent process and inherit its address space via copy-on-write: reads
    never trigger physical copies, so the memory cost stays flat regardless of
    the number of workers.  Every scenario — training, validation, evaluation
    — benefits equally with zero disk I/O for embeddings.

Lazy LRU mode  (``node_emb_preload: false``, default for large datasets)
    Embedding files are loaded on demand and cached per worker in an LRU
    dict (size ``node_emb_cache_size``).  Useful when the total embedding
    data exceeds available RAM.  Validation workers are ephemeral so they
    do not benefit from caching; training workers (persistent) do.

In both modes, three additional mitigations apply:

* **Memory-mapped I/O** (PyTorch ≥ 2.1): ``torch.load(..., mmap=True)`` pages
  in only the accessed frames instead of the whole file.  Reduces the peak
  memory spike during cold loads.  Falls back silently on older PyTorch.

* **Per-worker LRU cache** (lazy mode): trajectories yielding multiple clips
  are read from disk only once per worker lifetime.  Cache size is bounded by
  ``node_emb_cache_size`` to cap per-worker RAM usage.

* **Slice clone**: ``node_emb[start:end].clone()`` gives the returned slice
  its own storage, independent of the cached/mmap-backed source tensor.
  Without this, the slice would pin the full trajectory storage in the
  DataLoader's shared-memory queue and prevent LRU eviction from freeing RAM.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from omegaconf import DictConfig
from utils.distributed_utils import rank_zero_print
from utils.print_utils import cyan

from .base_video import SPLIT
from .file_explorer import (
    FileExplorerAdvancedVideoDataset,
    FileExplorerSimpleVideoDataset,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NODE_EMB_SUFFIX = "_node_emb.pt"

# Memory-mapped torch.load is available from PyTorch 2.1.  It lets the OS
# page in only the accessed frames instead of reading the whole file into RAM.
_TORCH_MMAP_SUPPORTED: bool = tuple(
    int(x) for x in torch.__version__.split(".")[:2] if x.isdigit()
)[:2] >= (2, 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node_emb_path(mp4_path: Path) -> Path:
    return mp4_path.with_name(mp4_path.stem + _NODE_EMB_SUFFIX)


def _load_node_emb_file(path: Path) -> torch.Tensor:
    """
    Load a ``*_node_emb.pt`` file from disk.

    Uses memory-mapped I/O when available (PyTorch ≥ 2.1) so that only the
    pages corresponding to the requested slice ever have to be read from disk.
    This function is intentionally thin — caching and pre-loading are handled
    by the caller.
    """
    if _TORCH_MMAP_SUPPORTED:
        return torch.load(str(path), weights_only=True, mmap=True)
    return torch.load(str(path), weights_only=True)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FileExplorerNodeCondAdvancedVideoDataset(FileExplorerAdvancedVideoDataset):
    """
    File-explorer advanced dataset that appends precomputed node-structure
    embeddings to the action conditioning vector.

    Expected dataset config additions:

    .. code-block:: yaml

        node_emb_dim: 768           # must match the sentence-transformer model used
        external_cond_dim: 771      # = 3 (action) + node_emb_dim
        node_emb_preload: false     # true = load all files at init (see module docstring)
        node_emb_cache_size: 256    # per-worker LRU capacity; ignored when preload=true

    Choosing between pre-load and lazy mode
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Estimate the total size of your node embedding files::

        find <data_dir> -name '*_node_emb.pt' | xargs du -ch | tail -1

    If it comfortably fits in system RAM, set ``node_emb_preload: true``.
    This makes training, validation, and evaluation equally fast.

    If it does not fit, keep ``node_emb_preload: false`` and tune
    ``node_emb_cache_size`` (lazy LRU mode).  Memory estimate per worker::

        node_emb_cache_size × avg_trajectory_frames × node_emb_dim × 4 bytes
        e.g.  256 × 160 × 768 × 4 B ≈ 125 MB/worker  →  2 GB for 16 workers
    """

    def __init__(
        self,
        cfg: DictConfig,
        split: SPLIT = "training",
        current_epoch: Optional[int] = None,
    ) -> None:
        self.node_emb_dim: int = cfg.node_emb_dim
        preload: bool = getattr(cfg, "node_emb_preload", False)
        cache_size: int = getattr(cfg, "node_emb_cache_size", 256)

        # Both stores start empty; they are populated after super().__init__()
        # has run on_before_prepare_clips() and finalised self.metadata.
        self._emb_store: Optional[Dict[str, torch.Tensor]] = None
        self._emb_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._emb_cache_size: int = cache_size
        self._preload_requested: bool = preload

        super().__init__(cfg, split, current_epoch)

        # Pre-load after super().__init__ so that self.metadata is already
        # filtered (missing files excluded by on_before_prepare_clips).
        if self._preload_requested:
            self._emb_store = self._preload_embeddings()

    # ------------------------------------------------------------------
    # Embedding storage
    # ------------------------------------------------------------------

    def _preload_embeddings(self) -> Dict[str, torch.Tensor]:
        """
        Load every node-embedding file referenced by this split into RAM.

        Called once in the main process before any worker is spawned.  On
        Linux, DataLoader workers are forked from this process and inherit
        the populated dict via copy-on-write, so the physical pages are
        shared for free as long as workers only read.
        """
        paths: List[Path] = sorted(
            {_node_emb_path(Path(vm["video_paths"])) for vm in self.metadata}
        )
        rank_zero_print(
            cyan(
                f"[node_emb] Pre-loading {len(paths)} embedding files "
                f"for the '{self.split}' split …"
            )
        )
        store: Dict[str, torch.Tensor] = {}
        for path in paths:
            store[str(path)] = _load_node_emb_file(path)

        total_mb = sum(t.nbytes for t in store.values()) / 1024 ** 2
        rank_zero_print(cyan(f"[node_emb] Pre-load complete — {total_mb:.0f} MB in RAM."))
        return store

    def _get_node_emb(self, path: Path) -> torch.Tensor:
        """
        Return the full node-embedding tensor for *path*.

        Checks the pre-loaded store first; falls back to the per-worker LRU
        cache.  The returned tensor may be mmap-backed or a reference to a
        pre-loaded tensor — callers must ``.clone()`` any slice they intend to
        retain so that neither the mmap pages nor the cached storage is pinned
        by downstream DataLoader references.
        """
        key = str(path)

        # Pre-load mode: O(1) dict lookup, always a hit after __init__.
        if self._emb_store is not None:
            return self._emb_store[key]

        # Lazy LRU mode: populated per worker, bounded by _emb_cache_size.
        if key in self._emb_cache:
            self._emb_cache.move_to_end(key)
            return self._emb_cache[key]

        tensor = _load_node_emb_file(path)
        self._emb_cache[key] = tensor
        if len(self._emb_cache) > self._emb_cache_size:
            self._emb_cache.popitem(last=False)  # evict least-recently used
        return tensor

    def _put_node_emb(self, path: Path, tensor: torch.Tensor) -> None:
        """
        Write a (corrected) tensor back into whichever store is active.

        Used after the legacy fixup so that subsequent clips from the same
        trajectory skip the correction branch entirely.
        """
        key = str(path)
        if self._emb_store is not None:
            self._emb_store[key] = tensor
        elif key in self._emb_cache:
            self._emb_cache[key] = tensor
            self._emb_cache.move_to_end(key)

    # ------------------------------------------------------------------
    # Dataset hooks
    # ------------------------------------------------------------------

    def on_before_prepare_clips(self) -> None:
        """
        Drop trajectories that are missing a companion ``_node_emb.pt`` file.

        A small number of raw trajectories may be missing the upstream
        node-action-map artifacts, which means precomputation cannot emit an
        embedding file for them.  Excluding those samples keeps training from
        crashing on startup while still surfacing the data issue clearly.
        """
        super().on_before_prepare_clips()
        kept: list[Dict[str, Any]] = []
        missing: list[str] = []

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
                    f"Excluding {n} trajectory/ies missing a _node_emb.pt file.\n"
                    f"First {min(n, 5)}:\n  {examples}"
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
        Return ``[action ‖ node_emb]`` concatenated along the feature axis.

        Shape: ``(T, 3 + node_emb_dim)``
        """
        # Action slice — shape (T, 3), loaded from a small JSON sidecar.
        action = super().load_cond(video_metadata, start_frame, end_frame)

        # Node embedding slice — shape (T, node_emb_dim).
        mp4_path = Path(video_metadata["video_paths"])
        emb_path = _node_emb_path(mp4_path)

        full_emb = self._get_node_emb(emb_path)

        # Legacy support: older precompute runs emitted one embedding per
        # transition instead of one per frame, missing only the initial state.
        # Duplicate the first embedding so frames stay aligned and the initial
        # frame gets the best available approximation.
        # The corrected tensor is written back so subsequent clips from this
        # trajectory skip this branch entirely (both in pre-load and LRU mode).
        if full_emb.shape[0] + 1 == self.video_length(video_metadata):
            full_emb = torch.cat([full_emb[:1], full_emb], dim=0)
            self._put_node_emb(emb_path, full_emb)

        # .clone() detaches the slice from the source tensor's storage so that
        # (a) mmap pages / cached tensors can be reclaimed independently of the
        # batch that is about to be enqueued into the DataLoader, and (b) the
        # DataLoader's shared-memory mechanism only touches the small slice.
        node_emb = full_emb[start_frame:end_frame].clone()

        if node_emb.shape[0] != action.shape[0]:
            raise RuntimeError(
                f"Length mismatch for {mp4_path.name}: "
                f"action={action.shape[0]}, node_emb={node_emb.shape[0]}"
            )

        return torch.cat([action, node_emb], dim=-1)   # (T, 3 + node_emb_dim)


class FileExplorerNodeCondSimpleVideoDataset(FileExplorerSimpleVideoDataset):
    """
    Simple (full-video) variant for latent pre-processing.
    Identical to ``FileExplorerSimpleVideoDataset``; included for symmetry so
    that the experiment registry can resolve the dataset class by name.
    """
