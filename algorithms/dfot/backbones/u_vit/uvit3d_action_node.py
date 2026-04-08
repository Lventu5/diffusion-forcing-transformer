"""
UViT3DActionNode — UViT3D backbone conditioned on both actions and
precomputed node-structure embeddings.

Expected external_cond format  (B, T, 3 + node_emb_dim):
  dims   0–2 : action   [action_type_float, x/1024, y/768]
  dims 3–end : node embedding (e.g. 768-dim from all-mpnet-base-v2)

To train *without* node conditioning, use UViT3DAction (dataset external_cond_dim=3).
To train *with*    node conditioning, use UViT3DActionNode (external_cond_dim=3+D).
The two are architecturally independent — no checkpoint is shared.
"""

from __future__ import annotations

from .uvit3d_action import UViT3DAction
from ..modules.action_node_embedding import ActionNodeCondEmbedding


class UViT3DActionNode(UViT3DAction):
    """
    UViT3D with structured action conditioning AND node-structure conditioning.

    The ``external_cond_dim`` received from the dataset config is expected to be
    ``3 + node_emb_dim``.  The backbone splits the condition internally:
    the first 3 dims go through the action-embedding path; the remaining dims
    go through a two-layer MLP.  Both projections target ``emb_dim`` and are
    summed, so the interface to the rest of the backbone is identical to
    UViT3DAction.
    """

    def _build_external_cond_embedding(self) -> ActionNodeCondEmbedding:
        node_emb_dim = self.external_cond_dim - ActionNodeCondEmbedding.ACTION_DIMS
        if node_emb_dim <= 0:
            raise ValueError(
                f"UViT3DActionNode requires external_cond_dim > 3 "
                f"(got {self.external_cond_dim}).  "
                "Set dataset.external_cond_dim = 3 + <node_embedding_dim>."
            )
        return ActionNodeCondEmbedding(
            node_emb_dim        = node_emb_dim,
            n_action_types      = self.N_ACTION_TYPES,
            type_emb_dim        = self.TYPE_EMB_DIM,
            n_freqs             = self.N_FOURIER_FREQ,
            emb_dim             = self.external_cond_emb_dim,
            action_dropout_prob = self.cfg.get("external_cond_dropout", 0.0),
            node_dropout_prob   = self.cfg.get("node_cond_dropout", 0.0),
        )
