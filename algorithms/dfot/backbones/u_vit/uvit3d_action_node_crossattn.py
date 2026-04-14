"""
UViT3DActionNodeCrossAttn — UViT3D with action conditioning (addition) and
text/node conditioning via cross-attention.

Compared to UViT3DActionNode which projects node embeddings to emb_dim and
*adds* them to the noise embedding, this variant passes the raw node embeddings
as cross-attention context tokens.  The CrossAttnTransformerBlock's kv_proj
handles the projection from node_emb_dim → model dim internally.

Expected external_cond format (same as UViT3DActionNode, shape (B, T, 3 + node_emb_dim)):
  dims   0–2 : action   [action_type_float, x/1024, y/768]
  dims 3–end : node embedding (e.g. 768-dim from all-mpnet-base-v2)

Required backbone config keys:
  cross_attn_context_dim: <node_emb_dim>   (e.g. 768)
  block_types: [..., "CrossAttnTransformerBlock", ...]

Optional config keys:
  external_cond_dropout: float  action CFG dropout probability (default 0.0)
  node_cond_dropout:     float  text CFG dropout probability   (default 0.0)
"""

from __future__ import annotations

from typing import Optional

import torch
from omegaconf import DictConfig
from torch import Tensor

from .uvit3d_action import UViT3DAction
from .u_vit3d import UViT3D
from ..modules.embeddings import RandomEmbeddingDropout


class UViT3DActionNodeCrossAttn(UViT3DAction):
    """
    UViT3DAction with text/node conditioning via cross-attention instead of
    addition to the noise-level embedding.

    Action (dims 0:3 of external_cond) continues to be embedded and added to
    the noise-level embedding — unchanged from UViT3DAction.

    Text/node embeddings (dims 3: of external_cond) are passed as cross-attention
    context tokens (B, T, node_emb_dim) to every CrossAttnTransformerBlock.
    The backbone's CrossAttnBlock.kv_proj projects from node_emb_dim to model
    dim, so no additional projection layer is needed here.
    """

    _ACTION_DIMS: int = 3  # [action_type, x/1024, y/768]

    def __init__(
        self,
        cfg: DictConfig,
        x_shape: torch.Size,
        max_tokens: int,
        external_cond_dim: int,
        use_causal_mask: bool = True,
    ):
        node_emb_dim = external_cond_dim - self._ACTION_DIMS
        if node_emb_dim <= 0:
            raise ValueError(
                f"UViT3DActionNodeCrossAttn requires external_cond_dim > {self._ACTION_DIMS} "
                f"(got {external_cond_dim}).  "
                f"Set dataset.external_cond_dim = 3 + <node_emb_dim>."
            )
        self._node_emb_dim = node_emb_dim

        # Pass the full external_cond_dim up so BaseBackbone stores it, but
        # UViT3DAction._build_external_cond_embedding builds an ActionCondEmbedding
        # that always uses only the first 3 dims — the node dims are ignored there.
        super().__init__(cfg, x_shape, max_tokens, external_cond_dim, use_causal_mask)

        # Independent dropout on the text path for classifier-free guidance.
        self._node_dropout = RandomEmbeddingDropout(
            p=cfg.get("node_cond_dropout", 0.0)
        )

    def forward(
        self,
        x: Tensor,
        noise_levels: Tensor,
        external_cond: Optional[Tensor] = None,
        external_cond_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            x:                  (B, T, C, H, W)
            noise_levels:       (B, T)
            external_cond:      (B, T, 3 + node_emb_dim)  or None
            external_cond_mask: (B,) optional CFG mask; zeros both action and
                                text conditioning when set.
        Returns:
            (B, T, C, H, W)
        """
        context_tokens = None
        if external_cond is not None:
            action_cond = external_cond[..., : self._ACTION_DIMS]   # (B, T, 3)
            node_emb    = external_cond[..., self._ACTION_DIMS :]   # (B, T, node_emb_dim)

            # Apply dropout for CFG; share the mask with the action path so
            # both signals are zeroed together at inference time.
            context_tokens = self._node_dropout(
                node_emb.float(), external_cond_mask
            )  # (B, T, node_emb_dim)

            external_cond = action_cond

        # Delegate to UViT3D.forward (UViT3DAction does not override forward).
        # Action cond is added to noise-level emb as usual; text goes via
        # cross-attention through context_tokens.
        return UViT3D.forward(
            self,
            x,
            noise_levels,
            external_cond=external_cond,
            external_cond_mask=external_cond_mask,
            context_tokens=context_tokens,
            context_mask=None,
        )
