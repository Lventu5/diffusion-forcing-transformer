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
  graph_tokens_per_frame: int    >1 for packed graph-token sidecars
  graph_token_dim:        int    per-token embedding dim when graph_tokens_per_frame > 1
  graph_token_has_mask:   bool   packed sidecar includes K mask bits (default true)
"""

from __future__ import annotations

from typing import Optional

import torch
from omegaconf import DictConfig
from torch import Tensor

from .uvit3d_action import UViT3DAction
from .u_vit3d import UViT3D
from ..modules.embeddings import RandomEmbeddingDropout


def unpack_graph_token_context(
    node_cond: Tensor,
    *,
    tokens_per_frame: int,
    token_dim: int,
    has_mask: bool = True,
) -> tuple[Tensor, Optional[Tensor]]:
    """Unpack ``[K*D flattened tokens | K mask]`` into cross-attn context."""
    if tokens_per_frame <= 1:
        return node_cond, None

    token_values_dim = tokens_per_frame * token_dim
    expected_dim = token_values_dim + (tokens_per_frame if has_mask else 0)
    if node_cond.shape[-1] != expected_dim:
        raise ValueError(
            "Graph-token condition dim mismatch: "
            f"got {node_cond.shape[-1]}, expected {expected_dim} "
            f"(K={tokens_per_frame}, D={token_dim}, mask={has_mask})."
        )

    B, T = node_cond.shape[:2]
    tokens = node_cond[..., :token_values_dim].reshape(
        B,
        T * tokens_per_frame,
        token_dim,
    )
    if not has_mask:
        return tokens, None

    mask = node_cond[..., token_values_dim:].reshape(B, T * tokens_per_frame)
    context_mask = mask > 0.5

    # A fully zero packed row is the graph-token representation of "no node
    # condition" after padding, context masking, or CFG-style zeroing. Cross
    # attention still needs one valid key/value slot; use slot 0 as a null token.
    frame_tokens = tokens.reshape(B, T, tokens_per_frame, token_dim)
    frame_mask = context_mask.reshape(B, T, tokens_per_frame)
    empty_frames = ~frame_mask.any(dim=-1)
    if empty_frames.any():
        frame_has_values = frame_tokens.abs().sum(dim=(-1, -2)) > 0
        inconsistent = empty_frames & frame_has_values
        if inconsistent.any():
            raise ValueError(
                "Graph-token condition contains non-zero token values but no valid mask bits. "
                "Check packed node embedding sidecars and graph_token_has_mask."
            )
        frame_mask = frame_mask.clone()
        frame_mask[..., 0] |= empty_frames
        context_mask = frame_mask.reshape(B, T * tokens_per_frame)

    return tokens, context_mask


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
        self._graph_tokens_per_frame = int(cfg.get("graph_tokens_per_frame", 1))
        self._graph_token_dim = int(cfg.get("graph_token_dim", node_emb_dim))
        self._graph_token_has_mask = bool(cfg.get("graph_token_has_mask", True))
        if self._graph_tokens_per_frame <= 0:
            raise ValueError("graph_tokens_per_frame must be >= 1")
        if self._graph_tokens_per_frame == 1:
            self._graph_token_dim = node_emb_dim

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
        node_cond_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            x:                  (B, T, C, H, W)
            noise_levels:       (B, T)
            external_cond:      (B, T, 3 + node_emb_dim)  or None
            external_cond_mask: (B,) CFG mask for action conditioning.
            node_cond_mask:     (B,) CFG mask for node/text conditioning.
                                When provided, action and node are dropped independently.
                                When None, falls back to external_cond_mask (joint drop).
        Returns:
            (B, T, C, H, W)
        """
        context_tokens = None
        context_mask = None
        if external_cond is not None:
            action_cond = external_cond[..., : self._ACTION_DIMS]   # (B, T, 3)
            node_emb    = external_cond[..., self._ACTION_DIMS :]   # (B, T, node_emb_dim)
            context_tokens, context_mask = unpack_graph_token_context(
                node_emb.float(),
                tokens_per_frame=self._graph_tokens_per_frame,
                token_dim=self._graph_token_dim,
                has_mask=self._graph_token_has_mask,
            )

            # Independent dropout: use node_cond_mask if provided, else fall back to
            # external_cond_mask (backward-compatible joint drop).
            context_tokens = self._node_dropout(
                context_tokens,
                node_cond_mask if node_cond_mask is not None else external_cond_mask,
            )  # (B, T*K, graph_token_dim) or legacy (B, T, node_emb_dim)

            external_cond = action_cond

        # [ctx-check] One-time shape/value check to verify context_tokens are correct.
        if not getattr(self, "_ctx_checked", False):
            self._ctx_checked = True
            if context_tokens is not None:
                var_across_t = context_tokens.std(dim=1).mean().item()
                print(
                    f"[ctx-check] context_tokens: shape={tuple(context_tokens.shape)} "
                    f"mask_shape={None if context_mask is None else tuple(context_mask.shape)} "
                    f"mean={context_tokens.mean().item():.4f} "
                    f"std={context_tokens.std().item():.4f} "
                    f"std_across_T={var_across_t:.4f} "
                    f"(near-zero std_across_T → all frames same embedding)"
                )
            else:
                print("[ctx-check] context_tokens is None — no node conditioning active")

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
            context_mask=context_mask,
        )
