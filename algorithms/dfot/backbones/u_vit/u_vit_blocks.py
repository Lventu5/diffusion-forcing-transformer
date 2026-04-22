from typing import Optional, Tuple
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from einops import rearrange


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)
from ..modules.normalization import (
    RMSNorm as Normalize,
)
from ..modules.embeddings import (
    RotaryEmbedding1D,
    RotaryEmbedding2D,
    RotaryEmbeddingND,
)
from ..modules.zero_module import zero_module


class EmbedInput(nn.Module):
    """
    Initial downsampling layer for U-ViT.
    One shall replace this with 5/3 DWT, which is fully invertible and may slightly improve performance, according to the Simple Diffusion paper.
    """

    def __init__(self, in_channels: int, dim: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        return x


class ProjectOutput(nn.Module):
    """
    Final upsampling layer for U-ViT.
    One shall replace this with IDWT, which is an inverse operation of DWT.
    """

    def __init__(self, dim: int, out_channels: int, patch_size: int):
        super().__init__()
        self.proj = zero_module(
            nn.ConvTranspose2d(
                dim, out_channels, kernel_size=patch_size, stride=patch_size
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        return x


# pylint: disable-next=invalid-name
def NormalizeWithBias(num_channels: int):
    return nn.GroupNorm(num_groups=32, num_channels=num_channels, eps=1e-6, affine=True)


class ResBlock(nn.Module):
    """
    Standard ResNet block.
    """

    def __init__(self, channels: int, emb_dim: int, dropout: float = 0.0):
        super().__init__()
        assert dropout == 0.0, "Dropout is not supported in ResBlock."
        self.emb_layer = nn.Conv2d(emb_dim, channels * 2, kernel_size=(1, 1))
        self.in_layers = nn.Sequential(
            NormalizeWithBias(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1)),
        )
        self.out_norm = NormalizeWithBias(channels)
        self.out_rest = nn.Sequential(
            nn.SiLU(),
            zero_module(
                nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1))
            ),
        )

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        """
        Forward pass of the ResNet block.
        Args:
            x: Input tensor of shape (B, C, H, W).
            emb: Embedding tensor of shape (B, C) or (B, C, H, W).
        Returns:
            Output tensor of shape (B, C, H, W).
        """
        h = self.in_layers(x)
        emb_out = self.emb_layer(emb if emb.dim() == 4 else emb[:, :, None, None])
        scale, shift = emb_out.chunk(2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_rest(h)
        return x + h


class NormalizeWithCond(nn.Module):
    """
    Conditioning block for U-ViT, that injects external conditions into the network using FiLM.
    """

    def __init__(self, dim: int, emb_dim: int):
        super().__init__()
        self.emb_layer = nn.Linear(emb_dim, dim * 2)
        self.norm = Normalize(dim)

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        """
        Forward pass of the conditioning block.
        Args:
            x: Input tensor of shape (B, N, C).
            emb: Embedding tensor of shape (B, N, C).
        Returns:
            Output tensor of shape (B, N, C).
        """
        scale, shift = self.emb_layer(emb).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class AttentionBlock(nn.Module):
    """
    Simple Attention block for axial attention.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        emb_dim: int,
        rope: Optional[RotaryEmbeddingND] = None,
    ):
        super().__init__()
        dim_head = dim // heads
        self.heads = heads
        self.rope = rope
        self.norm = NormalizeWithCond(dim, emb_dim)
        self.proj = nn.Linear(dim, dim * 3, bias=False)
        self.q_norm, self.k_norm = Normalize(dim_head), Normalize(dim_head)
        self.out = zero_module(nn.Linear(dim, dim, bias=False))

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        """
        Forward pass of the attention block.
        Args:
            x: Input tensor of shape (B, N, C).
            emb: Embedding tensor of shape (B, N, C).
        Returns:
            Output tensor of shape (B, N, C).
        """
        x = self.norm(x, emb)
        qkv = self.proj(x)
        q, k, v = rearrange(
            qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads
        ).unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)

        # pylint: disable-next=not-callable
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b h n d -> b n (h d)")
        return x + self.out(x)


class CrossAttnBlock(nn.Module):
    """
    Cross-attention block that attends input tokens to context tokens.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        emb_dim: int,
        context_dim: Optional[int] = None,
        dropout: float = 0.0,
        t_seq: int = 0,
    ):
        super().__init__()
        _ = emb_dim  # unused: CrossAttnBlock does not use FiLM conditioning
        self.heads = heads
        dim_head = dim // heads
        self.t_seq = t_seq
        self.norm = Normalize(dim)
        self.context_norm = Normalize(context_dim or dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(context_dim or dim, dim * 2, bias=False)
        self.q_norm, self.k_norm = Normalize(dim_head), Normalize(dim_head)
        self.out = zero_module(nn.Linear(dim, dim, bias=True))
        self.dropout = nn.Dropout(dropout)
        # Temporal 1D RoPE: shared T index for video Q tokens and context K tokens.
        self.rope_t = RotaryEmbedding1D(dim_head, t_seq) if t_seq > 0 else None
        # Set to True externally to capture last attention weights for visualization.
        self.store_attn_weights: bool = False
        self.last_attn_weights: Optional[Tensor] = None  # (B, heads, N, M)

    def forward(
        self,
        x: Tensor,
        emb: Tensor,
        context: Tensor,
        context_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        """
        Args:
            x: (B, N, C) input tokens.
            emb: (B, N, C) conditioning embedding (unused).
            context: (B, M, Cc) context tokens.
            context_mask: optional bool mask (B, M), True for valid tokens.
            is_causal: if True, frame t only attends to context tokens <= t.
        """
        _ = emb
        residual = x
        x = self.norm(x)
        context = self.context_norm(context)

        q = self.q_proj(x)
        kv = self.kv_proj(context)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k, v = rearrange(kv, "b m (kv h d) -> kv b h m d", kv=2, h=self.heads).unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope_t is not None:
            n_q, n_k = q.shape[2], k.shape[2]
            t_idx_q = torch.arange(n_q, device=q.device) // (n_q // self.t_seq)
            t_idx_k = torch.arange(n_k, device=k.device) // (n_k // self.t_seq)
            freqs_q = self.rope_t.freqs[t_idx_q][None, None]  # (1, 1, n_q, dim_head)
            freqs_k = self.rope_t.freqs[t_idx_k][None, None]  # (1, 1, n_k, dim_head)
            q = q * freqs_q.cos() + _rotate_half(q) * freqs_q.sin()
            k = k * freqs_k.cos() + _rotate_half(k) * freqs_k.sin()

        attn_mask = None
        if is_causal:
            n = x.shape[1]
            m = context.shape[1]
            if n % m != 0:
                raise ValueError(
                    f"Causal cross-attn expects N divisible by M (N={n}, M={m})."
                )
            patches_per_t = n // m
            t_q = torch.arange(n, device=x.device) // patches_per_t
            t_k = torch.arange(m, device=x.device)
            causal_allow = t_k[None, :] <= t_q[:, None]  # (N, M)
            attn_mask = ~causal_allow[None, None, :, :]  # True = masked
        elif context_mask is not None:
            attn_mask = ~context_mask[:, None, None, :]

        if self.store_attn_weights:
            scale = q.shape[-1] ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if attn_mask is not None:
                scores = scores.masked_fill(attn_mask, float("-inf"))
            attn_w = scores.softmax(dim=-1)
            self.last_attn_weights = attn_w.detach().cpu()
            x = torch.matmul(attn_w, v)
        else:
            # pylint: disable-next=not-callable
            # SDPA boolean convention: True = attend. Our attn_mask uses True = blocked,
            # so negate before passing (the masked_fill branch above is already correct).
            sdpa_mask = ~attn_mask if (attn_mask is not None and attn_mask.dtype == torch.bool) else attn_mask
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask, is_causal=False)
        x = rearrange(x, "b h n d -> b n (h d)")
        x = self.out(x)
        x = self.dropout(x)
        return residual + x


class AxialRotaryEmbedding(nn.Module):
    """
    Axial rotary embedding for axial attention.
    Composed of two rotary embeddings for each axis.
    """

    def __init__(
        self,
        dim: int,
        sizes: Tuple[int, int] | Tuple[int, int, int],
        theta: float = 10000.0,
        flatten: bool = True,
    ):
        """
        If len(sizes) == 2, each axis corresponds to each dimension.
        If len(sizes) == 3, the first dimension corresponds to the first axis, and the rest corresponds to the second axis.
        This enables to be compatible with the initializations of `.embeddings.RotaryEmbedding2D` and `.embeddings.RotaryEmbedding3D`.
        """
        super().__init__()
        self.ax1 = RotaryEmbedding1D(dim, sizes[0], theta, flatten)
        self.ax2 = (
            RotaryEmbedding1D(dim, sizes[1], theta, flatten)
            if len(sizes) == 2
            else RotaryEmbedding2D(dim, sizes[1:], theta, flatten)
        )


class TransformerBlock(nn.Module):
    """
    Efficient transformer block with parallel attention + MLP and Query-Key normalization,
    following https://arxiv.org/abs/2302.05442

    Supports axial attention.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        emb_dim: int,
        dropout: float,
        use_axial: bool = False,
        ax1_len: Optional[int] = None,
        rope: Optional[AxialRotaryEmbedding | RotaryEmbeddingND] = None,
        use_cross_attn: bool = False,
        cross_attn_is_causal: bool = False,
        cross_attn_context_dim: Optional[int] = None,
        cross_attn_t_seq: int = 0,
    ):
        super().__init__()
        self.rope = rope.ax2 if (rope is not None and use_axial) else rope
        self.norm = NormalizeWithCond(dim, emb_dim)

        self.heads = heads
        dim_head = dim // heads
        self.use_axial = use_axial
        self.ax1_len = ax1_len
        self.use_cross_attn = use_cross_attn
        self.cross_attn_is_causal = cross_attn_is_causal
        mlp_dim = 4 * dim
        self.fused_dims = (3 * dim, mlp_dim)
        self.fused_attn_mlp_proj = nn.Linear(dim, sum(self.fused_dims), bias=True)
        self.q_norm, self.k_norm = Normalize(dim_head), Normalize(dim_head)

        self.attn_out = zero_module(nn.Linear(dim, dim, bias=True))

        if self.use_axial:
            self.another_attn = AttentionBlock(
                dim, heads, emb_dim, rope.ax1 if rope is not None else None
            )

        if self.use_cross_attn:
            self.cross_attn = CrossAttnBlock(
                dim,
                heads,
                emb_dim,
                context_dim=cross_attn_context_dim or dim,
                dropout=dropout,
                t_seq=cross_attn_t_seq,
            )

        self.mlp_out = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_module(nn.Linear(mlp_dim, dim, bias=True)),
        )

    def forward(
        self,
        x: Tensor,
        emb: Tensor,
        context_tokens: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass of the transformer block.
        Args:
            x: Input tensor of shape (B, N, C).
            emb: Embedding tensor of shape (B, N, C).
            context_tokens: Optional context tokens (B, M, Cc) for cross-attention.
            context_mask: Optional bool mask (B, M) for context tokens.
        Returns:
            Output tensor of shape (B, N, C).
        """
        if self.use_axial:
            x, emb = map(
                lambda y: rearrange(
                    y, "b (ax1 ax2) d -> (b ax1) ax2 d", ax1=self.ax1_len
                ),
                (x, emb),
            )
        _x = x
        x = self.norm(x, emb)
        qkv, mlp_h = self.fused_attn_mlp_proj(x).split(self.fused_dims, dim=-1)
        qkv = rearrange(qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)

        # pylint: disable-next=not-callable
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b h n d -> b n (h d)")
        x = _x + self.attn_out(x)

        if self.use_axial:
            ax2_len = x.shape[1]
            x, emb = map(
                lambda y: rearrange(
                    y, "(b ax1) ax2 d -> (b ax2) ax1 d", ax1=self.ax1_len
                ),
                (x, emb),
            )
            x = self.another_attn(x, emb)
            x = rearrange(x, "(b ax2) ax1 d -> (b ax1) ax2 d", ax2=ax2_len)

        if self.use_cross_attn:
            if context_tokens is None:
                raise ValueError(
                    "TransformerBlock with use_cross_attn=True requires context_tokens, but none were provided."
                )
            x = self.cross_attn(
                x, emb, context_tokens, context_mask=context_mask,
                is_causal=self.cross_attn_is_causal,
            )

        x = x + self.mlp_out(mlp_h)

        if self.use_axial:
            x = rearrange(x, "(b ax1) ax2 d -> b (ax1 ax2) d", ax1=self.ax1_len)
        return x


class Downsample(nn.Module):
    """
    Downsample block for U-ViT.
    Done by average pooling + conv.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        # pylint: disable-next=not-callable
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    """
    Upsample block for U-ViT.
    Done by conv + nearest neighbor upsampling.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return x
