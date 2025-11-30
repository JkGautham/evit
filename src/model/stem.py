import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class OverlapPatchEmbedding(nn.Module):
    """
    Overlap patch embedding (SegFormer / EViT style) with flexible normalization.

    Behavior:
      - Applies a Conv2d(in_ch -> embed_dim) with kernel/stride/padding.
      - Optionally applies a map-normalization (BatchNorm2d / GroupNorm) OR
        token LayerNorm (SegFormer default). When using LayerNorm, we flatten to (B, N, C)
        and LN is applied over the last dim.
      - Returns: (tokens, H', W', feat_map)
          tokens: (B, N, C)   where N = H'*W'
          H', W': spatial dims after the conv
          feat_map: (B, C, H', W')  (useful for decoder skip connections)

    Args:
      in_channels: input image channels (usually 3)
      embed_dim: output embedding dimension / conv out channels
      kernel_size: conv kernel size (int)
      stride: conv stride (int)
      padding: conv padding (int)
      norm: 'ln' (LayerNorm on tokens, default), 'bn' (BatchNorm2d on feature map),
            or 'gn' (GroupNorm on feature map).
      gn_num_groups: number of groups if norm=='gn'
      activation: callable or name; 'gelu' | 'relu' | None
      dropout: dropout probability applied to tokens after norm (optional)
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 32,
        kernel_size: int = 7,
        stride: int = 4,
        padding: int = 3,
        norm: str = "ln",
        gn_num_groups: int = 1,
        activation: Optional[str] = "gelu",
        dropout: float = 0.0,
    ):

        super().__init__()

        assert norm in ("ln", "bn", "gn"), "norm must be one of ('ln','bn','gn')"

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=(norm != "bn"),  # if BN follows, bias can be omitted
        )


        # map-normalization options (applied to (B, C, H, W))
        if norm == "bn":
            self.map_norm = nn.BatchNorm2d(embed_dim)
            self.token_norm = None
        elif norm == "gn":
            # GroupNorm(num_groups, num_channels)
            # if gn_num_groups == 1, GroupNorm becomes LayerNorm over channel axis per spatial location
            self.map_norm = nn.GroupNorm(gn_num_groups, embed_dim)
            self.token_norm = None
        else:  # 'ln'
            self.map_norm = None
            # token LayerNorm operates on last dimension (embed_dim) after flattening to (B, N, C)
            self.token_norm = nn.LayerNorm(embed_dim, eps=1e-6)


        # activation
        if activation is None:
            self.act = None
        elif activation.lower() in ("gelu",):
            self.act = nn.GELU()
        elif activation.lower() in ("relu",):
            self.act = nn.ReLU(inplace=True)
        else:
            raise ValueError("unsupported activation: " + str(activation))


        # dropout on tokens (applied after norm)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else None
        self.norm_type = norm
        self.embed_dim = embed_dim

    def out_shape(self, H: int, W: int) -> Tuple[int, int]:
        """
        Compute spatial output shape (H', W') after the conv.
        Use this if you need to pre-allocate buffers.
        """
        # mimic Conv2d output formula
        k = self.proj.kernel_size
        s = self.proj.stride
        p = self.proj.padding
        # kernel_size, stride, padding can be tuples; handle both
        kh = k[0] if isinstance(k, tuple) else k
        kw = k[1] if isinstance(k, tuple) else k
        sh = s[0] if isinstance(s, tuple) else s
        sw = s[1] if isinstance(s, tuple) else s
        ph = p[0] if isinstance(p, tuple) else p
        pw = p[1] if isinstance(p, tuple) else p

        H_out = (H + 2 * ph - kh) // sh + 1
        W_out = (W + 2 * pw - kw) // sw + 1
        return int(H_out), int(W_out)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int, torch.Tensor]:
        """
        Forward pass.

        Input:
          x: (B, in_channels, H, W)

        Returns:
          tokens: (B, N, C)
          H', W': ints
          feat_map: (B, C, H', W')
        """
        # conv projection
        feat = self.proj(x)  # (B, C, H', W')
        B, C, H, W = feat.shape

        # apply map-level norm if configured (BN / GN)
        if self.map_norm is not None:
            feat = self.map_norm(feat)  # still (B, C, H', W')

        # activation on map (if using map norm or token norm both OK)
        if self.act is not None:
            feat = self.act(feat)

        # produce tokens: (B, N, C)
        tokens = feat.flatten(2).transpose(1, 2)  # (B, H'*W', C)

        # token normalization (if LN chosen)
        if self.token_norm is not None:
            tokens = self.token_norm(tokens)

        # dropout (optional) - use dropout on token embedding dimension
        if self.dropout is not None:
            # dropout expects (B, C, H, W) or (B, N, C) depending; use functional dropout on tokens
            tokens = F.dropout(tokens, p=self.dropout.p, training=self.training)

        # reconstruct feature map (useful for decoder)
        feat_map = tokens.transpose(1, 2).reshape(B, C, H, W)

        return tokens, H, W, feat_map