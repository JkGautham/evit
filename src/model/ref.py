"""
============================================================================
EViT: Efficient Vision and Tracking System
============================================================================
Complete production-ready implementation with corrected architecture
ARM Jetson Nano / Edge Device Optimized
============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import time
import os
from typing import Tuple, Dict, List, Optional
from collections import defaultdict


# ============================================================================
# PART 1: PATCH EMBEDDING WITH OVERLAP
# ============================================================================

class OverlapPatchEmbedding(nn.Module):
    """
    Overlapping Patch Embedding using strided convolution.
    Preserves spatial information without explicit positional encoding.

    Args:
        in_channels: Input channels (3 for RGB, or previous stage channels)
        embed_dim: Output embedding dimension
        kernel_size: Convolution kernel size (7 for first stage, 3 for others)
        stride: Stride for spatial reduction (4 for first stage, 2 for others)
    """

    def __init__(
            self,
            in_channels: int,
            embed_dim: int,
            kernel_size: int = 7,
            stride: int = 4
    ):
        super().__init__()
        padding = kernel_size // 2

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            tokens: Flattened tokens (B, N, C) where N = H' * W'
            H: Height after projection
            W: Width after projection
        """
        x = self.proj(x)  # (B, embed_dim, H', W')
        B, C, H, W = x.shape

        # Flatten spatial dimensions: (B, C, H, W) -> (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


# ============================================================================
# PART 2: EFFICIENT ATTENTION WITH SPATIAL REDUCTION
# ============================================================================

class EfficientAttention(nn.Module):
    """
    Efficient Multi-Head Self-Attention with Spatial Reduction.
    Reduces attention complexity from O(N²) to O(N²/R²).

    Key innovation: Apply spatial reduction to Key and Value only,
    keeping Query at full resolution for detailed feature representation.

    Args:
        dim: Input/output dimension
        num_heads: Number of attention heads
        reduction_ratio: Spatial reduction ratio for K,V (default: 2)
        qkv_bias: Whether to use bias in Q,K,V projections
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            reduction_ratio: int = 2,
            qkv_bias: bool = False
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.reduction_ratio = reduction_ratio

        # Query projection (full resolution)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        # Key-Value projection (will be applied after spatial reduction)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        # Output projection
        self.proj = nn.Linear(dim, dim)

        # Spatial reduction for K,V (if ratio > 1)
        if reduction_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=reduction_ratio,
                                stride=reduction_ratio, groups=dim)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: Input tokens (B, N, C) where N = H * W
            H: Spatial height
            W: Spatial width

        Returns:
            out: Attention output (B, N, C)
        """
        B, N, C = x.shape

        # Query: full resolution
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # (B, num_heads, N, head_dim)

        # Key-Value: with spatial reduction
        if self.reduction_ratio > 1:
            # Reshape to spatial: (B, N, C) -> (B, H, W, C) -> (B, C, H, W)
            x_spatial = x.permute(0, 2, 1).reshape(B, C, H, W)

            # Apply spatial reduction
            x_reduced = self.sr(x_spatial)  # (B, C, H/R, W/R)

            # Flatten back: (B, C, H/R, W/R) -> (B, C, N/R²) -> (B, N/R², C)
            x_reduced = x_reduced.flatten(2).transpose(1, 2)
            x_reduced = self.norm(x_reduced)

            # Project to K,V
            kv = self.kv(x_reduced)  # (B, N/R², 2*C)
        else:
            kv = self.kv(x)  # (B, N, 2*C)

        # Split into K and V
        kv = kv.reshape(B, -1, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)  # (2, B, num_heads, N', head_dim)
        k, v = kv[0], kv[1]

        # Attention: Q @ K^T
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N')
        attn = F.softmax(attn, dim=-1)

        # Aggregate: Attn @ V
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # Output projection
        x = self.proj(x)

        return x


# ============================================================================
# PART 3: FEED-FORWARD NETWORK WITH DEPTHWISE CONV
# ============================================================================

class FeedForward(nn.Module):
    """
    Feed-Forward Network with Depthwise Convolution for spatial mixing.
    Structure: Linear -> GELU -> DWConv -> Linear

    Args:
        dim: Input/output dimension
        hidden_dim: Hidden layer dimension (typically 4x dim)
        drop: Dropout rate
    """

    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1,
                                groups=hidden_dim, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: Input (B, N, C)
            H: Spatial height
            W: Spatial width

        Returns:
            out: Output (B, N, C)
        """
        B, N, C = x.shape

        # First linear + activation
        x = self.fc1(x)
        x = self.act(x)

        # Reshape for depthwise conv
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)  # Back to (B, N, C)

        # Second linear
        x = self.fc2(x)
        x = self.drop(x)

        return x


# ============================================================================
# PART 4: TRANSFORMER BLOCK
# ============================================================================

class TransformerBlock(nn.Module):
    """
    Single Transformer Block with efficient attention and FFN.
    Structure:
        Input -> [LN -> Efficient Attn -> +] -> [LN -> FFN -> +] -> Output

    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
        mlp_ratio: Ratio of hidden dim to input dim in FFN
        reduction_ratio: Spatial reduction for attention
        drop: Dropout rate
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            reduction_ratio: int = 2,
            drop: float = 0.0
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(dim, num_heads, reduction_ratio)

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.ffn = FeedForward(dim, hidden_dim, drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: Input (B, N, C)
            H: Spatial height
            W: Spatial width

        Returns:
            out: Output (B, N, C)
        """
        # Attention with residual
        x = x + self.attn(self.norm1(x), H, W)

        # FFN with residual
        x = x + self.ffn(self.norm2(x), H, W)

        return x

# ============================================================================
# ASPP-LITE MODULE (EDGE OPTIMIZED)
# ============================================================================

class ASPPLite(nn.Module):
    """
    Lightweight ASPP module for edge devices.
    Uses depthwise dilated convolutions instead of full atrous convs.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # 1x1 conv branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Dilated DWConv rate=2
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3,
                      padding=2, dilation=2, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Dilated DWConv rate=4
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3,
                      padding=4, dilation=4, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Global pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Final fusion
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        b4 = self.global_pool(x)
        b4 = F.interpolate(b4, size=(H, W),
                           mode="bilinear", align_corners=False)

        x = torch.cat([b1, b2, b3, b4], dim=1)
        x = self.fuse(x)

        return x



# ============================================================================
# PART 5: HIERARCHICAL ENCODER (4 STAGES)
# ============================================================================

class EViTEncoder(nn.Module):
    """
    4-Stage Hierarchical Transformer Encoder.
    Based on SegFormer architecture with overlapping patch embeddings.

    Each stage progressively:
    - Reduces spatial resolution (H/4 -> H/8 -> H/16 -> H/32)
    - Increases channel dimension
    - Applies multiple transformer blocks

    Args:
        in_channels: Input image channels (3 for RGB)
        embed_dims: List of embedding dimensions for each stage
        num_heads: List of attention heads for each stage
        depths: List of transformer block depths for each stage
        mlp_ratios: List of MLP expansion ratios for each stage
        reduction_ratios: List of spatial reduction ratios for each stage
    """

    def __init__(
            self,
            in_channels: int = 3,
            embed_dims: List[int] = [64, 128, 256, 512],
            num_heads: List[int] = [1, 2, 4, 8],
            depths: List[int] = [3, 4, 6, 3],
            mlp_ratios: List[float] = [4.0, 4.0, 4.0, 4.0],
            reduction_ratios: List[int] = [8, 4, 2, 1],
            drop_rate: float = 0.0
    ):
        super().__init__()

        self.depths = depths
        self.num_stages = 4

        # Patch embedding layers for each stage
        self.patch_embeds = nn.ModuleList()

        # Stage 1: RGB -> First embedding
        self.patch_embeds.append(
            OverlapPatchEmbedding(in_channels, embed_dims[0],
                                  kernel_size=7, stride=4)
        )

        # Stages 2-4: Previous stage features -> Next stage
        for i in range(1, self.num_stages):
            self.patch_embeds.append(
                OverlapPatchEmbedding(embed_dims[i - 1], embed_dims[i],
                                      kernel_size=3, stride=2)
            )

        # Transformer blocks for each stage
        self.blocks = nn.ModuleList()
        for stage_idx in range(self.num_stages):
            stage_blocks = nn.ModuleList([
                TransformerBlock(
                    dim=embed_dims[stage_idx],
                    num_heads=num_heads[stage_idx],
                    mlp_ratio=mlp_ratios[stage_idx],
                    reduction_ratio=reduction_ratios[stage_idx],
                    drop=drop_rate
                )
                for _ in range(depths[stage_idx])
            ])
            self.blocks.append(stage_blocks)

        # Layer norms for each stage output
        self.norms = nn.ModuleList([
            nn.LayerNorm(embed_dims[i]) for i in range(self.num_stages)
        ])
        self.aspp = ASPPLite(
            in_channels=embed_dims[2],
            out_channels=embed_dims[2]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through all 4 stages.

        Args:
            x: Input image (B, 3, H, W)

        Returns:
            features: List of 4 feature maps from each stage
                     [(B, N1, C1), (B, N2, C2), (B, N3, C3), (B, N4, C4)]
        """
        B = x.shape[0]
        features = []

        for stage_idx in range(self.num_stages):
            x, H, W = self.patch_embeds[stage_idx](x)

            for block in self.blocks[stage_idx]:
                x = block(x, H, W)

            x = self.norms[stage_idx](x)

            # ===== ASPP after Stage 3 =====
            if stage_idx == 2:
                # Token → spatial
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)

                # ASPP Lite context
                x = self.aspp(x)

                # Spatial → token
                x = x.flatten(2).transpose(1, 2)

            features.append(x)

            # Prepare for next stage
            if stage_idx < self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        return features


# ============================================================================
# PART 6: ALL-MLP DECODER (SEGFORMER STYLE)
# ============================================================================

class MLPDecoder(nn.Module):
    """
    All-MLP Decoder for efficient multi-scale feature fusion.
    Inspired by SegFormer's lightweight decoder design.

    Process:
    1. Project all stage features to same channel dimension
    2. Upsample to common resolution
    3. Concatenate
    4. Fuse with 1x1 conv
    5. Generate segmentation logits

    Args:
        in_channels: List of input channels from each encoder stage
        num_classes: Number of segmentation classes
        embed_dim: Unified embedding dimension for fusion
    """

    def __init__(
            self,
            in_channels: List[int],
            num_classes: int,
            embed_dim: int = 256
    ):
        super().__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Linear projections for each stage
        self.linear_projections = nn.ModuleList([
            nn.Linear(in_dim, embed_dim) for in_dim in in_channels
        ])

        # Fusion convolution
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels), embed_dim,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

        # Segmentation head
        self.segmentation_head = nn.Conv2d(embed_dim, num_classes,
                                           kernel_size=1)

    def forward(
            self,
            features: List[torch.Tensor],
            spatial_shapes: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """
        Args:
            features: List of encoder features [(B, N1, C1), ..., (B, N4, C4)]
            spatial_shapes: List of (H, W) for each feature map

        Returns:
            logits: Segmentation logits (B, num_classes, H, W)
        """
        B = features[0].shape[0]
        target_H, target_W = spatial_shapes[0]  # Upsample to first stage resolution

        upsampled_features = []

        for i, (feat, (H, W)) in enumerate(zip(features, spatial_shapes)):
            # Project to unified dimension
            feat = self.linear_projections[i](feat)  # (B, N, embed_dim)

            # Reshape to spatial
            feat = feat.reshape(B, H, W, self.embed_dim)
            feat = feat.permute(0, 3, 1, 2)  # (B, embed_dim, H, W)

            # Upsample to target resolution
            if H != target_H or W != target_W:
                feat = F.interpolate(feat, size=(target_H, target_W),
                                     mode='bilinear', align_corners=False)

            upsampled_features.append(feat)

        # Concatenate all features
        x = torch.cat(upsampled_features, dim=1)  # (B, embed_dim*4, H, W)

        # Fuse features
        x = self.fusion(x)  # (B, embed_dim, H, W)

        # Generate segmentation logits
        logits = self.segmentation_head(x)  # (B, num_classes, H, W)

        return logits


# ============================================================================
# PART 7: COMPLETE EViT MODEL
# ============================================================================

class EViT(nn.Module):
    """
    Complete EViT: Efficient Vision and Tracking Model.
    Combines hierarchical transformer encoder with lightweight MLP decoder.

    Args:
        num_classes: Number of segmentation classes
        in_channels: Input image channels
        encoder_embed_dims: Embedding dimensions for encoder stages
        encoder_num_heads: Attention heads for encoder stages
        encoder_depths: Transformer block depths for encoder stages
        encoder_mlp_ratios: MLP expansion ratios for encoder stages
        encoder_reduction_ratios: Spatial reduction ratios for encoder stages
        decoder_embed_dim: Unified dimension for decoder
        drop_rate: Dropout rate
    """

    def __init__(
            self,
            num_classes: int = 10,
            in_channels: int = 3,
            encoder_embed_dims: List[int] = [64, 128, 256, 512],
            encoder_num_heads: List[int] = [1, 2, 4, 8],
            encoder_depths: List[int] = [3, 4, 6, 3],
            encoder_mlp_ratios: List[float] = [4.0, 4.0, 4.0, 4.0],
            encoder_reduction_ratios: List[int] = [8, 4, 2, 1],
            decoder_embed_dim: int = 256,
            drop_rate: float = 0.0
    ):
        super().__init__()

        # Encoder
        self.encoder = EViTEncoder(
            in_channels=in_channels,
            embed_dims=encoder_embed_dims,
            num_heads=encoder_num_heads,
            depths=encoder_depths,
            mlp_ratios=encoder_mlp_ratios,
            reduction_ratios=encoder_reduction_ratios,
            drop_rate=drop_rate
        )

        # Decoder
        self.decoder = MLPDecoder(
            in_channels=encoder_embed_dims,
            num_classes=num_classes,
            embed_dim=decoder_embed_dim
        )

        # Store config for spatial shape calculation
        self.encoder_embed_dims = encoder_embed_dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input image (B, 3, H, W)

        Returns:
            logits: Segmentation logits (B, num_classes, H', W')
        """
        B, C, H, W = x.shape

        # Encoder: Extract multi-scale features
        features = self.encoder(x)

        # Calculate spatial shapes for each stage
        spatial_shapes = [
            (H // 4, W // 4),  # Stage 1: stride 4
            (H // 8, W // 8),  # Stage 2: stride 8
            (H // 16, W // 16),  # Stage 3: stride 16
            (H // 32, W // 32)  # Stage 4: stride 32
        ]

        # Decoder: Fuse features and generate segmentation
        logits = self.decoder(features, spatial_shapes)

        # Upsample logits to input resolution
        if logits.shape[-2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W),
                                   mode='bilinear', align_corners=False)

        return logits


# ============================================================================
# PART 8: KALMAN FILTER FOR TRACKING
# ============================================================================

class KalmanTracker:
    """
    Kalman Filter for object tracking with constant velocity model.

    State vector: [x, y, w, h, vx, vy, vw, vh]
    - (x, y): Center coordinates
    - (w, h): Width and height
    - (vx, vy): Velocity in x and y
    - (vw, vh): Change rate of width and height

    Args:
        bbox: Initial bounding box [x, y, w, h]
        track_id: Unique identifier for this track
    """

    def __init__(self, bbox: List[float], track_id: int):
        self.track_id = track_id

        # Initialize state: [x, y, w, h, vx, vy, vw, vh]
        self.state = np.array(bbox + [0, 0, 0, 0], dtype=np.float32)

        # State covariance matrix
        self.P = np.eye(8, dtype=np.float32)
        self.P[4:, 4:] *= 10.0  # Higher uncertainty for velocities

        # Process noise covariance
        self.Q = np.eye(8, dtype=np.float32)
        self.Q[4:, 4:] *= 0.01  # Lower noise for velocities

        # Measurement noise covariance
        self.R = np.eye(4, dtype=np.float32) * 0.1

        # Track management
        self.age = 0
        self.hits = 0
        self.hit_streak = 0
        self.time_since_update = 0

    def predict(self) -> np.ndarray:
        """
        Predict next state using constant velocity model.

        Returns:
            predicted_bbox: Predicted bounding box [x, y, w, h]
        """
        # Constant velocity model: position += velocity
        self.state[:4] += self.state[4:]

        # Update covariance
        self.P += self.Q

        # Update track management
        self.age += 1
        self.time_since_update += 1

        return self.state[:4].copy()

    def update(self, bbox: List[float]) -> None:
        """
        Update state with measurement (detected bounding box).

        Args:
            bbox: Measured bounding box [x, y, w, h]
        """
        z = np.array(bbox, dtype=np.float32)

        # Innovation (measurement residual)
        y = z - self.state[:4]

        # Innovation covariance
        S = self.P[:4, :4] + self.R

        # Kalman gain
        K = self.P[:4, :4] @ np.linalg.inv(S)

        # Update state
        self.state[:4] += K @ y

        # Update velocity based on position change
        if self.time_since_update > 0:
            self.state[4:] = y / self.time_since_update

        # Update covariance
        self.P[:4, :4] = (np.eye(4) - K) @ self.P[:4, :4]

        # Update track management
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0

    def get_state(self) -> Dict:
        """
        Get current track state.

        Returns:
            state_dict: Dictionary with track information
        """
        return {
            'track_id': self.track_id,
            'bbox': self.state[:4].tolist(),
            'velocity': self.state[4:].tolist(),
            'age': self.age,
            'hits': self.hits,
            'hit_streak': self.hit_streak,
            'time_since_update': self.time_since_update
        }


# ============================================================================
# PART 9: IOU CALCULATION
# ============================================================================

def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    Box format: [x_center, y_center, width, height]

    Args:
        box1: First bounding box [x, y, w, h]
        box2: Second bounding box [x, y, w, h]

    Returns:
        iou: Intersection over Union score
    """
    # Convert center format to corner format
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2

    # Intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # Union area
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area

    # IoU
    iou = inter_area / union_area if union_area > 0 else 0

    return iou


# ============================================================================
# PART 10: MULTI-OBJECT TRACKER
# ============================================================================

class EViTTracker:
    """
    Multi-Object Tracker using Kalman Filter + IoU Association.
    Model-agnostic: works with any detection/segmentation output.

    Args:
        iou_threshold: Minimum IoU for matching detection to track
        max_age: Maximum frames to keep track without detection
        min_hits: Minimum hits before track is confirmed
    """

    def __init__(
            self,
            iou_threshold: float = 0.3,
            max_age: int = 30,
            min_hits: int = 3
    ):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits

        self.tracks: Dict[int, KalmanTracker] = {}
        self.next_id = 0
        self.frame_count = 0

    def update(self, detections: List[List[float]]) -> Dict[int, Dict]:
        """
        Update tracker with new detections.

        Args:
            detections: List of bounding boxes [[x, y, w, h], ...]

        Returns:
            active_tracks: Dictionary of active tracks {id: state_dict}
        """
        self.frame_count += 1

        # Predict all existing tracks
        predictions = {}
        for track_id, tracker in list(self.tracks.items()):
            pred_bbox = tracker.predict()
            predictions[track_id] = pred_bbox

        # Match detections to tracks using IoU
        matched_tracks = set()
        matched_detections = set()

        if len(detections) > 0 and len(predictions) > 0:
            # Compute IoU matrix
            iou_matrix = np.zeros((len(detections), len(predictions)))
            track_ids = list(predictions.keys())

            for i, detection in enumerate(detections):
                for j, track_id in enumerate(track_ids):
                    iou = compute_iou(
                        np.array(detection),
                        predictions[track_id]
                    )
                    iou_matrix[i, j] = iou

            # Greedy matching (can be replaced with Hungarian algorithm)
            while True:
                # Find maximum IoU
                max_iou = iou_matrix.max()
                if max_iou < self.iou_threshold:
                    break

                # Get indices of maximum
                i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                track_id = track_ids[j]

                # Update matched track
                self.tracks[track_id].update(detections[i])
                matched_tracks.add(track_id)
                matched_detections.add(i)

                # Zero out row and column to prevent re-matching
                iou_matrix[i, :] = 0
                iou_matrix[:, j] = 0

        # Create new tracks for unmatched detections
        for i, detection in enumerate(detections):
            if i not in matched_detections:
                self.tracks[self.next_id] = KalmanTracker(detection, self.next_id)
                self.next_id += 1

                # Remove dead tracks
            tracks_to_remove = []
            for track_id, tracker in self.tracks.items():
                if tracker.time_since_update > self.max_age:
                    tracks_to_remove.append(track_id)

            for track_id in tracks_to_remove:
                del self.tracks[track_id]

            # Return only confirmed tracks
            active_tracks = {}
            for track_id, tracker in self.tracks.items():
                if tracker.hits >= self.min_hits or tracker.hit_streak >= self.min_hits:
                    active_tracks[track_id] = tracker.get_state()

            return active_tracks

# ============================================================================
# PART 11: COMPLETE SYSTEM - TRAINING & INFERENCE
# ============================================================================

class EViTSystem:
    """
    Complete EViT System for training and inference.
    Combines segmentation model with tracking.
    """

    def __init__(
            self,
            model: EViT,
            tracker: EViTTracker,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.tracker = tracker
        self.device = device

    def extract_objects_from_mask(
            self,
            mask: torch.Tensor,
            conf_threshold: float = 0.5
    ) -> List[List[float]]:
        """
        Extract bounding boxes from segmentation mask.

        Args:
            mask: Segmentation mask (H, W)
            conf_threshold: Confidence threshold

        Returns:
            bboxes: List of [x, y, w, h] bounding boxes
        """
        mask_np = mask.cpu().numpy().astype(np.uint8)

        # Find contours
        from scipy import ndimage
        labeled_mask, num_objects = ndimage.label(mask_np > 0)

        bboxes = []
        for obj_id in range(1, num_objects + 1):
            obj_mask = (labeled_mask == obj_id)

            # Get bounding box
            rows = np.any(obj_mask, axis=1)
            cols = np.any(obj_mask, axis=0)

            if not rows.any() or not cols.any():
                continue

            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]

            # Convert to center format
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min

            bboxes.append([x_center, y_center, width, height])

        return bboxes

    def process_frame(
            self,
            frame: torch.Tensor,
            return_mask: bool = True
    ) -> Dict:
        """
        Process single frame: segment + track.

        Args:
            frame: Input frame (B, 3, H, W)
            return_mask: Whether to return segmentation mask

        Returns:
            results: Dictionary with segmentation and tracking results
        """
        self.model.eval()

        with torch.no_grad():
            # Segmentation
            logits = self.model(frame)
            mask = torch.argmax(logits, dim=1)[0]  # (H, W)

            # Extract objects
            detections = self.extract_objects_from_mask(mask)

            # Update tracker
            tracks = self.tracker.update(detections)

        results = {
            'tracks': tracks,
            'detections': detections,
            'num_objects': len(detections)
        }

        if return_mask:
            results['mask'] = mask

        return results

# ============================================================================
# PART 12: EXAMPLE USAGE & BENCHMARKING
# ============================================================================

def benchmark_model(
        model: EViT,
        input_size: Tuple[int, int] = (512, 512),
        device: str = 'cuda'
) -> Dict:
    """
    Benchmark model performance.
    """
    model.eval()
    model = model.to(device)

    # Create dummy input
    x = torch.randn(1, 3, *input_size).to(device)

    # Warmup
    for _ in range(10):
        _ = model(x)

    # Benchmark
    times = []
    for _ in range(100):
        start = time.time()
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize() if device == 'cuda' else None
        times.append(time.time() - start)

    # Calculate stats
    times = np.array(times) * 1000  # Convert to ms

    return {
        'mean_latency_ms': np.mean(times),
        'std_latency_ms': np.std(times),
        'fps': 1000 / np.mean(times),
        'min_latency_ms': np.min(times),
        'max_latency_ms': np.max(times)
    }

def main():
    """
    Example usage of EViT system.
    """
    print("=" * 80)
    print("EViT: Efficient Vision and Tracking System")
    print("=" * 80)

    # Model configuration for edge devices
    model = EViT(
        num_classes=10,
        encoder_embed_dims=[32, 64, 128, 256],  # Lighter for edge
        encoder_num_heads=[1, 2, 4, 8],
        encoder_depths=[2, 3, 4, 2],  # Fewer blocks
        encoder_mlp_ratios=[4.0, 4.0, 4.0, 4.0],
        encoder_reduction_ratios=[8, 4, 2, 1],
        decoder_embed_dim=128,
        drop_rate=0.1
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size (MB): {total_params * 4 / 1024 / 1024:.2f}")

    # Benchmark
    print(f"\nBenchmarking on input size (512, 512)...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    stats = benchmark_model(model, input_size=(512, 512), device=device)

    print(f"\nPerformance Metrics:")
    print(f"  Mean latency: {stats['mean_latency_ms']:.2f} ± {stats['std_latency_ms']:.2f} ms")
    print(f"  FPS: {stats['fps']:.2f}")
    print(f"  Min latency: {stats['min_latency_ms']:.2f} ms")
    print(f"  Max latency: {stats['max_latency_ms']:.2f} ms")

    # Create tracker
    tracker = EViTTracker(
        iou_threshold=0.3,
        max_age=30,
        min_hits=3
    )

    # Create complete system
    system = EViTSystem(model, tracker, device=device)

    # Test single frame
    print(f"\nTesting single frame processing...")
    test_frame = torch.randn(1, 3, 512, 512).to(device)
    results = system.process_frame(test_frame)

    print(f"  Detected objects: {results['num_objects']}")
    print(f"  Active tracks: {len(results['tracks'])}")

    print("\n" + "=" * 80)
    print("EViT System Ready for Deployment!")
    print("=" * 80)

if __name__ == "__main__":
    main()
