"""Point cloud encoder for tactile contact clouds."""

from __future__ import annotations

import torch
from torch import nn

from models.temporal_encoder import TemporalEncoder


class PointCloudEncoder(nn.Module):
    """Encode ``[B, T, 88, 6]`` XYZ plus contact features into one token."""

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.temporal = TemporalEncoder(hidden_dim * 2, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, point_cloud: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode point clouds with optional valid-point mask."""

        if point_cloud.ndim != 4 or point_cloud.shape[-2:] != (88, 6):
            raise ValueError(f"Expected point_cloud shape [B, T, 88, 6], got {tuple(point_cloud.shape)}.")
        features = self.point_mlp(point_cloud)
        if mask is None:
            max_pool = features.amax(dim=2)
            mean_pool = features.mean(dim=2)
        else:
            valid = mask.bool().unsqueeze(-1)
            masked_max = features.masked_fill(~valid, torch.finfo(features.dtype).min)
            max_pool = masked_max.amax(dim=2)
            valid_count = valid.sum(dim=2).clamp_min(1)
            mean_pool = (features * valid.float()).sum(dim=2) / valid_count
        return self.temporal(torch.cat([max_pool, mean_pool], dim=-1))
