"""Dual-head multimodal baseline classifier."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from models.fusion_transformer import FusionTransformer
from models.pointcloud_encoder import PointCloudEncoder
from models.temporal_encoder import TemporalEncoder


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the baseline classifier."""

    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    num_object_classes: int = 34
    num_weight_classes: int = 4
    modalities: str = "M12345"


class DualHeadHASClassifier(nn.Module):
    """Readable dual-head baseline for object and weight prediction."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        hidden = self.config.hidden_dim
        active = set(self.config.modalities.upper())

        self.use_m1 = "1" in active
        self.use_m2 = "2" in active
        self.use_m3 = "3" in active
        self.use_m4 = "4" in active
        self.use_m5 = "5" in active

        if self.use_m1:
            self.point_cloud = PointCloudEncoder(hidden_dim=hidden, dropout=self.config.dropout)
        if self.use_m2:
            self.skin = TemporalEncoder(44 * 3, hidden_dim=hidden, num_heads=self.config.num_heads, dropout=self.config.dropout)
        if self.use_m3:
            self.dami = TemporalEncoder(12, hidden_dim=hidden, num_heads=self.config.num_heads, dropout=self.config.dropout)
        if self.use_m4:
            self.pose = TemporalEncoder(7, hidden_dim=hidden, num_heads=self.config.num_heads, dropout=self.config.dropout)
        if self.use_m5:
            self.ft = TemporalEncoder(6, hidden_dim=hidden, num_heads=self.config.num_heads, dropout=self.config.dropout)

        self.fusion = FusionTransformer(
            hidden_dim=hidden,
            max_modalities=5,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        )
        self.shared = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )
        self.object_head = nn.Linear(hidden, self.config.num_object_classes)
        self.weight_head = nn.Linear(hidden, self.config.num_weight_classes)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(object_logits, weight_logits)`` for a window batch."""

        tokens: list[torch.Tensor] = []
        if self.use_m1:
            tokens.append(
                self.point_cloud(
                    batch["obs_point_cloud_skin_contact"],
                    batch.get("mask_point_cloud_skin_contact"),
                )
            )
        if self.use_m2:
            skin = torch.cat(
                [
                    batch["obs_skin_prox_left"],
                    batch["obs_skin_force_left"],
                    batch["obs_skin_dist_left"],
                ],
                dim=-1,
            )
            tokens.append(self.skin(skin))
        if self.use_m3:
            dami = torch.cat([batch["obs_dami_force_left"], batch["obs_dami_prox_left"]], dim=-1)
            tokens.append(self.dami(dami))
        if self.use_m4:
            tokens.append(self.pose(batch["obs_pose_left"]))
        if self.use_m5:
            tokens.append(self.ft(batch["obs_ft_left"]))

        features = self.shared(self.fusion(tokens))
        return self.object_head(features), self.weight_head(features)


def build_model(
    modalities: str = "M12345",
    hidden_dim: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    dropout: float = 0.1,
) -> DualHeadHASClassifier:
    """Build the default baseline model."""

    return DualHeadHASClassifier(
        ModelConfig(
            modalities=modalities,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
    )
