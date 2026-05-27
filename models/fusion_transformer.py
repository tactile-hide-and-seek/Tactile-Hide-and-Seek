"""Fusion transformer for modality tokens."""

from __future__ import annotations

import torch
from torch import nn


class FusionTransformer(nn.Module):
    """Fuse a variable set of modality tokens."""

    def __init__(
        self,
        hidden_dim: int = 128,
        max_modalities: int = 5,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.modality_embedding = nn.Parameter(torch.randn(max_modalities, hidden_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: list[torch.Tensor]) -> torch.Tensor:
        """Fuse modality tokens, each shaped ``[B, hidden_dim]``."""

        if not tokens:
            raise ValueError("At least one modality token is required for fusion.")
        x = torch.stack(tokens, dim=1)
        if x.shape[1] > self.modality_embedding.shape[0]:
            raise ValueError("Received more modality tokens than configured.")
        x = x + self.modality_embedding[: x.shape[1]].unsqueeze(0)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        fused = self.encoder(torch.cat([cls, x], dim=1))
        return self.norm(fused[:, 0])
