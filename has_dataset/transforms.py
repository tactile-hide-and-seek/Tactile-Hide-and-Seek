"""Small tensor transforms used by the HAS baselines."""

from __future__ import annotations

from typing import Any, Mapping

import torch


def normalize_with_stats(
    batch: dict[str, torch.Tensor],
    stats: Mapping[str, Mapping[str, Any]],
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Apply mean/std normalization to keys present in ``stats``.

    Expected stat format per key is ``{"mean": [...], "std": [...]}``. Unknown
    keys are ignored so a partial stats file can be used.
    """

    normalized = dict(batch)
    for key, values in stats.items():
        if key not in normalized or "mean" not in values or "std" not in values:
            continue
        tensor = normalized[key]
        mean = torch.as_tensor(values["mean"], dtype=tensor.dtype, device=tensor.device)
        std = torch.as_tensor(values["std"], dtype=tensor.dtype, device=tensor.device).clamp_min(eps)
        normalized[key] = (tensor - mean) / std
    return normalized
