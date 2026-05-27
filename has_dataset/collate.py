"""Collate functions for HAS window batches."""

from __future__ import annotations

import torch


def window_collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack fixed-shape window samples and report clear shape mismatches."""

    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    keys = batch[0].keys()
    collated: dict[str, torch.Tensor] = {}
    for key in keys:
        tensors = [sample[key] for sample in batch]
        first_shape = tensors[0].shape
        for item_index, tensor in enumerate(tensors[1:], start=1):
            if tensor.shape != first_shape:
                raise ValueError(
                    f"Inconsistent shapes for key '{key}': sample 0 has {first_shape}, "
                    f"sample {item_index} has {tensor.shape}."
                )
        collated[key] = torch.stack(tensors, dim=0)
    return collated
