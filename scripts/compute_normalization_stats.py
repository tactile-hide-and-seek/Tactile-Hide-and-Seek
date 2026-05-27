#!/usr/bin/env python
"""Compute simple mean/std statistics for selected HAS fields."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from has_dataset.hf_loader import DEFAULT_DATASET_NAME, load_has_dataset

FIELDS = [
    "obs_point_cloud_skin_contact",
    "obs_pose_left",
    "obs_skin_prox_left",
    "obs_skin_force_left",
    "obs_skin_dist_left",
    "obs_dami_force_left",
    "obs_dami_prox_left",
    "obs_ft_left",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=Path("data/normalization_stats.json"))
    args = parser.parse_args()

    dataset = load_has_dataset(args.dataset_name, split=args.split, max_samples=args.max_samples)
    stats = {}
    for field in FIELDS:
        values = np.asarray(dataset[field], dtype=np.float64)
        axes = tuple(range(values.ndim - 1))
        stats[field] = {
            "mean": values.mean(axis=axes).tolist(),
            "std": values.std(axis=axes).clip(min=1e-6).tolist(),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    print(f"Wrote normalization stats to {args.output}")


if __name__ == "__main__":
    main()
