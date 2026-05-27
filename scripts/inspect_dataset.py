#!/usr/bin/env python
"""Inspect HAS dataset split sizes and window counts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from has_dataset.hf_loader import DEFAULT_DATASET_NAME, load_has_dataset
from has_dataset.labels import combined_to_dual_lookup, load_label_mapping
from has_dataset.window_dataset import HASWindowDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", default="train")
    parser.add_argument("--window_length", type=int, default=100)
    parser.add_argument("--stride", type=int, default=40)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    dataset = load_has_dataset(args.dataset_name, split=args.split, max_samples=args.max_samples)
    windows = HASWindowDataset(
        dataset,
        window_length=args.window_length,
        stride=args.stride,
        label_lookup=combined_to_dual_lookup(load_label_mapping()),
    )
    print(f"split={args.split}")
    print(f"frames={len(dataset)}")
    print(f"episodes={len(windows.episodes)}")
    print(f"windows={len(windows)}")


if __name__ == "__main__":
    main()
