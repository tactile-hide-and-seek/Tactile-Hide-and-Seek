#!/usr/bin/env python
"""Evaluate a trained HAS baseline checkpoint."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from has_dataset.collate import window_collate
from has_dataset.hf_loader import DEFAULT_DATASET_NAME, load_has_dataset
from has_dataset.labels import combined_to_dual_lookup, load_label_mapping
from has_dataset.window_dataset import HASWindowDataset
from models.dual_head_classifier import build_model
from scripts.train import make_synthetic_rows, move_to_device


def parse_args() -> argparse.Namespace:
    """Parse evaluation arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--window_length", type=int, default=100)
    parser.add_argument("--stride", type=int, default=40)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--confusion_matrix", type=Path, default=None)
    parser.add_argument("--synthetic_data", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run checkpoint evaluation."""

    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = checkpoint.get("model_config", {})
    mapping = checkpoint.get("label_mapping") or load_label_mapping()
    lookup = combined_to_dual_lookup(mapping)

    rows = make_synthetic_rows() if args.synthetic_data else load_has_dataset(args.dataset_name, split=args.split, max_samples=args.max_samples)
    dataset = HASWindowDataset(rows, window_length=args.window_length, stride=args.stride, label_lookup=lookup)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=window_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total = object_correct = weight_correct = joint_correct = 0
    matrix = torch.zeros(34, 34, dtype=torch.long)
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            object_logits, weight_logits = model(batch)
            object_pred = object_logits.argmax(dim=-1)
            weight_pred = weight_logits.argmax(dim=-1)
            object_target = batch["object_label"]
            weight_target = batch["weight_label"]
            total += int(object_target.numel())
            object_correct += int((object_pred == object_target).sum().item())
            weight_correct += int((weight_pred == weight_target).sum().item())
            joint_correct += int(((object_pred == object_target) & (weight_pred == weight_target)).sum().item())
            for target, pred in zip(object_target.cpu(), object_pred.cpu()):
                matrix[int(target), int(pred)] += 1

    print("HAS evaluation summary")
    print(f"split: {args.split}")
    print(f"windows: {len(dataset)}")
    print(f"object_accuracy: {object_correct / max(total, 1):.4f}")
    print(f"weight_accuracy: {weight_correct / max(total, 1):.4f}")
    print(f"joint_accuracy: {joint_correct / max(total, 1):.4f}")

    if args.confusion_matrix:
        args.confusion_matrix.parent.mkdir(parents=True, exist_ok=True)
        with args.confusion_matrix.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerows(matrix.tolist())


if __name__ == "__main__":
    main()
