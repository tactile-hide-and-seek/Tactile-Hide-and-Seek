#!/usr/bin/env python
"""Evaluate a trained HAS baseline checkpoint (object-weight classification).

Reports object / weight / joint accuracy and
macro-F1 over the 61 joint object-weight categories (sklearn ``f1_score`` with
``average="macro"``). The joint prediction is derived from the two per-head
argmax predictions mapped back to a joint label.

``--per_mode`` additionally breaks the accuracies down by interaction mode. The
mode is read from the skin-point validity mask: single-hand push episodes have
the inactive hand's 44 points masked out (44 valid points), bimanual grasp
episodes use both hands (88 valid points).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from has_dataset.collate import window_collate
from has_dataset.hf_loader import DEFAULT_DATASET_NAME, load_has_dataset
from has_dataset.labels import combined_to_dual_lookup
from has_dataset.window_dataset import HASWindowDataset
from scripts.eval_utils import (
    DEFAULT_SEED,
    collect_window_outputs,
    load_checkpoint,
    resolve_device,
    set_seed,
)
from scripts.train import make_synthetic_rows


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
    parser.add_argument(
        "--per_mode",
        action="store_true",
        help="Also report accuracies per interaction mode (push vs. bimanual grasp), "
        "inferred from the skin-point validity mask (44 vs. 88 valid points).",
    )
    parser.add_argument("--confusion_matrix", type=Path, default=None)
    parser.add_argument("--results_json", type=Path, default=None, help="Optional path to dump the metrics as JSON.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--synthetic_data", action="store_true")
    return parser.parse_args()


def episode_modes(rows) -> dict[int, str]:
    """Infer the interaction mode of every episode from the skin-point mask.

    Single-hand push data has the inactive hand's 44 points masked out, bimanual
    grasp data uses all 88 points. One frame per episode suffices because the
    hand-activity mask is constant within an episode.
    """

    episode_column = rows["episode_index"]
    first_row: dict[int, int] = {}
    for row_index, episode in enumerate(episode_column):
        first_row.setdefault(int(episode), row_index)
    modes: dict[int, str] = {}
    for episode, row_index in first_row.items():
        valid_points = int(np.sum(rows[row_index]["mask_point_cloud_skin_contact"]))
        modes[episode] = "grasp" if valid_points > 44 else "push"
    return modes


def main() -> None:
    """Run checkpoint evaluation."""

    args = parse_args()
    set_seed(args.seed)
    device = resolve_device()

    model, mapping, joint_space = load_checkpoint(args.checkpoint, device)
    lookup = combined_to_dual_lookup(mapping)

    rows = make_synthetic_rows() if args.synthetic_data else load_has_dataset(args.dataset_name, split=args.split, max_samples=args.max_samples)
    dataset = HASWindowDataset(rows, window_length=args.window_length, stride=args.stride, label_lookup=lookup)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=window_collate)

    outputs = collect_window_outputs(model, loader, device)

    total = len(outputs)
    object_correct = int((outputs.object_pred == outputs.object_target).sum())
    weight_correct = int((outputs.weight_pred == outputs.weight_target).sum())
    joint_correct = int(((outputs.object_pred == outputs.object_target) & (outputs.weight_pred == outputs.weight_target)).sum())

    # Joint prediction over the 61 categories, from the two per-head argmaxes.
    joint_pred = joint_space.joint_pred_from_heads(outputs.object_pred, outputs.weight_pred)
    joint_macro_f1 = float(
        f1_score(
            outputs.class_target,
            joint_pred,
            labels=list(range(joint_space.num_classes)),
            average="macro",
            zero_division=0,
        )
    )

    metrics = {
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "windows": total,
        "object_accuracy": object_correct / max(total, 1),
        "weight_accuracy": weight_correct / max(total, 1),
        "joint_accuracy": joint_correct / max(total, 1),
        "joint_macro_f1": joint_macro_f1,
    }

    if args.per_mode:
        modes = episode_modes(rows)
        window_mode = np.array([modes[int(e)] for e in outputs.episode_index])
        per_mode: dict[str, dict] = {}
        for mode in ("push", "grasp"):
            selected = window_mode == mode
            count = int(selected.sum())
            per_mode[mode] = {
                "windows": count,
                "episodes": sum(1 for m in modes.values() if m == mode),
                "object_accuracy": float((outputs.object_pred == outputs.object_target)[selected].mean()) if count else float("nan"),
                "weight_accuracy": float((outputs.weight_pred == outputs.weight_target)[selected].mean()) if count else float("nan"),
                "joint_accuracy": float(
                    ((outputs.object_pred == outputs.object_target) & (outputs.weight_pred == outputs.weight_target))[selected].mean()
                )
                if count
                else float("nan"),
            }
        metrics["per_mode"] = per_mode

    print("HAS evaluation summary")
    print(f"split: {metrics['split']}")
    print(f"windows: {metrics['windows']}")
    print(f"object_accuracy: {metrics['object_accuracy']:.4f}")
    print(f"weight_accuracy: {metrics['weight_accuracy']:.4f}")
    print(f"joint_accuracy: {metrics['joint_accuracy']:.4f}")
    print(f"joint_macro_f1: {metrics['joint_macro_f1']:.4f}")
    if args.per_mode:
        for mode, stats in metrics["per_mode"].items():
            print(
                f"[{mode}] windows: {stats['windows']}  episodes: {stats['episodes']}"
                f"  object: {stats['object_accuracy']:.4f}  weight: {stats['weight_accuracy']:.4f}"
                f"  joint: {stats['joint_accuracy']:.4f}"
            )

    if args.confusion_matrix:
        num_objects = int(outputs.object_prob.shape[1])
        matrix = np.zeros((num_objects, num_objects), dtype=np.int64)
        np.add.at(matrix, (outputs.object_target, outputs.object_pred), 1)
        args.confusion_matrix.parent.mkdir(parents=True, exist_ok=True)
        with args.confusion_matrix.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerows(matrix.tolist())

    if args.results_json:
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        with args.results_json.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)


if __name__ == "__main__":
    main()
