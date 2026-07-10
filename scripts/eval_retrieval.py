#!/usr/bin/env python
"""Target retrieval evaluation.

Given a target object-weight category ``c* = (o*, w*)``, the baseline scores each
tactile window with the joint posterior ``sigma(c*, X) = P(object=o*) * P(weight=w*)``
read directly from the two softmax heads (no retraining needed).

For every positive object-weight category we form (target, window) pairs:
positives are windows carrying that label, negatives are all other windows.
Negatives are tagged into two groups for separate false-positive-rate reporting:

* ``same_object_diff_weight`` -- windows of the same object at a different weight,
* ``no_object`` -- negative (no-object) windows.

Reported metrics: AUROC, Average Precision, F1 at ``sigma >= 0.5`` (all pooled over
every (target, window) pair), MRR and the median rank of the first true positive
(per-target ranking of candidate windows), and FPR at ``sigma >= 0.5`` for the two
negative groups (and overall).

Three evaluation axes, freely combinable:

* default -- the standard stride-40 classification windows (T=100 frames).
* ``--hard_negatives`` -- restrict each target's candidate pool to its positives
  plus *same-object/different-weight* distractors (targets without such
  distractors are dropped). The full pool is dominated by easily rejected
  different-object windows and is near-saturated for the baseline; the hard pool
  isolates the weight-verification difficulty that matters for tactile search.
* ``--tau 25 50 100 200`` -- *early target retrieval*, combining this task with the
  time-budgeted early-recognition protocol: for each budget ``tau`` the candidate pool
  is one onset-prefix window (the first ``tau`` frames) per episode, fed at its
  true length so no padding and no future-frame access is possible.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
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
    FRAME_RATE_HZ,
    ONSET_STRIDE,
    WindowOutputs,
    collect_window_outputs,
    joint_posterior,
    load_checkpoint,
    resolve_device,
    set_seed,
)
from scripts.train import make_synthetic_rows

NEGATIVE_CLASS = 0
POSITIVE_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    """Parse retrieval evaluation arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--window_length", type=int, default=100)
    parser.add_argument("--stride", type=int, default=40)
    parser.add_argument(
        "--hard_negatives",
        action="store_true",
        help="Restrict each target's candidate pool to positives plus same-object/"
        "different-weight distractors (targets without such distractors are dropped).",
    )
    parser.add_argument(
        "--tau",
        type=int,
        nargs="+",
        default=None,
        help="Early-retrieval mode: onset-prefix lengths in frames (e.g. 25 50 100 200). "
        "Overrides --window_length/--stride with one onset window per episode and budget.",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--results_json", type=Path, default=None, help="Optional path to dump the metrics as JSON.")
    parser.add_argument("--curve_csv", type=Path, default=None, help="Early-retrieval mode: optional per-tau metrics CSV.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--synthetic_data", action="store_true")
    return parser.parse_args()


def compute_retrieval_metrics(outputs: WindowOutputs, joint_space, hard_negatives: bool = False) -> dict:
    """Compute all retrieval metrics over one pool of candidate windows.

    With ``hard_negatives`` each target is ranked only against its positives and
    its same-object/different-weight distractors. FPR is always computed on the
    tagged negative groups, which are independent of the pool restriction.
    """

    scores_by_class = joint_posterior(outputs, joint_space)  # [N, num_classes]
    class_target = outputs.class_target
    object_target = outputs.object_target

    # Positive targets are every joint category except the no-object class.
    targets = [c for c in range(joint_space.num_classes) if c != NEGATIVE_CLASS]

    pooled_true: list[np.ndarray] = []
    pooled_score: list[np.ndarray] = []
    first_ranks: list[int] = []
    # (score >= 0.5) counts for the two tagged negative groups and all negatives.
    fpr_counts = {
        "same_object_diff_weight": [0, 0],  # [false_positives, group_size]
        "no_object": [0, 0],
        "overall_negative": [0, 0],
    }

    for target in targets:
        object_id = int(joint_space.object_of_class[target])
        score = scores_by_class[:, target]
        y_true = class_target == target
        if not y_true.any():
            continue  # target absent from this pool -> undefined retrieval query
        same_object = (object_target == object_id) & (class_target != target)
        no_object = class_target == NEGATIVE_CLASS

        if hard_negatives:
            if not same_object.any():
                continue  # no hard distractors for this target
            pool = y_true | same_object
        else:
            pool = np.ones_like(y_true, dtype=bool)
        pool_score = score[pool]
        pool_true = y_true[pool]

        pooled_true.append(pool_true.astype(np.int64))
        pooled_score.append(pool_score)

        # Rank of the first true positive among the target's candidates.
        order = np.argsort(-pool_score, kind="stable")
        first_ranks.append(1 + int(np.flatnonzero(pool_true[order])[0]))

        predicted_positive = score >= POSITIVE_THRESHOLD
        for name, mask in (
            ("same_object_diff_weight", same_object),
            ("no_object", no_object),
            ("overall_negative", ~y_true),
        ):
            fpr_counts[name][0] += int(predicted_positive[mask].sum())
            fpr_counts[name][1] += int(mask.sum())

    y_true_all = np.concatenate(pooled_true)
    score_all = np.concatenate(pooled_score)
    predicted_all = (score_all >= POSITIVE_THRESHOLD).astype(np.int64)
    ranks = np.array(first_ranks, dtype=np.float64)

    return {
        "negatives": "same_object_diff_weight" if hard_negatives else "all",
        "num_targets": len(first_ranks),
        "num_windows": len(outputs),
        "num_pairs": int(y_true_all.shape[0]),
        "num_positive_pairs": int(y_true_all.sum()),
        "auroc": float(roc_auc_score(y_true_all, score_all)),
        "average_precision": float(average_precision_score(y_true_all, score_all)),
        "f1_at_0.5": float(f1_score(y_true_all, predicted_all, zero_division=0)),
        "mrr": float(np.mean(1.0 / ranks)),
        "median_first_rank": float(np.median(ranks)),
        "fpr_at_0.5": {
            name: (count[0] / count[1] if count[1] else float("nan")) for name, count in fpr_counts.items()
        },
    }


def print_metrics_block(metrics: dict) -> None:
    """Print one pool's metrics in the classic summary layout."""

    print(f"negatives: {metrics['negatives']}")
    print(f"targets: {metrics['num_targets']}  windows: {metrics['num_windows']}  pairs: {metrics['num_pairs']}")
    print(f"AUROC: {metrics['auroc']:.4f}")
    print(f"Average Precision: {metrics['average_precision']:.4f}")
    print(f"F1@0.5: {metrics['f1_at_0.5']:.4f}")
    print(f"MRR: {metrics['mrr']:.4f}")
    print(f"median first rank: {metrics['median_first_rank']:.1f}")
    print(f"FPR@0.5 (same-object/diff-weight): {metrics['fpr_at_0.5']['same_object_diff_weight']:.4f}")
    print(f"FPR@0.5 (no-object): {metrics['fpr_at_0.5']['no_object']:.4f}")
    print(f"FPR@0.5 (all negatives): {metrics['fpr_at_0.5']['overall_negative']:.4f}")


def main() -> None:
    """Run the target retrieval evaluation."""

    args = parse_args()
    set_seed(args.seed)
    device = resolve_device()

    model, mapping, joint_space = load_checkpoint(args.checkpoint, device)
    lookup = combined_to_dual_lookup(mapping)

    rows = make_synthetic_rows() if args.synthetic_data else load_has_dataset(args.dataset_name, split=args.split, max_samples=args.max_samples)

    if args.tau is None:
        # Standard full-window retrieval on the stride-40 classification windows.
        dataset = HASWindowDataset(rows, window_length=args.window_length, stride=args.stride, label_lookup=lookup)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=window_collate)
        outputs = collect_window_outputs(model, loader, device)
        metrics = {
            "split": args.split,
            "checkpoint": str(args.checkpoint),
            "mode": "full_window",
            **compute_retrieval_metrics(outputs, joint_space, hard_negatives=args.hard_negatives),
        }
        print("HAS target retrieval summary")
        print(f"split: {metrics['split']}")
        print_metrics_block(metrics)
    else:
        # Early target retrieval: one onset-prefix window per episode and budget.
        per_tau: list[dict] = []
        for tau in sorted(args.tau):
            try:
                dataset = HASWindowDataset(rows, window_length=tau, stride=ONSET_STRIDE, label_lookup=lookup)
            except ValueError:
                print(f"[skip] tau={tau}: no episode has at least {tau} frames.")
                continue
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=window_collate)
            outputs = collect_window_outputs(model, loader, device)
            per_tau.append(
                {
                    "tau_frames": tau,
                    "tau_seconds": tau / FRAME_RATE_HZ,
                    **compute_retrieval_metrics(outputs, joint_space, hard_negatives=args.hard_negatives),
                }
            )
        if not per_tau:
            raise SystemExit("No prefix length produced any windows; check --tau against episode lengths.")

        metrics = {
            "split": args.split,
            "checkpoint": str(args.checkpoint),
            "mode": "onset_prefix",
            "negatives": per_tau[0]["negatives"],
            "per_tau": per_tau,
        }
        print("HAS early target retrieval summary")
        print(f"split: {metrics['split']}  negatives: {metrics['negatives']}")
        print(
            f"{'tau(frames)':>12} {'tau(s)':>7} {'wins':>5} {'AUROC':>7} {'AP':>7} {'F1@.5':>7}"
            f" {'MRR':>7} {'MedR':>5} {'FPR-SO':>7} {'FPR-NO':>7}"
        )
        for m in per_tau:
            print(
                f"{m['tau_frames']:>12} {m['tau_seconds']:>7.2f} {m['num_windows']:>5}"
                f" {m['auroc']:>7.4f} {m['average_precision']:>7.4f} {m['f1_at_0.5']:>7.4f}"
                f" {m['mrr']:>7.4f} {m['median_first_rank']:>5.1f}"
                f" {m['fpr_at_0.5']['same_object_diff_weight']:>7.4f} {m['fpr_at_0.5']['no_object']:>7.4f}"
            )

        if args.curve_csv:
            args.curve_csv.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = [
                "tau_frames", "tau_seconds", "num_windows",
                "auroc", "average_precision", "f1_at_0.5",
                "mrr", "median_first_rank",
                "fpr_same_object_diff_weight", "fpr_no_object",
            ]
            with args.curve_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for m in per_tau:
                    writer.writerow(
                        {
                            "tau_frames": m["tau_frames"],
                            "tau_seconds": m["tau_seconds"],
                            "num_windows": m["num_windows"],
                            "auroc": m["auroc"],
                            "average_precision": m["average_precision"],
                            "f1_at_0.5": m["f1_at_0.5"],
                            "mrr": m["mrr"],
                            "median_first_rank": m["median_first_rank"],
                            "fpr_same_object_diff_weight": m["fpr_at_0.5"]["same_object_diff_weight"],
                            "fpr_no_object": m["fpr_at_0.5"]["no_object"],
                        }
                    )

    if args.results_json:
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        with args.results_json.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)


if __name__ == "__main__":
    main()
