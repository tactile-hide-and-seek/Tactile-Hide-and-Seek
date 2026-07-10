#!/usr/bin/env python
"""Early tactile recognition evaluation.

Feeds only the first ``tau`` frames from interaction onset (the start of each
episode) and measures how quickly the baseline recognizes the object-weight
category. Prefixes are fed at their true length (variable-length feed): within a
single ``tau`` every window is exactly ``tau`` frames long, so there is no padding
and therefore no possibility of future-frame leakage. The nominal training window
is T=100 frames (1.0 s at 100 Hz); ``tau`` may be shorter or longer.

Reported per time budget ``tau``: object / weight / joint accuracy. Aggregated
across budgets: the joint-accuracy-vs-time curve and its normalized AUC-tau,
time-to-correct-recognition (earliest ``tau`` from which the prediction stays
correct for all longer budgets) and time-to-confidence (earliest ``tau`` whose max
joint posterior reaches ``delta``), with the accuracy of that confident prediction.
The accuracy-vs-time curve is also exported as CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
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
    collect_window_outputs,
    joint_posterior,
    load_checkpoint,
    resolve_device,
    set_seed,
)
from scripts.train import make_synthetic_rows


def parse_args() -> argparse.Namespace:
    """Parse early-recognition evaluation arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tau", type=int, nargs="+", default=[25, 50, 100, 200], help="Prefix lengths in frames.")
    parser.add_argument("--confidence_delta", type=float, default=0.9)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--results_json", type=Path, default=None, help="Optional path to dump the metrics as JSON.")
    parser.add_argument("--curve_csv", type=Path, default=None, help="Optional path for the accuracy-vs-time CSV.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--synthetic_data", action="store_true")
    return parser.parse_args()


def evaluate_prefix(model, rows, tau, lookup, joint_space, batch_size, device) -> dict[str, np.ndarray] | None:
    """Evaluate one prefix length; return per-episode arrays or ``None`` if empty."""

    try:
        dataset = HASWindowDataset(rows, window_length=tau, stride=ONSET_STRIDE, label_lookup=lookup)
    except ValueError:
        return None  # no episode is at least ``tau`` frames long
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=window_collate)
    outputs = collect_window_outputs(model, loader, device)

    posterior = joint_posterior(outputs, joint_space)
    confidence = posterior / posterior.sum(axis=1, keepdims=True).clip(min=1e-12)
    joint_post_pred = posterior.argmax(axis=1)
    return {
        "episode_index": outputs.episode_index,
        "object_correct": (outputs.object_pred == outputs.object_target).astype(bool),
        "weight_correct": (outputs.weight_pred == outputs.weight_target).astype(bool),
        "joint_correct": ((outputs.object_pred == outputs.object_target) & (outputs.weight_pred == outputs.weight_target)).astype(bool),
        "confidence": confidence.max(axis=1),
        "joint_post_correct": (joint_post_pred == outputs.class_target).astype(bool),
    }


def main() -> None:
    """Run the early tactile recognition evaluation."""

    args = parse_args()
    set_seed(args.seed)
    device = resolve_device()

    model, mapping, joint_space = load_checkpoint(args.checkpoint, device)
    lookup = combined_to_dual_lookup(mapping)

    rows = make_synthetic_rows() if args.synthetic_data else load_has_dataset(args.dataset_name, split=args.split, max_samples=args.max_samples)

    per_tau: dict[int, dict[str, np.ndarray]] = {}
    for tau in sorted(args.tau):
        result = evaluate_prefix(model, rows, tau, lookup, joint_space, args.batch_size, device)
        if result is None:
            print(f"[skip] tau={tau}: no episode has at least {tau} frames.")
            continue
        per_tau[tau] = result

    if not per_tau:
        raise SystemExit("No prefix length produced any windows; check --tau against episode lengths.")

    taus = sorted(per_tau)
    seconds = [t / FRAME_RATE_HZ for t in taus]

    curve = []
    for tau in taus:
        r = per_tau[tau]
        curve.append(
            {
                "tau_frames": tau,
                "tau_seconds": tau / FRAME_RATE_HZ,
                "num_episodes": int(r["episode_index"].shape[0]),
                "object_acc": float(r["object_correct"].mean()),
                "weight_acc": float(r["weight_correct"].mean()),
                "joint_acc": float(r["joint_correct"].mean()),
            }
        )

    # Normalized AUC over the joint-accuracy-vs-time curve (trapezoidal rule,
    # written out to stay compatible across NumPy versions).
    joint_curve = np.array([c["joint_acc"] for c in curve])
    if len(taus) > 1:
        span = seconds[-1] - seconds[0]
        area = float(np.sum((joint_curve[:-1] + joint_curve[1:]) / 2.0 * np.diff(seconds)))
        auc_tau = area / span if span > 0 else float(joint_curve.mean())
    else:
        auc_tau = float(joint_curve[0])

    # Per-episode temporal metrics use episodes present at every budget.
    common = set.intersection(*[set(per_tau[t]["episode_index"].tolist()) for t in taus])
    common_episodes = sorted(common)
    index_maps = {t: {int(e): i for i, e in enumerate(per_tau[t]["episode_index"])} for t in taus}

    joint_correct_mat = np.array(
        [[per_tau[t]["joint_correct"][index_maps[t][e]] for t in taus] for e in common_episodes], dtype=bool
    )
    confidence_mat = np.array(
        [[per_tau[t]["confidence"][index_maps[t][e]] for t in taus] for e in common_episodes], dtype=float
    )
    post_correct_mat = np.array(
        [[per_tau[t]["joint_post_correct"][index_maps[t][e]] for t in taus] for e in common_episodes], dtype=bool
    )
    seconds_arr = np.array(seconds)

    # Time-to-correct: earliest budget after which the prediction stays correct.
    time_to_correct = np.full(len(common_episodes), np.nan)
    for e in range(len(common_episodes)):
        for i in range(len(taus)):
            if joint_correct_mat[e, i:].all():
                time_to_correct[e] = seconds_arr[i]
                break
    stable = ~np.isnan(time_to_correct)

    # Time-to-confidence: earliest budget whose max posterior reaches delta.
    time_to_confidence = np.full(len(common_episodes), np.nan)
    confident_correct = np.zeros(len(common_episodes), dtype=bool)
    reached = confidence_mat >= args.confidence_delta
    for e in range(len(common_episodes)):
        hits = np.where(reached[e])[0]
        if hits.size:
            first = hits[0]
            time_to_confidence[e] = seconds_arr[first]
            confident_correct[e] = post_correct_mat[e, first]
    confident = ~np.isnan(time_to_confidence)

    metrics = {
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "confidence_delta": args.confidence_delta,
        "curve": curve,
        "auc_tau_normalized": auc_tau,
        "num_common_episodes": len(common_episodes),
        "time_to_correct": {
            "fraction_ever_stable": float(stable.mean()) if stable.size else float("nan"),
            "mean_seconds_stable": float(np.nanmean(time_to_correct)) if stable.any() else float("nan"),
        },
        "time_to_confidence": {
            "fraction_confident": float(confident.mean()) if confident.size else float("nan"),
            "mean_seconds_confident": float(np.nanmean(time_to_confidence)) if confident.any() else float("nan"),
            "accuracy_at_confidence": float(confident_correct[confident].mean()) if confident.any() else float("nan"),
        },
    }

    print("HAS early tactile recognition summary")
    print(f"split: {metrics['split']}")
    print(f"{'tau(frames)':>12} {'tau(s)':>7} {'n':>5} {'object':>8} {'weight':>8} {'joint':>8}")
    for c in curve:
        print(f"{c['tau_frames']:>12} {c['tau_seconds']:>7.2f} {c['num_episodes']:>5} {c['object_acc']:>8.4f} {c['weight_acc']:>8.4f} {c['joint_acc']:>8.4f}")
    print(f"normalized AUC-tau (joint): {auc_tau:.4f}")
    print(f"common episodes: {metrics['num_common_episodes']}")
    print(f"time-to-correct: stable={metrics['time_to_correct']['fraction_ever_stable']:.4f} mean={metrics['time_to_correct']['mean_seconds_stable']:.4f}s")
    print(
        f"time-to-confidence (delta={args.confidence_delta}): "
        f"reached={metrics['time_to_confidence']['fraction_confident']:.4f} "
        f"mean={metrics['time_to_confidence']['mean_seconds_confident']:.4f}s "
        f"acc={metrics['time_to_confidence']['accuracy_at_confidence']:.4f}"
    )

    if args.curve_csv:
        args.curve_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.curve_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["tau_frames", "tau_seconds", "num_episodes", "object_acc", "weight_acc", "joint_acc"])
            writer.writeheader()
            writer.writerows(curve)

    if args.results_json:
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        with args.results_json.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)


if __name__ == "__main__":
    main()
