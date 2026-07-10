#!/usr/bin/env python
"""Render the early-recognition figure: joint accuracy and target-retrieval AP vs tau.

Reads the CSV outputs of ``scripts/eval_early.py`` (--curve_csv) and
``scripts/eval_retrieval.py --hard_negatives --tau`` (--curve_csv) from
``results/`` and writes a single-column vector figure
(``fig_early.pdf``) plus a PNG preview. Retrieval uses the official
hard-negative retrieval protocol.

Encoding: color = metric (joint accuracy vs. retrieval AP, colorblind-safe
Okabe-Ito pair), line style = configuration (M12345 solid, M123 dashed), so
neither identity relies on color alone.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

ACC_COLOR = "#0072B2"  # Okabe-Ito blue: joint accuracy
AP_COLOR = "#D55E00"  # Okabe-Ito vermillion: retrieval AP (hard negatives)
INK_MUTED = "#666666"
GRID = "#DDDDDD"


def read_csv_column(path: Path, column: str) -> tuple[list[float], list[float]]:
    """Return (tau_seconds, column values) from one results CSV."""

    taus: list[float] = []
    values: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            taus.append(float(row["tau_seconds"]))
            values.append(float(row[column]))
    return taus, values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "fig_early.pdf")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    series = {
        ("M12345", "joint_acc"): read_csv_column(args.results_dir / "early_m12345_test.csv", "joint_acc"),
        ("M123", "joint_acc"): read_csv_column(args.results_dir / "early_m123_test.csv", "joint_acc"),
        ("M12345", "ap"): read_csv_column(args.results_dir / "retrieval_early_hard_m12345_test.csv", "average_precision"),
        ("M123", "ap"): read_csv_column(args.results_dir / "retrieval_early_hard_m123_test.csv", "average_precision"),
    }

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,  # embed TrueType so ICRA/IEEE PDF checks pass
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(3.45, 2.3), dpi=200)

    style = {"M12345": "-", "M123": "--"}
    color = {"joint_acc": ACC_COLOR, "ap": AP_COLOR}
    label = {"joint_acc": "Joint acc", "ap": "Retr. AP (hard)"}
    for (config, metric), (taus, values) in series.items():
        ax.plot(
            taus,
            values,
            style[config],
            color=color[metric],
            marker="o",
            markersize=3.5,
            linewidth=1.4,
            label=f"{label[metric]} ({config})",
        )

    # Nominal training window length T = 100 frames (1 s).
    ax.axvline(1.0, color=INK_MUTED, linewidth=0.8, linestyle=":", zorder=0)
    ax.annotate("training window T", xy=(1.0, 0.5), xytext=(1.05, 0.49), color=INK_MUTED, fontsize=6.5)

    ax.set_xscale("log", base=2)
    ax.set_xticks([0.25, 0.5, 1.0, 2.0])
    ax.set_xticklabels(["0.25", "0.5", "1.0", "2.0"])
    ax.minorticks_off()
    ax.set_xlabel(r"Observation budget $\tau$ (s)")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.02)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 0.0),
        frameon=False,
        ncol=2,
        handlelength=2.8,
        columnspacing=1.0,
        labelspacing=0.3,
        fontsize=6.5,
    )

    fig.tight_layout(pad=0.4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    fig.savefig(args.output.with_suffix(".png"), dpi=300)
    print(f"wrote {args.output} and {args.output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
