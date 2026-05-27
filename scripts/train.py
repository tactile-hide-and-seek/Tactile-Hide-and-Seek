#!/usr/bin/env python
"""Train the public dual-head HAS baseline."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from has_dataset.collate import window_collate
from has_dataset.hf_loader import DEFAULT_DATASET_NAME, load_has_dataset
from has_dataset.labels import combined_to_dual_lookup, load_label_mapping
from has_dataset.window_dataset import HASWindowDataset
from models.dual_head_classifier import build_model


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline_m12345.yaml"))
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="validation")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--window_length", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--modalities", default=None, help="Examples: M12345, M123, M23")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--debug", action="store_true", help="Run one tiny train/validation pass.")
    parser.add_argument("--synthetic_data", action="store_true", help="Use generated data for smoke tests.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML config file if it exists."""

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def set_seed(seed: int) -> None:
    """Set common random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_synthetic_rows(num_episodes: int = 4, frames_per_episode: int = 120) -> list[dict[str, Any]]:
    """Create a tiny in-memory frame-wise dataset for smoke tests."""

    rows: list[dict[str, Any]] = []
    for episode in range(num_episodes):
        label = episode % 61
        for frame in range(frames_per_episode):
            rows.append(
                {
                    "timestamp": float(frame),
                    "episode_index": episode,
                    "obs_point_cloud_skin_contact": np.random.randn(88, 6).astype("float32"),
                    "mask_point_cloud_skin_contact": np.ones(88, dtype=bool),
                    "obs_pose_left": np.random.randn(7).astype("float32"),
                    "obs_skin_prox_left": np.random.randn(44).astype("float32"),
                    "obs_skin_force_left": np.random.randn(44).astype("float32"),
                    "obs_skin_dist_left": np.random.randn(44).astype("float32"),
                    "obs_dami_force_left": np.random.randn(6).astype("float32"),
                    "obs_dami_prox_left": np.random.randn(6).astype("float32"),
                    "obs_ft_left": np.random.randn(6).astype("float32"),
                    "classification_label": label,
                }
            )
    return rows


def build_dataset(
    split: str,
    dataset_name: str,
    lookup: dict[int, tuple[int, int]],
    window_length: int,
    stride: int,
    max_samples: int | None,
    synthetic: bool,
    max_windows: int | None,
) -> HASWindowDataset:
    """Load a split and wrap it as temporal windows."""

    if synthetic:
        rows = make_synthetic_rows(frames_per_episode=max(window_length + stride, window_length))
    else:
        rows = load_has_dataset(dataset_name, split=split, max_samples=max_samples)
    return HASWindowDataset(
        rows,
        window_length=window_length,
        stride=stride,
        label_lookup=lookup,
        max_windows=max_windows,
    )


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move tensor batch values to a device."""

    return {key: value.to(device) for key, value in batch.items()}


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_steps: int | None = None,
) -> dict[str, float]:
    """Run one train or evaluation epoch."""

    is_train = optimizer is not None
    model.train(is_train)
    totals = {"loss": 0.0, "object": 0.0, "weight": 0.0, "joint": 0.0, "count": 0.0}
    loss_fn = torch.nn.CrossEntropyLoss()
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for step, batch in enumerate(tqdm(loader, leave=False)):
            batch = move_to_device(batch, device)
            object_logits, weight_logits = model(batch)
            object_loss = loss_fn(object_logits, batch["object_label"])
            weight_loss = loss_fn(weight_logits, batch["weight_label"])
            loss = object_loss + 0.5 * weight_loss
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            object_pred = object_logits.argmax(dim=-1)
            weight_pred = weight_logits.argmax(dim=-1)
            count = float(batch["object_label"].numel())
            totals["loss"] += float(loss.item()) * count
            totals["object"] += float((object_pred == batch["object_label"]).sum().item())
            totals["weight"] += float((weight_pred == batch["weight_label"]).sum().item())
            totals["joint"] += float(((object_pred == batch["object_label"]) & (weight_pred == batch["weight_label"])).sum().item())
            totals["count"] += count
            if max_steps is not None and step + 1 >= max_steps:
                break
    count = max(totals.pop("count"), 1.0)
    return {
        "loss": totals["loss"] / count,
        "object_acc": totals["object"] / count,
        "weight_acc": totals["weight"] / count,
        "joint_acc": totals["joint"] / count,
    }


def main() -> None:
    """Train the model."""

    args = parse_args()
    config = load_config(args.config)
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})

    debug = args.debug
    dataset_name = args.dataset_name or dataset_cfg.get("name", DEFAULT_DATASET_NAME)
    batch_size = args.batch_size or dataset_cfg.get("batch_size", 32)
    lr = args.learning_rate or training_cfg.get("learning_rate", 1e-4)
    epochs = args.epochs or training_cfg.get("epochs", 20)
    window_length = args.window_length or dataset_cfg.get("window_length", 100)
    stride = args.stride or dataset_cfg.get("stride", 40)
    modalities = args.modalities or model_cfg.get("modalities", "M12345")
    num_workers = args.num_workers if args.num_workers is not None else dataset_cfg.get("num_workers", 4)
    output_dir = args.output_dir or Path(training_cfg.get("output_dir", "checkpoints"))
    seed = int(training_cfg.get("seed", 42))

    if debug:
        batch_size = min(batch_size, 2)
        epochs = 1
        num_workers = 0
        args.max_samples = args.max_samples or 240
        model_cfg["hidden_dim"] = min(int(model_cfg.get("hidden_dim", 128)), 32)

    set_seed(seed)
    mapping = load_label_mapping()
    lookup = combined_to_dual_lookup(mapping)
    synthetic = args.synthetic_data
    max_windows = 4 if debug else None

    train_data = build_dataset(args.train_split, dataset_name, lookup, window_length, stride, args.max_samples, synthetic, max_windows)
    val_data = build_dataset(args.val_split, dataset_name, lookup, window_length, stride, args.max_samples, synthetic, max_windows)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=window_collate)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=window_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        modalities=modalities,
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(training_cfg.get("weight_decay", 1e-3)))

    output_dir.mkdir(parents=True, exist_ok=True)
    best_joint = -1.0
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer, max_steps=1 if debug else None)
        val_metrics = run_epoch(model, val_loader, device, None, max_steps=1 if debug else None)
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"val_object_acc={val_metrics['object_acc']:.4f} "
            f"val_weight_acc={val_metrics['weight_acc']:.4f} "
            f"val_joint_acc={val_metrics['joint_acc']:.4f}"
        )
        if val_metrics["joint_acc"] >= best_joint:
            best_joint = val_metrics["joint_acc"]
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "modalities": modalities,
                    "hidden_dim": int(model_cfg.get("hidden_dim", 128)),
                    "num_heads": int(model_cfg.get("num_heads", 4)),
                    "num_layers": int(model_cfg.get("num_layers", 2)),
                    "dropout": float(model_cfg.get("dropout", 0.1)),
                },
                "label_mapping": mapping,
                "epoch": epoch,
                "validation": val_metrics,
            }
            torch.save(checkpoint, output_dir / "best_model.pt")
    with (output_dir / "last_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"validation": val_metrics}, handle, indent=2)


if __name__ == "__main__":
    main()
