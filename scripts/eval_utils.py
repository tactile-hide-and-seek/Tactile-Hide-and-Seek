#!/usr/bin/env python
"""Shared helpers for the HAS evaluation scripts.

These utilities are used by ``eval.py`` (object-weight classification),
``eval_retrieval.py`` (target retrieval), and ``eval_early.py`` (early tactile
recognition). They keep a single, readable definition of how a checkpoint is
loaded, how the dual softmax heads are turned into a joint object-weight
posterior, and how per-window model outputs are collected.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from has_dataset.labels import combined_to_dual_lookup, load_label_mapping
from models.dual_head_classifier import build_model
from scripts.train import move_to_device

DEFAULT_SEED = 42
FRAME_RATE_HZ = 100.0
# A stride larger than any episode keeps exactly the first (onset) window per
# episode; used by the early-recognition and early-retrieval prefix modes.
ONSET_STRIDE = 10_000_000


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed Python, NumPy, and torch RNGs for reproducible evaluation."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    """Return CUDA when available, otherwise CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class JointLabelSpace:
    """Mapping between the 61 joint labels and the two per-head labels.

    ``object_of_class[c]`` and ``weight_of_class[c]`` give the object and weight
    label indices for joint class ``c``. ``dual_to_joint[(o, w)]`` is the reverse
    lookup used to turn per-head argmax predictions back into a joint label; pairs
    that are not a valid joint category are simply absent.
    """

    class_names: list[str]
    object_of_class: np.ndarray
    weight_of_class: np.ndarray
    dual_to_joint: dict[tuple[int, int], int]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def joint_pred_from_heads(self, object_pred: np.ndarray, weight_pred: np.ndarray) -> np.ndarray:
        """Map per-head argmax predictions to joint labels (``-1`` if invalid).

        A predicted ``(object, weight)`` pair that is not one of the valid joint
        categories yields ``-1``, which counts as a miss for every true class.
        """

        return np.array(
            [self.dual_to_joint.get((int(o), int(w)), -1) for o, w in zip(object_pred, weight_pred)],
            dtype=np.int64,
        )


def build_joint_label_space(mapping: dict[str, dict[str, int]]) -> JointLabelSpace:
    """Build the :class:`JointLabelSpace` from a public label mapping."""

    lookup = combined_to_dual_lookup(mapping)
    num_classes = len(mapping["classification_labels"])
    object_of_class = np.zeros(num_classes, dtype=np.int64)
    weight_of_class = np.zeros(num_classes, dtype=np.int64)
    dual_to_joint: dict[tuple[int, int], int] = {}
    for combined_id, (object_id, weight_id) in lookup.items():
        object_of_class[combined_id] = object_id
        weight_of_class[combined_id] = weight_id
        dual_to_joint[(object_id, weight_id)] = combined_id
    id_to_name = {idx: name for name, idx in mapping["classification_labels"].items()}
    class_names = [id_to_name[i] for i in range(num_classes)]
    return JointLabelSpace(
        class_names=class_names,
        object_of_class=object_of_class,
        weight_of_class=weight_of_class,
        dual_to_joint=dual_to_joint,
    )


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any], JointLabelSpace]:
    """Load a trained checkpoint and return ``(model, mapping, joint_space)``."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = checkpoint.get("model_config", {})
    mapping = checkpoint.get("label_mapping") or load_label_mapping()
    model = build_model(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    joint_space = build_joint_label_space(mapping)
    return model, mapping, joint_space


@dataclass
class WindowOutputs:
    """Per-window model outputs and targets collected over a data loader.

    All fields are NumPy arrays with a leading ``N`` window dimension.
    ``object_prob`` / ``weight_prob`` are softmax posteriors from the two heads.
    """

    object_prob: np.ndarray  # [N, num_object_classes]
    weight_prob: np.ndarray  # [N, num_weight_classes]
    object_pred: np.ndarray  # [N]
    weight_pred: np.ndarray  # [N]
    object_target: np.ndarray  # [N]
    weight_target: np.ndarray  # [N]
    class_target: np.ndarray  # [N] joint (61-way) target
    episode_index: np.ndarray  # [N]
    window_start: np.ndarray  # [N]

    def __len__(self) -> int:
        return int(self.class_target.shape[0])


@torch.no_grad()
def collect_window_outputs(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> WindowOutputs:
    """Run ``model`` over ``loader`` and gather per-window posteriors and targets."""

    object_prob: list[np.ndarray] = []
    weight_prob: list[np.ndarray] = []
    object_target: list[np.ndarray] = []
    weight_target: list[np.ndarray] = []
    class_target: list[np.ndarray] = []
    episode_index: list[np.ndarray] = []
    window_start: list[np.ndarray] = []

    model.eval()
    for batch in loader:
        batch = move_to_device(batch, device)
        object_logits, weight_logits = model(batch)
        object_prob.append(torch.softmax(object_logits, dim=-1).cpu().numpy())
        weight_prob.append(torch.softmax(weight_logits, dim=-1).cpu().numpy())
        object_target.append(batch["object_label"].cpu().numpy())
        weight_target.append(batch["weight_label"].cpu().numpy())
        class_target.append(batch["classification_label"].cpu().numpy())
        episode_index.append(batch["episode_index"].cpu().numpy())
        window_start.append(batch["window_start"].cpu().numpy())

    object_prob_arr = np.concatenate(object_prob, axis=0)
    weight_prob_arr = np.concatenate(weight_prob, axis=0)
    return WindowOutputs(
        object_prob=object_prob_arr,
        weight_prob=weight_prob_arr,
        object_pred=object_prob_arr.argmax(axis=-1),
        weight_pred=weight_prob_arr.argmax(axis=-1),
        object_target=np.concatenate(object_target, axis=0),
        weight_target=np.concatenate(weight_target, axis=0),
        class_target=np.concatenate(class_target, axis=0),
        episode_index=np.concatenate(episode_index, axis=0),
        window_start=np.concatenate(window_start, axis=0),
    )


def joint_posterior(outputs: WindowOutputs, joint_space: JointLabelSpace) -> np.ndarray:
    """Return the joint object-weight posterior ``P(o_c) * P(w_c)`` per class.

    Shape ``[N, num_classes]``. This is the target score used for retrieval and,
    once renormalized, the confidence used for early recognition. It is *not*
    renormalized here so that a single window's scores across classes are directly
    comparable to the per-target ``sigma`` in the retrieval task.
    """

    object_component = outputs.object_prob[:, joint_space.object_of_class]
    weight_component = outputs.weight_prob[:, joint_space.weight_of_class]
    return object_component * weight_component
