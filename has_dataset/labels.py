"""Label mapping helpers for HAS classification labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_LABEL_MAPPING_PATH = Path(__file__).resolve().parents[1] / "data" / "label_mapping.json"


def load_label_mapping(path: str | Path = DEFAULT_LABEL_MAPPING_PATH) -> dict[str, dict[str, int]]:
    """Load combined, object, and weight label mappings from JSON."""

    with Path(path).open("r", encoding="utf-8") as handle:
        mapping: dict[str, dict[str, int]] = json.load(handle)
    validate_label_mapping(mapping)
    return mapping


def validate_label_mapping(mapping: dict[str, Any]) -> None:
    """Validate the public label mapping."""

    required = {"classification_labels", "object_labels", "weight_labels"}
    missing = required.difference(mapping)
    if missing:
        raise ValueError(f"Label mapping is missing sections: {sorted(missing)}")
    if len(mapping["classification_labels"]) != 61:
        raise ValueError("Expected 61 classification labels.")
    if len(mapping["object_labels"]) != 34:
        raise ValueError("Expected 34 object labels including negative.")
    if len(mapping["weight_labels"]) != 4:
        raise ValueError("Expected 4 weight labels: none, light, medium, heavy.")


def combined_to_dual_lookup(mapping: dict[str, dict[str, int]]) -> dict[int, tuple[int, int]]:
    """Return ``classification_label -> (object_label, weight_label)`` lookup."""

    object_labels = mapping["object_labels"]
    weight_labels = mapping["weight_labels"]
    lookup: dict[int, tuple[int, int]] = {}
    for combined_name, combined_id in mapping["classification_labels"].items():
        if combined_name == "negative":
            object_name, weight_name = "negative", "none"
        else:
            object_name, weight_name = combined_name.rsplit("_", 1)
        if object_name not in object_labels:
            raise ValueError(f"Object label '{object_name}' is not defined.")
        if weight_name not in weight_labels:
            raise ValueError(f"Weight label '{weight_name}' is not defined.")
        lookup[int(combined_id)] = (object_labels[object_name], weight_labels[weight_name])
    return lookup
