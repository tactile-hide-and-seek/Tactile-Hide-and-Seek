"""Windowed PyTorch dataset for frame-wise tactile examples."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from has_dataset.transforms import normalize_with_stats

POINT_CLOUD_KEY = "obs_point_cloud_skin_contact"
POINT_MASK_KEY = "mask_point_cloud_skin_contact"
REQUIRED_FIELDS: dict[str, tuple[int, ...]] = {
    POINT_CLOUD_KEY: (88, 6),
    POINT_MASK_KEY: (88,),
    "obs_pose_left": (7,),
    "obs_skin_prox_left": (44,),
    "obs_skin_force_left": (44,),
    "obs_skin_dist_left": (44,),
    "obs_dami_force_left": (6,),
    "obs_dami_prox_left": (6,),
    "obs_ft_left": (6,),
}


@dataclass(frozen=True)
class WindowIndex:
    """Metadata for one temporal window."""

    episode_index: int
    start: int
    frame_indices: tuple[int, ...]


class HASWindowDataset(Dataset):
    """Group frames by episode and expose fixed-length temporal windows.

    The wrapper never crosses episode boundaries. Labels are read from the last
    frame in each window by default, which is suitable for frame-wise annotations
    and equivalent to episode-level labels when all frames in an episode match.
    """

    def __init__(
        self,
        hf_dataset: Sequence[Mapping[str, Any]],
        window_length: int = 100,
        stride: int = 40,
        label_lookup: Mapping[int, tuple[int, int]] | None = None,
        max_windows: int | None = None,
        strict: bool = True,
        normalization_stats: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        if window_length <= 0:
            raise ValueError("window_length must be positive.")
        if stride <= 0:
            raise ValueError("stride must be positive.")
        self.dataset = hf_dataset
        self.window_length = window_length
        self.stride = stride
        self.label_lookup = dict(label_lookup or {})
        self.strict = strict
        self.normalization_stats = normalization_stats
        self.episodes = self._group_by_episode()
        self.windows = self._build_windows(max_windows=max_windows)
        if not self.windows:
            raise ValueError(
                "No valid temporal windows were created. "
                f"Check split size, episode lengths, window_length={window_length}, stride={stride}."
            )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        window = self.windows[index]
        frames = [self._validate_and_tensorize(self.dataset[i], i) for i in window.frame_indices]
        batch: dict[str, torch.Tensor] = {}
        for key in REQUIRED_FIELDS:
            batch[key] = torch.stack([frame[key] for frame in frames], dim=0)

        label = int(frames[-1]["classification_label"].item())
        object_label, weight_label = self.label_lookup.get(label, (0, 0))
        batch["classification_label"] = torch.tensor(label, dtype=torch.long)
        batch["object_label"] = torch.tensor(object_label, dtype=torch.long)
        batch["weight_label"] = torch.tensor(weight_label, dtype=torch.long)
        batch["episode_index"] = torch.tensor(window.episode_index, dtype=torch.long)
        batch["window_start"] = torch.tensor(window.start, dtype=torch.long)
        if self.normalization_stats:
            batch = normalize_with_stats(batch, self.normalization_stats)
        return batch

    def _group_by_episode(self) -> dict[int, list[int]]:
        try:
            episode_indices = self.dataset["episode_index"]  # type: ignore[index]
        except Exception:
            episode_indices = [self.dataset[i]["episode_index"] for i in range(len(self.dataset))]

        episodes: dict[int, list[int]] = defaultdict(list)
        for row_index, episode_index in enumerate(episode_indices):
            episodes[int(episode_index)].append(row_index)
        return dict(sorted(episodes.items()))

    def _build_windows(self, max_windows: int | None) -> list[WindowIndex]:
        windows: list[WindowIndex] = []
        for episode_index, frame_indices in self.episodes.items():
            if len(frame_indices) < self.window_length:
                continue
            for start in range(0, len(frame_indices) - self.window_length + 1, self.stride):
                selected = tuple(frame_indices[start : start + self.window_length])
                windows.append(WindowIndex(episode_index=episode_index, start=start, frame_indices=selected))
                if max_windows is not None and len(windows) >= max_windows:
                    return windows
        return windows

    def _validate_and_tensorize(self, row: Mapping[str, Any], row_index: int) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for key, shape in REQUIRED_FIELDS.items():
            if key not in row or row[key] is None:
                if self.strict:
                    raise ValueError(f"Row {row_index} is missing required field '{key}'.")
                out[key] = torch.zeros(shape, dtype=torch.bool if key == POINT_MASK_KEY else torch.float32)
                continue

            array = np.asarray(row[key])
            if array.shape != shape:
                raise ValueError(
                    f"Row {row_index} field '{key}' has shape {array.shape}; expected {shape}."
                )
            dtype = torch.bool if key == POINT_MASK_KEY else torch.float32
            out[key] = torch.as_tensor(array, dtype=dtype)

        if "classification_label" not in row or row["classification_label"] is None:
            raise ValueError(f"Row {row_index} is missing required field 'classification_label'.")
        out["classification_label"] = torch.tensor(int(row["classification_label"]), dtype=torch.long)
        return out
