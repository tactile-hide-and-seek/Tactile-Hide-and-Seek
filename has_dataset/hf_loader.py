"""Hugging Face loading helpers for the Hide-and-Seek tactile dataset."""

from __future__ import annotations

from typing import Optional

from datasets import Array2D, Dataset, DatasetDict, Features, Sequence, Value, load_dataset

DEFAULT_DATASET_NAME = "TUM-ICS/Hide-and-Seek"

DATASET_FEATURES = Features(
    {
        "timestamp": Value("float64"),
        "episode_index": Value("int64"),
        "obs_point_cloud_skin_contact": Array2D((88, 6), dtype="float64"),
        "mask_point_cloud_skin_contact": Sequence(Value("bool"), length=88),
        "obs_pose_left": Sequence(Value("float64"), length=7),
        "obs_skin_prox_left": Sequence(Value("float64"), length=44),
        "obs_skin_force_left": Sequence(Value("float64"), length=44),
        "obs_skin_dist_left": Sequence(Value("float64"), length=44),
        "obs_dami_force_left": Sequence(Value("float64"), length=6),
        "obs_dami_prox_left": Sequence(Value("float64"), length=6),
        "obs_ft_left": Sequence(Value("float64"), length=6),
        "classification_label": Value("int64"),
    }
)


def load_has_dataset(
    name: str = DEFAULT_DATASET_NAME,
    split: Optional[str] = None,
    max_samples: Optional[int] = None,
    trust_remote_code: bool = False,
) -> Dataset | DatasetDict:
    """Load the public HAS dataset from Hugging Face.

    Args:
        name: Hugging Face dataset repository name.
        split: Optional split name. If omitted, returns a DatasetDict.
        max_samples: Optional number of rows to keep from the selected split.
        trust_remote_code: Forwarded to ``datasets.load_dataset``.

    Returns:
        A Hugging Face ``Dataset`` or ``DatasetDict``.
    """

    dataset = load_dataset(
        name,
        split=split,
        features=DATASET_FEATURES,
        trust_remote_code=trust_remote_code,
    )
    if max_samples is not None:
        if isinstance(dataset, DatasetDict):
            return DatasetDict(
                {
                    key: value.select(range(min(max_samples, len(value))))
                    for key, value in dataset.items()
                }
            )
        return dataset.select(range(min(max_samples, len(dataset))))
    return dataset
