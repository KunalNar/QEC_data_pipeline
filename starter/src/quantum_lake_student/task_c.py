"""Task C: bounded raw-detector MLP prototype.

Builds the distance-three detector-bit feature matrix from the Google ML input
table. Only rows with ``distance = 3 AND shot_index < 12_500`` are used, with
the supplied ``data_split``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .ml import MODEL_SPLITS, unpack_little_endian_bits


TASK_C_DISTANCE = 3
TASK_C_SHOT_LIMIT = 12_500
TASK_C_DETECTOR_COUNT = 200
FEATURE_NAMES = tuple(f"detector_{index}" for index in range(TASK_C_DETECTOR_COUNT))


@dataclass(frozen=True)
class TaskCFeatures:
    """Row-aligned model inputs: row ``i`` of every array is the same shot."""

    example_id: np.ndarray
    features: np.ndarray
    label: np.ndarray
    split: np.ndarray

    def part(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        """Return the features and labels of one supplied partition."""
        if split not in MODEL_SPLITS:
            raise ValueError(f"unexpected split: {split!r}")
        mask = self.split == split
        return self.features[mask], self.label[mask]


def select_task_c_rows(google: pd.DataFrame) -> pd.DataFrame:
    """Return the bounded distance-three subset in a repeatable row order."""
    mask = (google["distance"] == TASK_C_DISTANCE) & (
        google["shot_index"] < TASK_C_SHOT_LIMIT
    )
    return (
        google.loc[mask]
        .sort_values(["experiment_id", "shot_index"])
        .reset_index(drop=True)
    )


def build_features(google: pd.DataFrame) -> TaskCFeatures:
    """Unpack the Task C subset into a checked 200-bit feature matrix."""
    subset = select_task_c_rows(google)
    if subset.empty:
        raise ValueError("Task C subset is empty")

    if not (subset["detector_count"] == TASK_C_DETECTOR_COUNT).all():
        raise ValueError("distance-three rows must have 200 detectors")

    features = np.array(
        [
            unpack_little_endian_bits(bytes(packed), TASK_C_DETECTOR_COUNT)
            for packed in subset["detector_bits"]
        ],
        dtype=np.uint8,
    )
    event_counts = subset["detector_event_count"].to_numpy()
    if not (features.sum(axis=1) == event_counts).all():
        raise ValueError("unpacked bits do not match detector_event_count")

    split = subset["data_split"].astype(str).to_numpy()
    for name in MODEL_SPLITS:
        if not (split == name).any():
            raise ValueError(f"Task C {name} split is empty")
    if not np.isin(split, MODEL_SPLITS).all():
        raise ValueError("unexpected data_split value")

    return TaskCFeatures(
        example_id=subset["example_id"].astype(str).to_numpy(),
        features=features,
        label=subset["actual_observable_flip"].astype(bool).to_numpy(dtype=np.uint8),
        split=split,
    )
