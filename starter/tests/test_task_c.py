import numpy as np
import pandas as pd
import pytest

from quantum_lake_student.ml import google_data_split
from quantum_lake_student.task_c import (
    FEATURE_NAMES,
    build_features,
    select_task_c_rows,
)


def _pack(bits):
    """Pack 0/1 values into little-endian Stim b8 bytes."""
    packed = bytearray((len(bits) + 7) // 8)
    for index, bit in enumerate(bits):
        if bit:
            packed[index // 8] |= 1 << (index % 8)
    return bytes(packed)


def _row(experiment, shot, set_bits=(), distance=3, flip=False):
    detector_count = 200 if distance == 3 else 600
    bits = [0] * detector_count
    for index in set_bits:
        bits[index] = 1
    return {
        "example_id": f"{experiment}-{shot}",
        "experiment_id": experiment,
        "shot_index": shot,
        "distance": distance,
        "detector_count": detector_count,
        "detector_event_count": len(set_bits),
        "detector_bits": _pack(bits),
        "actual_observable_flip": flip,
        "data_split": google_data_split(shot),
    }


def _table(rows):
    return pd.DataFrame(rows)


def _all_splits(experiment="d3_a"):
    # shot 0 -> train, 1 -> test, 8 -> validation
    return [_row(experiment, 0), _row(experiment, 1), _row(experiment, 8)]


def test_feature_names_cover_200_detectors():
    assert len(FEATURE_NAMES) == 200
    assert FEATURE_NAMES[0] == "detector_0"
    assert FEATURE_NAMES[-1] == "detector_199"


def test_filter_keeps_only_distance_three_below_shot_limit():
    rows = _all_splits() + [
        _row("d3_a", 12_499),
        _row("d3_a", 12_500),
        _row("d5", 0, distance=5),
    ]
    subset = select_task_c_rows(_table(rows))
    assert set(subset["distance"]) == {3}
    assert subset["shot_index"].max() == 12_499
    assert len(subset) == 4


def test_unpacking_uses_little_endian_bit_order():
    rows = _all_splits()
    rows[0] = _row("d3_a", 0, set_bits=(0, 9, 199))
    result = build_features(_table(rows))
    first = result.features[result.example_id == "d3_a-0"][0]
    assert result.features.shape == (3, 200)
    assert np.flatnonzero(first).tolist() == [0, 9, 199]


def test_event_count_mismatch_is_rejected():
    rows = _all_splits()
    rows[0]["detector_event_count"] = 5
    with pytest.raises(ValueError, match="detector_event_count"):
        build_features(_table(rows))


def test_wrong_detector_count_is_rejected():
    rows = _all_splits()
    rows[0]["detector_count"] = 199
    with pytest.raises(ValueError, match="200 detectors"):
        build_features(_table(rows))


def test_missing_split_is_rejected():
    rows = [_row("d3_a", 0), _row("d3_a", 1)]
    with pytest.raises(ValueError, match="validation split is empty"):
        build_features(_table(rows))


def test_rows_stay_aligned_and_order_is_repeatable():
    rows = _all_splits("d3_b") + _all_splits("d3_a")
    rows[1] = _row("d3_b", 1, set_bits=(3,), flip=True)
    forward = build_features(_table(rows))
    backward = build_features(_table(list(reversed(rows))))

    assert forward.example_id.tolist() == backward.example_id.tolist()
    assert (forward.features == backward.features).all()

    flipped = forward.example_id == "d3_b-1"
    assert forward.label[flipped].tolist() == [1]
    assert forward.split[flipped].tolist() == ["test"]
    assert forward.features[flipped][0, 3] == 1


def test_part_returns_only_the_requested_split():
    result = build_features(_table(_all_splits("d3_a") + _all_splits("d3_b")))
    features, labels = result.part("validation")
    assert features.shape == (2, 200)
    assert labels.shape == (2,)
    with pytest.raises(ValueError):
        result.part("holdout")
