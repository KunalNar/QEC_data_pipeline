import numpy as np
import pandas as pd
import pytest

from quantum_lake_student import task_c
from quantum_lake_student.ml import google_data_split
from quantum_lake_student.task_c import (
    FEATURE_NAMES,
    TaskCFeatures,
    TrainedMLP,
    build_features,
    choose_threshold,
    classification_metrics,
    evaluate,
    select_task_c_rows,
    train_mlp,
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


def _random_features(seed=0, per_split=20):
    generator = np.random.default_rng(seed)
    count = per_split * 3
    return TaskCFeatures(
        example_id=np.array([f"shot-{index}" for index in range(count)]),
        features=generator.integers(0, 2, size=(count, 200), dtype=np.uint8),
        label=generator.integers(0, 2, size=count, dtype=np.uint8),
        split=np.array(["train", "validation", "test"] * per_split),
    )


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_same_seed_gives_identical_model():
    features = _random_features()
    first = train_mlp(features, seed=7)
    second = train_mlp(features, seed=7)
    probabilities = [
        trained.model.predict_proba(features.features)[:, 1]
        for trained in (first, second)
    ]
    assert np.array_equal(*probabilities)
    assert first.train_seconds >= 0


def test_mlp_is_fit_on_train_rows_only(monkeypatch):
    seen = {}

    class RecordingClassifier:
        def __init__(self, **settings):
            seen["settings"] = settings

        def fit(self, features, labels):
            seen["rows"] = len(features)
            seen["labels"] = len(labels)
            return self

    monkeypatch.setattr(task_c, "MLPClassifier", RecordingClassifier)
    features = _random_features(per_split=5)
    train_mlp(features, seed=3)

    assert seen["rows"] == seen["labels"] == 5
    assert seen["settings"]["random_state"] == 3


def test_metrics_on_a_known_example():
    result = classification_metrics(
        np.array([0, 1, 1, 0]),
        np.array([0, 1, 0, 0]),
        np.array([0.1, 0.9, 0.4, 0.2]),
    )
    assert result["logical_error_rate"] == 0.25
    assert result["balanced_accuracy"] == 0.75
    assert result["brier_score"] == pytest.approx(0.105)


def test_brier_score_is_not_applicable_without_probabilities():
    result = classification_metrics(np.array([0, 1]), np.array([0, 1]))
    assert result["brier_score"] is None


def test_threshold_minimises_error_and_prefers_half_on_ties():
    labels = np.array([0, 0, 1, 1])
    assert choose_threshold(np.array([0.2, 0.4, 0.6, 0.8]), labels) == 0.5
    assert choose_threshold(np.array([0.2, 0.3, 0.35, 0.8]), labels) == 0.35


class _FirstBitModel:
    """Fake model: probability 0.8 when detector 0 fired, otherwise 0.2."""

    def predict_proba(self, features):
        probability = features[:, 0] * 0.6 + 0.2
        return np.column_stack([1 - probability, probability])


def _first_bit_features(validation_inverted=False, test_inverted=False):
    first_bit = np.array([0, 1] * 6, dtype=np.uint8)
    split = np.repeat(["train", "validation", "test"], 4)
    label = first_bit.copy()
    if validation_inverted:
        label[split == "validation"] ^= 1
    if test_inverted:
        label[split == "test"] ^= 1
    features = np.zeros((12, 200), dtype=np.uint8)
    features[:, 0] = first_bit
    return TaskCFeatures(
        example_id=np.array([f"shot-{index}" for index in range(12)]),
        features=features,
        label=label,
        split=split,
    )


def test_threshold_is_chosen_on_validation_only():
    trained = TrainedMLP(model=_FirstBitModel(), train_seconds=0.0)
    base = evaluate(_first_bit_features(), trained).threshold
    test_changed = evaluate(_first_bit_features(test_inverted=True), trained).threshold
    validation_changed = evaluate(
        _first_bit_features(validation_inverted=True), trained
    ).threshold

    assert base == test_changed == 0.5
    assert validation_changed != base


def test_evaluate_scores_mlp_and_majority_on_the_same_rows():
    trained = TrainedMLP(model=_FirstBitModel(), train_seconds=1.5)
    result = evaluate(_first_bit_features(), trained)

    assert list(result.predictions.columns) == [
        "example_id", "model_id", "label", "prediction", "probability", "split"
    ]
    assert set(result.predictions["split"]) == {"validation", "test"}
    by_model = result.predictions.groupby("model_id")["example_id"].apply(list)
    assert by_model["C_mlp"] == by_model["C_majority"]

    test_metrics = {
        row["model_id"]: row for row in result.metrics if row["split"] == "test"
    }
    assert test_metrics["C_mlp"]["logical_error_rate"] == 0.0
    assert test_metrics["C_mlp"]["train_seconds"] == 1.5
    assert test_metrics["C_mlp"]["n_examples"] == 4
    assert test_metrics["C_majority"]["balanced_accuracy"] == 0.5


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_train_task_c_merges_without_touching_other_tasks(monkeypatch, tmp_path):
    import json
    from io import BytesIO

    generator = np.random.default_rng(0)
    rows = [
        _row(
            experiment,
            shot,
            set_bits=tuple(np.flatnonzero(generator.integers(0, 2, 200))),
            flip=bool(shot % 3),
        )
        for experiment in ("d3_a", "d3_b")
        for shot in range(20)
    ]
    buffer = BytesIO()
    _table(rows).to_parquet(buffer, index=False)
    monkeypatch.setattr(task_c, "read_lake_object", lambda *_: buffer.getvalue())

    pd.DataFrame(
        {
            "example_id": ["s-1"],
            "model_id": ["weighted_logistic_regression"],
            "label": [True],
            "prediction": [False],
            "probability": [0.3],
            "split": ["test"],
        }
    ).to_parquet(tmp_path / "predictions.parquet", index=False)
    (tmp_path / "metrics.json").write_text(json.dumps({"task_a_syndrome_decoder": {}}))
    (tmp_path / "run.json").write_text(json.dumps({"random_seed": 42}))

    for _ in range(2):
        result = task_c.train_task_c(None, run_id="test", results_root=tmp_path)

    predictions = pd.read_parquet(tmp_path / "predictions.parquet")
    counts = predictions["model_id"].value_counts().to_dict()
    assert counts == {"weighted_logistic_regression": 1, "C_mlp": 20, "C_majority": 20}
    assert set(predictions.loc[predictions["model_id"] != "weighted_logistic_regression", "split"]) == {"test"}
    assert result.output_count == 40

    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert set(metrics) == {"task_a_syndrome_decoder", task_c.TASK_C_KEY}
    assert metrics[task_c.TASK_C_KEY]["C_mlp"]["test_examples"] == 20

    run_record = json.loads((tmp_path / "run.json").read_text())
    assert run_record["random_seed"] == 42
    assert run_record["feature_order"][task_c.TASK_C_KEY] == list(FEATURE_NAMES)
    assert (tmp_path / "task_c_model.joblib").exists()
