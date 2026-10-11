"""Task C: bounded raw-detector MLP prototype.

Builds the distance-three detector-bit feature matrix from the Google ML input
table. Only rows with ``distance = 3 AND shot_index < 12_500`` are used, with
the supplied ``data_split``.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import balanced_accuracy_score, brier_score_loss
from sklearn.neural_network import MLPClassifier

from .config import Settings
from .connections import read_lake_object
from .ml import MODEL_SPLITS, unpack_little_endian_bits, weighted_logical_error_rate
from .models import StageResult


TASK_C_DISTANCE = 3
TASK_C_SHOT_LIMIT = 12_500
TASK_C_DETECTOR_COUNT = 200
FEATURE_NAMES = tuple(f"detector_{index}" for index in range(TASK_C_DETECTOR_COUNT))

MLP_SETTINGS = {"hidden_layer_sizes": (64,), "alpha": 1.0, "max_iter": 200}
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)

# Same seed Task A records as ``random_seed`` in run.json.
SEED = 42
GOOGLE_ML_OBJECT = "ml/ml_google_decoder_example.parquet"
TASK_C_KEY = "task_c_raw_detector_mlp"
TASK_C_MODEL_IDS = ("C_mlp", "C_majority")


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


@dataclass(frozen=True)
class TrainedMLP:
    model: MLPClassifier
    train_seconds: float


def train_mlp(features: TaskCFeatures, seed: int) -> TrainedMLP:
    """Fit the MLP on the supplied train rows only and time the fit."""
    train_features, train_labels = features.part("train")
    model = MLPClassifier(**MLP_SETTINGS, random_state=seed)
    started = perf_counter()
    model.fit(train_features, train_labels)
    return TrainedMLP(model=model, train_seconds=perf_counter() - started)


def choose_threshold(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Return the grid threshold with the lowest error; ties go nearest 0.5."""
    errors = [np.mean((probabilities >= value) != labels) for value in THRESHOLD_GRID]
    best = min(
        range(len(THRESHOLD_GRID)),
        key=lambda index: (errors[index], abs(THRESHOLD_GRID[index] - 0.5)),
    )
    return float(THRESHOLD_GRID[best])


def classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> dict[str, float | None]:
    """Logical-error rate, balanced accuracy, and Brier score for one shot set."""
    return {
        "logical_error_rate": weighted_logical_error_rate(
            labels, predictions, np.ones(len(labels))
        ),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "brier_score": (
            None
            if probabilities is None
            else float(brier_score_loss(labels, probabilities))
        ),
    }


@dataclass(frozen=True)
class TaskCEvaluation:
    threshold: float
    predictions: pd.DataFrame
    metrics: list[dict]


def evaluate(features: TaskCFeatures, trained: TrainedMLP) -> TaskCEvaluation:
    """Choose the MLP threshold on validation, then score validation and test.

    A majority baseline, fitted on the train labels, is scored on the same rows.
    """
    validation_features, validation_labels = features.part("validation")
    threshold = choose_threshold(
        trained.model.predict_proba(validation_features)[:, 1], validation_labels
    )

    started = perf_counter()
    flip_rate = float(features.part("train")[1].mean())
    majority_train_seconds = perf_counter() - started

    frames = []
    metrics = []
    for split in ("validation", "test"):
        mask = features.split == split
        labels = features.label[mask]

        started = perf_counter()
        mlp_probability = trained.model.predict_proba(features.features[mask])[:, 1]
        mlp_predict_seconds = perf_counter() - started

        started = perf_counter()
        majority_probability = np.full(len(labels), flip_rate)
        majority_predict_seconds = perf_counter() - started

        scored = (
            (
                "C_mlp",
                (mlp_probability >= threshold).astype(np.uint8),
                mlp_probability,
                trained.train_seconds,
                mlp_predict_seconds,
                threshold,
            ),
            (
                "C_majority",
                (majority_probability >= 0.5).astype(np.uint8),
                majority_probability,
                majority_train_seconds,
                majority_predict_seconds,
                0.5,
            ),
        )
        for model_id, prediction, probability, train_s, predict_s, cutoff in scored:
            frames.append(
                pd.DataFrame(
                    {
                        "example_id": features.example_id[mask],
                        "model_id": model_id,
                        "label": labels.astype(bool),
                        "prediction": prediction.astype(bool),
                        "probability": probability,
                        "split": split,
                    }
                )
            )
            metrics.append(
                {
                    "task": "C",
                    "model_id": model_id,
                    "distance": TASK_C_DISTANCE,
                    "split": split,
                    **classification_metrics(labels, prediction, probability),
                    "train_seconds": train_s,
                    "predict_seconds": predict_s,
                    "n_examples": int(mask.sum()),
                    "threshold": cutoff,
                }
            )

    return TaskCEvaluation(
        threshold=threshold,
        predictions=pd.concat(frames, ignore_index=True),
        metrics=metrics,
    )


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def train_task_c(
    settings: Settings,
    *,
    run_id: str,
    results_root: Path = Path("results/part2"),
) -> StageResult:
    """Train Task C and merge its results into the shared results/part2/ files.

    Only Task C's own rows and keys are replaced, so rerunning keeps the other
    tasks' results and never duplicates Task C predictions.
    """
    result = StageResult(stage="train_task_c", run_id=run_id)

    google_bytes = read_lake_object(settings, GOOGLE_ML_OBJECT)
    google = pq.read_table(io.BytesIO(google_bytes)).to_pandas()

    features = build_features(google)
    trained = train_mlp(features, SEED)
    evaluation = evaluate(features, trained)

    results_root.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained.model, results_root / "task_c_model.joblib")

    # predictions.parquet: replace only Task C's test predictions
    task_c_predictions = evaluation.predictions[
        evaluation.predictions["split"] == "test"
    ]
    predictions_path = results_root / "predictions.parquet"
    merged = task_c_predictions
    if predictions_path.exists():
        existing = pd.read_parquet(predictions_path)
        others = existing[~existing["model_id"].isin(TASK_C_MODEL_IDS)]
        merged = pd.concat([others, task_c_predictions], ignore_index=True)
    merged.to_parquet(predictions_path, index=False)

    # metrics.json: one entry for Task C, test metrics per model
    test_metrics = [row for row in evaluation.metrics if row["split"] == "test"]
    metrics_path = results_root / "metrics.json"
    metrics = _read_json(metrics_path)
    metrics[TASK_C_KEY] = {
        "distance": TASK_C_DISTANCE,
        "subset": f"distance = {TASK_C_DISTANCE} AND shot_index < {TASK_C_SHOT_LIMIT}",
        **{
            row["model_id"]: {
                "logical_error_rate": row["logical_error_rate"],
                "balanced_accuracy": row["balanced_accuracy"],
                "brier_score": row["brier_score"],
                "training_time_seconds": row["train_seconds"],
                "prediction_time_seconds": row["predict_seconds"],
                "threshold": row["threshold"],
                "test_examples": row["n_examples"],
            }
            for row in test_metrics
        },
    }
    _write_json(metrics_path, metrics)

    # run.json: add Task C's hashes, feature order, split rule, settings, timings
    mlp_test = next(row for row in test_metrics if row["model_id"] == "C_mlp")
    run_path = results_root / "run.json"
    run_record = _read_json(run_path)
    run_record.setdefault("ml_input_hashes", {})[GOOGLE_ML_OBJECT] = (
        hashlib.sha256(google_bytes).hexdigest()
    )
    run_record["random_seed"] = SEED
    run_record.setdefault("feature_order", {})[TASK_C_KEY] = list(FEATURE_NAMES)
    run_record.setdefault("split_rules", {})[TASK_C_KEY] = (
        "google_data_split on distance = 3 AND shot_index < 12500: odd test, "
        "shot_index % 10 == 8 validation, other even train"
    )
    run_record.setdefault("timings", {}).update(
        {
            "task_c_training_seconds": trained.train_seconds,
            "task_c_prediction_seconds": mlp_test["predict_seconds"],
        }
    )
    run_record.setdefault("task_settings", {})[TASK_C_KEY] = {
        "mlp_settings": {
            **MLP_SETTINGS,
            "hidden_layer_sizes": list(MLP_SETTINGS["hidden_layer_sizes"]),
        },
        "mlp_iterations": int(trained.model.n_iter_),
        "threshold": evaluation.threshold,
        "threshold_rule": "lowest validation error on a 0.05-0.95 grid",
        "bit_order": "Stim b8, little-endian within each byte",
        "rows": {
            split: int((features.split == split).sum()) for split in MODEL_SPLITS
        },
    }
    _write_json(run_path, run_record)

    result.input_count = len(features.example_id)
    result.output_count = len(task_c_predictions)
    result.finish()
    return result
