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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss

from .config import Settings
from .connections import read_lake_object
from .ml import (
    GOOGLE_META_PREDICTION_COLUMNS,
    google_meta_model_input,
    weighted_logical_error_rate,
)
from .models import StageResult


DISTANCES = (3, 5)
SEED = 42

GOOGLE_ML_OBJECT = "ml/ml_google_decoder_example.parquet"
TASK_B_KEY = "task_b_supplied_and_combined_decoders"

TASK_B_MODEL_IDS = (
    "B_prior_baseline",
    "B_belief_matching",
    "B_correlated_matching",
    "B_pymatching",
    "B_tensor_network",
    "B_combined_logistic",
)

FEATURE_NAMES = (
    "detector_event_density",
    *GOOGLE_META_PREDICTION_COLUMNS,
)

THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


@dataclass(frozen=True)
class DistanceData:
    """Prepared Task B data for one code distance."""

    example_id: np.ndarray
    features: np.ndarray
    label: np.ndarray
    split: np.ndarray


def build_distance_data(
    google: pd.DataFrame,
    distance: int,
) -> DistanceData:
    """Build the required five-feature matrix for one code distance."""

    subset = (
        google.loc[google["distance"] == distance]
        .sort_values(["experiment_id", "shot_index"])
        .reset_index(drop=True)
    )

    if subset.empty:
        raise ValueError(f"No rows found for distance {distance}")

    features = np.asarray(
        [
            google_meta_model_input(row)
            for row in subset.to_dict("records")
        ],
        dtype=float,
    )

    if features.shape[1] != 5:
        raise ValueError(
            "Task B must use exactly five input features"
        )

    split = subset["data_split"].astype(str).to_numpy()

    for required_split in ("train", "validation", "test"):
        if not np.any(split == required_split):
            raise ValueError(
                f"Distance {distance} has no {required_split} rows"
            )

    return DistanceData(
        example_id=subset["example_id"].astype(str).to_numpy(),
        features=features,
        label=(
            subset["actual_observable_flip"]
            .astype(bool)
            .to_numpy()
        ),
        split=split,
    )


def logical_error_rate(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> float:
    """Return the unweighted logical-error rate for hardware shots."""

    weights = np.ones(len(labels), dtype=float)

    return weighted_logical_error_rate(
        labels,
        predictions,
        weights,
    )


def classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> dict[str, float | None]:
    """Return the required Task B evaluation metrics."""

    return {
        "logical_error_rate": logical_error_rate(
            labels,
            predictions,
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                labels,
                predictions,
            )
        ),
        "brier_score": (
            None
            if probabilities is None
            else float(
                brier_score_loss(
                    labels,
                    probabilities,
                )
            )
        ),
    }


def choose_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Choose the classification threshold on validation data only."""

    errors = [
        logical_error_rate(
            labels,
            probabilities >= threshold,
        )
        for threshold in THRESHOLD_GRID
    ]

    best_index = min(
        range(len(THRESHOLD_GRID)),
        key=lambda index: (
            errors[index],
            abs(THRESHOLD_GRID[index] - 0.5),
        ),
    )

    return float(THRESHOLD_GRID[best_index])


def train_combined_decoder(
    data: DistanceData,
) -> tuple[LogisticRegression, float, float]:
    """Fit the five-feature linear combined decoder."""

    train_mask = data.split == "train"
    validation_mask = data.split == "validation"

    model = LogisticRegression(
        random_state=SEED,
        max_iter=1000,
    )

    started = perf_counter()

    model.fit(
        data.features[train_mask],
        data.label[train_mask],
    )

    train_seconds = perf_counter() - started

    validation_probabilities = model.predict_proba(
        data.features[validation_mask]
    )[:, 1]

    threshold = choose_threshold(
        validation_probabilities,
        data.label[validation_mask],
    )

    return model, threshold, train_seconds


SUPPLIED_DECODER_MODEL_IDS = {
    "belief_matching_prediction": "B_belief_matching",
    "correlated_matching_prediction": "B_correlated_matching",
    "pymatching_prediction": "B_pymatching",
    "tensor_network_contraction_prediction": "B_tensor_network",
}

COMBINED_MODEL_ID = "B_combined_logistic"
BASELINE_MODEL_ID = "B_prior_baseline"


@dataclass(frozen=True)
class TaskBDistanceEvaluation:
    """Evaluation results for one Google code distance."""

    distance: int
    threshold: float
    predictions: pd.DataFrame
    metrics: list[dict]


def evaluate_distance(
    data: DistanceData,
    distance: int,
) -> tuple[LogisticRegression, TaskBDistanceEvaluation]:
    """Evaluate all required Task B models on one distance.

    The prior baseline is estimated from the training split.
    The combined decoder is fitted on training data and its threshold is
    selected using validation data. Final metrics are computed on test data.
    """

    train_mask = data.split == "train"
    test_mask = data.split == "test"

    train_labels = data.label[train_mask]
    test_labels = data.label[test_mask]
    test_ids = data.example_id[test_mask]
    test_features = data.features[test_mask]

    # -------------------------------------------------------------
    # 1. Prior / majority baseline
    # -------------------------------------------------------------
    started = perf_counter()

    prior_probability = float(train_labels.mean())
    baseline_prediction_value = prior_probability >= 0.5

    baseline_train_seconds = perf_counter() - started

    started = perf_counter()

    baseline_probabilities = np.full(
        len(test_labels),
        prior_probability,
        dtype=float,
    )

    baseline_predictions = np.full(
        len(test_labels),
        baseline_prediction_value,
        dtype=bool,
    )

    baseline_predict_seconds = perf_counter() - started

    # -------------------------------------------------------------
    # 2. Combined five-feature logistic regression
    # -------------------------------------------------------------
    combined_model, threshold, combined_train_seconds = (
        train_combined_decoder(data)
    )

    started = perf_counter()

    combined_probabilities = combined_model.predict_proba(
        test_features
    )[:, 1]

    combined_predictions = (
        combined_probabilities >= threshold
    )

    combined_predict_seconds = perf_counter() - started

    # -------------------------------------------------------------
    # 3. Collect predictions and metrics
    # -------------------------------------------------------------
    prediction_frames = []
    metrics = []

    # Prior baseline
    prediction_frames.append(
        pd.DataFrame(
            {
                "example_id": test_ids,
                "model_id": BASELINE_MODEL_ID,
                "label": test_labels.astype(bool),
                "prediction": baseline_predictions.astype(bool),
                "probability": baseline_probabilities,
                "split": "test",
            }
        )
    )

    metrics.append(
        {
            "task": "B",
            "distance": distance,
            "model_id": BASELINE_MODEL_ID,
            **classification_metrics(
                test_labels,
                baseline_predictions,
                baseline_probabilities,
            ),
            "training_time_seconds": baseline_train_seconds,
            "prediction_time_seconds": baseline_predict_seconds,
            "threshold": 0.5,
            "test_examples": len(test_labels),
        }
    )

    # -------------------------------------------------------------
    # Four supplied decoder predictions
    #
    # Columns 1-4 of the Task B feature matrix are exactly the four
    # supplied decoder predictions. Column 0 is detector-event density.
    # -------------------------------------------------------------
    for feature_index, decoder_column in enumerate(
        GOOGLE_META_PREDICTION_COLUMNS,
        start=1,
    ):
        model_id = SUPPLIED_DECODER_MODEL_IDS[decoder_column]

        supplied_predictions = (
            test_features[:, feature_index]
            .astype(bool)
        )

        prediction_frames.append(
            pd.DataFrame(
                {
                    "example_id": test_ids,
                    "model_id": model_id,
                    "label": test_labels.astype(bool),
                    "prediction": supplied_predictions,
                    "probability": np.nan,
                    "split": "test",
                }
            )
        )

        metrics.append(
            {
                "task": "B",
                "distance": distance,
                "model_id": model_id,
                **classification_metrics(
                    test_labels,
                    supplied_predictions,
                    probabilities=None,
                ),
                # These predictions were supplied in the ML table.
                # We did not train or run the original decoder ourselves.
                "training_time_seconds": None,
                "prediction_time_seconds": None,
                "threshold": None,
                "test_examples": len(test_labels),
            }
        )

    # -------------------------------------------------------------
    # Combined logistic regression
    # -------------------------------------------------------------
    prediction_frames.append(
        pd.DataFrame(
            {
                "example_id": test_ids,
                "model_id": COMBINED_MODEL_ID,
                "label": test_labels.astype(bool),
                "prediction": combined_predictions.astype(bool),
                "probability": combined_probabilities,
                "split": "test",
            }
        )
    )

    metrics.append(
        {
            "task": "B",
            "distance": distance,
            "model_id": COMBINED_MODEL_ID,
            **classification_metrics(
                test_labels,
                combined_predictions,
                combined_probabilities,
            ),
            "training_time_seconds": combined_train_seconds,
            "prediction_time_seconds": combined_predict_seconds,
            "threshold": threshold,
            "test_examples": len(test_labels),
        }
    )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    evaluation = TaskBDistanceEvaluation(
        distance=distance,
        threshold=threshold,
        predictions=predictions,
        metrics=metrics,
    )

    return combined_model, evaluation


def decoder_error_overlap(
    data: DistanceData,
) -> dict[str, object]:
    """Describe how often the four supplied decoders make the same mistakes."""

    test_mask = data.split == "test"
    labels = data.label[test_mask]

    # Columns 1-4 are the four supplied decoder predictions.
    supplied_predictions = data.features[
        test_mask,
        1:5,
    ].astype(bool)

    errors = supplied_predictions != labels[:, None]

    decoder_names = list(GOOGLE_META_PREDICTION_COLUMNS)

    error_rates = {
        decoder_names[index]: float(
            errors[:, index].mean()
        )
        for index in range(len(decoder_names))
    }

    pairwise_overlap = {}

    for left_index in range(len(decoder_names)):
        for right_index in range(
            left_index + 1,
            len(decoder_names),
        ):
            left_name = decoder_names[left_index]
            right_name = decoder_names[right_index]

            both_wrong = (
                errors[:, left_index]
                & errors[:, right_index]
            )

            either_wrong = (
                errors[:, left_index]
                | errors[:, right_index]
            )

            intersection = int(both_wrong.sum())
            union = int(either_wrong.sum())

            jaccard = (
                float(intersection / union)
                if union > 0
                else 1.0
            )

            pairwise_overlap[
                f"{left_name}__{right_name}"
            ] = {
                "both_wrong": intersection,
                "either_wrong": union,
                "error_jaccard": jaccard,
            }

    all_wrong = int(
        errors.all(axis=1).sum()
    )

    any_wrong = int(
        errors.any(axis=1).sum()
    )

    exactly_one_wrong = int(
        (errors.sum(axis=1) == 1).sum()
    )

    return {
        "decoder_error_rates": error_rates,
        "pairwise_error_overlap": pairwise_overlap,
        "all_four_wrong": all_wrong,
        "any_decoder_wrong": any_wrong,
        "exactly_one_decoder_wrong": exactly_one_wrong,
        "test_examples": int(len(labels)),
    }


def _read_json(path: Path) -> dict:
    """Read a JSON file if it exists, otherwise return an empty dict."""

    if not path.exists():
        return {}

    return json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )


def _write_json(
    path: Path,
    value: dict,
) -> None:
    """Write a JSON object using the shared Part II formatting."""

    path.write_text(
        json.dumps(
            value,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def train_task_b(
    settings: Settings,
    *,
    run_id: str,
    results_root: Path = Path("results/part2"),
) -> StageResult:
    """Train and evaluate Task B for distances 3 and 5."""

    result = StageResult(
        stage="train_task_b",
        run_id=run_id,
    )

    google_bytes = read_lake_object(
        settings,
        GOOGLE_ML_OBJECT,
    )

    google = pq.read_table(
        io.BytesIO(google_bytes)
    ).to_pandas()

    all_predictions = []
    all_metrics = []

    overlap_by_distance = {}
    trained_models = {}
    evaluations_by_distance = {}

    for distance in DISTANCES:
        data = build_distance_data(
            google,
            distance,
        )

        model, evaluation = evaluate_distance(
            data,
            distance,
        )

        trained_models[distance] = model
        evaluations_by_distance[distance] = evaluation

        all_predictions.append(
            evaluation.predictions
        )

        all_metrics.extend(
            evaluation.metrics
        )

        overlap_by_distance[
            str(distance)
        ] = decoder_error_overlap(data)

    task_b_predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    results_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Save one combined logistic-regression model per distance.
    # -------------------------------------------------------------
    for distance, model in trained_models.items():
        joblib.dump(
            model,
            results_root
            / f"task_b_distance_{distance}_model.joblib",
        )

    # -------------------------------------------------------------
    # predictions.parquet
    #
    # Replace only Task B rows so rerunning does not create duplicates.
    # -------------------------------------------------------------
    predictions_path = (
        results_root
        / "predictions.parquet"
    )

    merged_predictions = task_b_predictions

    if predictions_path.exists():
        existing = pd.read_parquet(
            predictions_path
        )

        other_tasks = existing[
            ~existing["model_id"].isin(
                TASK_B_MODEL_IDS
            )
        ]

        merged_predictions = pd.concat(
            [
                other_tasks,
                task_b_predictions,
            ],
            ignore_index=True,
        )

    merged_predictions.to_parquet(
        predictions_path,
        index=False,
    )

    # -------------------------------------------------------------
    # metrics.json
    # -------------------------------------------------------------
    metrics_path = (
        results_root
        / "metrics.json"
    )

    metrics_record = _read_json(
        metrics_path
    )

    distance_metrics = {}

    for distance in DISTANCES:
        rows = [
            row
            for row in all_metrics
            if row["distance"] == distance
        ]

        distance_metrics[
            str(distance)
        ] = {
            row["model_id"]: {
                "logical_error_rate": (
                    row["logical_error_rate"]
                ),
                "balanced_accuracy": (
                    row["balanced_accuracy"]
                ),
                "brier_score": (
                    row["brier_score"]
                ),
                "training_time_seconds": (
                    row["training_time_seconds"]
                ),
                "prediction_time_seconds": (
                    row["prediction_time_seconds"]
                ),
                "threshold": (
                    row["threshold"]
                ),
                "test_examples": (
                    row["test_examples"]
                ),
            }
            for row in rows
        }

    metrics_record[
        TASK_B_KEY
    ] = {
        "distances": distance_metrics,
        "decoder_error_overlap": (
            overlap_by_distance
        ),
    }

    _write_json(
        metrics_path,
        metrics_record,
    )

    # -------------------------------------------------------------
    # run.json
    # -------------------------------------------------------------
    run_path = (
        results_root
        / "run.json"
    )

    run_record = _read_json(
        run_path
    )

    run_record.setdefault(
        "ml_input_hashes",
        {},
    )[GOOGLE_ML_OBJECT] = hashlib.sha256(
        google_bytes
    ).hexdigest()

    run_record["random_seed"] = SEED

    run_record.setdefault(
        "feature_order",
        {},
    )[TASK_B_KEY] = list(
        FEATURE_NAMES
    )

    run_record.setdefault(
        "split_rules",
        {},
    )[TASK_B_KEY] = (
        "google_data_split separately for distances 3 and 5: "
        "odd shot_index test, shot_index % 10 == 8 validation, "
        "other even shot_index train"
    )

    run_record.setdefault(
        "task_settings",
        {},
    )[TASK_B_KEY] = {
        "model": "LogisticRegression",
        "max_iter": 1000,
        "threshold_grid": [
            float(value)
            for value in THRESHOLD_GRID
        ],
        "threshold_rule": (
            "lowest validation logical-error rate; "
            "ties go to threshold nearest 0.5"
        ),
        "distances": list(
            DISTANCES
        ),
        "selected_thresholds": {
            str(distance): (
                evaluations_by_distance[
                    distance
                ].threshold
            )
            for distance in DISTANCES
        },
    }

    # -------------------------------------------------------------
    # Task B timings
    # -------------------------------------------------------------
    task_b_timings = {}

    for distance in DISTANCES:
        rows_by_model = {
            row["model_id"]: row
            for row in all_metrics
            if row["distance"] == distance
        }

        task_b_timings[
            str(distance)
        ] = {
            "prior_baseline_training_seconds": (
                rows_by_model[
                    BASELINE_MODEL_ID
                ]["training_time_seconds"]
            ),
            "prior_baseline_prediction_seconds": (
                rows_by_model[
                    BASELINE_MODEL_ID
                ]["prediction_time_seconds"]
            ),
            "combined_training_seconds": (
                rows_by_model[
                    COMBINED_MODEL_ID
                ]["training_time_seconds"]
            ),
            "combined_prediction_seconds": (
                rows_by_model[
                    COMBINED_MODEL_ID
                ]["prediction_time_seconds"]
            ),
        }

    run_record.setdefault(
        "timings",
        {},
    )[TASK_B_KEY] = task_b_timings

    _write_json(
        run_path,
        run_record,
    )

    result.input_count = len(google)
    result.output_count = len(
        task_b_predictions
    )

    result.finish()

    return result