"""Required Part II model-training stage.

Consume the two required ML input tables produced by Part I through the
supplied model-input and partition helpers. Publish repeatable model files,
predictions, metrics, run settings, and the exact feature order under
``results/part2/``; ``report.md`` there is written by hand. Numerical
performance is not graded.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import read_lake_object
from quantum_lake_student.ml import (
    syndrome_model_input,
    weighted_logical_error_rate,
)
from quantum_lake_student.models import StageResult
from quantum_lake_student.part1_results import code_revision
from quantum_lake_student.part2_report import write_report
from quantum_lake_student.source_validation import (
    MANIFEST_OBJECT,
    read_release_manifest,
)
from quantum_lake_student.task_c import train_task_c


def run(model_run_id: str) -> StageResult:
    """Main Part II training orchestrator called by `make train`."""
    settings = Settings.from_environment()

    # Run Task A (weighted syndrome decoder)
    result_a = train_task_a(settings, run_id=model_run_id)

    # Run Task C (raw-detector MLP)
    train_task_c(settings, run_id=model_run_id)

    # Assemble report.md from docs/part2/ and the metrics written above
    write_report()

    return result_a


def train_task_a(
    settings: Settings,
    *,
    run_id: str,
    results_root: Path = Path("results/part2"),
) -> StageResult:
    """Train Task A models and save results to results/part2/."""
    result = StageResult(stage="train_task_a", run_id=run_id)

    # -------------------------------------------------------------
    # SECTION 1: Load Data from MinIO (Notebook Step 2)
    # -------------------------------------------------------------
    syndrome_bytes = read_lake_object(
        settings, "ml/ml_syndrome_decoder_example.parquet"
    )
    table = pq.read_table(io.BytesIO(syndrome_bytes))
    df = table.to_pandas()

    # -------------------------------------------------------------
    # SECTION 2: Extract X, y, w, and example_ids (Notebook Step 4)
    # -------------------------------------------------------------
    X = np.array([syndrome_model_input(row) for row in df["syndrome_bits"]])
    y = df["logical_error_label"].to_numpy().astype(bool)
    w = df["sample_weight"].to_numpy().astype(np.int64)

    # -------------------------------------------------------------
    # SECTION 3: Create Train / Val / Test Splits (Notebook Step 5)
    # -------------------------------------------------------------
    train_mask = (df["data_split"] == "train").to_numpy()
    val_mask = (df["data_split"] == "validation").to_numpy()
    test_mask = (df["data_split"] == "test").to_numpy()

    # 1. Training Set
    X_train = X[train_mask]
    y_train = y[train_mask]
    w_train = w[train_mask]

    # 2. Validation Set
    X_val = X[val_mask]
    y_val = y[val_mask]
    w_val = w[val_mask]

    # 3. Test Set
    X_test = X[test_mask]
    y_test = y[test_mask]
    w_test = w[test_mask]

    # -------------------------------------------------------------
    # SECTION 4: Fit Baseline & Logistic Regression (Notebook Steps 6 & 7)
    # -------------------------------------------------------------
    # 1. Dummy prior baseline
    baseline = DummyClassifier(strategy="prior")
    t0 = time.perf_counter()
    baseline.fit(X_train, y_train, sample_weight=w_train)
    base_train_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    test_preds_baseline = baseline.predict(X_test)
    test_probs_baseline = baseline.predict_proba(X_test)[:, 1]
    base_pred_time = time.perf_counter() - t0
    base_ler_baseline = weighted_logical_error_rate(
        y_test, test_preds_baseline, w_test
    )
    base_balanced_acc = balanced_accuracy_score(
        y_test, test_preds_baseline, sample_weight=w_test
    )
    base_brier = brier_score_loss(
        y_test, test_probs_baseline, sample_weight=w_test
    )

    # 2. Weighted Logistic Regression
    clf = LogisticRegression(random_state=42)
    t0 = time.perf_counter()
    clf.fit(X_train, y_train, sample_weight=w_train)
    train_time = time.perf_counter() - t0

    # -------------------------------------------------------------
    # SECTION 5: Tune Threshold on Validation (Notebook Step 8)
    # -------------------------------------------------------------
    val_probs = clf.predict_proba(X_val)[:, 1]
    best_threshold = 0.5
    best_val_ler = float("inf")

    for threshold in np.arange(0.05, 0.95, 0.05):
        val_preds = val_probs >= threshold
        val_ler = weighted_logical_error_rate(y_val, val_preds, w_val)
        if val_ler < best_val_ler:
            best_val_ler = val_ler
            best_threshold = float(round(threshold, 2))

    # -------------------------------------------------------------
    # SECTION 6: Evaluate Test Set & Save Files (Notebook Steps 9 & 11)
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    test_probs = clf.predict_proba(X_test)[:, 1]
    pred_time = time.perf_counter() - t0

    test_preds = test_probs >= best_threshold
    test_ler = weighted_logical_error_rate(y_test, test_preds, w_test)
    test_balanced_acc = balanced_accuracy_score(
        y_test, test_preds, sample_weight=w_test
    )
    test_brier = brier_score_loss(y_test, test_probs, sample_weight=w_test)

    results_root.mkdir(parents=True, exist_ok=True)

    # Save fitted model file (rubric requirement)
    joblib.dump(clf, results_root / "task_a_model.joblib")

    # Build predictions DataFrame for Test Set
    df_preds = pd.DataFrame(
        {
            "example_id": df.loc[test_mask, "example_id"].values,
            "model_id": "weighted_logistic_regression",
            "label": y_test.astype(bool),
            "prediction": test_preds.astype(bool),
            "probability": test_probs.astype(float),
            "split": "test",
        }
    )

    # Save / merge predictions.parquet
    predictions_path = results_root / "predictions.parquet"
    if predictions_path.exists():
        try:
            existing_preds = pd.read_parquet(predictions_path)
            other_preds = existing_preds[
                existing_preds["model_id"] != "weighted_logistic_regression"
            ]
            merged_preds = pd.concat([other_preds, df_preds], ignore_index=True)
        except Exception:
            merged_preds = df_preds
    else:
        merged_preds = df_preds
    merged_preds.to_parquet(predictions_path, index=False)

    # Build metrics dict
    task_a_metrics = {
        "task_a_syndrome_decoder": {
            "baseline_prior": {
                "weighted_logical_error_rate": float(base_ler_baseline),
                "weighted_balanced_accuracy": float(base_balanced_acc),
                "weighted_brier_score": float(base_brier),
                "training_time_seconds": float(base_train_time),
                "prediction_time_seconds": float(base_pred_time),
            },
            "weighted_logistic_regression": {
                "optimal_threshold": float(best_threshold),
                "weighted_logical_error_rate": float(test_ler),
                "weighted_balanced_accuracy": float(test_balanced_acc),
                "weighted_brier_score": float(test_brier),
                "training_time_seconds": float(train_time),
                "prediction_time_seconds": float(pred_time),
            },
        }
    }

    # Save / merge metrics.json
    metrics_path = results_root / "metrics.json"
    existing_metrics = {}
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                existing_metrics = json.load(f)
        except Exception:
            existing_metrics = {}
    existing_metrics.update(task_a_metrics)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(existing_metrics, f, indent=2)

    # Save / merge run.json
    run_path = results_root / "run.json"
    run_record = {}
    if run_path.exists():
        try:
            with open(run_path, "r", encoding="utf-8") as f:
                run_record = json.load(f)
        except Exception:
            run_record = {}

    manifest = read_release_manifest(
        read_lake_object(settings, MANIFEST_OBJECT)
    )
    run_record["data_release"] = {
        "name": manifest.get("release_name"),
        "version": manifest.get("bundle_version"),
        "date": manifest.get("release_date"),
    }
    run_record["code_revision_kind"], run_record["code_revision"] = (
        code_revision()
    )
    ml_input_hashes = run_record.get("ml_input_hashes", {})
    ml_input_hashes["ml/ml_syndrome_decoder_example.parquet"] = hashlib.sha256(
        syndrome_bytes
    ).hexdigest()
    try:
        google_bytes = read_lake_object(
            settings, "ml/ml_google_decoder_example.parquet"
        )
        ml_input_hashes["ml/ml_google_decoder_example.parquet"] = hashlib.sha256(
            google_bytes
        ).hexdigest()
    except Exception:
        pass
    run_record["ml_input_hashes"] = ml_input_hashes
    run_record["random_seed"] = 42

    feature_order = run_record.get("feature_order", {})
    feature_order["task_a_syndrome_decoder"] = [
        f"round_{r}_check_{c}" for r in range(4) for c in range(4)
    ]
    run_record["feature_order"] = feature_order

    split_rules = run_record.get("split_rules", {})
    split_rules["task_a_syndrome_decoder"] = (
        "syndrome_data_split: validation=0.0005, test=0.005, train=others"
    )
    run_record["split_rules"] = split_rules

    timings = run_record.get("timings", {})
    timings["task_a_training_seconds"] = float(train_time)
    timings["task_a_prediction_seconds"] = float(pred_time)
    run_record["timings"] = timings

    run_record["dependency_versions"] = {
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pa.__version__,
    }

    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(run_record, f, indent=2)

    # -------------------------------------------------------------
    # FINISH & RETURN
    # -------------------------------------------------------------
    result.input_count = len(df)
    result.output_count = len(df_preds)
    result.finish()
    return result
