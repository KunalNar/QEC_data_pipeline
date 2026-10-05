"""Automated test suite for Part II Task A model training."""

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from quantum_lake_student.config import Settings
from quantum_lake_student.stages.train import train_task_a, run


def test_train_task_a_execution(tmp_path: Path) -> None:
    """Verify that train_task_a runs, generates valid files, and outputs correct schemas."""
    settings = Settings.from_environment()
    result = train_task_a(settings, run_id="test-task-a-1", results_root=tmp_path)

    # Check stage result
    assert result.stage == "train_task_a"
    assert result.input_count == 75598
    assert result.output_count == 20887
    assert result.finished_at is not None

    # Check predictions.parquet
    preds_path = tmp_path / "predictions.parquet"
    assert preds_path.exists()
    df_preds = pd.read_parquet(preds_path)
    assert len(df_preds) == 20887
    expected_cols = {"example_id", "model_id", "label", "prediction", "probability", "split"}
    assert set(df_preds.columns) == expected_cols
    assert (df_preds["split"] == "test").all()
    assert (df_preds["model_id"] == "weighted_logistic_regression").all()
    assert df_preds["probability"].between(0.0, 1.0).all()

    # Check metrics.json
    metrics_path = tmp_path / "metrics.json"
    assert metrics_path.exists()
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert "task_a_syndrome_decoder" in metrics
    task_a = metrics["task_a_syndrome_decoder"]
    assert "baseline_prior" in task_a
    assert "weighted_logistic_regression" in task_a

    base_metrics = task_a["baseline_prior"]
    clf_metrics = task_a["weighted_logistic_regression"]

    # Verify baseline logical error rate is ~0.103950 (testing the fix for the previous bug)
    assert pytest.approx(base_metrics["weighted_logical_error_rate"], abs=1e-4) == 0.103950
    assert pytest.approx(base_metrics["weighted_balanced_accuracy"], abs=1e-4) == 0.5
    assert pytest.approx(base_metrics["weighted_brier_score"], abs=1e-4) == 0.096910

    # Verify classifier metrics
    assert pytest.approx(clf_metrics["weighted_logical_error_rate"], abs=1e-4) == 0.104156
    assert pytest.approx(clf_metrics["optimal_threshold"], abs=1e-2) == 0.90
    assert clf_metrics["weighted_balanced_accuracy"] > 0.50
    assert clf_metrics["weighted_brier_score"] > 0.0
    assert clf_metrics["training_time_seconds"] > 0.0
    assert clf_metrics["prediction_time_seconds"] > 0.0

    # Check model file
    model_path = tmp_path / "task_a_model.joblib"
    assert model_path.exists()
    loaded_model = joblib.load(model_path)
    assert hasattr(loaded_model, "predict")

    # Check run.json
    run_path = tmp_path / "run.json"
    assert run_path.exists()
    with open(run_path, "r", encoding="utf-8") as f:
        run_info = json.load(f)
    assert run_info["data_release"]["name"] == "quantum-data-core"
    assert run_info["code_revision"]
    assert run_info["random_seed"] == 42
    assert "task_a_syndrome_decoder" in run_info["feature_order"]
    assert len(run_info["feature_order"]["task_a_syndrome_decoder"]) == 16
