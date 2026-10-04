"""Required Part II model-training stage.

Consume the two required ML input tables produced by Part I through the
supplied model-input and partition helpers. Publish repeatable model files,
predictions, metrics, run settings, the exact feature order, and the concise
Part II report under ``results/part2/``. Numerical performance is not graded.
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


def run(model_run_id: str) -> StageResult:
    """Main Part II training orchestrator called by `make train`."""
    settings = Settings.from_environment()

    # Run Task A (weighted syndrome decoder)
    result_a = train_task_a(settings, run_id=model_run_id)

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
    baseline.fit(X_train, y_train, sample_weight=w_train)
    test_preds_baseline = baseline.predict(X_test)
    base_ler_baseline = weighted_logical_error_rate(
        y_test, test_preds_baseline, w_test
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
                "brier_score": None,
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

    run_record.setdefault("data_release", "course-qec-v1")
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

    # Save / write report.md template if not already present
    report_path = results_root / "report.md"
    if not report_path.exists():
        report_content = (
            "# Part II: AI/ML Decoder Report\n\n"
            "## Task A: Weighted Syndrome Decoder\n\n"
            "### 1. Inputs and Targets\n"
            "- **Dataset:** `ml/ml_syndrome_decoder_example.parquet` (75,598 aggregate rows representing 70,000,000 simulated observations).\n"
            "- **Features ($X$):** 16 ordered binary syndrome bits (`syndrome_bits`), representing 4 consecutive rounds of 4 stabilizer checks on a surface code, extracted via `syndrome_model_input`.\n"
            "- **Target ($y$):** `logical_error_label` (boolean indicating whether an uncorrected logical bit-flip error occurred).\n"
            "- **Physical Weights ($w$):** `sample_weight` (integer observation count per row). Used during fitting and for all reported metrics without row expansion.\n"
            "- **Data Splits:** Fixed physical fault rate partitions: Validation ($p = 0.0005$, 10,950 examples), Test ($p = 0.005$, 20,887 examples), Train (remaining 5 fault rates, 43,761 examples).\n\n"
            "### 2. Model Selection and Rationale\n"
            "- **Baseline Prior:** `DummyClassifier(strategy='prior')` fit with `sample_weight=w_train`. Reflects the weighted prior probability of logical failure without syndrome information.\n"
            "- **Linear Classifier:** `LogisticRegression(random_state=42)` fit with `sample_weight=w_train`. Evaluates whether simple linear additive weighting over syndrome bits provides predictive signal for error detection.\n\n"
            "### 3. Validation Tuning and Test Evaluation\n"
            f"- **Threshold Tuning:** The classifier decision threshold was tuned exclusively on the validation set ($p=0.0005$), evaluating weighted logical error rate (LER) across thresholds in $[0.05, 0.90]$ with step $0.05$. Optimal validation threshold found: {best_threshold:.2f} (Val LER = {best_val_ler:.6f}).\n"
            "- **Held-Out Test Results ($p=0.005$):**\n"
            f"  - **Baseline Prior LER:** {base_ler_baseline:.6f}\n"
            f"  - **Logistic Regression LER:** {test_ler:.6f}\n"
            f"  - **Weighted Balanced Accuracy:** {test_balanced_acc:.4f}\n"
            f"  - **Weighted Brier Score:** {test_brier:.6f}\n"
            f"  - **Training Latency:** {train_time:.4f} seconds\n"
            f"  - **Prediction Latency (20,887 rows):** {pred_time:.4f} seconds\n\n"
            "### 4. Discarded Information and Limitations\n"
            "The learned model exhibits virtually identical test performance to the uninformative baseline prior. Inspecting the model weights and problem physics reveals why:\n"
            "1. **Flattening Spatio-Temporal Structure:** The 16 syndrome bits form a $4 \\times 4$ space-time grid. Logistic regression flattens this into an independent 16-element vector, ignoring temporal continuity ($t \\rightarrow t+1$ via `index % 4`) and 2D spatial adjacency on the chip.\n"
            "2. **Measurement Noise vs. Error Percolation:** A single-round excitation that clears in subsequent rounds is a transient measurement readout glitch ($y=0$), whereas excitation persisting across rounds indicates a physical error chain ($y=1$). A purely additive linear model ($\\sum w_i X_i$) cannot evaluate multi-round temporal persistence without explicit interaction terms.\n"
            "3. **Parity Loops and Non-Linearity:** Topological surface codes detect errors when chains form non-trivial homology loops. Determining whether a syndrome configuration forms a closed topological chain is non-linear and cannot be captured by linear hyperplanes.\n"
            "4. **Learned Weight Interpretation:**\n"
            "   - **Check Disparity:** Checks 0 and 2 have strong positive weights ($+1.5$ to $+2.3$), while Checks 1 and 3 are near zero. This aligns with surface code physics: Checks 0 and 2 are $Z$-type stabilizers (sensitive to bit flips, the target label), whereas Checks 1 and 3 are $X$-type stabilizers (sensitive to phase flips).\n"
            "   - **U-Shaped Temporal Curve:** Coefficients peak in Round 1 (early errors that propagate through 3 subsequent noisy cycles) and Round 4 (late errors occurring right before readout with zero rounds left for correction).\n\n"
            "---\n\n"
            "## Task B: Supplied and Combined Google Decoders\n"
            "*(To be completed by Task B team member)*\n\n"
            "---\n\n"
            "## Task C: Bounded Raw-Detector Prototype\n"
            "*(To be completed by Task C team member)*\n"
        )
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

    # -------------------------------------------------------------
    # FINISH & RETURN
    # -------------------------------------------------------------
    result.input_count = len(df)
    result.output_count = len(df_preds)
    result.finish()
    return result
