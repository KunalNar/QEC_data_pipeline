"""Connect saved Part II predictions to the existing Part I source evidence."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import atomic_write, read_lake_object
from quantum_lake_student.stages.build_google_ml_table import GOOGLE_ML_OBJECT
from quantum_lake_student.stages.build_ml_tables import SYNDROME_ML_OBJECT
from quantum_lake_student.stages.run_part1 import _google_trace_example
from quantum_lake_student.stages.run_syndrome_qasm import _json_bytes, _trace_example


PREDICTION_COLUMNS = (
    "example_id", "model_id", "label", "prediction", "probability", "split",
)


def _select_prediction(
    predictions: list[dict],
    examples: dict[str, dict],
    label_column: str,
) -> dict:
    candidates = [
        row for row in predictions
        if row["split"] == "test" and row["example_id"] in examples
    ]
    if not candidates:
        raise ValueError(f"No test prediction available for {label_column}")
    selected = min(candidates, key=lambda row: (row["model_id"], row["example_id"]))
    example = examples[selected["example_id"]]
    if (
        type(selected["label"]) is not bool
        or type(selected["prediction"]) is not bool
        or selected["label"] != example[label_column]
        or selected["split"] != example["data_split"]
    ):
        raise ValueError("Prediction label or split does not match its ML example")
    return selected


def complete_traces(
    settings: Settings,
    *,
    results_root: Path = Path("results/part1"),
    part2_root: Path = Path("results/part2"),
) -> dict[str, object]:
    """Save one complete test prediction trace per dataset, without retraining."""
    run_path = results_root / "run.json"
    part1_run = json.loads(run_path.read_text(encoding="utf-8"))
    source_trace_bytes = (results_root / "source_trace.parquet").read_bytes()
    if (
        hashlib.sha256(source_trace_bytes).hexdigest()
        != part1_run["outputs"]["results/part1/source_trace.parquet"]["sha256"]
    ):
        raise ValueError("Source trace hash does not match the Part I run")
    part2_run_bytes = (part2_root / "run.json").read_bytes()
    part2_run = json.loads(part2_run_bytes)
    prediction_bytes = (part2_root / "predictions.parquet").read_bytes()
    prediction_table = pq.read_table(BytesIO(prediction_bytes))
    if not set(PREDICTION_COLUMNS).issubset(prediction_table.column_names):
        raise ValueError("Predictions lack required assignment columns")
    predictions = prediction_table.select(PREDICTION_COLUMNS).to_pylist()
    keys = [
        (row["model_id"], row["example_id"], row["split"])
        for row in predictions
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate model/example/split predictions")

    examples = {}
    for object_name in (SYNDROME_ML_OBJECT, GOOGLE_ML_OBJECT):
        data = read_lake_object(settings, object_name)
        digest = hashlib.sha256(data).hexdigest()
        if (
            part2_run.get("ml_input_hashes", {}).get(object_name) != digest
            or part1_run["outputs"][object_name]["sha256"] != digest
        ):
            raise ValueError(f"ML input hash mismatch: {object_name}")
        table = pq.read_table(BytesIO(data))
        examples[object_name] = {row["example_id"]: row for row in table.to_pylist()}

    syndrome_prediction = _select_prediction(
        predictions, examples[SYNDROME_ML_OBJECT], "logical_error_label",
    )
    google_prediction = _select_prediction(
        predictions, examples[GOOGLE_ML_OBJECT], "actual_observable_flip",
    )
    trace = _trace_example(settings, results_root, syndrome_prediction["example_id"])
    trace.pop("pending_full_part1")
    trace["scope"] = "all_three_sources"
    trace["syndrome"]["prediction"] = syndrome_prediction
    trace["google_qec"] = _google_trace_example(
        settings, results_root, google_prediction["example_id"],
    )
    trace["google_qec"]["prediction"] = google_prediction
    trace["part2_evidence"] = {
        "predictions_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
        "run_sha256": hashlib.sha256(part2_run_bytes).hexdigest(),
    }

    # Confirm the trace still points to the exact unchanged Bronze objects.
    bronze_hashes = {}
    for name in ("syndrome", "google_qec"):
        for source in trace[name]["bronze_sources"]:
            object_name = source["bronze_object"]
            if object_name not in bronze_hashes:
                bronze_hashes[object_name] = hashlib.sha256(
                    read_lake_object(settings, object_name)
                ).hexdigest()
            if source["input_sha256"] != bronze_hashes[object_name]:
                raise ValueError(f"Bronze input hash mismatch: {object_name}")

    data = _json_bytes(trace)
    part1_run["outputs"]["results/part1/trace_examples.json"] = {
        "rows": 2, "sha256": hashlib.sha256(data).hexdigest(),
    }
    part1_run["checks"]["gold_and_ml"]["prediction_traces"] = {"status": "passed"}
    atomic_write(results_root / "trace_examples.json", data)
    atomic_write(run_path, _json_bytes(part1_run))
    return trace
