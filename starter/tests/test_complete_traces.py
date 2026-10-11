"""Complete prediction traces without changing trained models or source data."""

import hashlib
import json
from copy import deepcopy
from io import BytesIO

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quantum_lake_student.stages import complete_traces as stage


def parquet_bytes(rows):
    buffer = BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), buffer)
    return buffer.getvalue()


def digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    part1 = tmp_path / "part1"
    part2 = tmp_path / "part2"
    part1.mkdir()
    part2.mkdir()
    objects = {
        stage.SYNDROME_ML_OBJECT: parquet_bytes([{
            "example_id": "syn-test", "logical_error_label": False,
            "data_split": "test",
        }]),
        stage.GOOGLE_ML_OBJECT: parquet_bytes([{
            "example_id": "google-test", "actual_observable_flip": True,
            "data_split": "test",
        }]),
        "bronze/syndrome.zip": b"unchanged syndrome archive",
        "bronze/google.zip": b"unchanged google archive",
    }
    predictions = [{
        "example_id": example_id, "model_id": model_id,
        "label": label, "prediction": False, "probability": probability,
        "split": "test",
    } for example_id, model_id, label, probability in (
        ("syn-test", "weighted_logistic_regression", False, 0.2),
        ("google-test", "C_mlp", True, 0.4),
    )]
    (part2 / "predictions.parquet").write_bytes(parquet_bytes(predictions))
    (part2 / "run.json").write_text(json.dumps({
        "ml_input_hashes": {name: digest(data) for name, data in objects.items()
                            if name.startswith("ml/")},
    }))
    (part1 / "source_trace.parquet").write_bytes(b"verified source trace")
    (part1 / "run.json").write_text(json.dumps({
        "outputs": {
            **{name: {"rows": 1, "sha256": digest(data)}
               for name, data in objects.items() if name.startswith("ml/")},
            "results/part1/source_trace.parquet": {
                "rows": 9, "sha256": digest(b"verified source trace"),
            },
            "results/part1/trace_examples.json": {"rows": 2, "sha256": "old"},
        },
        "checks": {"gold_and_ml": {}}, "run_id": "original-part1",
    }))
    (part1 / "trace_examples.json").write_text('{"original": true}')

    def source(object_name, member):
        return {"bronze_object": object_name, "archive_member": member,
                "input_sha256": digest(objects[object_name])}

    trace = {
        "scope": "syndrome_qasm_only", "pending_full_part1": ["google_qec"],
        "syndrome": {
            "example_id": "syn-test", "prediction": "pending Part II",
            "bronze_sources": [source("bronze/syndrome.zip", "syndrome.csv")],
        },
        "qasmbench": {"circuit_id": "preserved-circuit"},
    }
    google = {
        "example_id": "google-test", "prediction": "pending Part II",
        "bronze_sources": [source("bronze/google.zip", f"companion-{i}")
                           for i in range(8)],
    }
    calls = []

    def syndrome_trace(_settings, root, example_id):
        calls.append(("syndrome", root, example_id))
        return deepcopy(trace)

    def google_trace(_settings, root, example_id):
        calls.append(("google", root, example_id))
        return deepcopy(google)

    monkeypatch.setattr(stage, "read_lake_object", lambda _, name: objects[name])
    monkeypatch.setattr(stage, "_trace_example", syndrome_trace)
    monkeypatch.setattr(stage, "_google_trace_example", google_trace)
    return part1, part2, objects, predictions, calls, trace


def complete(evidence):
    part1, part2, *_ = evidence
    return stage.complete_traces(None, results_root=part1, part2_root=part2)


def test_completes_both_predictions_preserves_qasm_and_updates_hash(evidence):
    part1, part2, _, predictions, calls, _ = evidence
    trace = complete(evidence)
    assert trace["syndrome"]["prediction"] == predictions[0]
    assert trace["google_qec"]["prediction"] == predictions[1]
    assert trace["qasmbench"] == {"circuit_id": "preserved-circuit"}
    assert trace["scope"] == "all_three_sources"
    assert "pending_full_part1" not in trace
    assert calls == [("syndrome", part1, "syn-test"), ("google", part1, "google-test")]
    run = json.loads((part1 / "run.json").read_text())
    assert run["run_id"] == "original-part1"
    assert run["outputs"]["results/part1/trace_examples.json"] == {
        "rows": 2, "sha256": digest((part1 / "trace_examples.json").read_bytes()),
    }
    assert run["checks"]["gold_and_ml"]["prediction_traces"] == {"status": "passed"}
    assert trace["part2_evidence"]["predictions_sha256"] == digest(
        (part2 / "predictions.parquet").read_bytes()
    )


def test_repeat_is_identical_and_does_not_modify_part2(evidence):
    part1, part2, *_ = evidence
    originals = {path.name: path.read_bytes() for path in part2.iterdir()}
    complete(evidence)
    first = {path.name: path.read_bytes() for path in part1.iterdir()}
    complete(evidence)
    assert first == {path.name: path.read_bytes() for path in part1.iterdir()}
    assert originals == {path.name: path.read_bytes() for path in part2.iterdir()}


@pytest.mark.parametrize("problem", [
    "missing_google", "wrong_label", "wrong_split", "duplicate", "unknown_ids",
    "missing_column", "wrong_ml_hash", "wrong_part1_ml_hash", "wrong_bronze_hash",
    "wrong_trace_hash", "missing_predictions",
])
def test_invalid_evidence_leaves_previous_outputs_unchanged(evidence, problem):
    part1, part2, _, predictions, _, trace = evidence
    if problem == "missing_google":
        predictions.pop()
    elif problem == "wrong_label":
        predictions[0]["label"] = True
    elif problem == "wrong_split":
        predictions[0]["split"] = "train"
    elif problem == "duplicate":
        predictions.append(predictions[0])
    elif problem == "unknown_ids":
        predictions[0]["example_id"] = "missing"
    elif problem == "missing_column":
        for row in predictions:
            del row["model_id"]
    elif problem == "wrong_ml_hash":
        run = json.loads((part2 / "run.json").read_text())
        run["ml_input_hashes"][stage.SYNDROME_ML_OBJECT] = "wrong"
        (part2 / "run.json").write_text(json.dumps(run))
    elif problem == "wrong_part1_ml_hash":
        run = json.loads((part1 / "run.json").read_text())
        run["outputs"][stage.SYNDROME_ML_OBJECT]["sha256"] = "wrong"
        (part1 / "run.json").write_text(json.dumps(run))
    elif problem == "wrong_bronze_hash":
        trace["syndrome"]["bronze_sources"][0]["input_sha256"] = "wrong"
    elif problem == "wrong_trace_hash":
        (part1 / "source_trace.parquet").write_bytes(b"modified pointers")
    (part2 / "predictions.parquet").write_bytes(parquet_bytes(predictions))
    if problem == "missing_predictions":
        (part2 / "predictions.parquet").unlink()
    original_trace = (part1 / "trace_examples.json").read_bytes()
    original_run = (part1 / "run.json").read_bytes()
    with pytest.raises((ValueError, FileNotFoundError)):
        complete(evidence)
    assert (part1 / "trace_examples.json").read_bytes() == original_trace
    assert (part1 / "run.json").read_bytes() == original_run


def test_selection_is_stable_accepts_nullable_probability_and_multiple_models():
    examples = {"one": {"logical_error_label": False, "data_split": "test"}}
    predictions = [{
        "example_id": "one", "model_id": model_id, "label": False,
        "prediction": False, "probability": None, "split": "test",
    } for model_id in ("z-model", "a-model")]
    assert stage._select_prediction(predictions, examples, "logical_error_label")[
        "model_id"
    ] == "a-model"
    assert stage._select_prediction(
        predictions[::-1], examples, "logical_error_label",
    ) == predictions[1]
