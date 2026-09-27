"""Run all three sources through Silver, Gold, analyses, and ML handoff."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import (
    atomic_write,
    postgres_connection,
    read_lake_object,
)
from quantum_lake_student.google_qec import (
    EXPERIMENT_PATH,
    SHOT_PATH,
    SHOT_TRACE_MEMBER_COUNT,
)
from quantum_lake_student.part1_results import code_revision
from quantum_lake_student.qasmbench import (
    CIRCUIT_OBJECT,
    CONDITIONAL_CORRECTION_OBJECT,
    STABILIZER_CHECK_OBJECT,
)
from quantum_lake_student.stages.build_google_ml_table import (
    GOOGLE_ML_OBJECT,
    build_google_ml_table,
)
from quantum_lake_student.stages.build_ml_tables import (
    SYNDROME_ML_OBJECT,
    build_syndrome_ml_table,
)
from quantum_lake_student.stages.load_postgres import (
    SyndromeGoldRows,
    replace_syndrome_gold,
    syndrome_gold_rows,
)
from quantum_lake_student.stages.load_postgres_google import (
    GoogleGoldRows,
    google_gold_rows,
    replace_google_gold,
)
from quantum_lake_student.stages.load_qasmbench_gold import (
    QasmGoldRows,
    qasm_gold_rows,
    replace_qasm_gold,
)
from quantum_lake_student.stages.prepare_qasmbench import prepare_qasmbench
from quantum_lake_student.stages.prepare_syndromes import prepare_syndromes
from quantum_lake_student.stages.run_syndrome_qasm import (
    ANALYSES,
    _analysis_outputs,
    _file_output,
    _json_bytes,
    _trace_example,
)
from quantum_lake_student.stages.silvergoogle import prepare_google_data
from quantum_lake_student.syndromes import SILVER_OBJECT


RESULTS_ROOT = Path("results/part1")
ALL_ANALYSES = {
    **ANALYSES,
    "google_decoders": "analysis_google_decoders.sql",
}


def load_gold_all(
    settings: Settings,
    syndrome_rows: SyndromeGoldRows,
    qasm_rows: QasmGoldRows,
    google_rows: GoogleGoldRows,
) -> None:
    """Commit all three Gold models together or retain the previous state."""
    with postgres_connection(settings) as connection:
        replace_syndrome_gold(connection, syndrome_rows)
        replace_qasm_gold(connection, qasm_rows)
        replace_google_gold(connection, google_rows)


def _trace_count(root: Path, expected_ids: set[str], google_rows: GoogleGoldRows) -> int:
    trace = pq.read_table(
        root / "source_trace.parquet",
        columns=["source_record_id", "source_name"],
    )
    actual_ids = set(pc.unique(trace["source_record_id"]).to_pylist())
    if actual_ids != expected_ids:
        raise RuntimeError("Part I source trace does not cover exactly the Silver rows")
    google_count = pc.sum(
        pc.equal(trace["source_name"], "google_qec")
    ).as_py()
    expected_google_count = len(google_rows.experiments) + (
        len(google_rows.shots) * SHOT_TRACE_MEMBER_COUNT
    )
    if google_count != expected_google_count:
        raise RuntimeError("Google companion-file trace rows do not reconcile")
    return trace.num_rows


def _google_trace_example(settings: Settings, root: Path) -> dict[str, object]:
    ml = pq.read_table(BytesIO(read_lake_object(settings, GOOGLE_ML_OBJECT)))
    if not ml.num_rows:
        raise RuntimeError("Google ML table is empty")
    example = ml.slice(0, 1).to_pylist()[0]
    example_id = example["example_id"]
    with postgres_connection(settings) as connection:
        connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        gold_row = connection.execute(
            "SELECT s.source_record_id, s.experiment_id, s.shot_index, "
            "e.distance, s.detector_event_count, encode(s.detector_bits, 'hex') "
            "FROM gold.google_shot AS s "
            "JOIN gold.google_experiment AS e USING (experiment_id) "
            "WHERE s.source_record_id = %s",
            (example_id,),
        ).fetchone()
        link = connection.execute(
            "SELECT source_record_id FROM gold.google_ml_source "
            "WHERE example_id = %s",
            (example_id,),
        ).fetchone()
    if gold_row is None or link != (example_id,) or (
        gold_row[1] != example["experiment_id"]
        or gold_row[2] != example["shot_index"]
        or gold_row[3] != example["distance"]
        or gold_row[4] != example["detector_event_count"]
        or gold_row[5] != example["detector_bits"].hex()
    ):
        raise RuntimeError("Google ML example does not resolve to its Gold shot")
    source_rows = pq.read_table(
        root / "source_trace.parquet",
        filters=[("source_record_id", "=", example_id)],
    ).to_pylist()
    if len(source_rows) != SHOT_TRACE_MEMBER_COUNT:
        raise RuntimeError("Google Gold shot lacks companion-file trace rows")
    return {
        "example_id": example_id,
        "experiment_id": example["experiment_id"],
        "shot_index": example["shot_index"],
        "ml_data_split": example["data_split"],
        "gold_source_record_id": gold_row[0],
        "bronze_sources": source_rows,
        "prediction": "pending Part II",
    }


def run_part1(
    settings: Settings,
    *,
    run_id: str,
    results_root: Path = RESULTS_ROOT,
) -> dict[str, object]:
    """Rebuild the three-source Part I handoff with one Gold transaction."""
    started_at = datetime.now(UTC).isoformat()
    syndrome = prepare_syndromes(settings, run_id=run_id, results_root=results_root)
    qasm = prepare_qasmbench(settings, run_id=run_id, results_root=results_root)
    google = prepare_google_data(settings, run_id=run_id, results_root=results_root)

    syndrome_table = pq.read_table(BytesIO(read_lake_object(settings, SILVER_OBJECT)))
    qasm_tables = [
        pq.read_table(BytesIO(read_lake_object(settings, name)))
        for name in (
            CIRCUIT_OBJECT,
            STABILIZER_CHECK_OBJECT,
            CONDITIONAL_CORRECTION_OBJECT,
        )
    ]
    google_experiments = pq.read_table(
        BytesIO(read_lake_object(settings, EXPERIMENT_PATH))
    )
    google_shots = pq.read_table(BytesIO(read_lake_object(settings, SHOT_PATH)))
    syndrome_rows = syndrome_gold_rows(syndrome_table)
    qasm_rows = qasm_gold_rows(*qasm_tables)
    google_rows = google_gold_rows(google_experiments, google_shots)
    silver_ids = (
        {row[0] for row in syndrome_rows.observations}
        | {row[1] for row in qasm_rows.circuits}
        | {row[0] for row in qasm_rows.checks + qasm_rows.corrections}
        | {row[1] for row in google_rows.experiments}
        | {row[0] for row in google_rows.shots}
    )
    expected_id_count = (
        len(syndrome_rows.observations)
        + len(qasm_rows.circuits)
        + len(qasm_rows.checks)
        + len(qasm_rows.corrections)
        + len(google_rows.experiments)
        + len(google_rows.shots)
    )
    if len(silver_ids) != expected_id_count:
        raise RuntimeError("Source IDs overlap across the three datasets")
    trace_count = _trace_count(results_root, silver_ids, google_rows)
    load_gold_all(settings, syndrome_rows, qasm_rows, google_rows)

    syndrome_ml = build_syndrome_ml_table(settings, run_id=run_id)
    google_ml = build_google_ml_table(settings, run_id=run_id)
    analyses = _analysis_outputs(settings, results_root, ALL_ANALYSES)
    trace_example = _trace_example(settings, results_root)
    trace_example.pop("pending_full_part1")
    trace_example["scope"] = "all_three_sources"
    trace_example["google_qec"] = _google_trace_example(settings, results_root)
    trace_output = _file_output(
        results_root / "trace_examples.json",
        _json_bytes(trace_example),
        2,
    )

    counts_path = results_root / "row_counts.json"
    counts = json.loads(counts_path.read_text(encoding="utf-8"))
    counts["qec_syndromes"].update({
        "gold_experiment_rows": len(syndrome_rows.experiments),
        "gold_pattern_rows": len(syndrome_rows.patterns),
        "gold_observation_rows": len(syndrome_rows.observations),
        "ml_example_rows": syndrome_ml.output_count,
    })
    counts["qasmbench"].update({
        "gold_circuit_rows": len(qasm_rows.circuits),
        "gold_register_rows": len(qasm_rows.registers),
        "gold_stabilizer_check_rows": len(qasm_rows.checks),
        "gold_check_data_qubit_rows": len(qasm_rows.participants),
        "gold_conditional_correction_rows": len(qasm_rows.corrections),
    })
    counts["google_qec"].update({
        "gold_experiment_rows": len(google_rows.experiments),
        "gold_shot_rows": len(google_rows.shots),
        "gold_decoder_rows": 4,
        "gold_prediction_rows": len(google_rows.predictions),
        "ml_example_rows": google_ml.output_count,
    })
    atomic_write(counts_path, _json_bytes(counts))

    run_path = results_root / "run.json"
    record = json.loads(run_path.read_text(encoding="utf-8"))
    outputs = {name: value for name, value in record["outputs"].items()
               if name.startswith("silver/")}
    for name, size in (
        ("syndrome_experiment", len(syndrome_rows.experiments)),
        ("syndrome_pattern", len(syndrome_rows.patterns)),
        ("syndrome_observation", len(syndrome_rows.observations)),
        ("qasm_circuit", len(qasm_rows.circuits)),
        ("qasm_register", len(qasm_rows.registers)),
        ("qasm_stabilizer_check", len(qasm_rows.checks)),
        ("qasm_check_data_qubit", len(qasm_rows.participants)),
        ("qasm_conditional_correction", len(qasm_rows.corrections)),
        ("google_experiment", len(google_rows.experiments)),
        ("google_shot", len(google_rows.shots)),
        ("google_decoder", 4),
        ("google_decoder_prediction", len(google_rows.predictions)),
    ):
        outputs[f"gold.{name}"] = {"rows": size}
    for name, size in (
        (SYNDROME_ML_OBJECT, syndrome_ml.output_count),
        (GOOGLE_ML_OBJECT, google_ml.output_count),
    ):
        value = read_lake_object(settings, name)
        outputs[name] = {"rows": size, "sha256": hashlib.sha256(value).hexdigest()}
    for name, size in (
        ("source_trace.parquet", trace_count),
        ("data_issues.parquet", syndrome.issue_count + qasm.issue_count + google.issue_count),
    ):
        data = (results_root / name).read_bytes()
        outputs[f"results/part1/{name}"] = {
            "rows": size, "sha256": hashlib.sha256(data).hexdigest()
        }
    outputs["results/part1/trace_examples.json"] = trace_output
    outputs.update(analyses)
    revision_kind, revision = code_revision()
    record["checks"]["gold_and_ml"] = {
        "all_three_gold_loads_atomic": {"status": "passed"},
        "syndrome_gold_to_ml": {"status": "passed"},
        "google_gold_to_ml": {"status": "passed"},
        "source_trace": {"status": "passed"},
    }
    record.update({
        "stage": "run_part1",
        "scope": ["qec_syndromes", "qasmbench", "google_qec"],
        "status": "complete",
        "run_id": run_id,
        "code_revision_kind": revision_kind,
        "code_revision": revision,
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "outputs": outputs,
    })
    atomic_write(run_path, _json_bytes(record))
    return record
