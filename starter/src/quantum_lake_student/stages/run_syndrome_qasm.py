"""Repeatable Part I checkpoint for syndromes and QASMBench only."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, datetime
from importlib.resources import files
from io import BytesIO, StringIO
from pathlib import Path

import pyarrow.parquet as pq

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import (
    atomic_write,
    postgres_connection,
    read_lake_object,
)
from quantum_lake_student.part1_results import code_revision
from quantum_lake_student.qasmbench import (
    CIRCUIT_OBJECT,
    CONDITIONAL_CORRECTION_OBJECT,
    STABILIZER_CHECK_OBJECT,
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
from quantum_lake_student.stages.load_qasmbench_gold import (
    QasmGoldRows,
    qasm_gold_rows,
    replace_qasm_gold,
)
from quantum_lake_student.stages.prepare_qasmbench import prepare_qasmbench
from quantum_lake_student.stages.prepare_syndromes import prepare_syndromes
from quantum_lake_student.syndromes import SILVER_OBJECT


RESULTS_ROOT = Path("results/part1/syndrome_qasm")
ANALYSES = {
    "syndrome_fault_rate": "analysis_syndrome_fault_rate.sql",
    "syndrome_patterns": "analysis_syndrome_patterns.sql",
    "qasmbench": "analysis_qasmbench.sql",
}


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _file_output(path: Path, data: bytes, rows: int) -> dict[str, object]:
    atomic_write(path, data)
    return {"rows": rows, "sha256": hashlib.sha256(data).hexdigest()}


def _analysis_outputs(settings: Settings, root: Path) -> dict[str, dict[str, object]]:
    outputs = {}
    with postgres_connection(settings) as connection:
        connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        for name, sql_file in ANALYSES.items():
            query = files("quantum_lake_student").joinpath(
                f"sql/{sql_file}"
            ).read_text(encoding="utf-8")
            with connection.cursor() as cursor:
                cursor.execute(query)
                header = [column.name for column in cursor.description]
                rows = cursor.fetchall()
            buffer = StringIO(newline="")
            writer = csv.writer(buffer)
            writer.writerow(header)
            writer.writerows(rows)
            path = root / "analysis" / f"{name}.csv"
            outputs[path.as_posix()] = (
                _file_output(path, buffer.getvalue().encode("utf-8"), len(rows))
            )
    return outputs


def _trace_example(settings: Settings, root: Path) -> dict[str, object]:
    trace_rows = pq.read_table(root / "source_trace.parquet").to_pylist()
    trace_by_id = {row["source_record_id"]: row for row in trace_rows}
    ml_table = pq.read_table(BytesIO(read_lake_object(settings, SYNDROME_ML_OBJECT)))
    ml_examples = {row["example_id"]: row for row in ml_table.to_pylist()}

    with postgres_connection(settings) as connection:
        connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        example_id, experiment_id, label, weight = connection.execute(
            "SELECT example_id, experiment_id, logical_error_label, sample_weight "
            "FROM gold.syndrome_ml_example ORDER BY example_id LIMIT 1"
        ).fetchone()
        observations = [
            {
                "source_record_id": source_id,
                "syndrome_hex": bits_hex,
                "quantity": quantity,
            }
            for source_id, bits_hex, quantity in connection.execute(
                "SELECT o.source_record_id, encode(o.syndrome_bits, 'hex'), "
                "o.quantity FROM gold.syndrome_ml_source AS link "
                "JOIN gold.syndrome_observation AS o USING (source_record_id) "
                "WHERE link.example_id = %s ORDER BY o.source_record_id",
                (example_id,),
            )
        ]
        source_ids = [row["source_record_id"] for row in observations]
        circuit_id, circuit_source_id = connection.execute(
            "SELECT circuit_id, source_record_id FROM gold.qasm_circuit "
            "WHERE benchmark_name = 'qec_sm_n5' "
            "ORDER BY circuit_id LIMIT 1"
        ).fetchone()
        (check_source_id,) = connection.execute(
            "SELECT source_record_id FROM gold.qasm_stabilizer_check "
            "WHERE circuit_id = %s ORDER BY check_id LIMIT 1",
            (circuit_id,),
        ).fetchone()
        (correction_source_id,) = connection.execute(
            "SELECT source_record_id FROM gold.qasm_conditional_correction "
            "WHERE circuit_id = %s ORDER BY condition_value LIMIT 1",
            (circuit_id,),
        ).fetchone()
        check_id, ancilla, syndrome_bit = connection.execute(
            "SELECT check_id, ancilla_qubit, syndrome_bit "
            "FROM gold.qasm_stabilizer_check WHERE source_record_id = %s",
            (check_source_id,),
        ).fetchone()
        condition_register, condition_value, gate, target = connection.execute(
            "SELECT condition_register, condition_value, gate, target_qubit "
            "FROM gold.qasm_conditional_correction WHERE source_record_id = %s",
            (correction_source_id,),
        ).fetchone()

    ml_row = ml_examples.get(example_id)
    if ml_row is None or (
        ml_row["experiment_id"] != experiment_id
        or ml_row["logical_error_label"] != label
        or ml_row["sample_weight"] != weight
    ):
        raise RuntimeError("Syndrome Gold example missing from ML Parquet")
    selected_ids = source_ids + [
        circuit_source_id, check_source_id, correction_source_id
    ]
    if (
        not source_ids
        or sum(row["quantity"] for row in observations) != weight
        or any(source_id not in trace_by_id for source_id in selected_ids)
    ):
        raise RuntimeError("Gold source row missing from scoped source trace")
    return {
        "scope": "syndrome_qasm_only",
        "syndrome": {
            "example_id": example_id,
            "experiment_id": experiment_id,
            "logical_error_label": label,
            "sample_weight": weight,
            "ml_data_split": ml_row["data_split"],
            "ml_syndrome_hex": ml_row["syndrome_bits"].hex(),
            "gold_observations": observations,
            "bronze_sources": [trace_by_id[source_id] for source_id in source_ids],
            "prediction": "pending Part II",
        },
        "qasmbench": {
            "circuit_id": circuit_id,
            "circuit": trace_by_id[circuit_source_id],
            "stabilizer_check": {
                "check_id": check_id,
                "ancilla_qubit": ancilla,
                "syndrome_bit": syndrome_bit,
                "source": trace_by_id[check_source_id],
            },
            "conditional_correction": {
                "condition_register": condition_register,
                "condition_value": condition_value,
                "gate": gate,
                "target_qubit": target,
                "source": trace_by_id[correction_source_id],
            },
        },
        "pending_full_part1": "Google Gold, Google ML export, and Google trace",
    }


def load_gold_pair(
    settings: Settings,
    syndrome_rows: SyndromeGoldRows,
    qasm_rows: QasmGoldRows,
) -> None:
    """Commit both Gold models together or leave both previous versions."""
    with postgres_connection(settings) as connection:
        replace_syndrome_gold(connection, syndrome_rows)
        replace_qasm_gold(connection, qasm_rows)


def _check_source_trace(root: Path, silver_ids: set[str]) -> int:
    trace = pq.read_table(root / "source_trace.parquet")
    trace_ids = trace["source_record_id"].to_pylist()
    if len(trace_ids) != len(silver_ids) or set(trace_ids) != silver_ids:
        raise RuntimeError("Scoped source trace does not cover exactly both Silver sources")
    return trace.num_rows


def run_syndrome_qasm(
    settings: Settings,
    *,
    run_id: str,
    results_root: Path = RESULTS_ROOT,
) -> dict[str, object]:
    """Build two sources from Bronze and publish a clearly scoped checkpoint."""
    started_at = datetime.now(UTC).isoformat()
    syndrome = prepare_syndromes(settings, run_id=run_id, results_root=results_root)
    qasm = prepare_qasmbench(settings, run_id=run_id, results_root=results_root)

    syndrome_table = pq.read_table(BytesIO(read_lake_object(settings, SILVER_OBJECT)))
    qasm_tables = [
        pq.read_table(BytesIO(read_lake_object(settings, object_name)))
        for object_name in (
            CIRCUIT_OBJECT,
            STABILIZER_CHECK_OBJECT,
            CONDITIONAL_CORRECTION_OBJECT,
        )
    ]
    syndrome_rows = syndrome_gold_rows(syndrome_table)
    qasm_rows = qasm_gold_rows(*qasm_tables)
    silver_ids = {
        row[0] for row in syndrome_rows.observations
    } | {
        row[1] for row in qasm_rows.circuits
    } | {
        row[0] for row in qasm_rows.checks + qasm_rows.corrections
    }
    trace_count = _check_source_trace(results_root, silver_ids)

    load_gold_pair(settings, syndrome_rows, qasm_rows)

    ml = build_syndrome_ml_table(settings, run_id=run_id)
    ml_bytes = read_lake_object(settings, SYNDROME_ML_OBJECT)
    analysis_outputs = _analysis_outputs(settings, results_root)
    trace_example = _trace_example(settings, results_root)
    trace_path = results_root / "trace_examples.json"
    trace_output = _file_output(trace_path, _json_bytes(trace_example), 2)

    counts_path = results_root / "row_counts.json"
    counts = json.loads(counts_path.read_text(encoding="utf-8"))
    counts["qec_syndromes"].update({
        "gold_experiment_rows": len(syndrome_rows.experiments),
        "gold_pattern_rows": len(syndrome_rows.patterns),
        "gold_observation_rows": len(syndrome_rows.observations),
        "ml_example_rows": ml.output_count,
    })
    counts["qasmbench"].update({
        "gold_circuit_rows": len(qasm_rows.circuits),
        "gold_register_rows": len(qasm_rows.registers),
        "gold_stabilizer_check_rows": len(qasm_rows.checks),
        "gold_check_data_qubit_rows": len(qasm_rows.participants),
        "gold_conditional_correction_rows": len(qasm_rows.corrections),
    })
    atomic_write(counts_path, _json_bytes(counts))

    run_path = results_root / "run.json"
    record = json.loads(run_path.read_text(encoding="utf-8"))
    outputs = {key: value for key, value in record["outputs"].items()
               if key.startswith("silver/")}
    for name, size in (
        ("syndrome_experiment", len(syndrome_rows.experiments)),
        ("syndrome_pattern", len(syndrome_rows.patterns)),
        ("syndrome_observation", len(syndrome_rows.observations)),
        ("qasm_circuit", len(qasm_rows.circuits)),
        ("qasm_register", len(qasm_rows.registers)),
        ("qasm_stabilizer_check", len(qasm_rows.checks)),
        ("qasm_check_data_qubit", len(qasm_rows.participants)),
        ("qasm_conditional_correction", len(qasm_rows.corrections)),
    ):
        outputs[f"gold.{name}"] = {"rows": size}
    outputs[SYNDROME_ML_OBJECT] = {
        "rows": ml.output_count,
        "sha256": hashlib.sha256(ml_bytes).hexdigest(),
    }
    for name, size in (
        ("source_trace.parquet", trace_count),
        ("data_issues.parquet", syndrome.issue_count + qasm.issue_count),
    ):
        path = results_root / name
        data = path.read_bytes()
        outputs[f"results/part1/syndrome_qasm/{name}"] = {
            "rows": size, "sha256": hashlib.sha256(data).hexdigest()
        }
    outputs["results/part1/syndrome_qasm/trace_examples.json"] = trace_output
    outputs.update(analysis_outputs)
    revision_kind, revision = code_revision()
    record["checks"]["gold_and_ml"] = {
        "syndrome_silver_to_gold": {"status": "passed"},
        "qasmbench_silver_to_gold": {"status": "passed"},
        "syndrome_gold_to_ml": {"status": "passed"},
        "scoped_source_trace": {"status": "passed"},
    }
    record.update({
        "stage": "run_syndrome_qasm",
        "scope": ["qec_syndromes", "qasmbench"],
        "status": "partial_part1_google_pending",
        "run_id": run_id,
        "code_revision_kind": revision_kind,
        "code_revision": revision,
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "outputs": outputs,
    })
    atomic_write(run_path, _json_bytes(record))
    return record
