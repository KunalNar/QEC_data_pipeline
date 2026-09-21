"""Prepare QASMBench Silver tables and shared Part I evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import read_lake_object, write_lake_object
from quantum_lake_student.evidence import parquet_bytes
from quantum_lake_student.models import StageResult
from quantum_lake_student.part1_results import (
    merge_evidence,
    update_row_counts,
    update_run_record,
)
from quantum_lake_student.qasmbench import (
    BRONZE_OBJECT,
    CIRCUIT_OBJECT,
    CONDITIONAL_CORRECTION_OBJECT,
    MANIFEST_OBJECT,
    QasmBenchTables,
    SOURCE_NAME,
    STABILIZER_CHECK_OBJECT,
    build_qasmbench_tables,
)
from quantum_lake_student.source_validation import (
    SourceObjectSpec,
    source_specs,
)


def _release_manifest(manifest_bytes: bytes) -> dict[str, object]:
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Release manifest is not valid UTF-8 JSON") from error
    if not isinstance(manifest, dict):
        raise ValueError("Release manifest root must be a JSON object")
    return manifest


def _qasmbench_spec(
    manifest: dict[str, object],
) -> SourceObjectSpec:
    try:
        spec = next(
            item for item in source_specs(manifest)
            if item.source_name == SOURCE_NAME
        )
    except StopIteration as error:
        raise ValueError("Release manifest has no qasmbench object") from error
    if spec.bronze_object != BRONZE_OBJECT:
        raise ValueError(
            "Release manifest has an unexpected qasmbench object path"
        )
    return spec


def _check_results(
    tables: QasmBenchTables,
) -> dict[str, dict[str, object]]:
    circuit_rows = tables.circuits.to_pylist()
    check_rows = tables.stabilizer_checks.to_pylist()
    correction_rows = tables.conditional_corrections.to_pylist()
    silver_ids = {
        row["source_record_id"]
        for rows in (circuit_rows, check_rows, correction_rows)
        for row in rows
    }
    trace_ids = set(
        tables.source_trace.column("source_record_id").to_pylist()
    )
    variants_by_benchmark: dict[str, set[str]] = {}
    for row in circuit_rows:
        variants_by_benchmark.setdefault(
            row["benchmark_name"], set()
        ).add(row["variant"])
    circuit_counts = {
        f"{row['benchmark_name']}:{row['variant']}": {
            "operations": row["operation_count"],
            "measurements": row["measurement_count"],
            "two_qubit_gates": row["two_qubit_gate_count"],
            "qubits": row["qubit_count"],
            "classical_bits": row["classical_bit_count"],
        }
        for row in circuit_rows
    }

    qec_sm_circuit_ids = {
        row["circuit_id"]
        for row in circuit_rows
        if row["benchmark_name"] == "qec_sm_n5"
    }
    check_counts = {
        circuit_id: sum(
            row["circuit_id"] == circuit_id for row in check_rows
        )
        for circuit_id in qec_sm_circuit_ids
    }
    correction_counts = {
        circuit_id: sum(
            row["circuit_id"] == circuit_id for row in correction_rows
        )
        for circuit_id in qec_sm_circuit_ids
    }

    return {
        "bronze_size_and_sha256": {"status": "passed"},
        "archive_member_paths": {"status": "passed"},
        "archive_member_names_unique": {"status": "passed"},
        "archive_crc": {"status": "passed"},
        "qasm_member_set": {
            "status": "passed",
            "actual": tables.input_count,
        },
        "openqasm_structure": {
            "status": (
                "passed" if tables.rejected_count == 0 else "failed"
            )
        },
        "source_and_transpiled_variants": {
            "status": (
                "passed"
                if all(
                    variants == {"source", "transpiled"}
                    for variants in variants_by_benchmark.values()
                )
                and len(variants_by_benchmark) == 3
                else "failed"
            )
        },
        "declared_qubit_counts": {
            "status": (
                "passed"
                if all(row["qubit_count"] == 5 for row in circuit_rows)
                else "failed"
            ),
            "circuits": circuit_counts,
        },
        "executed_measurement_counts": {
            "status": (
                "passed"
                if all(row["measurement_count"] == 5 for row in circuit_rows)
                else "failed"
            )
        },
        "operation_and_two_qubit_counts": {
            "status": (
                "passed"
                if all(
                    row["operation_count"] >= row["two_qubit_gate_count"]
                    for row in circuit_rows
                )
                else "failed"
            ),
            "circuits": circuit_counts,
        },
        "explicit_stabilizer_checks": {
            "status": (
                "passed"
                if check_counts
                and set(check_counts.values()) == {2}
                else "failed"
            ),
            "rows": len(check_rows),
        },
        "syndrome_controlled_corrections": {
            "status": (
                "passed"
                if correction_counts
                and set(correction_counts.values()) == {3}
                else "failed"
            ),
            "rows": len(correction_rows),
        },
        "row_reconciliation": {
            "status": (
                "passed"
                if (
                    tables.accepted_count + tables.rejected_count
                    == tables.input_count
                )
                else "failed"
            )
        },
        "source_record_id_uniqueness": {
            "status": (
                "passed"
                if len(silver_ids) == tables.silver_row_count
                else "failed"
            )
        },
        "source_trace_coverage": {
            "status": "passed" if silver_ids == trace_ids else "failed"
        },
    }


def prepare_qasmbench(
    settings: Settings,
    *,
    run_id: str,
    results_root: Path = Path("results/part1"),
) -> StageResult:
    """Build and publish all required QASMBench Silver tables."""
    result = StageResult(stage="prepare_qasmbench", run_id=run_id)

    manifest = _release_manifest(
        read_lake_object(settings, MANIFEST_OBJECT)
    )
    spec = _qasmbench_spec(manifest)
    archive_bytes = read_lake_object(settings, BRONZE_OBJECT)
    tables = build_qasmbench_tables(
        archive_bytes,
        expected_bytes=spec.expected_bytes,
        expected_sha256=spec.expected_sha256,
        run_id=run_id,
    )

    circuit_output = parquet_bytes(tables.circuits)
    check_output = parquet_bytes(tables.stabilizer_checks)
    correction_output = parquet_bytes(tables.conditional_corrections)
    write_lake_object(settings, CIRCUIT_OBJECT, circuit_output)
    write_lake_object(settings, STABILIZER_CHECK_OBJECT, check_output)
    write_lake_object(
        settings,
        CONDITIONAL_CORRECTION_OBJECT,
        correction_output,
    )

    (
        merged_trace,
        trace_output,
        merged_issues,
        issues_output,
    ) = merge_evidence(
        results_root,
        source_name=SOURCE_NAME,
        source_trace=tables.source_trace,
        data_issues=tables.data_issues,
    )
    update_row_counts(
        results_root,
        source_name=SOURCE_NAME,
        counts={
            "bronze_rows": tables.input_count,
            "circuit_rows": tables.circuits.num_rows,
            "stabilizer_check_rows": tables.stabilizer_checks.num_rows,
            "conditional_correction_rows": (
                tables.conditional_corrections.num_rows
            ),
            "silver_rows": tables.silver_row_count,
            "rejected_rows": tables.rejected_count,
            "issue_rows": tables.data_issues.num_rows,
        },
    )

    result.input_count = tables.input_count
    result.output_count = tables.silver_row_count
    result.issue_count = tables.data_issues.num_rows
    result.finish()

    update_run_record(
        results_root,
        run_id=result.run_id,
        stage=result.stage,
        release={
            "name": manifest.get("release_name"),
            "version": manifest.get("bundle_version"),
            "date": manifest.get("release_date"),
        },
        started_at=result.started_at.isoformat(),
        finished_at=result.finished_at.isoformat(),
        source_name=SOURCE_NAME,
        input_hashes={BRONZE_OBJECT: tables.input_sha256},
        outputs={
            CIRCUIT_OBJECT: {
                "rows": tables.circuits.num_rows,
                "sha256": hashlib.sha256(circuit_output).hexdigest(),
            },
            STABILIZER_CHECK_OBJECT: {
                "rows": tables.stabilizer_checks.num_rows,
                "sha256": hashlib.sha256(check_output).hexdigest(),
            },
            CONDITIONAL_CORRECTION_OBJECT: {
                "rows": tables.conditional_corrections.num_rows,
                "sha256": hashlib.sha256(correction_output).hexdigest(),
            },
            "results/part1/source_trace.parquet": {
                "rows": merged_trace.num_rows,
                "sha256": hashlib.sha256(trace_output).hexdigest(),
            },
            "results/part1/data_issues.parquet": {
                "rows": merged_issues.num_rows,
                "sha256": hashlib.sha256(issues_output).hexdigest(),
            },
        },
        checks=_check_results(tables),
    )
    return result


def run(run_id: str) -> StageResult:
    """Run the QASMBench portion of Silver preparation."""
    return prepare_qasmbench(
        Settings.from_environment(),
        run_id=run_id,
    )
