"""Prepare QASMBench Silver tables and shared Part I evidence."""

from __future__ import annotations

from pathlib import Path

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import read_lake_object
from quantum_lake_student.models import StageResult
from quantum_lake_student.part1_results import (
    check_result,
    passed_checks,
    publish_part1_results,
)
from quantum_lake_student.qasmbench import (
    BRONZE_OBJECT,
    CIRCUIT_OBJECT,
    CONDITIONAL_CORRECTION_OBJECT,
    EXPECTED_QASM_MEMBERS,
    SOURCE_NAME,
    STABILIZER_CHECK_OBJECT,
    QasmBenchTables,
    build_qasmbench_tables,
)
from quantum_lake_student.source_validation import (
    MANIFEST_OBJECT,
    read_release_manifest,
    source_spec,
)


def _check_results(
    tables: QasmBenchTables,
) -> dict[str, dict[str, object]]:
    circuits = tables.circuits.to_pylist()
    stabilizers = tables.stabilizer_checks.to_pylist()
    corrections = tables.conditional_corrections.to_pylist()
    silver_ids = {
        row["source_record_id"]
        for rows in (circuits, stabilizers, corrections)
        for row in rows
    }
    trace_ids = set(tables.source_trace["source_record_id"].to_pylist())

    variants: dict[str, set[str]] = {}
    for row in circuits:
        variants.setdefault(row["benchmark_name"], set()).add(row["variant"])

    circuit_counts = {
        f"{row['benchmark_name']}:{row['variant']}": {
            "operations": row["operation_count"],
            "measurements": row["measurement_count"],
            "two_qubit_gates": row["two_qubit_gate_count"],
            "qubits": row["qubit_count"],
            "classical_bits": row["classical_bit_count"],
        }
        for row in circuits
    }
    qec_sm_ids = {
        row["circuit_id"]
        for row in circuits
        if row["benchmark_name"] == "qec_sm_n5"
    }
    stabilizer_counts = {
        circuit_id: sum(
            row["circuit_id"] == circuit_id for row in stabilizers
        )
        for circuit_id in qec_sm_ids
    }
    correction_counts = {
        circuit_id: sum(
            row["circuit_id"] == circuit_id for row in corrections
        )
        for circuit_id in qec_sm_ids
    }

    checks = passed_checks(
        "bronze_size_and_sha256",
        "archive_member_paths",
        "archive_member_names_unique",
        "archive_crc",
    )
    checks.update(
        {
            "qasm_member_set": check_result(
                tables.input_count == len(EXPECTED_QASM_MEMBERS),
                actual=tables.input_count,
            ),
            "openqasm_structure": check_result(tables.rejected_count == 0),
            "source_and_transpiled_variants": check_result(
                len(variants) == 3
                and all(
                    value == {"source", "transpiled"}
                    for value in variants.values()
                )
            ),
            "declared_qubit_counts": check_result(
                all(row["qubit_count"] == 5 for row in circuits),
                circuits=circuit_counts,
            ),
            "executed_measurement_counts": check_result(
                all(row["measurement_count"] == 5 for row in circuits)
            ),
            "operation_and_two_qubit_counts": check_result(
                all(
                    row["operation_count"] >= row["two_qubit_gate_count"]
                    for row in circuits
                ),
                circuits=circuit_counts,
            ),
            "explicit_stabilizer_checks": check_result(
                bool(stabilizer_counts)
                and set(stabilizer_counts.values()) == {2},
                rows=len(stabilizers),
            ),
            "syndrome_controlled_corrections": check_result(
                bool(correction_counts)
                and set(correction_counts.values()) == {3},
                rows=len(corrections),
            ),
            "row_reconciliation": check_result(
                tables.accepted_count + tables.rejected_count
                == tables.input_count
            ),
            "source_record_id_uniqueness": check_result(
                len(silver_ids) == tables.silver_row_count
            ),
            "source_trace_coverage": check_result(silver_ids == trace_ids),
        }
    )
    return checks


def prepare_qasmbench(
    settings: Settings,
    *,
    run_id: str,
    results_root: Path = Path("results/part1"),
) -> StageResult:
    """Build and publish all required QASMBench Silver tables."""
    result = StageResult(stage="prepare_qasmbench", run_id=run_id)
    manifest = read_release_manifest(
        read_lake_object(settings, MANIFEST_OBJECT)
    )
    spec = source_spec(
        manifest,
        SOURCE_NAME,
        expected_object=BRONZE_OBJECT,
    )
    tables = build_qasmbench_tables(
        read_lake_object(settings, BRONZE_OBJECT),
        spec=spec,
        run_id=run_id,
    )

    result.input_count = tables.input_count
    result.output_count = tables.silver_row_count
    publish_part1_results(
        settings,
        result,
        results_root,
        source_name=SOURCE_NAME,
        manifest=manifest,
        input_hashes={BRONZE_OBJECT: tables.input_sha256},
        silver_tables={
            CIRCUIT_OBJECT: tables.circuits,
            STABILIZER_CHECK_OBJECT: tables.stabilizer_checks,
            CONDITIONAL_CORRECTION_OBJECT: tables.conditional_corrections,
        },
        source_trace=tables.source_trace,
        data_issues=tables.data_issues,
        row_counts={
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
        checks=_check_results(tables),
    )
    return result


def run(run_id: str) -> StageResult:
    """Run the QASMBench Silver stage using environment-based settings."""
    return prepare_qasmbench(
        Settings.from_environment(),
        run_id=run_id,
    )
