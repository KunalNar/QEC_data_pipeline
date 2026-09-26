"""Load QASMBench circuit relationships from the three Silver tables."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib.resources import files
from io import BytesIO

import pyarrow as pa
import pyarrow.parquet as pq
import psycopg

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import postgres_connection, read_lake_object
from quantum_lake_student.models import StageResult
from quantum_lake_student.qasmbench import (
    CIRCUIT_OBJECT,
    CIRCUIT_SCHEMA,
    CONDITIONAL_CORRECTION_OBJECT,
    CONDITIONAL_CORRECTION_SCHEMA,
    STABILIZER_CHECK_OBJECT,
    STABILIZER_CHECK_SCHEMA,
)


BIT_REFERENCE = re.compile(r"([A-Za-z_][A-Za-z_0-9]*)\[(\d+)\]")


@dataclass
class QasmGoldRows:
    circuits: list[tuple]
    registers: list[tuple]
    checks: list[tuple]
    participants: list[tuple]
    corrections: list[tuple]


def _check_table(table: pa.Table, schema: pa.Schema, name: str) -> None:
    if not table.schema.equals(schema, check_metadata=False):
        raise ValueError(f"Unexpected QASMBench {name} Silver schema")
    if any(column.null_count for column in table.columns):
        raise ValueError(f"QASMBench {name} Silver contains null values")
    if not table.num_rows:
        raise ValueError(f"QASMBench {name} Silver is empty")


def _require_bit(
    reference: str,
    registers: dict[str, tuple[str, int]],
    kind: str,
) -> None:
    match = BIT_REFERENCE.fullmatch(reference) if isinstance(reference, str) else None
    if match is None:
        raise ValueError(f"Invalid QASM bit reference: {reference}")
    name, index = match.group(1), int(match.group(2))
    declaration = registers.get(name)
    if declaration is None or declaration[0] != kind or index >= declaration[1]:
        raise ValueError(f"Undeclared or out-of-range QASM bit: {reference}")


def qasm_gold_rows(
    circuits: pa.Table,
    checks: pa.Table,
    corrections: pa.Table,
) -> QasmGoldRows:
    """Normalize register and participant lists while preserving source IDs."""
    _check_table(circuits, CIRCUIT_SCHEMA, "circuit")
    _check_table(checks, STABILIZER_CHECK_SCHEMA, "stabilizer check")
    _check_table(
        corrections,
        CONDITIONAL_CORRECTION_SCHEMA,
        "conditional correction",
    )

    output = QasmGoldRows([], [], [], [], [])
    source_ids: set[str] = set()
    circuit_ids: set[str] = set()
    register_by_circuit: dict[str, dict[str, tuple[str, int]]] = {}

    def claim_source_id(source_id: str) -> None:
        if not source_id or source_id in source_ids:
            raise ValueError("Missing or duplicate QASMBench source_record_id")
        source_ids.add(source_id)

    for row in circuits.to_pylist():
        circuit_id = row["circuit_id"]
        claim_source_id(row["source_record_id"])
        if not circuit_id or circuit_id in circuit_ids:
            raise ValueError("Missing or duplicate QASMBench circuit_id")
        if row["variant"] not in {"source", "transpiled"}:
            raise ValueError("Invalid QASMBench variant")
        if not row["benchmark_name"] or not re.fullmatch(
            r"[0-9a-f]{64}", row["member_sha256"]
        ):
            raise ValueError("Invalid QASMBench circuit metadata")
        if (
            row["qubit_count"] <= 0
            or row["classical_bit_count"] < 0
            or row["operation_count"] < 0
            or row["measurement_count"] < 0
            or not 0 <= row["two_qubit_gate_count"] <= row["operation_count"]
        ):
            raise ValueError("Invalid QASMBench operation or register counts")

        try:
            declarations = json.loads(row["register_declarations"])
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid QASMBench register declarations") from error
        if not isinstance(declarations, list) or not declarations:
            raise ValueError("Invalid QASMBench register declarations")

        registers: dict[str, tuple[str, int]] = {}
        for order, declaration in enumerate(declarations, start=1):
            if not isinstance(declaration, dict) or set(declaration) != {
                "kind", "name", "size"
            }:
                raise ValueError("Invalid QASMBench register declaration")
            kind = declaration["kind"]
            name = declaration["name"]
            size = declaration["size"]
            if (
                kind not in {"qreg", "creg"}
                or not isinstance(name, str)
                or not name
                or name in registers
                or type(size) is not int
                or size <= 0
            ):
                raise ValueError("Invalid QASMBench register declaration")
            registers[name] = (kind, size)
            output.registers.append((circuit_id, name, kind, size, order))

        if sum(size for kind, size in registers.values() if kind == "qreg") != row[
            "qubit_count"
        ] or sum(size for kind, size in registers.values() if kind == "creg") != row[
            "classical_bit_count"
        ]:
            raise ValueError("QASMBench register counts do not reconcile")

        circuit_ids.add(circuit_id)
        register_by_circuit[circuit_id] = registers
        output.circuits.append(
            (
                circuit_id,
                row["source_record_id"],
                row["benchmark_name"],
                row["variant"],
                row["member_sha256"],
                row["qubit_count"],
                row["classical_bit_count"],
                row["operation_count"],
                row["measurement_count"],
                row["two_qubit_gate_count"],
            )
        )

    check_keys: set[tuple[str, str]] = set()
    for row in checks.to_pylist():
        claim_source_id(row["source_record_id"])
        circuit_id = row["circuit_id"]
        registers = register_by_circuit.get(circuit_id)
        if registers is None:
            raise ValueError("QASMBench check references a missing circuit")
        check_key = (circuit_id, row["check_id"])
        if not row["check_id"] or check_key in check_keys:
            raise ValueError("Missing or duplicate QASMBench check_id")
        _require_bit(row["ancilla_qubit"], registers, "qreg")
        _require_bit(row["syndrome_bit"], registers, "creg")
        data_qubits = row["data_qubits"]
        if len(data_qubits) < 2 or len(data_qubits) != len(set(data_qubits)):
            raise ValueError("Invalid QASMBench check participants")
        if row["ancilla_qubit"] in data_qubits:
            raise ValueError("QASMBench ancilla cannot be its own data qubit")
        for order, data_qubit in enumerate(data_qubits, start=1):
            _require_bit(data_qubit, registers, "qreg")
            output.participants.append((row["source_record_id"], order, data_qubit))
        check_keys.add(check_key)
        output.checks.append(
            (
                row["source_record_id"],
                circuit_id,
                row["check_id"],
                row["ancilla_qubit"],
                row["syndrome_bit"],
            )
        )

    for row in corrections.to_pylist():
        claim_source_id(row["source_record_id"])
        circuit_id = row["circuit_id"]
        registers = register_by_circuit.get(circuit_id)
        if registers is None:
            raise ValueError("QASMBench correction references a missing circuit")
        condition = registers.get(row["condition_register"])
        value = row["condition_value"]
        if (
            condition is None
            or condition[0] != "creg"
            or not 0 <= value < 2 ** condition[1]
            or not row["gate"]
        ):
            raise ValueError("Invalid QASMBench correction condition or gate")
        _require_bit(row["target_qubit"], registers, "qreg")
        output.corrections.append(
            (
                row["source_record_id"],
                circuit_id,
                row["condition_register"],
                value,
                row["gate"],
                row["target_qubit"],
            )
        )

    return output


def load_qasmbench_gold(settings: Settings, *, run_id: str) -> StageResult:
    """Replace only QASMBench Gold tables inside one transaction."""
    result = StageResult(stage="load_qasmbench_gold", run_id=run_id)
    circuit_table = pq.read_table(
        BytesIO(read_lake_object(settings, CIRCUIT_OBJECT))
    )
    check_table = pq.read_table(
        BytesIO(read_lake_object(settings, STABILIZER_CHECK_OBJECT))
    )
    correction_table = pq.read_table(
        BytesIO(read_lake_object(settings, CONDITIONAL_CORRECTION_OBJECT))
    )
    rows = qasm_gold_rows(circuit_table, check_table, correction_table)
    with postgres_connection(settings) as connection:
        replace_qasm_gold(connection, rows)

    result.input_count = sum(
        table.num_rows for table in (circuit_table, check_table, correction_table)
    )
    result.output_count = result.input_count
    result.finish()
    return result


def replace_qasm_gold(connection: psycopg.Connection, rows: QasmGoldRows) -> None:
    """Replace QASMBench Gold within the caller's transaction."""
    schema_sql = files("quantum_lake_student").joinpath(
        "sql/gold_qasmbench.sql"
    ).read_text(encoding="utf-8")

    for statement in schema_sql.split(";"):
        if statement.strip():
            connection.execute(statement)

    for name in (
        "qasm_conditional_correction",
        "qasm_check_data_qubit",
        "qasm_stabilizer_check",
        "qasm_register",
        "qasm_circuit",
    ):
        connection.execute(f"DELETE FROM gold.{name}")

    with connection.cursor() as cursor:
        for name, values in (
            ("qasm_circuit", rows.circuits),
            ("qasm_register", rows.registers),
            ("qasm_stabilizer_check", rows.checks),
            ("qasm_check_data_qubit", rows.participants),
            ("qasm_conditional_correction", rows.corrections),
        ):
            placeholders = ", ".join(["%s"] * len(values[0]))
            cursor.executemany(
                f"INSERT INTO gold.{name} VALUES ({placeholders})",
                values,
            )

    for name, values in (
        ("qasm_circuit", rows.circuits),
        ("qasm_register", rows.registers),
        ("qasm_stabilizer_check", rows.checks),
        ("qasm_check_data_qubit", rows.participants),
        ("qasm_conditional_correction", rows.corrections),
    ):
        (count,) = connection.execute(
            f"SELECT count(*) FROM gold.{name}"
        ).fetchone()
        if count != len(values):
            raise RuntimeError(f"QASMBench Gold {name} count mismatch")

    for name, values, source_column in (
        ("qasm_circuit", rows.circuits, 1),
        ("qasm_stabilizer_check", rows.checks, 0),
        ("qasm_conditional_correction", rows.corrections, 0),
    ):
        actual_ids = {
            source_id
            for (source_id,) in connection.execute(
                f"SELECT source_record_id FROM gold.{name}"
            )
        }
        if actual_ids != {row[source_column] for row in values}:
            raise RuntimeError(f"QASMBench Gold {name} source IDs mismatch")


def run(run_id: str) -> StageResult:
    """Run the QASMBench Gold loader using environment-based settings."""
    return load_qasmbench_gold(Settings.from_environment(), run_id=run_id)
