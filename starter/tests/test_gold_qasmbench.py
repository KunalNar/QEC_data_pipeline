"""QASMBench Silver-to-Gold normalization tests."""

import json

import pyarrow as pa
import pytest

from quantum_lake_student.qasmbench import (
    CIRCUIT_SCHEMA,
    CONDITIONAL_CORRECTION_SCHEMA,
    STABILIZER_CHECK_SCHEMA,
)
from quantum_lake_student.stages.load_qasmbench_gold import qasm_gold_rows


REGISTERS = [
    {"kind": "qreg", "name": "q", "size": 3},
    {"kind": "qreg", "name": "a", "size": 2},
    {"kind": "creg", "name": "c", "size": 3},
    {"kind": "creg", "name": "syn", "size": 2},
]


def circuit(**changes):
    row = {
        "source_record_id": "circuit-source-1",
        "circuit_id": "circuit-1",
        "benchmark_name": "qec_sm_n5",
        "variant": "source",
        "register_declarations": json.dumps(REGISTERS),
        "qubit_count": 5,
        "measurement_count": 5,
        "two_qubit_gate_count": 4,
        "member_sha256": "a" * 64,
        "classical_bit_count": 5,
        "operation_count": 5,
    }
    row.update(changes)
    return row


def check(**changes):
    row = {
        "source_record_id": "check-source-1",
        "circuit_id": "circuit-1",
        "check_id": "check-1",
        "ancilla_qubit": "a[0]",
        "data_qubits": ["q[0]", "q[1]"],
        "syndrome_bit": "syn[0]",
    }
    row.update(changes)
    return row


def correction(**changes):
    row = {
        "source_record_id": "correction-source-1",
        "circuit_id": "circuit-1",
        "condition_register": "syn",
        "condition_value": 1,
        "gate": "x",
        "target_qubit": "q[2]",
    }
    row.update(changes)
    return row


def build(circuit_rows=None, check_rows=None, correction_rows=None):
    return qasm_gold_rows(
        pa.Table.from_pylist(
            [circuit()] if circuit_rows is None else circuit_rows,
            schema=CIRCUIT_SCHEMA,
        ),
        pa.Table.from_pylist(
            [check()] if check_rows is None else check_rows,
            schema=STABILIZER_CHECK_SCHEMA,
        ),
        pa.Table.from_pylist(
            [correction()] if correction_rows is None else correction_rows,
            schema=CONDITIONAL_CORRECTION_SCHEMA,
        ),
    )


def test_qasm_gold_separates_registers_and_ordered_participants():
    rows = build()
    assert len(rows.circuits) == 1
    assert rows.circuits[0][1] == "circuit-source-1"
    assert rows.registers == [
        ("circuit-1", "q", "qreg", 3, 1),
        ("circuit-1", "a", "qreg", 2, 2),
        ("circuit-1", "c", "creg", 3, 3),
        ("circuit-1", "syn", "creg", 2, 4),
    ]
    assert rows.checks == [
        ("check-source-1", "circuit-1", "check-1", "a[0]", "syn[0]")
    ]
    assert rows.participants == [
        ("check-source-1", 1, "q[0]"),
        ("check-source-1", 2, "q[1]"),
    ]
    assert rows.corrections == [
        ("correction-source-1", "circuit-1", "syn", 1, "x", "q[2]")
    ]


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        ({"ancilla_qubit": "a[2]"}, "out-of-range"),
        ({"ancilla_qubit": "syn[0]"}, "Undeclared"),
        ({"syndrome_bit": "q[0]"}, "Undeclared"),
        ({"data_qubits": ["q[0]", "q[0]"]}, "participants"),
        ({"data_qubits": ["q[0]", "q[3]"]}, "out-of-range"),
        ({"data_qubits": ["q[0]", "a[0]"]}, "ancilla"),
        ({"syndrome_bit": "syn"}, "Invalid QASM bit"),
        ({"circuit_id": "missing"}, "missing circuit"),
    ],
)
def test_qasm_gold_rejects_invalid_check_references(changed, message):
    with pytest.raises(ValueError, match=message):
        build(check_rows=[check(**changed)])


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        ({"condition_register": "q"}, "condition"),
        ({"condition_register": "missing"}, "condition"),
        ({"condition_value": 4}, "condition"),
        ({"target_qubit": "a[2]"}, "out-of-range"),
    ],
)
def test_qasm_gold_rejects_invalid_correction_references(changed, message):
    with pytest.raises(ValueError, match=message):
        build(correction_rows=[correction(**changed)])


def test_qasm_gold_rejects_bad_register_counts_and_duplicate_sources():
    with pytest.raises(ValueError, match="register counts"):
        build(circuit_rows=[circuit(qubit_count=4)])
    with pytest.raises(ValueError, match="source_record_id"):
        build(check_rows=[check(source_record_id="circuit-source-1")])


@pytest.mark.parametrize(
    "declarations",
    [
        "not JSON",
        json.dumps(REGISTERS + [{"kind": "qreg", "name": "q", "size": 1}]),
        json.dumps([{**REGISTERS[0], "size": 0}, *REGISTERS[1:]]),
        json.dumps([{**REGISTERS[0], "kind": "other"}, *REGISTERS[1:]]),
    ],
)
def test_qasm_gold_rejects_bad_register_declarations(declarations):
    with pytest.raises(ValueError, match="register declaration"):
        build(circuit_rows=[circuit(register_declarations=declarations)])


def test_qasm_gold_rejects_duplicate_check_id_and_null_source():
    with pytest.raises(ValueError, match="duplicate QASMBench check_id"):
        build(check_rows=[
            check(),
            check(source_record_id="check-source-2"),
        ])
    with pytest.raises(ValueError, match="null"):
        build(circuit_rows=[circuit(source_record_id=None)])


def test_qasm_gold_rejects_missing_child_table():
    with pytest.raises(ValueError, match="empty"):
        build(check_rows=[])
