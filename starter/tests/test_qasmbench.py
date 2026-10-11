import hashlib
import io
import json
import warnings
from pathlib import Path
from zipfile import ZipFile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quantum_lake_student.config import Settings
from quantum_lake_student.evidence import (
    DATA_ISSUES_SCHEMA,
    SOURCE_TRACE_SCHEMA,
    parquet_bytes,
)
from quantum_lake_student.qasmbench import (
    BRONZE_OBJECT,
    CIRCUIT_OBJECT,
    CIRCUIT_SCHEMA,
    CONDITIONAL_CORRECTION_OBJECT,
    CONDITIONAL_CORRECTION_SCHEMA,
    EXPECTED_QASM_MEMBERS,
    QasmBenchArchiveError,
    QasmParseError,
    STABILIZER_CHECK_OBJECT,
    STABILIZER_CHECK_SCHEMA,
    build_qasmbench_tables,
    extract_stabilizer_checks,
    parse_qasm,
)
from quantum_lake_student.stages.prepare_qasmbench import prepare_qasmbench
from quantum_lake_student.source_validation import (
    SourceObjectSpec,
    source_spec,
)


BASIC_QASM = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
x q[0];
measure q -> c;
"""

QEC_SM_SOURCE = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
qreg a[2];
creg c[3];
creg syn[2];
gate syndrome d1,d2,d3,a1,a2 {
  cx d1,a1; cx d2,a1;
  cx d2,a2; cx d3,a2;
}
x q[0];
syndrome q[0],q[1],q[2],a[0],a[1];
measure a -> syn;
if(syn==1) x q[0];
if(syn==2) x q[2];
if(syn==3) x q[1];
measure q -> c;
"""

QEC_SM_TRANSPILED = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
qreg a[2];
creg c[3];
creg syn[2];
x q[0];
cx q[0],a[0];
cx q[1],a[0];
cx q[1],a[1];
cx q[2],a[1];
measure a[0] -> syn[0];
measure a[1] -> syn[1];
if(syn==1) x q[0];
if(syn==2) x q[2];
if(syn==3) x q[1];
measure q -> c;
"""


def _valid_members() -> dict[str, str]:
    return {
        member: (
            QEC_SM_SOURCE
            if member == "small/qec_sm_n5/qec_sm_n5.qasm"
            else QEC_SM_TRANSPILED
            if member == "small/qec_sm_n5/qec_sm_n5_transpiled.qasm"
            else BASIC_QASM
        )
        for member in EXPECTED_QASM_MEMBERS
    }


def _archive_bytes(
    *,
    overrides: dict[str, str | bytes] | None = None,
    remove: str | None = None,
    extra: tuple[str, str] | None = None,
    duplicate: str | None = None,
) -> bytes:
    members = _valid_members()
    if overrides:
        members.update(overrides)
    if remove:
        members.pop(remove)

    output = io.BytesIO()
    with ZipFile(output, mode="w") as archive:
        archive.writestr("README.md", "fixture")
        for member, source in sorted(members.items()):
            archive.writestr(member, source)
        if extra:
            archive.writestr(*extra)
        if duplicate:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                archive.writestr(duplicate, members[duplicate])
    return output.getvalue()


def _build(value: bytes, *, run_id: str = "test"):
    return build_qasmbench_tables(
        value,
        spec=SourceObjectSpec(
            source_name="qasmbench",
            bronze_object=BRONZE_OBJECT,
            expected_bytes=len(value),
            expected_sha256=hashlib.sha256(value).hexdigest(),
        ),
        run_id=run_id,
    )


def _local_settings(root: Path) -> Settings:
    return Settings(
        lake_backend="local",
        local_lake_root=root,
        s3_endpoint="",
        s3_access_key="",
        s3_secret_key="",
        s3_bucket="",
        postgres_host="",
        postgres_port=5432,
        postgres_db="",
        postgres_user="",
        postgres_password="",
    )


def test_custom_gate_is_expanded_only_when_called() -> None:
    parsed = parse_qasm(QEC_SM_SOURCE)
    checks = extract_stabilizer_checks(parsed)

    assert parsed.qubit_count == 5
    assert parsed.classical_bit_count == 5
    assert parsed.operation_count == 5
    assert parsed.two_qubit_gate_count == 4
    assert parsed.measurement_count == 5
    assert len(parsed.gate_definitions) == 1
    assert len(parsed.corrections) == 3
    assert [(item.ancilla_qubit, item.data_qubits, item.syndrome_bit) for item in checks] == [
        ("a[0]", ("q[0]", "q[1]"), "syn[0]"),
        ("a[1]", ("q[1]", "q[2]"), "syn[1]"),
    ]


def test_unused_gate_definition_is_not_an_executed_operation() -> None:
    source = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
gate pair left,right { cx left,right; }
x q[0];
measure q -> c;
"""
    parsed = parse_qasm(source)

    assert parsed.operation_count == 1
    assert parsed.two_qubit_gate_count == 0
    assert parsed.measurement_count == 2


def test_register_operations_and_measurements_are_expanded() -> None:
    source = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
x q;
measure q -> c;
"""
    parsed = parse_qasm(source)

    assert [operation.qubits for operation in parsed.operations] == [
        ("q[0]",),
        ("q[1]",),
        ("q[2]",),
    ]
    assert [measurement.qubit for measurement in parsed.measurements] == [
        "q[0]",
        "q[1]",
        "q[2]",
    ]


def test_register_local_indices_remain_distinct() -> None:
    parsed = parse_qasm(QEC_SM_TRANSPILED)
    all_qubits = {
        qubit
        for operation in parsed.operations
        for qubit in operation.qubits
    }

    assert "q[0]" in all_qubits
    assert "a[0]" in all_qubits
    assert "q[0]" != "a[0]"


def test_comments_whitespace_case_and_parameterized_gates_are_supported() -> None:
    source = '''// leading comment
openqasm 2.0;
INCLUDE "qelib1.inc";

QREG q[2]; // declaration comment
CREG c[2];
u3(pi/2, 0, pi) q[0];
barrier q;
CX q[0], q[1];
measure q -> c;
'''
    parsed = parse_qasm(source)

    assert [operation.name for operation in parsed.operations] == [
        "u3",
        "cx",
    ]
    assert parsed.operation_count == 2
    assert parsed.two_qubit_gate_count == 1
    assert parsed.measurement_count == 2


def test_scalar_argument_is_broadcast_across_a_register() -> None:
    source = """OPENQASM 2.0;
include "qelib1.inc";
qreg control[1];
qreg target[3];
creg result[3];
cx control[0],target;
measure target -> result;
"""
    parsed = parse_qasm(source)

    assert [operation.qubits for operation in parsed.operations] == [
        ("control[0]", "target[0]"),
        ("control[0]", "target[1]"),
        ("control[0]", "target[2]"),
    ]


def test_nested_custom_gates_expand_to_concrete_operations() -> None:
    source = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
gate pair left,right { cx left,right; }
gate wrapped left,right { h left; pair left,right; }
wrapped q[0],q[1];
measure q -> c;
"""
    parsed = parse_qasm(source)

    assert [
        (operation.name, operation.qubits)
        for operation in parsed.operations
    ] == [
        ("h", ("q[0]",)),
        ("cx", ("q[0]", "q[1]")),
    ]


def test_barriers_are_validated_but_not_counted_as_operations() -> None:
    source = BASIC_QASM.replace("x q[0];", "barrier q[0],q[1];")
    parsed = parse_qasm(source)

    assert parsed.operation_count == 0
    assert parsed.measurement_count == 5


def test_single_qubit_measurement_preserves_exact_register_identity() -> None:
    source = """OPENQASM 2.0;
include "qelib1.inc";
qreg data[1];
qreg ancilla[1];
creg output[1];
measure ancilla[0] -> output[0];
"""
    parsed = parse_qasm(source)

    assert parsed.measurements[0].qubit == "ancilla[0]"
    assert parsed.measurements[0].classical_bit == "output[0]"


def test_conditional_single_qubit_gate_broadcasts_across_register() -> None:
    source = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg syn[2];
if(syn==1) x q;
"""
    parsed = parse_qasm(source)

    assert [correction.target_qubit for correction in parsed.corrections] == [
        "q[0]",
        "q[1]",
        "q[2]",
    ]


def test_stabilizer_check_accepts_syndrome_register_name() -> None:
    source = """OPENQASM 2.0;
include "qelib1.inc";
qreg data[2];
qreg ancilla[1];
creg syndrome[1];
cx data[0],ancilla[0];
cx data[1],ancilla[0];
measure ancilla[0] -> syndrome[0];
"""
    checks = extract_stabilizer_checks(parse_qasm(source))

    assert len(checks) == 1
    assert checks[0].ancilla_qubit == "ancilla[0]"
    assert checks[0].data_qubits == ("data[0]", "data[1]")
    assert checks[0].syndrome_bit == "syndrome[0]"
    assert checks[0].source_lines == (6, 7, 8)


def test_non_syndrome_measurement_does_not_create_a_stabilizer_check() -> None:
    source = """OPENQASM 2.0;
include "qelib1.inc";
qreg data[2];
qreg ancilla[1];
creg output[1];
cx data[0],ancilla[0];
cx data[1],ancilla[0];
measure ancilla[0] -> output[0];
"""

    assert extract_stabilizer_checks(parse_qasm(source)) == ()


@pytest.mark.parametrize(
    ("source", "rule_id"),
    [
        (
            BASIC_QASM.replace('include "qelib1.inc";', 'include "qelib1.inc;'),
            "qasm.string",
        ),
        (
            BASIC_QASM.replace("measure q -> c;", "measure q -> c"),
            "qasm.statement_termination",
        ),
        (
            BASIC_QASM.replace("OPENQASM 2.0;", "OPENQASM 3.0;"),
            "qasm.version",
        ),
        (
            BASIC_QASM.replace(
                'include "qelib1.inc";',
                'OPENQASM 2.0;\ninclude "qelib1.inc";',
            ),
            "qasm.version",
        ),
        (
            BASIC_QASM.replace("qreg q[5];", "qreg q[0];"),
            "qasm.register_size",
        ),
        (
            """OPENQASM 2.0;
include "qelib1.inc";
creg c[1];
""",
            "qasm.register_declaration",
        ),
        (
            """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
""",
            "qasm.register_declaration",
        ),
        (
            BASIC_QASM.replace("x q[0];", "x q[];"),
            "qasm.operand",
        ),
        (
            BASIC_QASM.replace("x q[0];", "x;"),
            "qasm.operation",
        ),
        (
            BASIC_QASM.replace("x q[0];", "x c[0];"),
            "qasm.register_reference",
        ),
        (
            BASIC_QASM.replace("x q[0];", "barrier c;"),
            "qasm.register_reference",
        ),
        (
            BASIC_QASM.replace(
                "x q[0];",
                "measure q[0] c[0];",
            ),
            "qasm.measurement",
        ),
        (
            BASIC_QASM.replace(
                "x q[0];",
                "gate pair left,left { cx left,left; }",
            ),
            "qasm.gate_definition",
        ),
        (
            BASIC_QASM.replace(
                "x q[0];",
                "gate pair left,right { cx left,missing; }",
            ),
            "qasm.gate_definition",
        ),
        (
            BASIC_QASM.replace(
                "x q[0];",
                "gate empty value { }",
            ),
            "qasm.gate_definition",
        ),
        (
            BASIC_QASM.replace(
                "x q[0];",
                "gate indexed value[0] { x value[0]; }",
            ),
            "qasm.gate_definition",
        ),
        (
            BASIC_QASM.replace(
                "x q[0];",
                "gate first value { x value; }\n"
                "gate first value { h value; }",
            ),
            "qasm.gate_definition",
        ),
        (
            BASIC_QASM.replace(
                "x q[0];",
                "gate first value { second value; }\n"
                "gate second value { first value; }\n"
                "first q[0];",
            ),
            "qasm.gate_recursion",
        ),
        (
            BASIC_QASM.replace(
                "x q[0];",
                "gate outer value { gate inner item { x item; } }",
            ),
            "qasm.gate_definition",
        ),
        (
            BASIC_QASM.replace("x q[0];", "unexpected q[0] { x q[0]; }"),
            "qasm.block",
        ),
        (
            BASIC_QASM.replace("x q[0];", "}"),
            "qasm.block",
        ),
        (
            BASIC_QASM.replace("x q[0];", "gate open value { x value;"),
            "qasm.gate_definition",
        ),
        (
            QEC_SM_SOURCE.replace("if(syn==1) x q[0];", "if(c==8) x q[0];"),
            "qasm.condition_value",
        ),
        (
            QEC_SM_SOURCE.replace("if(syn==1) x q[0];", "if(q==1) x q[0];"),
            "qasm.condition_register",
        ),
        (
            QEC_SM_SOURCE.replace(
                "if(syn==1) x q[0];",
                "if(syn==1) cx q[0],q[1];",
            ),
            "qasm.correction_arity",
        ),
    ],
)
def test_additional_invalid_syntax_reports_specific_rule(
    source: str,
    rule_id: str,
) -> None:
    with pytest.raises(QasmParseError) as error:
        parse_qasm(source)

    assert error.value.rule_id == rule_id


@pytest.mark.parametrize(
    "operations",
    [
        "cx data[0],ancilla[0];",
        "cx ancilla[0],data[0];\ncx ancilla[0],data[1];",
        "cx data[0],ancilla[0];\ncx data[0],ancilla[0];",
    ],
)
def test_invalid_stabilizer_shapes_are_rejected(operations: str) -> None:
    source = f'''OPENQASM 2.0;
include "qelib1.inc";
qreg data[2];
qreg ancilla[1];
creg syn[1];
{operations}
measure ancilla[0] -> syn[0];
'''

    with pytest.raises(QasmParseError) as error:
        extract_stabilizer_checks(parse_qasm(source))

    assert error.value.rule_id == "qasm.stabilizer_structure"


@pytest.mark.parametrize(
    ("member", "source", "rule_id"),
    [
        (
            "small/qec_en_n5/qec_en_n5.qasm",
            BASIC_QASM.replace("qreg q[5]", "qreg q[4]").replace(
                "creg c[5]",
                "creg c[4]",
            ),
            "qasm.qubit_count",
        ),
        (
            "small/qec_en_n5/qec_en_n5.qasm",
            BASIC_QASM.replace(
                "measure q -> c",
                "measure q[0] -> c[0]",
            ),
            "qasm.measurement_count",
        ),
        (
            "small/qec_sm_n5/qec_sm_n5.qasm",
            QEC_SM_SOURCE.replace("syn", "signal"),
            "qasm.stabilizer_count",
        ),
        (
            "small/qec_sm_n5/qec_sm_n5.qasm",
            QEC_SM_SOURCE.replace(
                "if(syn==1) x q[0];",
                "if(syn==1) x q[1];",
            ),
            "qasm.correction_structure",
        ),
        (
            "small/qec_en_n5/qec_en_n5.qasm",
            BASIC_QASM.replace(
                "x q[0];",
                "x q[0];\nif(c==1) x q[0];",
            ),
            "qasm.unexpected_recovery",
        ),
    ],
)
def test_curated_circuit_contract_failures_become_data_issues(
    member: str,
    source: str,
    rule_id: str,
) -> None:
    tables = _build(_archive_bytes(overrides={member: source}))

    assert tables.input_count == 6
    assert tables.accepted_count == 5
    assert tables.rejected_count == 1
    assert tables.data_issues.num_rows == 1
    issue = tables.data_issues.to_pylist()[0]
    assert issue["archive_member"] == member
    assert issue["rule_id"] == rule_id
    assert issue["action"] == "excluded_from_silver"


def test_non_utf8_qasm_member_is_excluded_with_encoding_issue() -> None:
    member = "small/qec_en_n5/qec_en_n5.qasm"
    tables = _build(_archive_bytes(overrides={member: b"\xff\xfe"}))

    assert tables.accepted_count == 5
    assert tables.rejected_count == 1
    assert tables.data_issues.num_rows == 1
    issue = tables.data_issues.to_pylist()[0]
    assert issue["archive_member"] == member
    assert issue["rule_id"] == "qasm.encoding"
    assert issue["action"] == "excluded_from_silver"


@pytest.mark.parametrize(
    ("source", "rule_id"),
    [
        (
            BASIC_QASM.replace("x q[0]", "x q[5]"),
            "qasm.register_index",
        ),
        (
            BASIC_QASM.replace("measure q -> c", "measure q[0] -> c"),
            "qasm.measurement_width",
        ),
        (
            QEC_SM_SOURCE.replace("if(syn==3)", "if(syn==4)"),
            "qasm.condition_value",
        ),
        (
            BASIC_QASM.replace("OPENQASM 2.0;\n", ""),
            "qasm.version",
        ),
        (
            BASIC_QASM.replace('include "qelib1.inc";\n', ""),
            "qasm.include",
        ),
        (
            BASIC_QASM.replace("x q[0]", "x missing[0]"),
            "qasm.register_reference",
        ),
        (
            BASIC_QASM.replace(
                "qreg q[5];",
                "qreg q[5];\nqreg other[3];",
            ).replace("x q[0]", "cx q,other"),
            "qasm.register_width",
        ),
        (
            BASIC_QASM.replace(
                "x q[0];",
                "gate pair left,right { cx left,right; }\npair q[0];",
            ),
            "qasm.gate_arity",
        ),
        (
            BASIC_QASM.replace(
                "x q[0];",
                "gate loop value { loop value; }\nloop q[0];",
            ),
            "qasm.gate_recursion",
        ),
        (
            BASIC_QASM.replace("creg c[5];", "creg c[5];\ncreg c[1];"),
            "qasm.register_declaration",
        ),
    ],
)
def test_invalid_qasm_is_rejected_with_specific_rule(
    source: str,
    rule_id: str,
) -> None:
    with pytest.raises(QasmParseError) as error:
        parse_qasm(source)

    assert error.value.rule_id == rule_id


def test_valid_archive_builds_exact_schemas_and_complete_trace() -> None:
    value = _archive_bytes()
    tables = _build(value)

    assert tables.circuits.schema == CIRCUIT_SCHEMA
    assert tables.stabilizer_checks.schema == STABILIZER_CHECK_SCHEMA
    assert (
        tables.conditional_corrections.schema
        == CONDITIONAL_CORRECTION_SCHEMA
    )
    assert tables.source_trace.schema == SOURCE_TRACE_SCHEMA
    assert tables.data_issues.schema == DATA_ISSUES_SCHEMA
    assert tables.input_count == 6
    assert tables.accepted_count == 6
    assert tables.rejected_count == 0
    assert tables.circuits.num_rows == 6
    assert tables.stabilizer_checks.num_rows == 4
    assert tables.conditional_corrections.num_rows == 6
    assert tables.silver_row_count == 16
    assert tables.source_trace.num_rows == 16
    assert tables.data_issues.num_rows == 0

    silver_ids = {
        value
        for table in (
            tables.circuits,
            tables.stabilizer_checks,
            tables.conditional_corrections,
        )
        for value in table.column("source_record_id").to_pylist()
    }
    assert silver_ids == set(
        tables.source_trace.column("source_record_id").to_pylist()
    )
    assert len(silver_ids) == 16
    assert set(tables.circuits.column("variant").to_pylist()) == {
        "source",
        "transpiled",
    }
    assert all(
        len(value) == 64
        for value in tables.circuits.column("member_sha256").to_pylist()
    )


def test_ids_and_parquet_values_are_stable_across_runs() -> None:
    value = _archive_bytes()
    first = _build(value, run_id="first")
    second = _build(value, run_id="second")

    for first_table, second_table in (
        (first.circuits, second.circuits),
        (first.stabilizer_checks, second.stabilizer_checks),
        (first.conditional_corrections, second.conditional_corrections),
        (first.source_trace, second.source_trace),
    ):
        assert first_table.equals(second_table)
        round_trip = pq.read_table(
            pa.BufferReader(parquet_bytes(first_table))
        )
        assert round_trip.equals(first_table)


def test_malformed_member_is_excluded_and_preserved_as_issue() -> None:
    member = "small/qec_en_n5/qec_en_n5.qasm"
    value = _archive_bytes(
        overrides={
            member: BASIC_QASM.replace("OPENQASM 2.0;\n", "")
        }
    )
    tables = _build(value)

    assert tables.input_count == 6
    assert tables.accepted_count == 5
    assert tables.rejected_count == 1
    assert tables.circuits.num_rows == 5
    assert tables.data_issues.num_rows == 1
    issue = tables.data_issues.to_pylist()[0]
    assert issue["rule_id"] == "qasm.version"
    assert issue["source_record_id"] is not None
    assert issue["archive_member"] == member
    assert issue["action"] == "excluded_from_silver"
    assert "qelib1.inc" in issue["source_record"]


def test_missing_or_unexpected_qasm_member_stops_the_run() -> None:
    missing = "small/qec_en_n5/qec_en_n5.qasm"
    with pytest.raises(QasmBenchArchiveError, match="member mismatch"):
        _build(_archive_bytes(remove=missing))

    with pytest.raises(QasmBenchArchiveError, match="member mismatch"):
        _build(
            _archive_bytes(
                extra=("small/extra/extra.qasm", BASIC_QASM)
            )
        )


def test_checksum_size_duplicate_and_unsafe_members_stop_the_run() -> None:
    value = _archive_bytes()
    digest = hashlib.sha256(value).hexdigest()

    with pytest.raises(QasmBenchArchiveError, match="Byte size"):
        build_qasmbench_tables(
            value,
            spec=SourceObjectSpec(
                source_name="qasmbench",
                bronze_object=BRONZE_OBJECT,
                expected_bytes=len(value) + 1,
                expected_sha256=digest,
            ),
            run_id="test",
        )
    with pytest.raises(QasmBenchArchiveError, match="SHA-256"):
        build_qasmbench_tables(
            value,
            spec=SourceObjectSpec(
                source_name="qasmbench",
                bronze_object=BRONZE_OBJECT,
                expected_bytes=len(value),
                expected_sha256="0" * 64,
            ),
            run_id="test",
        )

    duplicate = next(iter(EXPECTED_QASM_MEMBERS))
    with pytest.raises(QasmBenchArchiveError, match="duplicate"):
        _build(_archive_bytes(duplicate=duplicate))

    unsafe_value = _archive_bytes(extra=("../outside.txt", "bad"))
    with pytest.raises(QasmBenchArchiveError, match="Unsafe"):
        _build(unsafe_value)


def test_local_stage_writes_replaces_and_merges_outputs(
    tmp_path: Path,
) -> None:
    archive_bytes = _archive_bytes()
    digest = hashlib.sha256(archive_bytes).hexdigest()
    lake_root = tmp_path / "lake"
    archive_path = (
        lake_root / "raw" / "source=qasmbench" / "qasmbench-qec.zip"
    )
    archive_path.parent.mkdir(parents=True)
    archive_path.write_bytes(archive_bytes)

    manifest_path = lake_root / "metadata" / "bundle-manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "release_name": "test-release",
                "bundle_version": 1,
                "objects": [
                    {
                        "source": "qasmbench",
                        "path": "raw/source=qasmbench/qasmbench-qec.zip",
                        "bytes": len(archive_bytes),
                        "sha256": digest,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    results_root = tmp_path / "results" / "part1"
    results_root.mkdir(parents=True)
    existing_trace = pa.Table.from_pylist(
        [
            {
                "source_record_id": "syndrome-existing",
                "source_name": "qec_syndromes",
                "bronze_object": "bronze/syndrome.zip",
                "archive_member": "data.csv",
                "record_locator": "csv_data_row=1",
                "input_sha256": "a" * 64,
            }
        ],
        schema=SOURCE_TRACE_SCHEMA,
    )
    (results_root / "source_trace.parquet").write_bytes(
        parquet_bytes(existing_trace)
    )
    (results_root / "data_issues.parquet").write_bytes(
        parquet_bytes(pa.Table.from_pylist([], schema=DATA_ISSUES_SCHEMA))
    )
    (results_root / "row_counts.json").write_text(
        json.dumps({"qec_syndromes": {"silver_rows": 1}}),
        encoding="utf-8",
    )

    settings = _local_settings(lake_root)
    first = prepare_qasmbench(
        settings,
        run_id="first",
        results_root=results_root,
    )
    first_circuits = pq.read_table(lake_root / CIRCUIT_OBJECT)
    first_ids = first_circuits.column("source_record_id").to_pylist()
    second = prepare_qasmbench(
        settings,
        run_id="second",
        results_root=results_root,
    )

    assert first.input_count == 6
    assert first.output_count == 16
    assert first.issue_count == 0
    assert second.output_count == 16
    assert first_ids == pq.read_table(
        lake_root / CIRCUIT_OBJECT
    ).column("source_record_id").to_pylist()
    assert pq.read_table(lake_root / STABILIZER_CHECK_OBJECT).num_rows == 4
    assert pq.read_table(
        lake_root / CONDITIONAL_CORRECTION_OBJECT
    ).num_rows == 6

    trace = pq.read_table(results_root / "source_trace.parquet")
    assert trace.num_rows == 17
    assert set(trace.column("source_name").to_pylist()) == {
        "qasmbench",
        "qec_syndromes",
    }
    assert trace.column("source_record_id").to_pylist().count(
        "syndrome-existing"
    ) == 1

    counts = json.loads(
        (results_root / "row_counts.json").read_text(encoding="utf-8")
    )
    assert counts["qec_syndromes"] == {"silver_rows": 1}
    assert counts["qasmbench"] == {
        "bronze_rows": 6,
        "circuit_rows": 6,
        "conditional_correction_rows": 6,
        "issue_rows": 0,
        "rejected_rows": 0,
        "silver_rows": 16,
        "stabilizer_check_rows": 4,
    }

    run = json.loads((results_root / "run.json").read_text(encoding="utf-8"))
    assert run["run_id"] == "second"
    assert run["input_hashes"][BRONZE_OBJECT] == digest
    assert run["outputs"][CIRCUIT_OBJECT]["rows"] == 6
    assert all(
        check["status"] == "passed"
        for check in run["checks"]["qasmbench"].values()
    )


def test_real_release_exact_counts_and_semantics() -> None:
    archive_path = Path(
        "/course-data/raw/source=qasmbench/qasmbench-qec.zip"
    )
    manifest_path = Path("/course-data/metadata/bundle-manifest.json")
    if not archive_path.is_file() or not manifest_path.is_file():
        pytest.skip("Course data mount is unavailable")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spec = source_spec(
        manifest,
        "qasmbench",
        expected_object=BRONZE_OBJECT,
    )
    tables = build_qasmbench_tables(
        archive_path.read_bytes(),
        spec=spec,
        run_id="real-release",
    )

    assert tables.input_count == 6
    assert tables.accepted_count == 6
    assert tables.rejected_count == 0
    assert tables.circuits.num_rows == 6
    assert tables.stabilizer_checks.num_rows == 4
    assert tables.conditional_corrections.num_rows == 6
    assert tables.source_trace.num_rows == 16
    assert tables.data_issues.num_rows == 0

    counts = {
        (row["benchmark_name"], row["variant"]): (
            row["operation_count"],
            row["measurement_count"],
            row["two_qubit_gate_count"],
        )
        for row in tables.circuits.to_pylist()
    }
    assert counts == {
        ("error_correctiond3_n5", "source"): (114, 5, 49),
        ("error_correctiond3_n5", "transpiled"): (237, 5, 49),
        ("qec_en_n5", "source"): (25, 5, 10),
        ("qec_en_n5", "transpiled"): (46, 5, 10),
        ("qec_sm_n5", "source"): (5, 5, 4),
        ("qec_sm_n5", "transpiled"): (5, 5, 4),
    }
