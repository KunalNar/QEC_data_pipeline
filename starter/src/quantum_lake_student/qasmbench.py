"""Parse the curated QASMBench QEC archive into Silver tables."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePosixPath
from zipfile import ZipFile

import pyarrow as pa

from .evidence import (
    DATA_ISSUES_SCHEMA,
    SOURCE_TRACE_SCHEMA,
    data_issue_row,
    source_trace_row,
)
from .models import stable_record_hash
from .source_validation import (
    SourceObjectSpec,
    SourceValidationError,
    sha256_bytes,
    validate_source_object,
)


SOURCE_NAME = "qasmbench"
BRONZE_OBJECT = "bronze/source=qasmbench/qasmbench-qec.zip"
MANIFEST_OBJECT = "metadata/course-release/bundle-manifest.json"
CIRCUIT_OBJECT = "silver/qasmbench/circuit.parquet"
STABILIZER_CHECK_OBJECT = "silver/qasmbench/stabilizer_check.parquet"
CONDITIONAL_CORRECTION_OBJECT = (
    "silver/qasmbench/conditional_correction.parquet"
)

EXPECTED_QASM_MEMBERS = frozenset(
    {
        "small/error_correctiond3_n5/error_correctiond3_n5.qasm",
        (
            "small/error_correctiond3_n5/"
            "error_correctiond3_n5_transpiled.qasm"
        ),
        "small/qec_en_n5/qec_en_n5.qasm",
        "small/qec_en_n5/qec_en_n5_transpiled.qasm",
        "small/qec_sm_n5/qec_sm_n5.qasm",
        "small/qec_sm_n5/qec_sm_n5_transpiled.qasm",
    }
)


CIRCUIT_SCHEMA = pa.schema(
    [
        pa.field("source_record_id", pa.string(), nullable=False),
        pa.field("circuit_id", pa.string(), nullable=False),
        pa.field("benchmark_name", pa.string(), nullable=False),
        pa.field("variant", pa.string(), nullable=False),
        pa.field("register_declarations", pa.string(), nullable=False),
        pa.field("qubit_count", pa.int32(), nullable=False),
        pa.field("measurement_count", pa.int32(), nullable=False),
        pa.field("two_qubit_gate_count", pa.int32(), nullable=False),
        pa.field("member_sha256", pa.string(), nullable=False),
        pa.field("classical_bit_count", pa.int32(), nullable=False),
        pa.field("operation_count", pa.int32(), nullable=False),
    ]
)

STABILIZER_CHECK_SCHEMA = pa.schema(
    [
        pa.field("source_record_id", pa.string(), nullable=False),
        pa.field("circuit_id", pa.string(), nullable=False),
        pa.field("check_id", pa.string(), nullable=False),
        pa.field("ancilla_qubit", pa.string(), nullable=False),
        pa.field(
            "data_qubits",
            pa.list_(pa.field("item", pa.string(), nullable=False)),
            nullable=False,
        ),
        pa.field("syndrome_bit", pa.string(), nullable=False),
    ]
)

CONDITIONAL_CORRECTION_SCHEMA = pa.schema(
    [
        pa.field("source_record_id", pa.string(), nullable=False),
        pa.field("circuit_id", pa.string(), nullable=False),
        pa.field("condition_register", pa.string(), nullable=False),
        pa.field("condition_value", pa.int64(), nullable=False),
        pa.field("gate", pa.string(), nullable=False),
        pa.field("target_qubit", pa.string(), nullable=False),
    ]
)


class QasmBenchArchiveError(ValueError):
    """Raised when the curated archive itself cannot be processed safely."""


class QasmParseError(ValueError):
    """A record-level QASM validation failure."""

    def __init__(self, rule_id: str, message: str) -> None:
        super().__init__(message)
        self.rule_id = rule_id


@dataclass(frozen=True)
class RegisterDeclaration:
    kind: str
    name: str
    size: int


@dataclass(frozen=True)
class GateTemplate:
    name: str
    operands: tuple[str, ...]
    line_number: int


@dataclass(frozen=True)
class GateDefinition:
    name: str
    parameters: tuple[str, ...]
    body: tuple[GateTemplate, ...]


@dataclass(frozen=True)
class GateOperation:
    name: str
    qubits: tuple[str, ...]
    statement_line: int
    source_lines: tuple[int, ...]


@dataclass(frozen=True)
class Measurement:
    qubit: str
    classical_bit: str
    line_number: int


@dataclass(frozen=True)
class ConditionalCorrection:
    condition_register: str
    condition_value: int
    gate: str
    target_qubit: str
    line_number: int


@dataclass(frozen=True)
class StabilizerCheck:
    ancilla_qubit: str
    data_qubits: tuple[str, ...]
    syndrome_bit: str
    source_lines: tuple[int, ...]


@dataclass(frozen=True)
class ParsedCircuit:
    registers: tuple[RegisterDeclaration, ...]
    gate_definitions: tuple[GateDefinition, ...]
    operations: tuple[GateOperation, ...]
    measurements: tuple[Measurement, ...]
    corrections: tuple[ConditionalCorrection, ...]

    @property
    def qubit_count(self) -> int:
        return sum(
            register.size
            for register in self.registers
            if register.kind == "qreg"
        )

    @property
    def classical_bit_count(self) -> int:
        return sum(
            register.size
            for register in self.registers
            if register.kind == "creg"
        )

    @property
    def measurement_count(self) -> int:
        return len(self.measurements)

    @property
    def operation_count(self) -> int:
        # Conditional recovery operations have their own Silver table.
        return len(self.operations)

    @property
    def two_qubit_gate_count(self) -> int:
        return sum(len(operation.qubits) == 2 for operation in self.operations)

    @property
    def register_declarations(self) -> str:
        return json.dumps(
            [
                {
                    "kind": register.kind,
                    "name": register.name,
                    "size": register.size,
                }
                for register in self.registers
            ],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


@dataclass(frozen=True)
class QasmBenchTables:
    circuits: pa.Table
    stabilizer_checks: pa.Table
    conditional_corrections: pa.Table
    source_trace: pa.Table
    data_issues: pa.Table
    input_sha256: str
    input_count: int
    accepted_count: int
    rejected_count: int

    @property
    def silver_row_count(self) -> int:
        return (
            self.circuits.num_rows
            + self.stabilizer_checks.num_rows
            + self.conditional_corrections.num_rows
        )


@dataclass(frozen=True)
class _Token:
    text: str
    terminator: str
    line_number: int


_REGISTER_PATTERN = re.compile(
    r"^(?P<kind>qreg|creg)\s+(?P<name>[A-Za-z_]\w*)"
    r"\[(?P<size>\d+)\]$",
    flags=re.IGNORECASE,
)
_REFERENCE_PATTERN = re.compile(
    r"^(?P<name>[A-Za-z_]\w*)"
    r"(?:\[(?P<index>\d+)\])?$"
)
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_]\w*$")
_GATE_CALL_PATTERN = re.compile(
    r"^(?P<name>[A-Za-z_]\w*)"
    r"(?:\s*\((?P<arguments>[^)]*)\))?"
    r"\s+(?P<operands>.+)$"
)
_CONDITION_PATTERN = re.compile(
    r"^if\s*\(\s*(?P<register>[A-Za-z_]\w*)\s*==\s*"
    r"(?P<value>\d+)\s*\)\s*(?P<operation>.+)$",
    flags=re.IGNORECASE,
)
_MEMBER_PATTERN = re.compile(
    r"^small/(?P<benchmark>[A-Za-z0-9_]+)/"
    r"(?P<filename>[A-Za-z0-9_]+)\.qasm$"
)


def _fail(rule_id: str, message: str) -> QasmParseError:
    return QasmParseError(rule_id, message)


def _without_line_comments(source: str) -> str:
    lines: list[str] = []
    for line in source.splitlines(keepends=True):
        newline = "\n" if line.endswith("\n") else ""
        body = line[:-1] if newline else line
        lines.append(body.split("//", 1)[0] + newline)
    return "".join(lines)


def _tokenize(source: str) -> tuple[_Token, ...]:
    """Split QASM statements while retaining their source line numbers."""
    cleaned = _without_line_comments(source)
    tokens: list[_Token] = []
    buffer: list[str] = []
    current_line = 1
    start_line: int | None = None
    in_string = False

    for character in cleaned:
        if start_line is None and not character.isspace():
            start_line = current_line

        if character == '"':
            in_string = not in_string

        if not in_string and character in ";{}":
            text = "".join(buffer).strip()
            if text or character in "{}":
                tokens.append(
                    _Token(
                        text=text,
                        terminator=character,
                        line_number=start_line or current_line,
                    )
                )
            buffer = []
            start_line = None
        else:
            buffer.append(character)

        if character == "\n":
            current_line += 1

    if in_string:
        raise _fail("qasm.string", "Unterminated quoted string")
    trailing = "".join(buffer).strip()
    if trailing:
        raise _fail(
            "qasm.statement_termination",
            f"Statement on line {start_line or current_line} has no terminator",
        )
    return tuple(tokens)


def _split_operands(value: str) -> tuple[str, ...]:
    operands = tuple(item.strip() for item in value.split(","))
    if not operands or any(not item for item in operands):
        raise _fail("qasm.operation", f"Invalid operand list: {value}")
    return operands


def _parse_gate_call(statement: str, line_number: int) -> tuple[str, tuple[str, ...]]:
    match = _GATE_CALL_PATTERN.fullmatch(statement.strip())
    if match is None:
        raise _fail(
            "qasm.operation",
            f"Cannot parse operation on line {line_number}: {statement}",
        )
    return (
        match.group("name").lower(),
        _split_operands(match.group("operands")),
    )


def _parse_reference(
    value: str,
    registers: dict[str, RegisterDeclaration],
    *,
    expected_kind: str,
    line_number: int,
) -> tuple[str, ...]:
    match = _REFERENCE_PATTERN.fullmatch(value.strip())
    if match is None:
        raise _fail(
            "qasm.operand",
            f"Invalid register reference on line {line_number}: {value}",
        )
    name = match.group("name")
    register = registers.get(name)
    if register is None or register.kind != expected_kind:
        raise _fail(
            "qasm.register_reference",
            f"Unknown {expected_kind} register on line {line_number}: {name}",
        )
    index_text = match.group("index")
    if index_text is None:
        return tuple(f"{name}[{index}]" for index in range(register.size))

    index = int(index_text)
    if index >= register.size:
        raise _fail(
            "qasm.register_index",
            f"Register index out of range on line {line_number}: {value}",
        )
    return (f"{name}[{index}]",)


def _broadcast_arguments(
    arguments: tuple[tuple[str, ...], ...],
    *,
    line_number: int,
) -> tuple[tuple[str, ...], ...]:
    width = max(len(argument) for argument in arguments)
    invalid_widths = [
        len(argument)
        for argument in arguments
        if len(argument) not in {1, width}
    ]
    if invalid_widths:
        raise _fail(
            "qasm.register_width",
            f"Register arguments have incompatible sizes on line {line_number}",
        )
    return tuple(
        tuple(
            argument[0] if len(argument) == 1 else argument[index]
            for argument in arguments
        )
        for index in range(width)
    )


def _parse_gate_definition(
    header: _Token,
    body_tokens: tuple[_Token, ...],
) -> GateDefinition:
    match = re.fullmatch(
        r"gate\s+(?P<name>[A-Za-z_]\w*)"
        r"(?:\s*\([^)]*\))?\s+(?P<parameters>.+)",
        header.text,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise _fail(
            "qasm.gate_definition",
            f"Invalid gate definition on line {header.line_number}",
        )
    parameters = _split_operands(match.group("parameters"))
    if any(
        _IDENTIFIER_PATTERN.fullmatch(parameter) is None
        for parameter in parameters
    ):
        raise _fail(
            "qasm.gate_definition",
            f"Invalid gate parameter on line {header.line_number}",
        )
    if len(parameters) != len(set(parameters)):
        raise _fail(
            "qasm.gate_definition",
            f"Duplicate gate parameter on line {header.line_number}",
        )

    body: list[GateTemplate] = []
    for token in body_tokens:
        if token.terminator != ";":
            raise _fail(
                "qasm.gate_definition",
                f"Nested block in gate definition on line {token.line_number}",
            )
        name, operands = _parse_gate_call(token.text, token.line_number)
        if any(operand not in parameters for operand in operands):
            raise _fail(
                "qasm.gate_definition",
                f"Unknown gate parameter on line {token.line_number}",
            )
        body.append(
            GateTemplate(
                name=name,
                operands=operands,
                line_number=token.line_number,
            )
        )
    if not body:
        raise _fail(
            "qasm.gate_definition",
            f"Empty gate definition on line {header.line_number}",
        )
    return GateDefinition(
        name=match.group("name").lower(),
        parameters=parameters,
        body=tuple(body),
    )


def _expand_gate(
    name: str,
    concrete_qubits: tuple[str, ...],
    *,
    statement_line: int,
    definitions: dict[str, GateDefinition],
    source_lines: tuple[int, ...],
    call_stack: tuple[str, ...] = (),
) -> tuple[GateOperation, ...]:
    definition = definitions.get(name)
    if definition is None:
        return (
            GateOperation(
                name=name,
                qubits=concrete_qubits,
                statement_line=statement_line,
                source_lines=tuple(sorted(set(source_lines))),
            ),
        )

    if name in call_stack:
        raise _fail(
            "qasm.gate_recursion",
            f"Recursive custom gate call: {' -> '.join(call_stack + (name,))}",
        )
    if len(concrete_qubits) != len(definition.parameters):
        raise _fail(
            "qasm.gate_arity",
            f"Custom gate {name} expects {len(definition.parameters)} qubits",
        )

    parameter_values = dict(zip(definition.parameters, concrete_qubits))
    expanded: list[GateOperation] = []
    for template in definition.body:
        body_qubits = tuple(
            parameter_values[operand] for operand in template.operands
        )
        expanded.extend(
            _expand_gate(
                template.name,
                body_qubits,
                statement_line=statement_line,
                definitions=definitions,
                source_lines=source_lines + (template.line_number,),
                call_stack=call_stack + (name,),
            )
        )
    return tuple(expanded)


def _expand_gate_statement(
    statement: str,
    *,
    line_number: int,
    registers: dict[str, RegisterDeclaration],
    definitions: dict[str, GateDefinition],
) -> tuple[GateOperation, ...]:
    name, operand_texts = _parse_gate_call(statement, line_number)
    arguments = tuple(
        _parse_reference(
            operand,
            registers,
            expected_kind="qreg",
            line_number=line_number,
        )
        for operand in operand_texts
    )
    operations: list[GateOperation] = []
    for concrete_qubits in _broadcast_arguments(
        arguments,
        line_number=line_number,
    ):
        operations.extend(
            _expand_gate(
                name,
                concrete_qubits,
                statement_line=line_number,
                definitions=definitions,
                source_lines=(line_number,),
            )
        )
    return tuple(operations)


def parse_qasm(source: str) -> ParsedCircuit:
    """Parse the OpenQASM-2 structures required by the curated release."""
    tokens = _tokenize(source)
    registers: list[RegisterDeclaration] = []
    register_map: dict[str, RegisterDeclaration] = {}
    definitions: dict[str, GateDefinition] = {}
    operations: list[GateOperation] = []
    measurements: list[Measurement] = []
    corrections: list[ConditionalCorrection] = []
    saw_header = False
    includes: set[str] = set()

    index = 0
    while index < len(tokens):
        token = tokens[index]
        statement = token.text.strip()

        if token.terminator == "{":
            if not statement.lower().startswith("gate "):
                raise _fail(
                    "qasm.block",
                    f"Unexpected block on line {token.line_number}",
                )
            body_tokens: list[_Token] = []
            index += 1
            while index < len(tokens) and tokens[index].terminator != "}":
                body_tokens.append(tokens[index])
                index += 1
            if index >= len(tokens):
                raise _fail(
                    "qasm.gate_definition",
                    f"Unclosed gate definition on line {token.line_number}",
                )
            definition = _parse_gate_definition(token, tuple(body_tokens))
            if definition.name in definitions:
                raise _fail(
                    "qasm.gate_definition",
                    f"Duplicate gate definition: {definition.name}",
                )
            definitions[definition.name] = definition
            index += 1
            continue

        if token.terminator == "}":
            raise _fail(
                "qasm.block",
                f"Unexpected closing brace on line {token.line_number}",
            )
        if token.terminator != ";":
            raise _fail(
                "qasm.statement_termination",
                f"Invalid statement ending on line {token.line_number}",
            )
        if not statement:
            index += 1
            continue

        if statement.lower().startswith("openqasm "):
            if statement.upper() != "OPENQASM 2.0":
                raise _fail(
                    "qasm.version",
                    f"Expected OPENQASM 2.0 on line {token.line_number}",
                )
            if saw_header:
                raise _fail("qasm.version", "Duplicate OPENQASM header")
            saw_header = True
            index += 1
            continue

        include_match = re.fullmatch(
            r'include\s+"(?P<path>[^"]+)"',
            statement,
            flags=re.IGNORECASE,
        )
        if include_match is not None:
            includes.add(include_match.group("path"))
            index += 1
            continue

        register_match = _REGISTER_PATTERN.fullmatch(statement)
        if register_match is not None:
            name = register_match.group("name")
            size = int(register_match.group("size"))
            if size <= 0:
                raise _fail(
                    "qasm.register_size",
                    f"Register {name} must have positive size",
                )
            if name in register_map:
                raise _fail(
                    "qasm.register_declaration",
                    f"Duplicate register name: {name}",
                )
            declaration = RegisterDeclaration(
                kind=register_match.group("kind").lower(),
                name=name,
                size=size,
            )
            registers.append(declaration)
            register_map[name] = declaration
            index += 1
            continue

        if statement.lower().startswith("barrier "):
            for operand in _split_operands(statement.split(None, 1)[1]):
                _parse_reference(
                    operand,
                    register_map,
                    expected_kind="qreg",
                    line_number=token.line_number,
                )
            index += 1
            continue

        if statement.lower().startswith("measure "):
            measurement_match = re.fullmatch(
                r"measure\s+(?P<qubits>.+?)\s*->\s*(?P<bits>.+)",
                statement,
                flags=re.IGNORECASE,
            )
            if measurement_match is None:
                raise _fail(
                    "qasm.measurement",
                    f"Invalid measurement on line {token.line_number}",
                )
            qubits = _parse_reference(
                measurement_match.group("qubits"),
                register_map,
                expected_kind="qreg",
                line_number=token.line_number,
            )
            bits = _parse_reference(
                measurement_match.group("bits"),
                register_map,
                expected_kind="creg",
                line_number=token.line_number,
            )
            if len(qubits) != len(bits):
                raise _fail(
                    "qasm.measurement_width",
                    f"Measurement widths differ on line {token.line_number}",
                )
            measurements.extend(
                Measurement(
                    qubit=qubit,
                    classical_bit=bit,
                    line_number=token.line_number,
                )
                for qubit, bit in zip(qubits, bits)
            )
            index += 1
            continue

        condition_match = _CONDITION_PATTERN.fullmatch(statement)
        if condition_match is not None:
            condition_register = condition_match.group("register")
            declaration = register_map.get(condition_register)
            if declaration is None or declaration.kind != "creg":
                raise _fail(
                    "qasm.condition_register",
                    f"Unknown condition register: {condition_register}",
                )
            condition_value = int(condition_match.group("value"))
            if condition_value >= 2 ** declaration.size:
                raise _fail(
                    "qasm.condition_value",
                    f"Condition value does not fit register {condition_register}",
                )
            conditional_operations = _expand_gate_statement(
                condition_match.group("operation"),
                line_number=token.line_number,
                registers=register_map,
                definitions=definitions,
            )
            for operation in conditional_operations:
                if len(operation.qubits) != 1:
                    raise _fail(
                        "qasm.correction_arity",
                        "Conditional correction must target one qubit",
                    )
                corrections.append(
                    ConditionalCorrection(
                        condition_register=condition_register,
                        condition_value=condition_value,
                        gate=operation.name,
                        target_qubit=operation.qubits[0],
                        line_number=token.line_number,
                    )
                )
            index += 1
            continue

        operations.extend(
            _expand_gate_statement(
                statement,
                line_number=token.line_number,
                registers=register_map,
                definitions=definitions,
            )
        )
        index += 1

    if not saw_header:
        raise _fail("qasm.version", "Missing OPENQASM 2.0 header")
    if "qelib1.inc" not in includes:
        raise _fail("qasm.include", "Missing qelib1.inc include")
    if not any(register.kind == "qreg" for register in registers):
        raise _fail("qasm.register_declaration", "No quantum register declared")
    if not any(register.kind == "creg" for register in registers):
        raise _fail("qasm.register_declaration", "No classical register declared")

    return ParsedCircuit(
        registers=tuple(registers),
        gate_definitions=tuple(definitions.values()),
        operations=tuple(operations),
        measurements=tuple(measurements),
        corrections=tuple(corrections),
    )


def extract_stabilizer_checks(
    circuit: ParsedCircuit,
) -> tuple[StabilizerCheck, ...]:
    """Identify explicit parity checks measured into syndrome registers."""
    checks: list[StabilizerCheck] = []
    for measurement in circuit.measurements:
        syndrome_register = measurement.classical_bit.split("[", 1)[0].lower()
        if syndrome_register not in {"syn", "syndrome"}:
            continue

        participant_qubits: list[str] = []
        source_lines = {measurement.line_number}
        for operation in circuit.operations:
            if (
                operation.name == "cx"
                and len(operation.qubits) == 2
                and operation.qubits[1] == measurement.qubit
                and operation.statement_line < measurement.line_number
            ):
                data_qubit = operation.qubits[0]
                if data_qubit not in participant_qubits:
                    participant_qubits.append(data_qubit)
                source_lines.update(operation.source_lines)

        if len(participant_qubits) < 2:
            raise _fail(
                "qasm.stabilizer_structure",
                "Syndrome measurement does not have two data participants: "
                f"{measurement.classical_bit}",
            )
        checks.append(
            StabilizerCheck(
                ancilla_qubit=measurement.qubit,
                data_qubits=tuple(participant_qubits),
                syndrome_bit=measurement.classical_bit,
                source_lines=tuple(sorted(source_lines)),
            )
        )
    return tuple(checks)


def member_metadata(member: str) -> tuple[str, str]:
    """Return benchmark name and source/transpiled variant from a member path."""
    match = _MEMBER_PATTERN.fullmatch(member)
    if match is None:
        raise QasmBenchArchiveError(f"Unexpected QASM member path: {member}")
    benchmark_name = match.group("benchmark")
    filename = PurePosixPath(member).stem
    if filename == benchmark_name:
        variant = "source"
    elif filename == f"{benchmark_name}_transpiled":
        variant = "transpiled"
    else:
        raise QasmBenchArchiveError(
            f"QASM filename does not match benchmark directory: {member}"
        )
    return benchmark_name, variant


def _circuit_source_record_id(
    input_sha256: str,
    member: str,
    member_sha256: str,
) -> str:
    return "qasm-circuit-source-" + stable_record_hash(
        {
            "input_sha256": input_sha256,
            "bronze_object": BRONZE_OBJECT,
            "archive_member": member,
            "member_sha256": member_sha256,
        }
    )


def _circuit_id(
    benchmark_name: str,
    variant: str,
    member_sha256: str,
) -> str:
    return "qasm-circuit-" + stable_record_hash(
        {
            "source_name": SOURCE_NAME,
            "benchmark_name": benchmark_name,
            "variant": variant,
            "member_sha256": member_sha256,
        }
    )


def _validate_curated_circuit(
    *,
    member: str,
    benchmark_name: str,
    circuit: ParsedCircuit,
    checks: tuple[StabilizerCheck, ...],
) -> None:
    if circuit.qubit_count != 5:
        raise _fail(
            "qasm.qubit_count",
            f"Curated n5 circuit must declare five qubits: {member}",
        )
    if circuit.measurement_count != 5:
        raise _fail(
            "qasm.measurement_count",
            f"Curated n5 circuit must execute five measurements: {member}",
        )
    if benchmark_name == "qec_sm_n5":
        if len(checks) != 2:
            raise _fail(
                "qasm.stabilizer_count",
                "qec_sm_n5 must contain two explicit parity checks",
            )
        correction_signature = {
            (
                correction.condition_register,
                correction.condition_value,
                correction.gate,
                correction.target_qubit,
            )
            for correction in circuit.corrections
        }
        expected_corrections = {
            ("syn", 1, "x", "q[0]"),
            ("syn", 2, "x", "q[2]"),
            ("syn", 3, "x", "q[1]"),
        }
        if correction_signature != expected_corrections:
            raise _fail(
                "qasm.correction_structure",
                "qec_sm_n5 syndrome-controlled corrections are incomplete",
            )
    elif checks or circuit.corrections:
        raise _fail(
            "qasm.unexpected_recovery",
            f"Unexpected explicit recovery structure in {member}",
        )


def build_qasmbench_tables(
    archive_bytes: bytes,
    *,
    expected_bytes: int,
    expected_sha256: str,
    run_id: str,
) -> QasmBenchTables:
    """Validate the Bronze archive and build the three QASMBench tables."""
    spec = SourceObjectSpec(
        source_name=SOURCE_NAME,
        bronze_object=BRONZE_OBJECT,
        expected_bytes=expected_bytes,
        expected_sha256=expected_sha256,
    )
    try:
        source_validation = validate_source_object(archive_bytes, spec)
    except SourceValidationError as error:
        raise QasmBenchArchiveError(str(error)) from error

    qasm_members = frozenset(
        member
        for member in source_validation.member_names
        if member.lower().endswith(".qasm")
    )
    if qasm_members != EXPECTED_QASM_MEMBERS:
        raise QasmBenchArchiveError(
            "QASM member mismatch; "
            f"missing={sorted(EXPECTED_QASM_MEMBERS - qasm_members)}, "
            f"unexpected={sorted(qasm_members - EXPECTED_QASM_MEMBERS)}"
        )

    circuit_rows: list[dict[str, object]] = []
    check_rows: list[dict[str, object]] = []
    correction_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    issue_rows: list[dict[str, object]] = []
    rejected_ids: set[str] = set()

    with ZipFile(BytesIO(archive_bytes)) as archive:
        for member in sorted(qasm_members):
            member_bytes = archive.read(member)
            member_sha256 = sha256_bytes(member_bytes)
            benchmark_name, variant = member_metadata(member)
            circuit_source_id = _circuit_source_record_id(
                source_validation.sha256,
                member,
                member_sha256,
            )

            try:
                source_text = member_bytes.decode("utf-8")
            except UnicodeDecodeError as error:
                issue_rows.append(
                    data_issue_row(
                        run_id=run_id,
                        source_name=SOURCE_NAME,
                        bronze_object=BRONZE_OBJECT,
                        input_sha256=source_validation.sha256,
                        archive_member=member,
                        record_locator="qasm_member",
                        source_record_id=circuit_source_id,
                        rule_id="qasm.encoding",
                        observed_value=str(error),
                        source_record={"member_sha256": member_sha256},
                        action="excluded_from_silver",
                        reason="QASM member must be valid UTF-8 text",
                    )
                )
                rejected_ids.add(circuit_source_id)
                continue

            try:
                circuit = parse_qasm(source_text)
                checks = extract_stabilizer_checks(circuit)
                _validate_curated_circuit(
                    member=member,
                    benchmark_name=benchmark_name,
                    circuit=circuit,
                    checks=checks,
                )
            except QasmParseError as error:
                issue_rows.append(
                    data_issue_row(
                        run_id=run_id,
                        source_name=SOURCE_NAME,
                        bronze_object=BRONZE_OBJECT,
                        input_sha256=source_validation.sha256,
                        archive_member=member,
                        record_locator="qasm_member",
                        source_record_id=circuit_source_id,
                        rule_id=error.rule_id,
                        observed_value=str(error),
                        source_record=source_text,
                        action="excluded_from_silver",
                        reason=str(error),
                    )
                )
                rejected_ids.add(circuit_source_id)
                continue

            circuit_id = _circuit_id(
                benchmark_name,
                variant,
                member_sha256,
            )
            circuit_rows.append(
                {
                    "source_record_id": circuit_source_id,
                    "circuit_id": circuit_id,
                    "benchmark_name": benchmark_name,
                    "variant": variant,
                    "register_declarations": circuit.register_declarations,
                    "qubit_count": circuit.qubit_count,
                    "measurement_count": circuit.measurement_count,
                    "two_qubit_gate_count": circuit.two_qubit_gate_count,
                    "member_sha256": member_sha256,
                    "classical_bit_count": circuit.classical_bit_count,
                    "operation_count": circuit.operation_count,
                }
            )
            trace_rows.append(
                source_trace_row(
                    source_record_id=circuit_source_id,
                    source_name=SOURCE_NAME,
                    bronze_object=BRONZE_OBJECT,
                    archive_member=member,
                    record_locator="qasm_member",
                    input_sha256=source_validation.sha256,
                )
            )

            for check in checks:
                check_id = "qasm-check-" + stable_record_hash(
                    {
                        "circuit_id": circuit_id,
                        "ancilla_qubit": check.ancilla_qubit,
                        "data_qubits": check.data_qubits,
                        "syndrome_bit": check.syndrome_bit,
                    }
                )
                source_record_id = "qasm-check-source-" + stable_record_hash(
                    {
                        "circuit_source_record_id": circuit_source_id,
                        "check_id": check_id,
                    }
                )
                check_rows.append(
                    {
                        "source_record_id": source_record_id,
                        "circuit_id": circuit_id,
                        "check_id": check_id,
                        "ancilla_qubit": check.ancilla_qubit,
                        "data_qubits": list(check.data_qubits),
                        "syndrome_bit": check.syndrome_bit,
                    }
                )
                trace_rows.append(
                    source_trace_row(
                        source_record_id=source_record_id,
                        source_name=SOURCE_NAME,
                        bronze_object=BRONZE_OBJECT,
                        archive_member=member,
                        record_locator=(
                            "qasm_lines="
                            + ",".join(str(line) for line in check.source_lines)
                        ),
                        input_sha256=source_validation.sha256,
                    )
                )

            for correction in circuit.corrections:
                source_record_id = "qasm-correction-source-" + stable_record_hash(
                    {
                        "circuit_source_record_id": circuit_source_id,
                        "line_number": correction.line_number,
                        "condition_register": correction.condition_register,
                        "condition_value": correction.condition_value,
                        "gate": correction.gate,
                        "target_qubit": correction.target_qubit,
                    }
                )
                correction_rows.append(
                    {
                        "source_record_id": source_record_id,
                        "circuit_id": circuit_id,
                        "condition_register": correction.condition_register,
                        "condition_value": correction.condition_value,
                        "gate": correction.gate,
                        "target_qubit": correction.target_qubit,
                    }
                )
                trace_rows.append(
                    source_trace_row(
                        source_record_id=source_record_id,
                        source_name=SOURCE_NAME,
                        bronze_object=BRONZE_OBJECT,
                        archive_member=member,
                        record_locator=f"qasm_line={correction.line_number}",
                        input_sha256=source_validation.sha256,
                    )
                )

    input_count = len(qasm_members)
    accepted_count = len(circuit_rows)
    rejected_count = len(rejected_ids)
    if accepted_count + rejected_count != input_count:
        raise RuntimeError(
            "Internal QASMBench reconciliation failure: "
            "accepted + rejected != input"
        )

    silver_ids = [
        str(row["source_record_id"])
        for rows in (circuit_rows, check_rows, correction_rows)
        for row in rows
    ]
    trace_ids = [str(row["source_record_id"]) for row in trace_rows]
    if len(silver_ids) != len(set(silver_ids)):
        raise RuntimeError("Generated duplicate QASMBench source_record_id values")
    if len(trace_ids) != len(set(trace_ids)):
        raise RuntimeError("Generated duplicate QASMBench trace rows")
    if set(silver_ids) != set(trace_ids):
        raise RuntimeError("Source trace does not cover every QASMBench Silver row")

    return QasmBenchTables(
        circuits=pa.Table.from_pylist(circuit_rows, schema=CIRCUIT_SCHEMA),
        stabilizer_checks=pa.Table.from_pylist(
            check_rows,
            schema=STABILIZER_CHECK_SCHEMA,
        ),
        conditional_corrections=pa.Table.from_pylist(
            correction_rows,
            schema=CONDITIONAL_CORRECTION_SCHEMA,
        ),
        source_trace=pa.Table.from_pylist(
            trace_rows,
            schema=SOURCE_TRACE_SCHEMA,
        ),
        data_issues=pa.Table.from_pylist(
            issue_rows,
            schema=DATA_ISSUES_SCHEMA,
        ),
        input_sha256=source_validation.sha256,
        input_count=input_count,
        accepted_count=accepted_count,
        rejected_count=rejected_count,
    )


__all__ = [
    "BRONZE_OBJECT",
    "CIRCUIT_OBJECT",
    "CIRCUIT_SCHEMA",
    "CONDITIONAL_CORRECTION_OBJECT",
    "CONDITIONAL_CORRECTION_SCHEMA",
    "EXPECTED_QASM_MEMBERS",
    "MANIFEST_OBJECT",
    "QasmBenchArchiveError",
    "QasmBenchTables",
    "QasmParseError",
    "STABILIZER_CHECK_OBJECT",
    "STABILIZER_CHECK_SCHEMA",
    "build_qasmbench_tables",
    "extract_stabilizer_checks",
    "member_metadata",
    "parse_qasm",
]
