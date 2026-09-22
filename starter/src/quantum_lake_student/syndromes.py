"""Build the qec_syndromes Silver, trace, and data-issue tables."""

from __future__ import annotations

import ast
import csv
import re
from dataclasses import dataclass
from io import BytesIO, TextIOWrapper
from pathlib import PurePosixPath
from zipfile import ZipFile

import pyarrow as pa

from .evidence import (
    DATA_ISSUES_SCHEMA,
    IssueLog,
    SOURCE_TRACE_SCHEMA,
    parquet_bytes,
    source_trace_row,
)
from .models import stable_record_hash
from .source_validation import (
    SourceObjectSpec,
    SourceValidationError,
    validate_source_object,
)


SOURCE_NAME = "qec_syndromes"
BRONZE_OBJECT = "bronze/source=qec_syndromes/syndromes_dataset.zip"
SILVER_OBJECT = "silver/qec_syndromes/syndrome_observation.parquet"
EXPECTED_COLUMNS = ("labels", "syndromes", "quantity")
EXPECTED_CSV_COUNT = 7

_FILENAME_PATTERN = re.compile(
    r"^d-3_pfr-(?P<fault_rate>\d+(?:\.\d+)?)_"
    r"nb-(?P<count>\d+)(?P<unit>[KMG]?)\.csv$",
    flags=re.IGNORECASE,
)

_COUNT_MULTIPLIERS = {
    "": 1,
    "K": 1_000,
    "M": 1_000_000,
    "G": 1_000_000_000,
}


SILVER_SCHEMA = pa.schema(
    [
        pa.field("source_record_id", pa.string(), nullable=False),
        pa.field("experiment_id", pa.string(), nullable=False),
        pa.field("physical_fault_rate", pa.float64(), nullable=False),
        pa.field("syndrome_bits", pa.binary(), nullable=False),
        pa.field("round_count", pa.int32(), nullable=False),
        pa.field("check_count", pa.int32(), nullable=False),
        pa.field("logical_error_label", pa.bool_(), nullable=False),
        pa.field("quantity", pa.int64(), nullable=False),
    ]
)


class SyndromeArchiveError(ValueError):
    """Raised when the archive cannot be processed safely."""


@dataclass(frozen=True)
class ExperimentMetadata:
    physical_fault_rate: float
    nominal_observations: int


@dataclass(frozen=True)
class SyndromeTables:
    silver: pa.Table
    source_trace: pa.Table
    data_issues: pa.Table
    input_sha256: str
    file_count: int
    input_count: int
    accepted_count: int
    rejected_count: int


def parse_filename(member: str) -> ExperimentMetadata:
    """Validate distance 3 and read fault rate and nominal count."""
    match = _FILENAME_PATTERN.fullmatch(PurePosixPath(member).name)
    if match is None:
        raise SyndromeArchiveError(
            f"Unexpected syndrome CSV filename: {member}"
        )

    fields = match.groupdict()
    unit = fields["unit"].upper()
    return ExperimentMetadata(
        physical_fault_rate=float(fields["fault_rate"]),
        nominal_observations=(
            int(fields["count"]) * _COUNT_MULTIPLIERS[unit]
        ),
    )


def parse_syndrome(value: object) -> tuple[bytes | None, str | None, str | None]:
    """Parse a 4-by-4 binary syndrome into sixteen one-byte values."""
    if not isinstance(value, str):
        return None, "syndrome.type", "Syndrome must be a string"

    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as error:
        return None, "syndrome.parse", f"Cannot parse syndrome: {error}"

    if not isinstance(parsed, (tuple, list)) or len(parsed) != 4:
        return (
            None,
            "syndrome.shape",
            "Syndrome must contain exactly four rounds",
        )

    if not all(
        isinstance(round_values, (tuple, list))
        and len(round_values) == 4
        for round_values in parsed
    ):
        return (
            None,
            "syndrome.shape",
            "Each syndrome round must contain exactly four values",
        )

    if not all(
        type(bit) is int and bit in (0, 1)
        for round_values in parsed
        for bit in round_values
    ):
        return (
            None,
            "syndrome.bit_domain",
            "Every syndrome value must be integer 0 or 1",
        )

    return (
        bytes(
            bit
            for round_values in parsed
            for bit in round_values
        ),
        None,
        None,
    )


def _parse_label(value: object) -> tuple[bool | None, str | None, str | None]:
    if not isinstance(value, str) or value.strip() not in {"0", "1"}:
        return (
            None,
            "syndrome.label_domain",
            "Logical-error label must be 0 or 1",
        )
    return value.strip() == "1", None, None


def _parse_quantity(value: object) -> tuple[int | None, str | None, str | None]:
    if not isinstance(value, str):
        return (
            None,
            "syndrome.quantity_integer",
            "Quantity must be an integer",
        )

    try:
        quantity = int(value.strip())
    except ValueError:
        return (
            None,
            "syndrome.quantity_integer",
            "Quantity must be an integer",
        )

    if quantity <= 0:
        return (
            None,
            "syndrome.quantity_positive",
            "Quantity must be greater than zero",
        )
    return quantity, None, None


def _source_record_id(
    input_sha256: str,
    archive_member: str,
    source_row_number: int,
) -> str:
    return "qec-syn-" + stable_record_hash(
        {
            "input_sha256": input_sha256,
            "bronze_object": BRONZE_OBJECT,
            "archive_member": archive_member,
            "source_row_number": source_row_number,
        }
    )


def _experiment_id(input_sha256: str, archive_member: str) -> str:
    return "qec-exp-" + stable_record_hash(
        {
            "input_sha256": input_sha256,
            "bronze_object": BRONZE_OBJECT,
            "archive_member": archive_member,
        }
    )


def build_syndrome_tables(
    archive_bytes: bytes,
    *,
    spec: SourceObjectSpec,
    run_id: str,
) -> SyndromeTables:
    """Validate the Bronze archive and construct all syndrome output tables."""
    try:
        source_validation = validate_source_object(archive_bytes, spec)
    except SourceValidationError as error:
        raise SyndromeArchiveError(str(error)) from error

    input_sha256 = source_validation.sha256

    silver_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    issues = IssueLog(
        run_id=run_id,
        source_name=SOURCE_NAME,
        bronze_object=BRONZE_OBJECT,
        input_sha256=input_sha256,
    )
    rejected_ids: set[str] = set()
    input_count = 0

    with ZipFile(BytesIO(archive_bytes)) as archive:
        member_names = source_validation.member_names
        csv_members = sorted(
            name for name in member_names
            if name.lower().endswith(".csv")
        )
        if len(csv_members) != EXPECTED_CSV_COUNT:
            raise SyndromeArchiveError(
                "Expected "
                f"{EXPECTED_CSV_COUNT} syndrome CSV files, "
                f"found {len(csv_members)}"
            )

        for member in csv_members:
            metadata = parse_filename(member)
            experiment_id = _experiment_id(input_sha256, member)
            file_weight_total = 0

            with archive.open(member) as binary_file:
                with TextIOWrapper(
                    binary_file,
                    encoding="utf-8-sig",
                    newline="",
                ) as text_file:
                    reader = csv.DictReader(text_file)
                    if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
                        raise SyndromeArchiveError(
                            f"Unexpected CSV header in {member}: "
                            f"{reader.fieldnames}"
                        )

                    for source_row_number, row in enumerate(reader, start=1):
                        input_count += 1
                        record_locator = (
                            f"csv_data_row={source_row_number}"
                        )
                        source_record_id = _source_record_id(
                            input_sha256,
                            member,
                            source_row_number,
                        )
                        source_record = row
                        issue_start = len(issues)

                        if (
                            None in row
                            or any(
                                row.get(column) is None
                                for column in EXPECTED_COLUMNS
                            )
                        ):
                            issues.add(
                                "syndrome.csv_row_shape",
                                (
                                    "CSV row must contain exactly the "
                                    "three documented values"
                                ),
                                member=member,
                                locator=record_locator,
                                record_id=source_record_id,
                                value=source_record,
                            )

                        label, label_rule, label_reason = _parse_label(
                            row.get("labels")
                        )
                        if label_rule is not None:
                            issues.add(
                                label_rule,
                                label_reason or "",
                                member=member,
                                locator=record_locator,
                                record_id=source_record_id,
                                value=row.get("labels"),
                                record=source_record,
                            )

                        syndrome_bits, syndrome_rule, syndrome_reason = (
                            parse_syndrome(row.get("syndromes"))
                        )
                        if syndrome_rule is not None:
                            issues.add(
                                syndrome_rule,
                                syndrome_reason or "",
                                member=member,
                                locator=record_locator,
                                record_id=source_record_id,
                                value=row.get("syndromes"),
                                record=source_record,
                            )

                        quantity, quantity_rule, quantity_reason = (
                            _parse_quantity(row.get("quantity"))
                        )
                        if quantity_rule is not None:
                            issues.add(
                                quantity_rule,
                                quantity_reason or "",
                                member=member,
                                locator=record_locator,
                                record_id=source_record_id,
                                value=row.get("quantity"),
                                record=source_record,
                            )
                        elif quantity is not None:
                            file_weight_total += quantity

                        if len(issues) > issue_start:
                            rejected_ids.add(source_record_id)
                            continue

                        assert label is not None
                        assert syndrome_bits is not None
                        assert quantity is not None

                        silver_rows.append(
                            {
                                "source_record_id": source_record_id,
                                "experiment_id": experiment_id,
                                "physical_fault_rate": (
                                    metadata.physical_fault_rate
                                ),
                                "syndrome_bits": syndrome_bits,
                                "round_count": 4,
                                "check_count": 4,
                                "logical_error_label": label,
                                "quantity": quantity,
                            }
                        )
                        trace_rows.append(
                            source_trace_row(
                                source_record_id=source_record_id,
                                source_name=SOURCE_NAME,
                                bronze_object=BRONZE_OBJECT,
                                archive_member=member,
                                record_locator=record_locator,
                                input_sha256=input_sha256,
                            )
                        )

            if file_weight_total != metadata.nominal_observations:
                issues.add(
                    "syndrome.weighted_total",
                    (
                        "Positive quantities do not reconcile to the "
                        "nominal observation count in the filename"
                    ),
                    member=member,
                    locator="archive_member",
                    value={
                        "actual": file_weight_total,
                        "expected": metadata.nominal_observations,
                    },
                    record={"archive_member": member},
                    action="published_valid_rows_only",
                )

    accepted_count = len(silver_rows)
    rejected_count = len(rejected_ids)

    if accepted_count + rejected_count != input_count:
        raise RuntimeError(
            "Internal reconciliation failure: accepted + rejected != input"
        )

    silver_ids = [str(row["source_record_id"]) for row in silver_rows]
    trace_ids = [str(row["source_record_id"]) for row in trace_rows]
    if len(silver_ids) != len(set(silver_ids)):
        raise RuntimeError("Generated duplicate syndrome source_record_id values")
    if set(silver_ids) != set(trace_ids):
        raise RuntimeError("Source trace does not cover every Silver row")

    return SyndromeTables(
        silver=pa.Table.from_pylist(silver_rows, schema=SILVER_SCHEMA),
        source_trace=pa.Table.from_pylist(
            trace_rows,
            schema=SOURCE_TRACE_SCHEMA,
        ),
        data_issues=issues.table(),
        input_sha256=input_sha256,
        file_count=len(csv_members),
        input_count=input_count,
        accepted_count=accepted_count,
        rejected_count=rejected_count,
    )
