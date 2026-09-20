"""Build the qec_syndromes Silver, trace, and data-issue tables."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from io import BytesIO, TextIOWrapper
from pathlib import PurePosixPath
from zipfile import BadZipFile, ZipFile

import pyarrow as pa
import pyarrow.parquet as pq

from .models import stable_record_hash


SOURCE_NAME = "qec_syndromes"
BRONZE_OBJECT = "bronze/source=qec_syndromes/syndromes_dataset.zip"
MANIFEST_OBJECT = "metadata/course-release/bundle-manifest.json"
SILVER_OBJECT = "silver/qec_syndromes/syndrome_observation.parquet"
EXPECTED_COLUMNS = ("labels", "syndromes", "quantity")
EXPECTED_CSV_COUNT = 7

_FILENAME_PATTERN = re.compile(
    r"^d-(?P<distance>\d+)_pfr-(?P<fault_rate>\d+(?:\.\d+)?)_"
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

SOURCE_TRACE_SCHEMA = pa.schema(
    [
        pa.field("source_record_id", pa.string(), nullable=False),
        pa.field("source_name", pa.string(), nullable=False),
        pa.field("bronze_object", pa.string(), nullable=False),
        pa.field("archive_member", pa.string(), nullable=False),
        pa.field("record_locator", pa.string(), nullable=False),
        pa.field("input_sha256", pa.string(), nullable=False),
    ]
)

DATA_ISSUES_SCHEMA = pa.schema(
    [
        pa.field("issue_id", pa.string(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("source_record_id", pa.string(), nullable=True),
        pa.field("source_name", pa.string(), nullable=False),
        pa.field("bronze_object", pa.string(), nullable=False),
        pa.field("archive_member", pa.string(), nullable=False),
        pa.field("record_locator", pa.string(), nullable=False),
        pa.field("rule_id", pa.string(), nullable=False),
        pa.field("severity", pa.string(), nullable=False),
        pa.field("observed_value", pa.string(), nullable=False),
        pa.field("source_record", pa.string(), nullable=False),
        pa.field("action", pa.string(), nullable=False),
        pa.field("reason", pa.string(), nullable=False),
    ]
)


class SyndromeArchiveError(ValueError):
    """Raised when the archive cannot be processed safely."""


@dataclass(frozen=True)
class ExperimentMetadata:
    distance: int
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


def sha256_bytes(value: bytes) -> str:
    """Return the SHA-256 digest for exact input bytes."""
    return hashlib.sha256(value).hexdigest()


def is_safe_member_path(name: str) -> bool:
    """Return whether an archive member stays inside an extraction root."""
    normalized = name.replace("\\", "/")
    member_path = PurePosixPath(normalized)
    has_drive = bool(
        member_path.parts
        and re.fullmatch(r"[A-Za-z]:", member_path.parts[0])
    )
    return (
        bool(member_path.parts)
        and not member_path.is_absolute()
        and ".." not in member_path.parts
        and not has_drive
    )


def parse_filename(member: str) -> ExperimentMetadata:
    """Read distance, fault rate, and nominal count from a CSV filename."""
    match = _FILENAME_PATTERN.fullmatch(PurePosixPath(member).name)
    if match is None:
        raise SyndromeArchiveError(
            f"Unexpected syndrome CSV filename: {member}"
        )

    fields = match.groupdict()
    unit = fields["unit"].upper()
    return ExperimentMetadata(
        distance=int(fields["distance"]),
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


def _json_value(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def _normalized_source_record(row: dict[object, object]) -> dict[str, object]:
    return {
        "__extra_fields__" if key is None else str(key): value
        for key, value in row.items()
    }


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


def _issue(
    *,
    run_id: str,
    input_sha256: str,
    archive_member: str,
    record_locator: str,
    source_record_id: str | None,
    rule_id: str,
    observed_value: object,
    source_record: object,
    action: str,
    reason: str,
) -> dict[str, object]:
    observed_text = (
        observed_value
        if isinstance(observed_value, str)
        else _json_value(observed_value)
    )
    issue_id = "issue-" + stable_record_hash(
        {
            "input_sha256": input_sha256,
            "archive_member": archive_member,
            "record_locator": record_locator,
            "source_record_id": source_record_id,
            "rule_id": rule_id,
            "observed_value": observed_text,
        }
    )
    return {
        "issue_id": issue_id,
        "run_id": run_id,
        "source_record_id": source_record_id,
        "source_name": SOURCE_NAME,
        "bronze_object": BRONZE_OBJECT,
        "archive_member": archive_member,
        "record_locator": record_locator,
        "rule_id": rule_id,
        "severity": "error",
        "observed_value": observed_text,
        "source_record": _json_value(source_record),
        "action": action,
        "reason": reason,
    }


def build_syndrome_tables(
    archive_bytes: bytes,
    *,
    expected_sha256: str,
    run_id: str,
) -> SyndromeTables:
    """Validate the Bronze archive and construct all syndrome output tables."""
    input_sha256 = sha256_bytes(archive_bytes)
    if input_sha256 != expected_sha256:
        raise SyndromeArchiveError(
            "Syndrome archive SHA-256 does not match the release manifest"
        )

    silver_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    issue_rows: list[dict[str, object]] = []
    rejected_ids: set[str] = set()
    input_count = 0

    try:
        archive = ZipFile(BytesIO(archive_bytes))
    except BadZipFile as error:
        raise SyndromeArchiveError("Syndrome object is not a valid ZIP") from error

    with archive:
        infos = archive.infolist()
        member_names = [info.filename for info in infos]

        unsafe = [
            name for name in member_names
            if not is_safe_member_path(name)
        ]
        if unsafe:
            raise SyndromeArchiveError(
                f"Unsafe archive member path: {unsafe[0]}"
            )

        if len(member_names) != len(set(member_names)):
            raise SyndromeArchiveError(
                "Syndrome archive contains duplicate member names"
            )

        bad_crc_member = archive.testzip()
        if bad_crc_member is not None:
            raise SyndromeArchiveError(
                f"CRC check failed for archive member: {bad_crc_member}"
            )

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
                        source_record = _normalized_source_record(row)
                        record_issues: list[dict[str, object]] = []

                        if (
                            None in row
                            or any(row.get(column) is None for column in EXPECTED_COLUMNS)
                        ):
                            record_issues.append(
                                _issue(
                                    run_id=run_id,
                                    input_sha256=input_sha256,
                                    archive_member=member,
                                    record_locator=record_locator,
                                    source_record_id=source_record_id,
                                    rule_id="syndrome.csv_row_shape",
                                    observed_value=source_record,
                                    source_record=source_record,
                                    action="excluded_from_silver",
                                    reason=(
                                        "CSV row must contain exactly the "
                                        "three documented values"
                                    ),
                                )
                            )

                        label, label_rule, label_reason = _parse_label(
                            row.get("labels")
                        )
                        if label_rule is not None:
                            record_issues.append(
                                _issue(
                                    run_id=run_id,
                                    input_sha256=input_sha256,
                                    archive_member=member,
                                    record_locator=record_locator,
                                    source_record_id=source_record_id,
                                    rule_id=label_rule,
                                    observed_value=row.get("labels"),
                                    source_record=source_record,
                                    action="excluded_from_silver",
                                    reason=label_reason or "",
                                )
                            )

                        syndrome_bits, syndrome_rule, syndrome_reason = (
                            parse_syndrome(row.get("syndromes"))
                        )
                        if syndrome_rule is not None:
                            record_issues.append(
                                _issue(
                                    run_id=run_id,
                                    input_sha256=input_sha256,
                                    archive_member=member,
                                    record_locator=record_locator,
                                    source_record_id=source_record_id,
                                    rule_id=syndrome_rule,
                                    observed_value=row.get("syndromes"),
                                    source_record=source_record,
                                    action="excluded_from_silver",
                                    reason=syndrome_reason or "",
                                )
                            )

                        quantity, quantity_rule, quantity_reason = (
                            _parse_quantity(row.get("quantity"))
                        )
                        if quantity_rule is not None:
                            record_issues.append(
                                _issue(
                                    run_id=run_id,
                                    input_sha256=input_sha256,
                                    archive_member=member,
                                    record_locator=record_locator,
                                    source_record_id=source_record_id,
                                    rule_id=quantity_rule,
                                    observed_value=row.get("quantity"),
                                    source_record=source_record,
                                    action="excluded_from_silver",
                                    reason=quantity_reason or "",
                                )
                            )
                        elif quantity is not None:
                            file_weight_total += quantity

                        if record_issues:
                            issue_rows.extend(record_issues)
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
                            {
                                "source_record_id": source_record_id,
                                "source_name": SOURCE_NAME,
                                "bronze_object": BRONZE_OBJECT,
                                "archive_member": member,
                                "record_locator": record_locator,
                                "input_sha256": input_sha256,
                            }
                        )

            if file_weight_total != metadata.nominal_observations:
                issue_rows.append(
                    _issue(
                        run_id=run_id,
                        input_sha256=input_sha256,
                        archive_member=member,
                        record_locator="archive_member",
                        source_record_id=None,
                        rule_id="syndrome.weighted_total",
                        observed_value={
                            "actual": file_weight_total,
                            "expected": metadata.nominal_observations,
                        },
                        source_record={"archive_member": member},
                        action="published_valid_rows_only",
                        reason=(
                            "Positive quantities do not reconcile to the "
                            "nominal observation count in the filename"
                        ),
                    )
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
        data_issues=pa.Table.from_pylist(
            issue_rows,
            schema=DATA_ISSUES_SCHEMA,
        ),
        input_sha256=input_sha256,
        file_count=len(csv_members),
        input_count=input_count,
        accepted_count=accepted_count,
        rejected_count=rejected_count,
    )


def parquet_bytes(table: pa.Table) -> bytes:
    """Serialize one Arrow table to deterministic compressed Parquet bytes."""
    output = pa.BufferOutputStream()
    pq.write_table(table, output, compression="zstd")
    return output.getvalue().to_pybytes()
