"""Shared Part I trace, data-issue, and Parquet helpers."""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq

from .models import stable_record_hash


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


def json_value(value: object) -> str:
    """Serialize evidence values consistently and readably."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def data_issue_row(
    *,
    run_id: str,
    source_name: str,
    bronze_object: str,
    input_sha256: str,
    archive_member: str,
    record_locator: str,
    source_record_id: str | None,
    rule_id: str,
    observed_value: object,
    source_record: object,
    action: str,
    reason: str,
    severity: str = "error",
) -> dict[str, object]:
    """Build one stable row for the shared Part I data-issues table."""
    observed_text = (
        observed_value
        if isinstance(observed_value, str)
        else json_value(observed_value)
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
        "source_name": source_name,
        "bronze_object": bronze_object,
        "archive_member": archive_member,
        "record_locator": record_locator,
        "rule_id": rule_id,
        "severity": severity,
        "observed_value": observed_text,
        "source_record": json_value(source_record),
        "action": action,
        "reason": reason,
    }


def source_trace_row(
    *,
    source_record_id: str,
    source_name: str,
    bronze_object: str,
    archive_member: str,
    record_locator: str,
    input_sha256: str,
) -> dict[str, object]:
    """Build one row for the shared Part I source-trace table."""
    return {
        "source_record_id": source_record_id,
        "source_name": source_name,
        "bronze_object": bronze_object,
        "archive_member": archive_member,
        "record_locator": record_locator,
        "input_sha256": input_sha256,
    }


def parquet_bytes(table: pa.Table) -> bytes:
    """Serialize one Arrow table to deterministic compressed Parquet bytes."""
    output = pa.BufferOutputStream()
    pq.write_table(table, output, compression="zstd")
    return output.getvalue().to_pybytes()
