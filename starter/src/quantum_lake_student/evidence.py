"""Part I trace, data-issue, and Parquet helpers."""

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


def _json_value(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


class IssueLog:
    """Collect assignment data issues with their shared source context."""

    def __init__(
        self,
        *,
        run_id: str,
        source_name: str,
        bronze_object: str,
        input_sha256: str,
        default_action: str = "excluded_from_silver",
        default_locator: str = "",
    ) -> None:
        self.run_id = run_id
        self.source_name = source_name
        self.bronze_object = bronze_object
        self.input_sha256 = input_sha256
        self.default_action = default_action
        self.default_locator = default_locator
        self.rows: list[dict[str, object]] = []

    def __len__(self) -> int:
        return len(self.rows)

    def add(
        self,
        rule_id: str,
        reason: str,
        *,
        member: str,
        value: object,
        locator: str | None = None,
        record_id: str | None = None,
        record: object | None = None,
        action: str | None = None,
        severity: str = "error",
    ) -> None:
        locator = self.default_locator if locator is None else locator
        action = self.default_action if action is None else action
        observed_value = (
            value if isinstance(value, str) else _json_value(value)
        )
        issue_id = "issue-" + stable_record_hash(
            {
                "input_sha256": self.input_sha256,
                "archive_member": member,
                "record_locator": locator,
                "source_record_id": record_id,
                "rule_id": rule_id,
                "observed_value": observed_value,
            }
        )
        self.rows.append(
            {
                "issue_id": issue_id,
                "run_id": self.run_id,
                "source_record_id": record_id,
                "source_name": self.source_name,
                "bronze_object": self.bronze_object,
                "archive_member": member,
                "record_locator": locator,
                "rule_id": rule_id,
                "severity": severity,
                "observed_value": observed_value,
                "source_record": _json_value(
                    value if record is None else record
                ),
                "action": action,
                "reason": reason,
            }
        )

    def table(self) -> pa.Table:
        return pa.Table.from_pylist(self.rows, schema=DATA_ISSUES_SCHEMA)


def source_trace_row(
    *,
    source_record_id: str,
    source_name: str,
    bronze_object: str,
    archive_member: str,
    record_locator: str,
    input_sha256: str,
) -> dict[str, object]:
    """Build one row for the source-trace table."""
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
