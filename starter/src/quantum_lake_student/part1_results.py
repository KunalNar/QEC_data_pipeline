"""Merge source-specific evidence into the shared Part I result files."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .connections import atomic_write
from .evidence import DATA_ISSUES_SCHEMA, SOURCE_TRACE_SCHEMA, parquet_bytes


def code_revision() -> tuple[str, str]:
    """Return a supplied Git revision or a deterministic source-tree hash."""
    supplied_revision = os.getenv("CODE_REVISION")
    if supplied_revision:
        return "git", supplied_revision

    package_root = Path(__file__).resolve().parent
    hasher = hashlib.sha256()
    for source_path in sorted(package_root.rglob("*.py")):
        relative_path = source_path.relative_to(package_root).as_posix()
        hasher.update(relative_path.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(source_path.read_bytes())
        hasher.update(b"\0")
    return "source_tree_sha256", hasher.hexdigest()


def _read_existing_table(path: Path, schema: pa.Schema) -> pa.Table:
    if not path.is_file():
        return pa.Table.from_pylist([], schema=schema)
    table = pq.read_table(path)
    if table.schema != schema:
        raise ValueError(
            f"Existing evidence schema does not match {path.name}: "
            f"{table.schema}"
        )
    return table


def _replace_source_rows(
    existing: pa.Table,
    replacement: pa.Table,
    *,
    source_name: str,
    sort_keys: list[tuple[str, str]],
) -> pa.Table:
    if existing.num_rows:
        keep = pc.not_equal(
            existing.column("source_name"),
            pa.scalar(source_name),
        )
        existing = existing.filter(keep)
    combined = pa.concat_tables([existing, replacement])
    if combined.num_rows:
        combined = combined.sort_by(sort_keys)
    return combined


def merge_evidence(
    results_root: Path,
    *,
    source_name: str,
    source_trace: pa.Table,
    data_issues: pa.Table,
) -> tuple[pa.Table, bytes, pa.Table, bytes]:
    """Replace one source's evidence while retaining the other sources."""
    if source_trace.schema != SOURCE_TRACE_SCHEMA:
        raise ValueError("Source-trace table does not match the shared schema")
    if data_issues.schema != DATA_ISSUES_SCHEMA:
        raise ValueError("Data-issues table does not match the shared schema")

    trace_path = results_root / "source_trace.parquet"
    issues_path = results_root / "data_issues.parquet"
    merged_trace = _replace_source_rows(
        _read_existing_table(trace_path, SOURCE_TRACE_SCHEMA),
        source_trace,
        source_name=source_name,
        sort_keys=[
            ("source_name", "ascending"),
            ("source_record_id", "ascending"),
            ("archive_member", "ascending"),
            ("record_locator", "ascending"),
        ],
    )
    merged_issues = _replace_source_rows(
        _read_existing_table(issues_path, DATA_ISSUES_SCHEMA),
        data_issues,
        source_name=source_name,
        sort_keys=[
            ("source_name", "ascending"),
            ("issue_id", "ascending"),
        ],
    )
    trace_output = parquet_bytes(merged_trace)
    issues_output = parquet_bytes(merged_issues)
    atomic_write(trace_path, trace_output)
    atomic_write(issues_path, issues_output)
    return merged_trace, trace_output, merged_issues, issues_output


def update_row_counts(
    results_root: Path,
    *,
    source_name: str,
    counts: dict[str, int],
) -> dict[str, object]:
    """Replace one source's counts in the shared reconciliation file."""
    path = results_root / "row_counts.json"
    if path.is_file():
        content = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(content, dict):
            raise ValueError("row_counts.json must contain a JSON object")
    else:
        content = {}
    content[source_name] = counts
    atomic_write(
        path,
        (json.dumps(content, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return content


def update_run_record(
    results_root: Path,
    *,
    run_id: str,
    stage: str,
    release: dict[str, object],
    started_at: str,
    finished_at: str,
    source_name: str,
    input_hashes: dict[str, str],
    outputs: dict[str, dict[str, object]],
    checks: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Update shared run evidence without discarding other source stages."""
    path = results_root / "run.json"
    previous: dict[str, object] = {}
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("run.json must contain a JSON object")
        previous = loaded

    previous_inputs = previous.get("input_hashes", {})
    previous_outputs = previous.get("outputs", {})
    previous_checks = previous.get("checks", {})
    if not isinstance(previous_inputs, dict):
        previous_inputs = {}
    if not isinstance(previous_outputs, dict):
        previous_outputs = {}
    if not isinstance(previous_checks, dict):
        previous_checks = {}

    # Migrate the earlier syndrome-only flat check layout when encountered.
    if previous_checks and all(
        isinstance(value, dict) and "status" in value
        for value in previous_checks.values()
    ):
        previous_checks = {"qec_syndromes": previous_checks}

    revision_kind, revision = code_revision()
    run_record = {
        "run_id": run_id,
        "stage": stage,
        "release": release,
        "code_revision": revision,
        "code_revision_kind": revision_kind,
        "started_at": started_at,
        "finished_at": finished_at,
        "input_hashes": {**previous_inputs, **input_hashes},
        "outputs": {**previous_outputs, **outputs},
        "checks": {**previous_checks, source_name: checks},
    }
    atomic_write(
        path,
        (json.dumps(run_record, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )
    return run_record
