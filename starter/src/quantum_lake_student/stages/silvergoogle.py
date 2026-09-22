"""Prepare the two required Google QEC Silver tables."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import atomic_write, read_lake_object
from quantum_lake_student.google_qec import (
    BRONZE_OBJECT,
    EXPERIMENT_PATH,
    SHOT_PATH,
    build_google_qec_tables,
    google_source_spec,
)
from quantum_lake_student.models import StageResult
from quantum_lake_student.source_validation import (
    MANIFEST_OBJECT,
    SourceValidationError,
)


ISSUES_PATH = Path("results/part1/data_issues_google_silver.json")


def _release_manifest(value: bytes) -> dict[str, object]:
    try:
        manifest = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceValidationError(
            "Release manifest is not valid UTF-8 JSON"
        ) from error
    if not isinstance(manifest, dict):
        raise SourceValidationError(
            "Release manifest root must be a JSON object"
        )
    return manifest


def _parquet_bytes(table: pa.Table) -> bytes:
    output = pa.BufferOutputStream()
    pq.write_table(table, output, compression="zstd")
    return output.getvalue().to_pybytes()


def prepare_google_data(
    settings: Settings,
    *,
    run_id: str,
    experiment_path: Path = Path(EXPERIMENT_PATH),
    shot_path: Path = Path(SHOT_PATH),
    issues_path: Path = ISSUES_PATH,
) -> StageResult:
    """Validate Google Bronze fully before publishing typed Parquet tables."""
    result = StageResult(stage="prepare_google_data", run_id=run_id)
    manifest = _release_manifest(read_lake_object(settings, MANIFEST_OBJECT))
    spec = google_source_spec(manifest)
    archive_bytes = read_lake_object(settings, BRONZE_OBJECT)
    tables = build_google_qec_tables(archive_bytes, spec=spec)

    experiment_output = _parquet_bytes(tables.experiments)
    shot_output = _parquet_bytes(tables.shots)
    issue_rows = [
        {
            "rule_id": finding.rule_id,
            "severity": finding.severity.value,
            "source_system": finding.source_system,
            "source_record_locator": finding.source_record_locator,
            "message": finding.message,
            "observed_value": finding.observed_value,
        }
        for finding in tables.findings
    ]
    issues_output = (
        json.dumps(issue_rows, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    atomic_write(experiment_path, experiment_output)
    atomic_write(shot_path, shot_output)
    atomic_write(issues_path, issues_output)

    result.input_count = tables.input_count
    result.output_count = (
        tables.experiments.num_rows + tables.shots.num_rows
    )
    result.issue_count = len(tables.findings)
    result.finish()
    return result


def run(run_id: str) -> StageResult:
    """Run the Google Silver stage using environment-based settings."""
    return prepare_google_data(
        Settings.from_environment(),
        run_id=run_id,
    )


__all__ = ["prepare_google_data", "run"]
