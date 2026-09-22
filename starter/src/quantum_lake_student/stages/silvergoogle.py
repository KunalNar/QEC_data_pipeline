"""Prepare Google QEC Silver tables and shared Part I evidence."""

from __future__ import annotations

from pathlib import Path

import pyarrow.compute as pc

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import read_lake_object
from quantum_lake_student.google_qec import (
    BRONZE_OBJECT,
    EXPECTED_EXPERIMENT_COUNT,
    EXPERIMENT_PATH,
    SHOT_PATH,
    SHOT_TRACE_MEMBER_COUNT,
    SOURCE_NAME,
    GoogleQecTables,
    build_google_qec_tables,
)
from quantum_lake_student.models import StageResult
from quantum_lake_student.part1_results import (
    check_result,
    passed_checks,
    publish_part1_results,
    rules_check,
)
from quantum_lake_student.source_validation import (
    MANIFEST_OBJECT,
    read_release_manifest,
    source_spec,
)


def _check_results(
    tables: GoogleQecTables,
) -> dict[str, dict[str, object]]:
    experiment_ids = set(
        tables.experiments.column("source_record_id").to_pylist()
    )
    shot_ids = set(tables.shots.column("source_record_id").to_pylist())
    silver_ids = experiment_ids | shot_ids
    trace_ids = set(
        pc.unique(
            tables.source_trace.column("source_record_id")
        ).to_pylist()
    )
    experiment_keys = set(
        tables.experiments.column("experiment_id").to_pylist()
    )
    shot_experiment_keys = set(
        tables.shots.column("experiment_id").to_pylist()
    )
    issue_rules = set(
        tables.data_issues.column("rule_id").to_pylist()
    )
    expected_trace_rows = tables.experiments.num_rows + (
        tables.shots.num_rows * SHOT_TRACE_MEMBER_COUNT
    )

    checks = passed_checks(
        "bronze_size_and_sha256",
        "archive_member_paths",
        "archive_member_names_unique",
        "archive_crc",
        "required_companion_files",
        "directory_metadata_consistency",
    )
    checks.update(
        {
            "experiment_directory_count": check_result(
                tables.experiment_count == EXPECTED_EXPERIMENT_COUNT,
                actual=tables.experiment_count,
            ),
            "packed_file_lengths": rules_check(
                issue_rules,
                "google.b8_length",
            ),
            "packed_padding_bits": rules_check(
                issue_rules,
                "google.b8_padding",
            ),
            "binary_text_domain": rules_check(
                issue_rules,
                "google.01_domain",
            ),
            "binary_text_lengths": rules_check(
                issue_rules,
                "google.01_length",
            ),
            "row_reconciliation": check_result(
                tables.accepted_count + tables.rejected_count
                == tables.input_count
            ),
            "experiment_shot_relationship": check_result(
                shot_experiment_keys == experiment_keys
                and sum(tables.experiments.column("shots").to_pylist())
                == tables.shots.num_rows
            ),
            "source_record_id_uniqueness": check_result(
                len(silver_ids)
                == tables.experiments.num_rows + tables.shots.num_rows
            ),
            "source_trace_coverage": check_result(
                silver_ids == trace_ids
            ),
            "source_trace_companion_rows": check_result(
                tables.source_trace.num_rows == expected_trace_rows,
                actual=tables.source_trace.num_rows,
                expected=expected_trace_rows,
            ),
        }
    )
    return checks


def prepare_google_data(
    settings: Settings,
    *,
    run_id: str,
    results_root: Path = Path("results/part1"),
) -> StageResult:
    """Build and publish Google Silver, trace, and issue tables."""
    result = StageResult(stage="prepare_google_data", run_id=run_id)
    manifest = read_release_manifest(
        read_lake_object(settings, MANIFEST_OBJECT)
    )
    spec = source_spec(
        manifest,
        SOURCE_NAME,
        expected_object=BRONZE_OBJECT,
    )
    archive_bytes = read_lake_object(settings, BRONZE_OBJECT)
    tables = build_google_qec_tables(
        archive_bytes,
        spec=spec,
        run_id=run_id,
    )

    result.input_count = tables.input_count
    result.output_count = (
        tables.experiments.num_rows + tables.shots.num_rows
    )
    publish_part1_results(
        settings,
        result,
        results_root,
        source_name=SOURCE_NAME,
        manifest=manifest,
        input_hashes={BRONZE_OBJECT: tables.input_sha256},
        silver_tables={
            EXPERIMENT_PATH: tables.experiments,
            SHOT_PATH: tables.shots,
        },
        source_trace=tables.source_trace,
        data_issues=tables.data_issues,
        row_counts={
            "bronze_rows": tables.input_count,
            "experiment_rows": tables.experiments.num_rows,
            "shot_rows": tables.shots.num_rows,
            "silver_rows": (
                tables.experiments.num_rows + tables.shots.num_rows
            ),
            "rejected_rows": tables.rejected_count,
            "issue_rows": tables.data_issues.num_rows,
            "trace_rows": tables.source_trace.num_rows,
        },
        checks=_check_results(tables),
    )
    return result


def run(run_id: str) -> StageResult:
    """Run the Google Silver stage using environment-based settings."""
    return prepare_google_data(
        Settings.from_environment(),
        run_id=run_id,
    )


__all__ = ["prepare_google_data", "run"]
