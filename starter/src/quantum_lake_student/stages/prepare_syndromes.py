"""Prepare the qec_syndromes Silver and evidence tables."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import read_lake_object, write_lake_object
from quantum_lake_student.models import StageResult
from quantum_lake_student.part1_results import (
    merge_evidence,
    update_row_counts,
    update_run_record,
)
from quantum_lake_student.syndromes import (
    BRONZE_OBJECT,
    MANIFEST_OBJECT,
    SILVER_OBJECT,
    build_syndrome_tables,
    parquet_bytes,
)

def _release_manifest(manifest_bytes: bytes) -> dict[str, object]:
    return json.loads(manifest_bytes.decode("utf-8"))


def _expected_syndrome_object(
    manifest: dict[str, object],
) -> dict[str, object]:
    try:
        return next(
            item
            for item in manifest["objects"]
            if item["source"] == "qec_syndromes"
        )
    except (KeyError, StopIteration) as error:
        raise ValueError(
            "Release manifest has no qec_syndromes object"
        ) from error

def _check_results(tables) -> dict[str, dict[str, object]]:
    issue_rules = set(
        tables.data_issues.column("rule_id").to_pylist()
    )
    source_ids = tables.silver.column(
        "source_record_id"
    ).to_pylist()
    trace_ids = tables.source_trace.column(
        "source_record_id"
    ).to_pylist()

    ambiguity: dict[tuple[float, bytes], set[bool]] = {}
    for row in tables.silver.select(
        [
            "physical_fault_rate",
            "syndrome_bits",
            "logical_error_label",
        ]
    ).to_pylist():
        key = (
            row["physical_fault_rate"],
            row["syndrome_bits"],
        )
        ambiguity.setdefault(key, set()).add(
            row["logical_error_label"]
        )
    ambiguous_groups = sum(
        len(labels) > 1
        for labels in ambiguity.values()
    )

    def status(*rules: str) -> str:
        return (
            "failed"
            if issue_rules.intersection(rules)
            else "passed"
        )

    return {
        "bronze_sha256": {"status": "passed"},
        "archive_member_paths": {"status": "passed"},
        "archive_member_names_unique": {"status": "passed"},
        "archive_crc": {"status": "passed"},
        "csv_file_count": {
            "status": "passed",
            "actual": tables.file_count,
        },
        "csv_headers": {"status": "passed"},
        "filename_metadata": {"status": "passed"},
        "syndrome_shape": {
            "status": status(
                "syndrome.type",
                "syndrome.parse",
                "syndrome.shape",
            )
        },
        "syndrome_bit_domain": {
            "status": status("syndrome.bit_domain")
        },
        "label_domain": {
            "status": status("syndrome.label_domain")
        },
        "positive_integer_quantities": {
            "status": status(
                "syndrome.quantity_integer",
                "syndrome.quantity_positive",
            )
        },
        "weighted_totals": {
            "status": status("syndrome.weighted_total")
        },
        "valid_same_syndrome_different_label": {
            "status": "passed",
            "groups": ambiguous_groups,
        },
        "row_reconciliation": {
            "status": (
                "passed"
                if (
                    tables.accepted_count
                    + tables.rejected_count
                    == tables.input_count
                )
                else "failed"
            )
        },
        "source_record_id_uniqueness": {
            "status": (
                "passed"
                if len(source_ids) == len(set(source_ids))
                else "failed"
            )
        },
        "source_trace_coverage": {
            "status": (
                "passed"
                if set(source_ids) == set(trace_ids)
                else "failed"
            )
        },
    }

def prepare_syndromes(
    settings: Settings,
    *,
    run_id: str,
    results_root: Path = Path("results/part1"),
) -> StageResult:
    """Build and publish the syndrome Silver, trace, and issue tables."""
    result = StageResult(stage="prepare_syndromes", run_id=run_id)

    manifest_bytes = read_lake_object(settings, MANIFEST_OBJECT)
    manifest = _release_manifest(manifest_bytes)
    expected_object = _expected_syndrome_object(manifest)
    archive_bytes = read_lake_object(settings, BRONZE_OBJECT)

    expected_bytes = int(expected_object["bytes"])
    if len(archive_bytes) != expected_bytes:
        raise ValueError(
            "Syndrome archive byte size does not match the release manifest"
        )

    tables = build_syndrome_tables(
        archive_bytes,
        expected_sha256=str(expected_object["sha256"]),
        run_id=run_id,
    )

    silver_output = parquet_bytes(tables.silver)

    write_lake_object(settings, SILVER_OBJECT, silver_output)
    (
        merged_trace,
        trace_output,
        merged_issues,
        issues_output,
    ) = merge_evidence(
        results_root,
        source_name="qec_syndromes",
        source_trace=tables.source_trace,
        data_issues=tables.data_issues,
    )

    update_row_counts(
        results_root,
        source_name="qec_syndromes",
        counts={
            "bronze_rows": tables.input_count,
            "silver_rows": tables.accepted_count,
            "rejected_rows": tables.rejected_count,
            "issue_rows": tables.data_issues.num_rows,
            "weighted_observations": sum(
                tables.silver.column("quantity").to_pylist()
            ),
        },
    )

    result.input_count = tables.input_count
    result.output_count = tables.accepted_count
    result.issue_count = tables.data_issues.num_rows
    result.finish()

    update_run_record(
        results_root,
        run_id=result.run_id,
        stage=result.stage,
        release={
            "name": manifest.get("release_name"),
            "version": manifest.get("bundle_version"),
            "date": manifest.get("release_date"),
        },
        started_at=result.started_at.isoformat(),
        finished_at=result.finished_at.isoformat(),
        source_name="qec_syndromes",
        input_hashes={
            BRONZE_OBJECT: tables.input_sha256,
        },
        outputs={
            SILVER_OBJECT: {
                "rows": tables.silver.num_rows,
                "sha256": hashlib.sha256(
                    silver_output
                ).hexdigest(),
            },
            "results/part1/source_trace.parquet": {
                "rows": merged_trace.num_rows,
                "sha256": hashlib.sha256(
                    trace_output
                ).hexdigest(),
            },
            "results/part1/data_issues.parquet": {
                "rows": merged_issues.num_rows,
                "sha256": hashlib.sha256(
                    issues_output
                ).hexdigest(),
            },
        },
        checks=_check_results(tables),
    )
    return result


def run(run_id: str) -> StageResult:
    """Run the currently implemented syndrome portion of Silver preparation."""
    return prepare_syndromes(
        Settings.from_environment(),
        run_id=run_id,
    )
