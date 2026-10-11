"""Prepare the qec_syndromes Silver and evidence tables."""

from __future__ import annotations

from pathlib import Path

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import read_lake_object
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
from quantum_lake_student.syndromes import (
    BRONZE_OBJECT,
    EXPECTED_CSV_COUNT,
    SILVER_OBJECT,
    SOURCE_NAME,
    SyndromeTables,
    build_syndrome_tables,
)


def _check_results(
    tables: SyndromeTables,
) -> dict[str, dict[str, object]]:
    issue_rules = set(tables.data_issues["rule_id"].to_pylist())
    source_ids = tables.silver["source_record_id"].to_pylist()
    trace_ids = tables.source_trace["source_record_id"].to_pylist()

    labels_by_syndrome: dict[tuple[float, bytes], set[bool]] = {}
    for row in tables.silver.select(
        ["physical_fault_rate", "syndrome_bits", "logical_error_label"]
    ).to_pylist():
        key = (row["physical_fault_rate"], row["syndrome_bits"])
        labels_by_syndrome.setdefault(key, set()).add(
            row["logical_error_label"]
        )
    ambiguous_groups = sum(
        len(labels) > 1 for labels in labels_by_syndrome.values()
    )

    checks = passed_checks(
        "bronze_size_and_sha256",
        "archive_member_paths",
        "archive_member_names_unique",
        "archive_crc",
        "csv_headers",
        "filename_metadata",
    )
    checks.update(
        {
            "csv_file_count": check_result(
                tables.file_count == EXPECTED_CSV_COUNT,
                actual=tables.file_count,
            ),
            "syndrome_shape": rules_check(
                issue_rules,
                "syndrome.type",
                "syndrome.parse",
                "syndrome.shape",
            ),
            "syndrome_bit_domain": rules_check(
                issue_rules,
                "syndrome.bit_domain",
            ),
            "label_domain": rules_check(
                issue_rules,
                "syndrome.label_domain",
            ),
            "positive_integer_quantities": rules_check(
                issue_rules,
                "syndrome.quantity_integer",
                "syndrome.quantity_positive",
            ),
            "weighted_totals": rules_check(
                issue_rules,
                "syndrome.weighted_total",
            ),
            "valid_same_syndrome_different_label": check_result(
                True,
                groups=ambiguous_groups,
            ),
            "row_reconciliation": check_result(
                tables.accepted_count + tables.rejected_count
                == tables.input_count
            ),
            "source_record_id_uniqueness": check_result(
                len(source_ids) == len(set(source_ids))
            ),
            "source_trace_coverage": check_result(
                set(source_ids) == set(trace_ids)
            ),
        }
    )
    return checks


def prepare_syndromes(
    settings: Settings,
    *,
    run_id: str,
    results_root: Path = Path("results/part1"),
) -> StageResult:
    """Build and publish the syndrome Silver and evidence tables."""
    result = StageResult(stage="prepare_syndromes", run_id=run_id)
    manifest = read_release_manifest(
        read_lake_object(settings, MANIFEST_OBJECT)
    )
    spec = source_spec(
        manifest,
        SOURCE_NAME,
        expected_object=BRONZE_OBJECT,
    )
    tables = build_syndrome_tables(
        read_lake_object(settings, BRONZE_OBJECT),
        spec=spec,
        run_id=run_id,
    )

    result.input_count = tables.input_count
    result.output_count = tables.accepted_count
    publish_part1_results(
        settings,
        result,
        results_root,
        source_name=SOURCE_NAME,
        manifest=manifest,
        input_hashes={BRONZE_OBJECT: tables.input_sha256},
        silver_tables={SILVER_OBJECT: tables.silver},
        source_trace=tables.source_trace,
        data_issues=tables.data_issues,
        row_counts={
            "bronze_rows": tables.input_count,
            "silver_rows": tables.accepted_count,
            "rejected_rows": tables.rejected_count,
            "issue_rows": tables.data_issues.num_rows,
            "weighted_observations": sum(tables.silver["quantity"].to_pylist()),
        },
        checks=_check_results(tables),
    )
    return result


def run(run_id: str) -> StageResult:
    """Run the syndrome Silver stage using environment-based settings."""
    return prepare_syndromes(
        Settings.from_environment(),
        run_id=run_id,
    )
