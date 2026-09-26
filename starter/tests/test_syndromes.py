import csv
import hashlib
import io
import json
import warnings
from pathlib import Path
from zipfile import ZipFile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quantum_lake_student.config import Settings
from quantum_lake_student.source_validation import (
    SourceObjectSpec,
    source_spec,
)
from quantum_lake_student.stages.prepare_syndromes import prepare_syndromes
from quantum_lake_student.syndromes import (
    BRONZE_OBJECT,
    DATA_ISSUES_SCHEMA,
    EXPECTED_COLUMNS,
    SILVER_OBJECT,
    SILVER_SCHEMA,
    SOURCE_TRACE_SCHEMA,
    SyndromeArchiveError,
    build_syndrome_tables,
    parquet_bytes,
)


VALID_SYNDROME = (
    (0, 1, 0, 1),
    (1, 0, 1, 0),
    (0, 0, 1, 1),
    (1, 1, 0, 0),
)
DEFAULT_ROWS = [
    {
        "labels": "0",
        "syndromes": repr(VALID_SYNDROME),
        "quantity": "1",
    },
    {
        "labels": "1",
        "syndromes": repr(VALID_SYNDROME),
        "quantity": "1",
    },
]


def _csv_text(
    rows: list[dict[str, str]],
    columns: tuple[str, ...] = EXPECTED_COLUMNS,
) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=columns,
        extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _member_name(index: int, nominal_count: int = 2) -> str:
    return (
        f"d-3_pfr-0.00{index + 1}000_"
        f"nb-{nominal_count}.csv"
    )


def _archive_bytes(
    *,
    first_rows: list[dict[str, str]] | None = None,
    first_name: str | None = None,
    first_nominal_count: int = 2,
    file_count: int = 7,
    columns: tuple[str, ...] = EXPECTED_COLUMNS,
    duplicate_first_member: bool = False,
) -> bytes:
    output = io.BytesIO()
    with ZipFile(output, mode="w") as archive:
        for index in range(file_count):
            name = _member_name(
                index,
                first_nominal_count if index == 0 else 2,
            )
            if index == 0 and first_name is not None:
                name = first_name
            rows = (
                first_rows
                if index == 0 and first_rows is not None
                else DEFAULT_ROWS
            )
            archive.writestr(name, _csv_text(rows, columns))

        if duplicate_first_member:
            duplicate_name = first_name or _member_name(
                0,
                first_nominal_count,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                archive.writestr(
                    duplicate_name,
                    _csv_text(DEFAULT_ROWS, columns),
                )

    return output.getvalue()


def _build(
    archive_bytes: bytes,
    *,
    run_id: str = "run-test",
):
    return build_syndrome_tables(
        archive_bytes,
        spec=SourceObjectSpec(
            source_name="qec_syndromes",
            bronze_object=BRONZE_OBJECT,
            expected_bytes=len(archive_bytes),
            expected_sha256=hashlib.sha256(archive_bytes).hexdigest(),
        ),
        run_id=run_id,
    )


def _column_values(table: pa.Table, name: str):
    return table.column(name).to_pylist()


def test_valid_archive_builds_exact_tables_without_data_loss() -> None:
    archive_bytes = _archive_bytes()
    tables = _build(archive_bytes)

    assert tables.silver.schema == SILVER_SCHEMA
    assert tables.source_trace.schema == SOURCE_TRACE_SCHEMA
    assert tables.data_issues.schema == DATA_ISSUES_SCHEMA

    assert tables.file_count == 7
    assert tables.input_count == 14
    assert tables.accepted_count == 14
    assert tables.rejected_count == 0
    assert tables.silver.num_rows == 14
    assert tables.source_trace.num_rows == 14
    assert tables.data_issues.num_rows == 0

    source_ids = _column_values(tables.silver, "source_record_id")
    trace_ids = _column_values(tables.source_trace, "source_record_id")
    assert len(source_ids) == len(set(source_ids))
    assert set(source_ids) == set(trace_ids)

    assert len(set(_column_values(tables.silver, "experiment_id"))) == 7
    assert sum(_column_values(tables.silver, "quantity")) == 14
    assert set(_column_values(tables.silver, "logical_error_label")) == {
        False,
        True,
    }
    assert all(
        len(value) == 16
        for value in _column_values(tables.silver, "syndrome_bits")
    )
    assert _column_values(tables.silver, "syndrome_bits")[0] == bytes(
        bit for round_values in VALID_SYNDROME for bit in round_values
    )

    # Same-syndrome/different-label records are both preserved.
    first_experiment = _column_values(
        tables.silver,
        "experiment_id",
    )[0]
    first_rows = tables.silver.filter(
        pa.compute.equal(
            tables.silver.column("experiment_id"),
            first_experiment,
        )
    )
    assert first_rows.num_rows == 2
    assert set(
        _column_values(first_rows, "logical_error_label")
    ) == {False, True}


@pytest.mark.parametrize(
    ("bad_row", "expected_rule"),
    [
        (
            {
                "labels": "2",
                "syndromes": repr(VALID_SYNDROME),
                "quantity": "1",
            },
            "syndrome.label_domain",
        ),
        (
            {
                "labels": "1",
                "syndromes": "not valid Python data",
                "quantity": "1",
            },
            "syndrome.parse",
        ),
        (
            {
                "labels": "1",
                "syndromes": repr(VALID_SYNDROME[:3]),
                "quantity": "1",
            },
            "syndrome.shape",
        ),
        (
            {
                "labels": "1",
                "syndromes": repr(
                    ((2, 1, 0, 1),) + VALID_SYNDROME[1:]
                ),
                "quantity": "1",
            },
            "syndrome.bit_domain",
        ),
        (
            {
                "labels": "1",
                "syndromes": repr(VALID_SYNDROME),
                "quantity": "1.5",
            },
            "syndrome.quantity_integer",
        ),
        (
            {
                "labels": "1",
                "syndromes": repr(VALID_SYNDROME),
                "quantity": "0",
            },
            "syndrome.quantity_positive",
        ),
    ],
)
def test_invalid_rows_are_excluded_and_preserved(
    bad_row: dict[str, str],
    expected_rule: str,
) -> None:
    archive_bytes = _archive_bytes(
        first_rows=[DEFAULT_ROWS[0], bad_row],
    )
    tables = _build(archive_bytes)

    assert tables.input_count == 14
    assert tables.accepted_count == 13
    assert tables.rejected_count == 1
    assert tables.accepted_count + tables.rejected_count == tables.input_count
    assert tables.source_trace.num_rows == tables.silver.num_rows

    rules = set(_column_values(tables.data_issues, "rule_id"))
    assert expected_rule in rules

    issues = tables.data_issues.to_pylist()
    rejected = [
        issue
        for issue in issues
        if issue["rule_id"] == expected_rule
    ]
    assert len(rejected) == 1
    assert rejected[0]["source_record_id"] is not None
    assert rejected[0]["action"] == "excluded_from_silver"
    assert rejected[0]["source_record"]


def test_weight_mismatch_is_reported_without_dropping_valid_rows() -> None:
    archive_bytes = _archive_bytes(first_nominal_count=3)
    tables = _build(archive_bytes)

    assert tables.input_count == 14
    assert tables.accepted_count == 14
    assert tables.rejected_count == 0

    issues = tables.data_issues.to_pylist()
    mismatch = [
        issue
        for issue in issues
        if issue["rule_id"] == "syndrome.weighted_total"
    ]
    assert len(mismatch) == 1
    assert mismatch[0]["source_record_id"] is None
    assert mismatch[0]["action"] == "published_valid_rows_only"


def test_ids_and_parquet_values_are_stable_across_runs() -> None:
    archive_bytes = _archive_bytes()
    first = _build(archive_bytes, run_id="run-one")
    second = _build(archive_bytes, run_id="run-two")

    assert first.silver.equals(second.silver)
    assert first.source_trace.equals(second.source_trace)

    round_trip = pq.read_table(
        pa.BufferReader(parquet_bytes(first.silver))
    )
    assert round_trip.equals(first.silver)
    assert round_trip.column_names == SILVER_SCHEMA.names


def test_checksum_mismatch_stops_the_run() -> None:
    archive_bytes = _archive_bytes()

    with pytest.raises(SyndromeArchiveError, match="SHA-256"):
        build_syndrome_tables(
            archive_bytes,
            spec=SourceObjectSpec(
                source_name="qec_syndromes",
                bronze_object=BRONZE_OBJECT,
                expected_bytes=len(archive_bytes),
                expected_sha256="0" * 64,
            ),
            run_id="run-test",
        )


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "../outside.csv",
        "/absolute.csv",
        r"C:\outside.csv",
    ],
)
def test_unsafe_archive_paths_stop_the_run(
    unsafe_name: str,
) -> None:
    archive_bytes = _archive_bytes(first_name=unsafe_name)

    with pytest.raises(SyndromeArchiveError, match="Unsafe"):
        _build(archive_bytes)


def test_duplicate_members_stop_the_run() -> None:
    archive_bytes = _archive_bytes(duplicate_first_member=True)

    with pytest.raises(SyndromeArchiveError, match="duplicate"):
        _build(archive_bytes)


def test_missing_csv_member_stops_the_run() -> None:
    archive_bytes = _archive_bytes(file_count=6)

    with pytest.raises(SyndromeArchiveError, match="Expected 7"):
        _build(archive_bytes)


def test_invalid_filename_stops_the_run() -> None:
    archive_bytes = _archive_bytes(first_name="unexpected.csv")

    with pytest.raises(SyndromeArchiveError, match="filename"):
        _build(archive_bytes)


def test_invalid_header_stops_the_run() -> None:
    archive_bytes = _archive_bytes(
        columns=("label", "syndromes", "quantity"),
    )

    with pytest.raises(SyndromeArchiveError, match="header"):
        _build(archive_bytes)


def test_local_stage_writes_and_replaces_all_outputs(
    tmp_path: Path,
) -> None:
    archive_bytes = _archive_bytes()
    digest = hashlib.sha256(archive_bytes).hexdigest()
    lake_root = tmp_path / "lake"
    archive_path = (
        lake_root
        / "raw"
        / "source=qec_syndromes"
        / "syndromes_dataset.zip"
    )
    archive_path.parent.mkdir(parents=True)
    archive_path.write_bytes(archive_bytes)

    manifest_path = lake_root / "metadata" / "bundle-manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "source": "qec_syndromes",
                        "path": (
                            "raw/source=qec_syndromes/"
                            "syndromes_dataset.zip"
                        ),
                        "bytes": len(archive_bytes),
                        "sha256": digest,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    settings = Settings(
        lake_backend="local",
        local_lake_root=lake_root,
        s3_endpoint="",
        s3_access_key="",
        s3_secret_key="",
        s3_bucket="",
        postgres_host="",
        postgres_port=5432,
        postgres_db="",
        postgres_user="",
        postgres_password="",
    )
    results_root = tmp_path / "results" / "part1"

    first_result = prepare_syndromes(
        settings,
        run_id="local-one",
        results_root=results_root,
    )
    first_table = pq.read_table(lake_root / SILVER_OBJECT)
    first_ids = _column_values(first_table, "source_record_id")

    second_result = prepare_syndromes(
        settings,
        run_id="local-two",
        results_root=results_root,
    )
    second_table = pq.read_table(lake_root / SILVER_OBJECT)

    assert first_result.input_count == 14
    assert first_result.output_count == 14
    assert first_result.issue_count == 0
    assert second_result.output_count == 14
    assert first_ids == _column_values(
        second_table,
        "source_record_id",
    )

    trace = pq.read_table(results_root / "source_trace.parquet")
    issues = pq.read_table(results_root / "data_issues.parquet")
    counts = json.loads(
        (results_root / "row_counts.json").read_text(
            encoding="utf-8"
        )
    )
    run_record = json.loads(
        (results_root / "run.json").read_text(
            encoding="utf-8"
        )
    )
    assert trace.num_rows == 14
    assert issues.num_rows == 0
    assert counts["qec_syndromes"] == {
        "bronze_rows": 14,
        "issue_rows": 0,
        "rejected_rows": 0,
        "silver_rows": 14,
        "weighted_observations": 14,
    }
    assert run_record["run_id"] == "local-two"
    assert run_record["input_hashes"][BRONZE_OBJECT] == digest
    assert run_record["outputs"][SILVER_OBJECT]["rows"] == 14
    assert run_record["code_revision"]
    assert run_record["code_revision_kind"] in {
        "git",
        "source_tree_sha256",
    }
    assert all(
        check["status"] == "passed"
        for check in run_record["checks"]["qec_syndromes"].values()
    )


def test_real_release_reconciles_without_information_loss() -> None:
    archive_path = Path(
        "/course-data/raw/source=qec_syndromes/"
        "syndromes_dataset.zip"
    )
    manifest_path = Path(
        "/course-data/metadata/bundle-manifest.json"
    )
    if not archive_path.is_file() or not manifest_path.is_file():
        pytest.skip("Course data mount is unavailable")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spec = source_spec(
        manifest,
        "qec_syndromes",
        expected_object=BRONZE_OBJECT,
    )
    tables = build_syndrome_tables(
        archive_path.read_bytes(),
        spec=spec,
        run_id="real-release-test",
    )

    assert tables.file_count == 7
    assert tables.input_count == 75_598
    assert tables.accepted_count == 75_598
    assert tables.rejected_count == 0
    assert tables.data_issues.num_rows == 0
    assert tables.source_trace.num_rows == 75_598
    assert sum(_column_values(tables.silver, "quantity")) == 70_000_000
    assert len(
        set(_column_values(tables.silver, "source_record_id"))
    ) == 75_598
    assert all(
        len(value) == 16
        for value in _column_values(tables.silver, "syndrome_bits")
    )
