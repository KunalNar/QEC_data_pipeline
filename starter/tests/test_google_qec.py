"""Google Bronze/Silver contracts and malformed companion-file cases."""

import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import pyarrow.parquet as pq
import pytest

from quantum_lake_student.config import Settings
from quantum_lake_student.evidence import DATA_ISSUES_SCHEMA, SOURCE_TRACE_SCHEMA
from quantum_lake_student.google_qec import (
    BRONZE_OBJECT,
    EXPERIMENT_PATH,
    EXPERIMENT_SCHEMA,
    GoogleQecArchiveError,
    SHOT_PATH,
    SHOT_SCHEMA,
    build_google_qec_tables,
)
from quantum_lake_student.source_validation import SourceObjectSpec
from quantum_lake_student.stages.silvergoogle import prepare_google_data


FOLDERS = (
    "surface_code_bX_d3_r25_center_1_1",
    "surface_code_bX_d3_r25_center_1_2",
    "surface_code_bX_d3_r25_center_2_1",
    "surface_code_bX_d3_r25_center_2_2",
    "surface_code_bX_d5_r25_center_3_3",
)


def _members(shots: int = 2) -> dict[str, bytes]:
    members = {}
    for folder in FOLDERS:
        distance = 5 if "_d5_" in folder else 3
        row, col = folder.rsplit("_", 2)[1:]
        properties = (
            "basis: X\n"
            f"distance: {distance}\n"
            "rounds: 25\n"
            f"shots: {shots}\n"
            f"center_data_qubit_row: {row}\n"
            f"center_data_qubit_col: {col}\n"
            "circuit_measurements: 9\n"
            "circuit_sweep_bits: 0\n"
            "circuit_detectors: 10\n"
        )
        members[f"{folder}/properties.yml"] = properties.encode()
        members[f"{folder}/measurements.b8"] = (
            b"\x01\x01\x00\x00" * (shots // 2)
            + (b"\x01\x01" if shots % 2 else b"")
        )
        members[f"{folder}/sweep.b8"] = b""
        members[f"{folder}/detection_events.b8"] = (
            b"\x01\x02\x02\x00" * (shots // 2)
            + (b"\x01\x02" if shots % 2 else b"")
        )
        actual = [b"0" if index % 2 == 0 else b"1" for index in range(shots)]
        predicted = [b"1" if index % 2 == 0 else b"0" for index in range(shots)]
        members[f"{folder}/obs_flips_actual.01"] = b"\n".join(actual) + b"\n"
        for decoder in (
            "belief_matching",
            "correlated_matching",
            "pymatching",
            "tensor_network_contraction",
        ):
            members[f"{folder}/obs_flips_predicted_by_{decoder}.01"] = (
                b"\n".join(predicted) + b"\n"
            )
    return members


def _archive(
    *,
    shots: int = 2,
    overrides: dict[str, bytes] | None = None,
    missing: str | None = None,
) -> bytes:
    members = _members(shots)
    members.update(overrides or {})
    if missing is not None:
        del members[missing]
    output = io.BytesIO()
    with ZipFile(output, "w") as archive:
        for name, value in sorted(members.items()):
            archive.writestr(name, value)
    return output.getvalue()


def _spec(value: bytes) -> SourceObjectSpec:
    return SourceObjectSpec(
        source_name="google_qec",
        bronze_object=BRONZE_OBJECT,
        expected_bytes=len(value),
        expected_sha256=hashlib.sha256(value).hexdigest(),
    )


def _build(value: bytes, *, run_id: str = "google-test"):
    return build_google_qec_tables(value, spec=_spec(value), run_id=run_id)


def _local_settings(root: Path) -> Settings:
    return Settings(
        lake_backend="local",
        local_lake_root=root,
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


def test_valid_archive_keeps_aligned_shots_and_complete_trace() -> None:
    tables = _build(_archive())

    assert tables.experiments.schema == EXPERIMENT_SCHEMA
    assert tables.shots.schema == SHOT_SCHEMA
    assert tables.source_trace.schema == SOURCE_TRACE_SCHEMA
    assert tables.data_issues.schema == DATA_ISSUES_SCHEMA
    assert (tables.experiment_count, tables.input_count) == (5, 10)
    assert (tables.accepted_count, tables.rejected_count) == (10, 0)
    assert tables.experiments.num_rows == 5
    assert tables.shots.num_rows == 10
    assert tables.source_trace.num_rows == 5 + 10 * 8
    assert tables.data_issues.num_rows == 0

    first, second = tables.shots.slice(0, 2).to_pylist()
    assert (first["shot_index"], second["shot_index"]) == (0, 1)
    assert first["measurement_bits"] == b"\x01\x01"
    assert first["sweep_bits"] == b""
    assert first["detector_bits"] == b"\x01\x02"
    assert (first["detector_event_count"], second["detector_event_count"]) == (2, 1)
    assert first["actual_observable_flip"] is False
    assert first["belief_matching_prediction"] is True
    assert second["actual_observable_flip"] is True

    source_ids = {
        *tables.experiments["source_record_id"].to_pylist(),
        *tables.shots["source_record_id"].to_pylist(),
    }
    trace_ids = set(tables.source_trace["source_record_id"].to_pylist())
    assert len(source_ids) == 15
    assert source_ids == trace_ids
    shot_trace = [
        row for row in tables.source_trace.to_pylist()
        if row["source_record_id"] == first["source_record_id"]
    ]
    assert len(shot_trace) == 8
    assert any("byte_offset=0" in row["record_locator"] for row in shot_trace)
    assert any("line=1" in row["record_locator"] for row in shot_trace)


@pytest.mark.parametrize(
    ("relative_name", "rule"),
    [
        ("measurements.b8", "google.b8_length"),
        ("obs_flips_actual.01", "google.01_length"),
    ],
)
def test_misaligned_companion_excludes_only_its_experiment(
    relative_name: str,
    rule: str,
) -> None:
    member = f"{FOLDERS[0]}/{relative_name}"
    bad_value = b"\x01" if relative_name.endswith(".b8") else b"0\n"
    tables = _build(_archive(overrides={member: bad_value}))

    assert (tables.input_count, tables.accepted_count, tables.rejected_count) == (
        10, 8, 2
    )
    assert (tables.experiments.num_rows, tables.shots.num_rows) == (4, 8)
    issues = tables.data_issues.to_pylist()
    assert len(issues) == 1
    assert issues[0]["rule_id"] == rule
    assert issues[0]["archive_member"] == member
    assert issues[0]["action"] == "exclude_experiment_from_silver"


def test_padding_issue_preserves_original_byte_and_shot() -> None:
    member = f"{FOLDERS[0]}/detection_events.b8"
    tables = _build(_archive(overrides={member: b"\x01\x82\x02\x00"}))

    assert (tables.accepted_count, tables.rejected_count) == (8, 2)
    issue, = tables.data_issues.to_pylist()
    assert issue["rule_id"] == "google.b8_padding"
    assert issue["archive_member"] == member
    assert issue["record_locator"] == "shot=0"
    assert issue["observed_value"] == "0x82"
    assert issue["source_record_id"] is not None


def test_late_invalid_01_value_is_preserved_not_truncated() -> None:
    member = f"{FOLDERS[0]}/obs_flips_actual.01"
    raw = b"0\n" * 49 + b"2\n"
    tables = _build(_archive(shots=50, overrides={member: raw}))

    assert (tables.input_count, tables.accepted_count, tables.rejected_count) == (
        250, 200, 50
    )
    issue, = tables.data_issues.to_pylist()
    assert issue["rule_id"] == "google.01_domain"
    assert issue["archive_member"] == member
    assert issue["record_locator"] == "shot=49;line=50"
    assert issue["observed_value"] == "b'2'"
    assert issue["source_record_id"] is not None


def test_missing_companion_and_metadata_mismatch_stop_loading() -> None:
    companion = f"{FOLDERS[0]}/obs_flips_actual.01"
    with pytest.raises(GoogleQecArchiveError, match="Missing required companion"):
        _build(_archive(missing=companion))

    properties = f"{FOLDERS[0]}/properties.yml"
    bad_properties = _members()[properties].replace(b"basis: X", b"basis: Z")
    with pytest.raises(GoogleQecArchiveError, match="disagree"):
        _build(_archive(overrides={properties: bad_properties}))


def test_identifiers_and_values_are_stable_across_runs() -> None:
    archive = _archive()
    first = _build(archive, run_id="first")
    second = _build(archive, run_id="second")

    assert first.experiments.equals(second.experiments)
    assert first.shots.equals(second.shots)
    assert first.source_trace.equals(second.source_trace)


def test_local_stage_publishes_and_replaces_google_rows(tmp_path: Path) -> None:
    archive = _archive()
    archive_path = tmp_path / BRONZE_OBJECT.replace("bronze/", "raw/")
    archive_path.parent.mkdir(parents=True)
    archive_path.write_bytes(archive)
    manifest_path = tmp_path / "metadata" / "bundle-manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps({
            "release_name": "fixture",
            "bundle_version": 1,
            "objects": [{
                "source": "google_qec",
                "path": "raw/source=google_qec/google-surface-code-curated.zip",
                "bytes": len(archive),
                "sha256": hashlib.sha256(archive).hexdigest(),
            }],
        }),
        encoding="utf-8",
    )
    results_root = tmp_path / "results" / "part1"
    settings = _local_settings(tmp_path)

    first = prepare_google_data(settings, run_id="first", results_root=results_root)
    second = prepare_google_data(settings, run_id="second", results_root=results_root)

    assert (first.output_count, second.output_count) == (15, 15)
    assert pq.read_table(tmp_path / EXPERIMENT_PATH).num_rows == 5
    assert pq.read_table(tmp_path / SHOT_PATH).num_rows == 10
    assert pq.read_table(results_root / "source_trace.parquet").num_rows == 85
    assert pq.read_table(results_root / "data_issues.parquet").num_rows == 0
    counts = json.loads((results_root / "row_counts.json").read_text())
    assert counts["google_qec"]["shot_rows"] == 10
    assert counts["google_qec"]["trace_rows"] == 85
