import hashlib
import io
import json
import warnings
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import pytest

from quantum_lake_student.config import Settings
from quantum_lake_student.source_validation import (
    SourceObjectSpec,
    SourceValidationError,
    is_safe_member_path,
    sha256_bytes,
    source_specs,
    validate_source_object,
)
from quantum_lake_student.stages.register_sources import register_sources


SOURCE_NAMES = ("qasmbench", "qec_syndromes", "google_qec")


def _zip_bytes(
    member_name: str = "folder/data.txt",
    *,
    duplicate: bool = False,
) -> bytes:
    output = io.BytesIO()
    with ZipFile(output, mode="w", compression=ZIP_STORED) as archive:
        archive.writestr(member_name, b"payload")
        if duplicate:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                archive.writestr(member_name, b"second")
    return output.getvalue()


def _spec(value: bytes) -> SourceObjectSpec:
    return SourceObjectSpec(
        source_name="example",
        bronze_object="bronze/source=example/example.zip",
        expected_bytes=len(value),
        expected_sha256=sha256_bytes(value),
    )


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


def _write_local_release(root: Path) -> dict[str, bytes]:
    objects: list[dict[str, object]] = []
    values: dict[str, bytes] = {}
    for source_name in SOURCE_NAMES:
        value = _zip_bytes(f"{source_name}/data.txt")
        manifest_path = f"raw/source={source_name}/{source_name}.zip"
        file_path = root / manifest_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(value)
        values[source_name] = value
        objects.append(
            {
                "source": source_name,
                "path": manifest_path,
                "bytes": len(value),
                "sha256": hashlib.sha256(value).hexdigest(),
            }
        )

    manifest_path = root / "metadata" / "bundle-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "release_name": "test-release",
                "bundle_version": 1,
                "objects": objects,
            }
        ),
        encoding="utf-8",
    )
    return values


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("folder/data.csv", True),
        ("../outside.csv", False),
        ("/absolute.csv", False),
        (r"C:\\outside.csv", False),
        ("", False),
    ],
)
def test_archive_member_path_safety(name: str, expected: bool) -> None:
    assert is_safe_member_path(name) is expected


def test_valid_source_object_returns_verified_inventory() -> None:
    value = _zip_bytes()

    result = validate_source_object(value, _spec(value))

    assert result.byte_count == len(value)
    assert result.sha256 == hashlib.sha256(value).hexdigest()
    assert result.member_names == ("folder/data.txt",)
    assert result.member_count == 1


def test_source_object_rejects_size_and_checksum_mismatches() -> None:
    value = _zip_bytes()
    spec = _spec(value)

    with pytest.raises(SourceValidationError, match="Byte size"):
        validate_source_object(
            value,
            SourceObjectSpec(
                source_name=spec.source_name,
                bronze_object=spec.bronze_object,
                expected_bytes=len(value) + 1,
                expected_sha256=spec.expected_sha256,
            ),
        )

    with pytest.raises(SourceValidationError, match="SHA-256"):
        validate_source_object(
            value,
            SourceObjectSpec(
                source_name=spec.source_name,
                bronze_object=spec.bronze_object,
                expected_bytes=len(value),
                expected_sha256="0" * 64,
            ),
        )


@pytest.mark.parametrize(
    "member_name",
    ["../outside.txt", "/absolute.txt", r"C:\\outside.txt"],
)
def test_source_object_rejects_unsafe_paths(member_name: str) -> None:
    value = _zip_bytes(member_name)

    with pytest.raises(SourceValidationError, match="Unsafe"):
        validate_source_object(value, _spec(value))


def test_source_object_rejects_duplicate_members() -> None:
    value = _zip_bytes(duplicate=True)

    with pytest.raises(SourceValidationError, match="duplicate"):
        validate_source_object(value, _spec(value))


def test_source_object_rejects_crc_failure() -> None:
    value = bytearray(_zip_bytes())
    payload_position = value.index(b"payload")
    value[payload_position] ^= 1
    corrupted = bytes(value)

    with pytest.raises(SourceValidationError, match="CRC"):
        validate_source_object(corrupted, _spec(corrupted))


def test_manifest_specs_reject_duplicate_sources() -> None:
    value = _zip_bytes()
    item = {
        "source": "duplicate",
        "path": "raw/source=duplicate/data.zip",
        "bytes": len(value),
        "sha256": sha256_bytes(value),
    }

    with pytest.raises(SourceValidationError, match="duplicate source"):
        source_specs({"objects": [item, item]})


def test_registration_verifies_all_sources_and_is_repeatable(
    tmp_path: Path,
) -> None:
    values = _write_local_release(tmp_path)
    settings = _local_settings(tmp_path)

    first = register_sources(settings, run_id="first")
    second = register_sources(settings, run_id="second")

    assert first.result.input_count == 3
    assert first.result.output_count == 3
    assert first.result.issue_count == 0
    assert first.result.finished_at is not None
    assert first.release_name == "test-release"
    assert first.bundle_version == 1
    assert {item.source_name for item in first.objects} == set(SOURCE_NAMES)
    assert first.input_hashes == second.input_hashes
    assert first.input_hashes == {
        f"bronze/source={name}/{name}.zip": sha256_bytes(value)
        for name, value in values.items()
    }


def test_registration_reports_missing_and_unexpected_objects(
    tmp_path: Path,
) -> None:
    _write_local_release(tmp_path)
    missing_path = (
        tmp_path
        / "raw"
        / "source=qasmbench"
        / "qasmbench.zip"
    )
    missing_path.unlink()
    unexpected_path = tmp_path / "raw" / "source=extra" / "extra.zip"
    unexpected_path.parent.mkdir(parents=True)
    unexpected_path.write_bytes(_zip_bytes())

    with pytest.raises(
        SourceValidationError,
        match="Bronze inventory mismatch",
    ) as error:
        register_sources(_local_settings(tmp_path), run_id="test")

    message = str(error.value)
    assert "qasmbench.zip" in message
    assert "source=extra/extra.zip" in message
