"""Shared validation for unchanged Bronze source objects."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePosixPath
from typing import Mapping
from zipfile import BadZipFile, ZipFile


MANIFEST_OBJECT = "metadata/course-release/bundle-manifest.json"


class SourceValidationError(ValueError):
    """Raised when a Bronze object or release manifest is invalid."""


@dataclass(frozen=True)
class SourceObjectSpec:
    """Expected properties of one object in the course release."""

    source_name: str
    bronze_object: str
    expected_bytes: int
    expected_sha256: str


@dataclass(frozen=True)
class ValidatedSourceObject:
    """Verified facts about one unchanged Bronze archive."""

    source_name: str
    bronze_object: str
    byte_count: int
    sha256: str
    member_names: tuple[str, ...]

    @property
    def member_count(self) -> int:
        return len(self.member_names)


def sha256_bytes(value: bytes) -> str:
    """Return the SHA-256 digest for exact input bytes."""
    return hashlib.sha256(value).hexdigest()


def is_safe_member_path(name: str) -> bool:
    """Return whether an archive member stays inside an extraction root."""
    normalized = name.replace("\\", "/")
    member_path = PurePosixPath(normalized)
    has_drive = bool(
        member_path.parts
        and re.fullmatch(r"[A-Za-z]:", member_path.parts[0])
    )
    return (
        bool(member_path.parts)
        and not member_path.is_absolute()
        and ".." not in member_path.parts
        and not has_drive
    )


def bronze_object_from_manifest_path(path: str) -> str:
    """Convert a release-manifest raw path to its Bronze object name."""
    source_prefix = "raw/"
    if not path.startswith(source_prefix):
        raise SourceValidationError(
            f"Manifest object path must start with {source_prefix!r}: {path}"
        )
    return "bronze/" + path.removeprefix(source_prefix)


def normalize_bronze_object(path: str) -> str:
    """Normalize local release paths and object-store paths for comparison."""
    if path.startswith("raw/"):
        return bronze_object_from_manifest_path(path)
    return path


def source_specs(
    manifest: Mapping[str, object],
) -> tuple[SourceObjectSpec, ...]:
    """Read and validate source-object specifications from a manifest."""
    objects = manifest.get("objects")
    if not isinstance(objects, list) or not objects:
        raise SourceValidationError(
            "Release manifest must contain a non-empty objects list"
        )

    specs: list[SourceObjectSpec] = []
    for index, item in enumerate(objects):
        if not isinstance(item, Mapping):
            raise SourceValidationError(
                f"Manifest object {index} must be a mapping"
            )
        try:
            source_name = str(item["source"])
            manifest_path = str(item["path"])
            expected_bytes = int(item["bytes"])
            expected_sha256 = str(item["sha256"])
        except (KeyError, TypeError, ValueError) as error:
            raise SourceValidationError(
                f"Manifest object {index} has invalid required fields"
            ) from error

        if not source_name:
            raise SourceValidationError(
                f"Manifest object {index} has an empty source name"
            )
        if expected_bytes < 0:
            raise SourceValidationError(
                f"Manifest object {source_name} has a negative byte count"
            )
        if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
            raise SourceValidationError(
                f"Manifest object {source_name} has an invalid SHA-256"
            )

        specs.append(
            SourceObjectSpec(
                source_name=source_name,
                bronze_object=bronze_object_from_manifest_path(manifest_path),
                expected_bytes=expected_bytes,
                expected_sha256=expected_sha256.lower(),
            )
        )

    source_names = [spec.source_name for spec in specs]
    object_names = [spec.bronze_object for spec in specs]
    if len(source_names) != len(set(source_names)):
        raise SourceValidationError(
            "Release manifest contains duplicate source names"
        )
    if len(object_names) != len(set(object_names)):
        raise SourceValidationError(
            "Release manifest contains duplicate object paths"
        )

    return tuple(specs)


def validate_source_object(
    value: bytes,
    spec: SourceObjectSpec,
) -> ValidatedSourceObject:
    """Verify one Bronze archive without extracting or changing it."""
    byte_count = len(value)
    if byte_count != spec.expected_bytes:
        raise SourceValidationError(
            f"Byte size mismatch for {spec.bronze_object}: "
            f"expected {spec.expected_bytes}, found {byte_count}"
        )

    actual_sha256 = sha256_bytes(value)
    if actual_sha256 != spec.expected_sha256:
        raise SourceValidationError(
            f"SHA-256 mismatch for {spec.bronze_object}"
        )

    try:
        archive = ZipFile(BytesIO(value))
    except BadZipFile as error:
        raise SourceValidationError(
            f"Bronze object is not a valid ZIP: {spec.bronze_object}"
        ) from error

    with archive:
        member_names = tuple(info.filename for info in archive.infolist())
        unsafe_members = [
            name for name in member_names if not is_safe_member_path(name)
        ]
        if unsafe_members:
            raise SourceValidationError(
                f"Unsafe archive member path in {spec.bronze_object}: "
                f"{unsafe_members[0]}"
            )

        if len(member_names) != len(set(member_names)):
            raise SourceValidationError(
                f"Archive contains duplicate member names: {spec.bronze_object}"
            )

        bad_crc_member = archive.testzip()
        if bad_crc_member is not None:
            raise SourceValidationError(
                f"CRC check failed for {spec.bronze_object}: "
                f"{bad_crc_member}"
            )

    return ValidatedSourceObject(
        source_name=spec.source_name,
        bronze_object=spec.bronze_object,
        byte_count=byte_count,
        sha256=actual_sha256,
        member_names=member_names,
    )
