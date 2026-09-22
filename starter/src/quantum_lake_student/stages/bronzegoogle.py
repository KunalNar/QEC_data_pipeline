"""Validate the unchanged Google QEC Bronze archive."""

from __future__ import annotations

import json
from pathlib import Path

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import atomic_write, read_lake_object
from quantum_lake_student.google_qec import (
    BRONZE_OBJECT,
    google_source_spec,
)
from quantum_lake_student.models import StageResult
from quantum_lake_student.source_validation import (
    MANIFEST_OBJECT,
    SourceValidationError,
    ValidatedSourceObject,
    validate_source_object,
)


REGISTRY_PATH = Path("results/part1/bronze_registry_google.json")
ISSUES_PATH = Path("results/part1/data_issues_google.json")


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


def validate_google_source(settings: Settings) -> ValidatedSourceObject:
    """Read the release manifest and validate the Google Bronze object."""
    manifest = _release_manifest(read_lake_object(settings, MANIFEST_OBJECT))
    spec = google_source_spec(manifest)
    archive_bytes = read_lake_object(settings, BRONZE_OBJECT)
    return validate_source_object(archive_bytes, spec)


def run(run_id: str) -> StageResult:
    """Validate Google Bronze and record the verified source facts."""
    result = StageResult(stage="register_sources_google", run_id=run_id)
    validated = validate_google_source(Settings.from_environment())
    registry = {
        validated.bronze_object: {
            "source_name": validated.source_name,
            "sha256": validated.sha256,
            "bytes": validated.byte_count,
            "member_count": validated.member_count,
            "members": list(validated.member_names),
        }
    }
    atomic_write(
        REGISTRY_PATH,
        (json.dumps(registry, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )
    atomic_write(ISSUES_PATH, b"[]\n")

    result.input_count = 1
    result.output_count = 1
    result.finish()
    return result


__all__ = ["run", "validate_google_source"]
