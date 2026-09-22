"""Register and verify the original Bronze source archives."""

from __future__ import annotations

from dataclasses import dataclass

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import (
    bronze_inventory,
    read_lake_object,
)
from quantum_lake_student.models import StageResult
from quantum_lake_student.source_validation import (
    MANIFEST_OBJECT,
    SourceValidationError,
    ValidatedSourceObject,
    normalize_bronze_object,
    read_release_manifest,
    source_specs,
    validate_source_object,
)


EXPECTED_SOURCES = frozenset(
    {"qasmbench", "qec_syndromes", "google_qec"}
)


@dataclass(frozen=True)
class SourceRegistration:
    """Verified release facts returned to the Part I orchestrator."""

    result: StageResult
    release_name: str | None
    bundle_version: object
    objects: tuple[ValidatedSourceObject, ...]

    @property
    def input_hashes(self) -> dict[str, str]:
        return {
            item.bronze_object: item.sha256
            for item in self.objects
        }


def register_sources(
    settings: Settings,
    *,
    run_id: str,
) -> SourceRegistration:
    """Verify that the complete supplied release is present and unchanged."""
    result = StageResult(stage="register_sources", run_id=run_id)

    manifest = read_release_manifest(
        read_lake_object(settings, MANIFEST_OBJECT)
    )

    specs = source_specs(manifest)
    manifest_sources = {spec.source_name for spec in specs}
    if manifest_sources != EXPECTED_SOURCES:
        missing_sources = sorted(EXPECTED_SOURCES - manifest_sources)
        unexpected_sources = sorted(manifest_sources - EXPECTED_SOURCES)
        raise SourceValidationError(
            "Release manifest source mismatch; "
            f"missing={missing_sources}, unexpected={unexpected_sources}"
        )

    expected_objects = {spec.bronze_object for spec in specs}
    actual_objects = {
        normalize_bronze_object(path)
        for path, _ in bronze_inventory(settings)
    }
    missing_objects = sorted(expected_objects - actual_objects)
    unexpected_objects = sorted(actual_objects - expected_objects)
    if missing_objects or unexpected_objects:
        raise SourceValidationError(
            "Bronze inventory mismatch; "
            f"missing={missing_objects}, unexpected={unexpected_objects}"
        )

    validated = tuple(
        validate_source_object(
            read_lake_object(settings, spec.bronze_object),
            spec,
        )
        for spec in specs
    )

    result.input_count = len(actual_objects)
    result.output_count = len(validated)
    result.finish()
    return SourceRegistration(
        result=result,
        release_name=(
            str(manifest["release_name"])
            if manifest.get("release_name") is not None
            else None
        ),
        bundle_version=manifest.get("bundle_version"),
        objects=validated,
    )


def run(run_id: str) -> StageResult:
    """Run source registration with environment-based settings."""
    return register_sources(
        Settings.from_environment(),
        run_id=run_id,
    ).result
