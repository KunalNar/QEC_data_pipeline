"""Register and verify the original source files.

Student responsibilities:

- verify checksums from the release description;
- list source files and archive members safely;
- keep the supplied bytes unchanged;
- report missing or unexpected source objects;
- make a second run safe: no duplicate source records.
"""

# Bronze, which is verify the unchanged bronze files (from rubric.md) and "archive member paths before reading"
# (according to brief.md)
# These paths from brief.md 
# From metadata/bundle-manifest.json
# but it's the same bytes, just under a different folder label in Bronze).
import hashlib
import io
import json
import zipfile
from pathlib import Path
from quantum_lake_student.config import Settings
from quantum_lake_student.connections import bronze_inventory, minio_client
from quantum_lake_student.models import QualityFinding, Severity, StageResult, stable_record_hash

registry_path = Path("results/part1/bronze_registry_google.json")
issues_path = Path("results/part1/data_issues_google.json")
object_name = "bronze/source=google_qec/google-surface-code-curated.zip"
expected_SHA256 = "5d6a24f89f055883a49910979490be4baef54d28bf9a0f8e096a1d6c46d1ea56"

def run(run_id: str) -> StageResult:
    settings = Settings.from_environment()
    client = minio_client(settings)

    inventory = dict(bronze_inventory(settings))
    registry = {}
    findings: list[QualityFinding] = []

    if object_name not in inventory:
        findings.append(
            QualityFinding(
                rule_id="missing_bronze_object",
                severity=Severity.ERROR,
                source_system="google_qec", # Will need to not hard code this later to work for all 3 datasets
                source_record_locator=object_name,
                message="Expected Bronze object was not found",
            )
        )
    else:
        response = client.get_object(settings.s3_bucket, object_name)
        data = response.read()
        response.close()
        response.release_conn()

        file_hash = hashlib.sha256(data).hexdigest()

        if file_hash != expected_SHA256:
            findings.append(
                QualityFinding(
                    rule_id="bronze_hash_error",
                    severity=Severity.ERROR,
                    source_system="google_qec",
                    source_record_locator=object_name,
                    message="File hash does not match the manifest",
                    observed_value=file_hash,
                )
            )

        zip_file = zipfile.ZipFile(io.BytesIO(data))
        member_names = zip_file.namelist()

        unsafe_found = False
        for name in member_names:
            if ".." in name:
                unsafe_found = True
                findings.append(
                    QualityFinding(
                        rule_id="unsafe_archive_member_path",
                        severity=Severity.ERROR,
                        source_system="google_qec",
                        source_record_locator=name,
                        message="Archive member path is unsafe to extract",
                        observed_value=name,
                    )
                )

        # A stable ID for this registration, built from the object name + its hash.
        # Same inputs will always produce the same ID, even across separate runs.
        source_record_id = stable_record_hash(
            {"object": object_name, "sha256": file_hash}
        )

        registry[object_name] = {
            "source_record_id": source_record_id,
            "sha256": file_hash,
            "hash_match_checker": file_hash == expected_SHA256,
            "members": member_names,
        }

        if unsafe_found:
            raise ValueError("Unsafe archive member path found — stopping run.")

    # Save the registry (what we found) and the issues (what went wrong)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2))

    issues_path.parent.mkdir(parents=True, exist_ok=True)
    issue_dicts = []
    for f in findings:
        issue_dicts.append({
            "rule_id": f.rule_id,
            "severity": f.severity.value,
            "source_system": f.source_system,
            "source_record_locator": f.source_record_locator,
            "message": f.message,
            "observed_value": f.observed_value,
        })    
    issues_path.write_text(json.dumps(issue_dicts, indent=2))
    
    result = StageResult(stage="register_sources_google", run_id=run_id)
    result.input_count = len(inventory)
    result.output_count = len(registry)
    result.issue_count = len(findings)
    result.finish()
    return result