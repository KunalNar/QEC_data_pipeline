#!/bin/sh
set -eu

alias_name="course"
endpoint="http://minio:9000"
bucket="${S3_BUCKET:-quantum-lake}"

until mc alias set "$alias_name" "$endpoint" "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD" >/dev/null 2>&1; do
  echo "Waiting for object storage..."
  sleep 1
done

mc mb --ignore-existing "$alias_name/$bucket"
mc version enable "$alias_name/$bucket" >/dev/null
mc anonymous set none "$alias_name/$bucket" >/dev/null

for archive in \
  source=qasmbench/qasmbench-qec.zip \
  source=qec_syndromes/syndromes_dataset.zip \
  source=google_qec/google-surface-code-curated.zip
do
  source_file="/seed/raw/$archive"
  if [ ! -f "$source_file" ]; then
    echo "Missing course archive: $source_file" >&2
    exit 1
  fi
  mc cp "$source_file" "$alias_name/$bucket/bronze/$archive"
done
mc mirror --overwrite /seed/metadata "$alias_name/$bucket/metadata/course-release"

echo "Copied the unchanged course inputs to $alias_name/$bucket."
mc ls --recursive "$alias_name/$bucket/bronze"
