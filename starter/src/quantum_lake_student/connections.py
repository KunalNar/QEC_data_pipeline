"""Supplied connection helpers for the local course platform."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import psycopg
from minio import Minio

from .config import Settings


def minio_client(settings: Settings) -> Minio:
    parsed = urlparse(settings.s3_endpoint)
    endpoint = parsed.netloc or parsed.path
    return Minio(
        endpoint,
        access_key=settings.s3_access_key,
        secret_key=settings.s3_secret_key,
        secure=parsed.scheme == "https",
    )


def local_lake_object_path(root: Path, object_name: str) -> Path:
    """Resolve an object name in either seeded or unpacked local layout."""
    candidates = [root / object_name]

    if object_name.startswith("bronze/"):
        candidates.append(root / "raw" / object_name.removeprefix("bronze/"))

    if object_name.startswith("metadata/course-release/"):
        candidates.append(
            root
            / "metadata"
            / object_name.removeprefix("metadata/course-release/")
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Lake object not found: {object_name}; checked {candidates}"
    )


def read_lake_object(settings: Settings, object_name: str) -> bytes:
    """Read exact object bytes from the configured local or MinIO lake."""
    if settings.lake_backend == "local":
        return local_lake_object_path(
            settings.local_lake_root,
            object_name,
        ).read_bytes()

    client = minio_client(settings)
    response = client.get_object(settings.s3_bucket, object_name)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def atomic_write(path: Path, value: bytes) -> None:
    """Replace a local file atomically after creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def write_lake_object(
    settings: Settings,
    object_name: str,
    value: bytes,
) -> None:
    """Write one object to the configured local or MinIO lake."""
    if settings.lake_backend == "local":
        atomic_write(settings.local_lake_root / object_name, value)
        return

    client = minio_client(settings)
    client.put_object(
        settings.s3_bucket,
        object_name,
        BytesIO(value),
        length=len(value),
        content_type="application/octet-stream",
    )


def bronze_inventory(settings: Settings) -> list[tuple[str, int]]:
    if settings.lake_backend == "local":
        raw = settings.local_lake_root / "bronze"
        if not raw.exists():
            # The downloadable archive retains its packaging directory name;
            # the course platform seeds that same content as Bronze.
            raw = settings.local_lake_root / "raw"
        return [
            (path.relative_to(settings.local_lake_root).as_posix(), path.stat().st_size)
            for path in sorted(raw.rglob("*"))
            if path.is_file()
        ]
    client = minio_client(settings)
    return [
        (item.object_name, item.size or 0)
        for item in client.list_objects(
            settings.s3_bucket, prefix="bronze/", recursive=True
        )
    ]


def postgres_connection(settings: Settings) -> psycopg.Connection:
    return psycopg.connect(settings.postgres_dsn)


def check_platform(settings: Settings) -> dict[str, str]:
    inventory = bronze_inventory(settings)
    if not inventory:
        raise RuntimeError(
            "No Bronze objects found. From infrastructure/, run `make seed`."
        )
    with postgres_connection(settings) as connection:
        row = connection.execute(
            "SELECT value FROM course_admin.platform_info WHERE key = 'platform'"
        ).fetchone()
    if not row:
        raise RuntimeError("The course platform metadata table is unavailable.")
    return {
        "object_store": f"{len(inventory)} Bronze object(s) available",
        "postgres": str(row[0]),
    }
