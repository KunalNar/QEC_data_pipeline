"""Small neutral records; these do not prescribe a target data model."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def stable_record_hash(value: Any) -> str:
    """Return the same SHA-256 hash for equivalent JSON-compatible data."""
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@dataclass
class StageResult:
    stage: str
    run_id: str
    input_count: int = 0
    output_count: int = 0
    issue_count: int = 0
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None

    def finish(self) -> None:
        self.finished_at = datetime.now(UTC)
