"""Three-source Gold transaction and unified trace checks."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quantum_lake_student.evidence import SOURCE_TRACE_SCHEMA
from quantum_lake_student.stages import run_part1 as stage


class Transaction:
    def __init__(self):
        self.committed = False
        self.rolled_back = False

    def __enter__(self):
        return self

    def __exit__(self, error_type, _error, _traceback):
        self.committed = error_type is None
        self.rolled_back = error_type is not None


def test_three_gold_loads_share_one_transaction(monkeypatch):
    transaction = Transaction()
    calls = []
    monkeypatch.setattr(stage, "postgres_connection", lambda _: transaction)
    for name in (
        "replace_syndrome_gold",
        "replace_qasm_gold",
        "replace_google_gold",
    ):
        monkeypatch.setattr(
            stage, name, lambda connection, rows: calls.append(connection)
        )

    stage.load_gold_all(None, None, None, None)

    assert calls == [transaction] * 3
    assert transaction.committed


def test_google_failure_rolls_back_the_other_gold_loads(monkeypatch):
    transaction = Transaction()
    monkeypatch.setattr(stage, "postgres_connection", lambda _: transaction)
    monkeypatch.setattr(stage, "replace_syndrome_gold", lambda *_: None)
    monkeypatch.setattr(stage, "replace_qasm_gold", lambda *_: None)

    def fail(*_):
        raise RuntimeError("Google load failed")

    monkeypatch.setattr(stage, "replace_google_gold", fail)
    with pytest.raises(RuntimeError, match="Google load failed"):
        stage.load_gold_all(None, None, None, None)
    assert transaction.rolled_back
    assert not transaction.committed


def test_unified_trace_allows_eight_google_companions(tmp_path: Path):
    rows = []
    for source_id, source_name, count in (
        ("syndrome-1", "qec_syndromes", 1),
        ("google-exp", "google_qec", 1),
        ("google-shot", "google_qec", 8),
    ):
        rows.extend({
            "source_record_id": source_id,
            "source_name": source_name,
            "bronze_object": "bronze/example.zip",
            "archive_member": f"member-{number}",
            "record_locator": "shot=0",
            "input_sha256": "0" * 64,
        } for number in range(count))
    pq.write_table(
        pa.Table.from_pylist(rows, schema=SOURCE_TRACE_SCHEMA),
        tmp_path / "source_trace.parquet",
    )
    google_rows = stage.GoogleGoldRows(
        experiments=[("google-exp",)],
        shots=[("google-shot",)],
        predictions=[],
    )
    expected = {"syndrome-1", "google-exp", "google-shot"}
    assert stage._trace_count(tmp_path, expected, google_rows) == 10
    with pytest.raises(RuntimeError, match="exactly the Silver"):
        stage._trace_count(tmp_path, expected | {"missing"}, google_rows)
