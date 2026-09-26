"""Two-source checkpoint transaction and trace-contract tests."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quantum_lake_student.evidence import SOURCE_TRACE_SCHEMA
from quantum_lake_student.stages import run_syndrome_qasm as stage


class Transaction:
    def __init__(self):
        self.committed = False
        self.rolled_back = False

    def __enter__(self):
        return self

    def __exit__(self, error_type, _error, _traceback):
        self.rolled_back = error_type is not None
        self.committed = error_type is None


def test_two_gold_loads_use_one_transaction(monkeypatch):
    transaction = Transaction()
    calls = []
    monkeypatch.setattr(stage, "postgres_connection", lambda _: transaction)
    monkeypatch.setattr(
        stage, "replace_syndrome_gold", lambda connection, _: calls.append(connection)
    )
    monkeypatch.setattr(
        stage, "replace_qasm_gold", lambda connection, _: calls.append(connection)
    )

    stage.load_gold_pair(None, None, None)

    assert calls == [transaction, transaction]
    assert transaction.committed
    assert not transaction.rolled_back


def test_second_gold_failure_rolls_back_first(monkeypatch):
    transaction = Transaction()
    monkeypatch.setattr(stage, "postgres_connection", lambda _: transaction)
    monkeypatch.setattr(stage, "replace_syndrome_gold", lambda *_: None)

    def fail(*_):
        raise RuntimeError("QASM load failed")

    monkeypatch.setattr(stage, "replace_qasm_gold", fail)
    with pytest.raises(RuntimeError, match="QASM load failed"):
        stage.load_gold_pair(None, None, None)

    assert transaction.rolled_back
    assert not transaction.committed


def test_scoped_trace_requires_exact_silver_ids(tmp_path: Path):
    rows = [{
        "source_record_id": "source-1",
        "source_name": "qec_syndromes",
        "bronze_object": "bronze/example.zip",
        "archive_member": "example.csv",
        "record_locator": "csv_data_row=1",
        "input_sha256": "0" * 64,
    }]
    pq.write_table(
        pa.Table.from_pylist(rows, schema=SOURCE_TRACE_SCHEMA),
        tmp_path / "source_trace.parquet",
    )

    assert stage._check_source_trace(tmp_path, {"source-1"}) == 1
    with pytest.raises(RuntimeError, match="exactly both Silver sources"):
        stage._check_source_trace(tmp_path, {"source-1", "source-2"})
