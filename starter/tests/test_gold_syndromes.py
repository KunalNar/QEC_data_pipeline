"""Syndrome Silver-to-Gold mapping tests."""

import pyarrow as pa
import pytest

from quantum_lake_student.stages.load_postgres import syndrome_gold_rows
from quantum_lake_student.syndromes import SILVER_SCHEMA


BITS = bytes([0, 1] * 8)


def silver_row(**changes):
    row = {
        "source_record_id": "source-1",
        "experiment_id": "experiment-1",
        "physical_fault_rate": 0.001,
        "syndrome_bits": BITS,
        "round_count": 4,
        "check_count": 4,
        "logical_error_label": False,
        "quantity": 3,
    }
    row.update(changes)
    return row


def table(*rows):
    return pa.Table.from_pylist(list(rows), schema=SILVER_SCHEMA)


def test_gold_keeps_each_source_row_and_both_labels():
    rows = syndrome_gold_rows(
        table(
            silver_row(),
            silver_row(source_record_id="source-2", quantity=5),
            silver_row(
                source_record_id="source-3",
                logical_error_label=True,
                quantity=7,
            ),
            silver_row(
                source_record_id="source-4",
                experiment_id="experiment-2",
                physical_fault_rate=0.002,
                quantity=11,
            ),
        )
    )
    assert rows.experiments == [
        ("experiment-1", 0.001),
        ("experiment-2", 0.002),
    ]
    assert rows.patterns == [(BITS, 4, 4)]
    assert len(rows.observations) == 4
    assert [row[0] for row in rows.observations] == [
        "source-1", "source-2", "source-3", "source-4"
    ]
    assert rows.total_weight == 26


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"source_record_id": ""}, "source_record_id"),
        ({"experiment_id": ""}, "experiment"),
        ({"physical_fault_rate": 0.0}, "experiment"),
        ({"syndrome_bits": bytes(15)}, "pattern"),
        ({"syndrome_bits": bytes([2] + [0] * 15)}, "pattern"),
        ({"round_count": 3}, "shape"),
        ({"quantity": 0}, "quantity"),
    ],
)
def test_gold_rejects_invalid_values(changes, message):
    with pytest.raises(ValueError, match=message):
        syndrome_gold_rows(table(silver_row(**changes)))


def test_gold_rejects_duplicate_source_id():
    with pytest.raises(ValueError, match="duplicate"):
        syndrome_gold_rows(table(silver_row(), silver_row(quantity=5)))


def test_gold_rejects_conflicting_experiment_rate():
    with pytest.raises(ValueError, match="Conflicting fault rates"):
        syndrome_gold_rows(
            table(
                silver_row(),
                silver_row(source_record_id="source-2", physical_fault_rate=0.002),
            )
        )


def test_gold_rejects_null_and_empty_silver():
    with pytest.raises(ValueError, match="null"):
        syndrome_gold_rows(table(silver_row(quantity=None)))
    with pytest.raises(ValueError, match="empty"):
        syndrome_gold_rows(table())
