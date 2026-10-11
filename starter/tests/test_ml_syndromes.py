"""Contract tests for the syndrome Gold-to-ML export."""

import pytest

from quantum_lake_student.stages.build_ml_tables import (
    SYNDROME_ML_SCHEMA,
    syndrome_ml_table,
)


BITS = bytes([0, 1] * 8)


def record(**changes):
    fields = {
        "example_id": "ml-syn-a",
        "experiment_id": "experiment-1",
        "fault_rate": 0.001,
        "bits": BITS,
        "round_count": 4,
        "check_count": 4,
        "label": False,
        "weight": 3,
    }
    fields.update(changes)
    return tuple(fields.values())


def valid_records():
    return [
        record(),
        record(
            example_id="ml-syn-b",
            experiment_id="experiment-2",
            fault_rate=0.0005,
            weight=5,
        ),
        record(
            example_id="ml-syn-c",
            experiment_id="experiment-3",
            fault_rate=0.005,
            label=True,
            weight=7,
        ),
    ]


def test_syndrome_ml_table_has_exact_schema_weights_and_splits():
    table = syndrome_ml_table(valid_records())
    assert table.schema == SYNDROME_ML_SCHEMA
    assert table.num_rows == 3
    assert table["sample_weight"].to_pylist() == [3, 5, 7]
    assert table["data_split"].to_pylist() == [
        "train", "validation", "test"
    ]
    assert table["syndrome_bits"].to_pylist() == [BITS] * 3


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"example_id": ""}, "example_id"),
        ({"experiment_id": ""}, "experiment"),
        ({"fault_rate": 0.0}, "experiment"),
        ({"bits": bytes(15)}, "bits"),
        ({"bits": bytes([2] + [0] * 15)}, "bits"),
        ({"round_count": 3}, "shape"),
        ({"label": 1}, "label or weight"),
        ({"weight": 0}, "label or weight"),
    ],
)
def test_syndrome_ml_table_rejects_invalid_values(changes, message):
    rows = valid_records()
    rows[0] = record(**changes)
    with pytest.raises(ValueError, match=message):
        syndrome_ml_table(rows)


def test_syndrome_ml_table_rejects_duplicate_example_id():
    rows = valid_records()
    rows[1] = record(experiment_id="experiment-2", fault_rate=0.0005)
    with pytest.raises(ValueError, match="duplicate syndrome ML example_id"):
        syndrome_ml_table(rows)


def test_syndrome_ml_table_rejects_duplicate_group():
    rows = valid_records()
    rows[1] = record(example_id="ml-syn-b", weight=5)
    with pytest.raises(ValueError, match="Duplicate syndrome ML"):
        syndrome_ml_table(rows)


def test_syndrome_ml_table_requires_all_splits():
    with pytest.raises(ValueError, match="train/validation/test"):
        syndrome_ml_table(valid_records()[:2])
