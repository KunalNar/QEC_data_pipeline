"""Google Silver-to-Gold mapping and Gold-to-ML contract tests."""

import pyarrow as pa
import pytest

from quantum_lake_student.google_qec import EXPERIMENT_SCHEMA, SHOT_SCHEMA
from quantum_lake_student.stages.build_google_ml_table import (
    GOOGLE_ML_SCHEMA,
    google_ml_table,
)
from quantum_lake_student.stages.load_postgres_google import google_gold_rows


def experiment(**changes):
    row = {
        "source_record_id": "experiment-source",
        "experiment_id": "experiment-1",
        "basis": "X",
        "distance": 3,
        "rounds": 25,
        "shots": 2,
        "center_row": 3,
        "center_col": 5,
        "measurement_count": 9,
        "detector_count": 10,
    }
    row.update(changes)
    return row


def shot(index, **changes):
    row = {
        "source_record_id": f"shot-{index}",
        "experiment_id": "experiment-1",
        "shot_index": index,
        "measurement_bits": b"\x01\x01",
        "sweep_bits": b"",
        "detector_bits": b"\x01\x02",
        "detector_event_count": 2,
        "actual_observable_flip": False,
        "belief_matching_prediction": True,
        "correlated_matching_prediction": False,
        "pymatching_prediction": True,
        "tensor_network_contraction_prediction": False,
    }
    row.update(changes)
    return row


def gold(experiments=None, shots=None):
    return google_gold_rows(
        pa.Table.from_pylist(
            [experiment()] if experiments is None else experiments,
            schema=EXPERIMENT_SCHEMA,
        ),
        pa.Table.from_pylist(
            [shot(0), shot(1)] if shots is None else shots,
            schema=SHOT_SCHEMA,
        ),
    )


def test_gold_separates_shots_from_decoder_predictions():
    rows = gold()
    assert len(rows.experiments) == 1
    assert len(rows.shots) == 2
    assert len(rows.predictions) == 8
    assert rows.experiments[0][:2] == ("experiment-1", "experiment-source")
    assert rows.shots[0][0:3] == ("shot-0", "experiment-1", 0)
    assert rows.shots[0][5] == b"\x01\x02"
    assert rows.predictions[0] == ("shot-0", "belief_matching", True)


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        ({"source_record_id": ""}, "source_record_id"),
        ({"experiment_id": "missing"}, "missing experiment"),
        ({"shot_index": -1}, "shot_index"),
        ({"measurement_bits": b"\x01"}, "packed bit"),
        ({"measurement_bits": b"\x01\x80"}, "packed bit"),
        ({"detector_bits": b"\x01"}, "packed bit"),
        ({"detector_bits": b"\x01\x80"}, "packed bit"),
        ({"detector_event_count": 1}, "event count"),
    ],
)
def test_gold_rejects_bad_shots(changed, message):
    with pytest.raises(ValueError, match=message):
        gold(shots=[shot(0, **changed), shot(1)])


def test_gold_rejects_duplicate_ids_indices_and_missing_shots():
    with pytest.raises(ValueError, match="source_record_id"):
        gold(shots=[shot(0), shot(1, source_record_id="shot-0")])
    with pytest.raises(ValueError, match="shot_index"):
        gold(shots=[shot(0), shot(0, source_record_id="shot-2")])
    with pytest.raises(ValueError, match="shot count"):
        gold(shots=[shot(0)])


def test_gold_rejects_bad_experiment_and_nulls():
    with pytest.raises(ValueError, match="experiment metadata"):
        gold(experiments=[experiment(detector_count=0)])
    with pytest.raises(ValueError, match="null"):
        gold(shots=[shot(0, actual_observable_flip=None), shot(1)])


def ml_row(index, **changes):
    row = {
        "example_id": f"shot-{index}",
        "experiment_id": "experiment-1",
        "shot_index": index,
        "distance": 3,
        "rounds": 25,
        "center_row": 3,
        "center_col": 5,
        "detector_count": 10,
        "detector_event_count": 2,
        "detector_bits": b"\x01\x02",
        "belief_matching_prediction": True,
        "correlated_matching_prediction": False,
        "pymatching_prediction": True,
        "tensor_network_contraction_prediction": False,
        "actual_observable_flip": False,
    }
    row.update(changes)
    return tuple(row[name] for name in GOOGLE_ML_SCHEMA.names[:-1])


def test_ml_table_has_exact_schema_and_course_splits():
    table = google_ml_table([ml_row(0), ml_row(1), ml_row(8)])
    assert table.schema == GOOGLE_ML_SCHEMA
    assert table["data_split"].to_pylist() == ["train", "test", "validation"]
    assert table["detector_bits"].to_pylist() == [b"\x01\x02"] * 3


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"example_id": "shot-0"}, "duplicate.*example_id"),
        ({"detector_bits": b"\x01"}, "byte length"),
        ({"detector_bits": b"\x01\x80"}, "padding"),
        ({"detector_event_count": 1}, "event count"),
        ({"pymatching_prediction": None}, "prediction"),
    ],
)
def test_ml_rejects_bad_gold_rows(changes, message):
    with pytest.raises(ValueError, match=message):
        google_ml_table([ml_row(0), ml_row(1, **changes), ml_row(8)])


def test_ml_requires_all_splits():
    with pytest.raises(ValueError, match="nonempty"):
        google_ml_table([ml_row(0), ml_row(2)])
