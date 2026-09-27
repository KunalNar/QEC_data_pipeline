"""Export the required Google shot examples from PostgreSQL Gold."""

from __future__ import annotations

from collections.abc import Iterable

import pyarrow as pa

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import postgres_connection, write_lake_object
from quantum_lake_student.evidence import parquet_bytes
from quantum_lake_student.ml import (
    GOOGLE_META_PREDICTION_COLUMNS,
    MODEL_SPLITS,
    google_data_split,
    google_meta_model_input,
    partition_records,
)
from quantum_lake_student.models import StageResult


GOOGLE_ML_OBJECT = "ml/ml_google_decoder_example.parquet"
GOOGLE_ML_SCHEMA = pa.schema(
    [
        pa.field("example_id", pa.string(), nullable=False),
        pa.field("experiment_id", pa.string(), nullable=False),
        pa.field("shot_index", pa.int64(), nullable=False),
        pa.field("distance", pa.int32(), nullable=False),
        pa.field("rounds", pa.int32(), nullable=False),
        pa.field("center_row", pa.int32(), nullable=False),
        pa.field("center_col", pa.int32(), nullable=False),
        pa.field("detector_count", pa.int32(), nullable=False),
        pa.field("detector_event_count", pa.int32(), nullable=False),
        pa.field("detector_bits", pa.binary(), nullable=False),
        *(pa.field(column, pa.bool_(), nullable=False)
          for column in GOOGLE_META_PREDICTION_COLUMNS),
        pa.field("actual_observable_flip", pa.bool_(), nullable=False),
        pa.field("data_split", pa.string(), nullable=False),
    ]
)


def google_ml_table(records: Iterable[tuple]) -> pa.Table:
    """Check Gold query rows and assign the course shot-based split."""
    output = []
    example_ids = set()
    shot_keys = set()

    for record in records:
        row = dict(zip(GOOGLE_ML_SCHEMA.names[:-1], record, strict=True))
        example_id = row["example_id"]
        shot_key = (row["experiment_id"], row["shot_index"])
        if not example_id or example_id in example_ids:
            raise ValueError("Missing or duplicate Google ML example_id")
        if not row["experiment_id"] or shot_key in shot_keys:
            raise ValueError("Missing or duplicate Google ML experiment shot")
        if (
            row["distance"] <= 0
            or row["rounds"] <= 0
            or row["center_row"] < 0
            or row["center_col"] < 0
            or row["detector_count"] <= 0
        ):
            raise ValueError("Invalid Google ML experiment metadata")
        bits = row["detector_bits"]
        bit_count = row["detector_count"]
        if not isinstance(bits, bytes) or len(bits) != (bit_count + 7) // 8:
            raise ValueError("Invalid Google ML detector byte length")
        if bit_count % 8 and bits[-1] >> (bit_count % 8):
            raise ValueError("Invalid Google ML detector padding")
        if row["detector_event_count"] != sum(byte.bit_count() for byte in bits):
            raise ValueError("Google ML detector event count disagrees with bits")
        if type(row["actual_observable_flip"]) is not bool or any(
            type(row[column]) is not bool
            for column in GOOGLE_META_PREDICTION_COLUMNS
        ):
            raise ValueError("Invalid Google ML label or decoder prediction")
        google_meta_model_input(row)
        row["data_split"] = google_data_split(row["shot_index"])
        example_ids.add(example_id)
        shot_keys.add(shot_key)
        output.append(row)

    partitions = partition_records(output)
    if any(not partitions[split] for split in MODEL_SPLITS):
        raise ValueError("Google ML table needs nonempty train/validation/test")
    return pa.Table.from_pylist(output, schema=GOOGLE_ML_SCHEMA)


def build_google_ml_table(settings: Settings, *, run_id: str) -> StageResult:
    """Publish a checked Google ML Parquet table without reading Silver."""
    result = StageResult(stage="build_google_ml_table", run_id=run_id)
    with postgres_connection(settings) as connection:
        connection.execute(
            "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"
        )
        records = connection.execute(
            "SELECT example_id, experiment_id, shot_index, distance, rounds, "
            "center_row, center_col, detector_count, detector_event_count, "
            "detector_bits, belief_matching_prediction, "
            "correlated_matching_prediction, pymatching_prediction, "
            "tensor_network_contraction_prediction, actual_observable_flip "
            "FROM gold.google_ml_example ORDER BY experiment_id, shot_index"
        ).fetchall()
        (shot_count,) = connection.execute(
            "SELECT count(*) FROM gold.google_shot"
        ).fetchone()
        (prediction_count,) = connection.execute(
            "SELECT count(*) FROM gold.google_decoder_prediction"
        ).fetchone()
        (link_count,) = connection.execute(
            "SELECT count(*) FROM gold.google_ml_source"
        ).fetchone()

    table = google_ml_table(records)
    if (
        table.num_rows != shot_count
        or link_count != shot_count
        or prediction_count != shot_count * len(GOOGLE_META_PREDICTION_COLUMNS)
    ):
        raise RuntimeError("Google Gold-to-ML reconciliation failed")
    write_lake_object(settings, GOOGLE_ML_OBJECT, parquet_bytes(table))
    result.input_count = shot_count
    result.output_count = table.num_rows
    result.finish()
    return result


def run(run_id: str) -> StageResult:
    return build_google_ml_table(Settings.from_environment(), run_id=run_id)
