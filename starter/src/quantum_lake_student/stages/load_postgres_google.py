"""Normalize Google Silver shots and decoder predictions into Gold."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from io import BytesIO

import pyarrow as pa
import pyarrow.parquet as pq
import psycopg

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import postgres_connection, read_lake_object
from quantum_lake_student.google_qec import (
    EXPERIMENT_PATH,
    EXPERIMENT_SCHEMA,
    SHOT_PATH,
    SHOT_SCHEMA,
)
from quantum_lake_student.ml import GOOGLE_META_PREDICTION_COLUMNS
from quantum_lake_student.models import StageResult


DECODERS = tuple(
    (column.removesuffix("_prediction"), column)
    for column in GOOGLE_META_PREDICTION_COLUMNS
)


@dataclass
class GoogleGoldRows:
    experiments: list[tuple]
    shots: list[tuple]
    predictions: list[tuple]


def _check_table(table: pa.Table, schema: pa.Schema, name: str) -> None:
    if not table.schema.equals(schema, check_metadata=False):
        raise ValueError(f"Unexpected Google {name} Silver schema")
    if any(column.null_count for column in table.columns):
        raise ValueError(f"Google {name} Silver contains null values")
    if not table.num_rows:
        raise ValueError(f"Google {name} Silver is empty")


def _packed_bits_valid(value: bytes, bit_count: int) -> bool:
    if len(value) != (bit_count + 7) // 8:
        return False
    remainder = bit_count % 8
    return not remainder or value[-1] >> remainder == 0


def google_gold_rows(experiments: pa.Table, shots: pa.Table) -> GoogleGoldRows:
    """Validate both Silver grains and separate decoder predictions."""
    _check_table(experiments, EXPERIMENT_SCHEMA, "experiment")
    _check_table(shots, SHOT_SCHEMA, "shot")
    output = GoogleGoldRows([], [], [])
    by_experiment = {}
    source_ids = set()

    for row in experiments.to_pylist():
        experiment_id = row["experiment_id"]
        source_id = row["source_record_id"]
        if not experiment_id or experiment_id in by_experiment:
            raise ValueError("Missing or duplicate Google experiment_id")
        if not source_id or source_id in source_ids:
            raise ValueError("Missing or duplicate Google source_record_id")
        if (
            not row["basis"]
            or row["distance"] <= 0
            or row["rounds"] <= 0
            or row["shots"] <= 0
            or row["center_row"] < 0
            or row["center_col"] < 0
            or row["measurement_count"] <= 0
            or row["detector_count"] <= 0
        ):
            raise ValueError("Invalid Google experiment metadata")
        by_experiment[experiment_id] = row
        source_ids.add(source_id)
        output.experiments.append((
            experiment_id,
            source_id,
            row["basis"],
            row["distance"],
            row["rounds"],
            row["shots"],
            row["center_row"],
            row["center_col"],
            row["measurement_count"],
            row["detector_count"],
        ))

    indices = {experiment_id: set() for experiment_id in by_experiment}
    for row in shots.to_pylist():
        source_id = row["source_record_id"]
        experiment_id = row["experiment_id"]
        experiment = by_experiment.get(experiment_id)
        if experiment is None:
            raise ValueError("Google shot references a missing experiment")
        if not source_id or source_id in source_ids:
            raise ValueError("Missing or duplicate Google source_record_id")
        index = row["shot_index"]
        if index < 0 or index >= experiment["shots"] or index in indices[experiment_id]:
            raise ValueError("Invalid or duplicate Google shot_index")
        if not _packed_bits_valid(
            row["measurement_bits"], experiment["measurement_count"]
        ) or not _packed_bits_valid(
            row["detector_bits"], experiment["detector_count"]
        ):
            raise ValueError("Invalid Google packed bit length or padding")
        if row["detector_event_count"] != sum(
            value.bit_count() for value in row["detector_bits"]
        ):
            raise ValueError("Google detector event count disagrees with packed bits")
        if not isinstance(row["sweep_bits"], bytes) or type(
            row["actual_observable_flip"]
        ) is not bool:
            raise ValueError("Invalid Google sweep bits or actual flip")

        indices[experiment_id].add(index)
        source_ids.add(source_id)
        output.shots.append((
            source_id,
            experiment_id,
            index,
            row["measurement_bits"],
            row["sweep_bits"],
            row["detector_bits"],
            row["detector_event_count"],
            row["actual_observable_flip"],
        ))
        for decoder_name, column in DECODERS:
            prediction = row[column]
            if type(prediction) is not bool:
                raise ValueError("Invalid Google decoder prediction")
            output.predictions.append((source_id, decoder_name, prediction))

    for experiment_id, experiment in by_experiment.items():
        if len(indices[experiment_id]) != experiment["shots"]:
            raise ValueError("Google experiment shot count does not reconcile")
    return output


def replace_google_gold(connection: psycopg.Connection, rows: GoogleGoldRows) -> None:
    """Replace Google Gold within the caller's transaction."""
    schema_sql = files("quantum_lake_student").joinpath(
        "sql/gold_google.sql"
    ).read_text(encoding="utf-8")
    for statement in schema_sql.split(";"):
        if statement.strip():
            connection.execute(statement)

    for name in (
        "google_decoder_prediction",
        "google_shot",
        "google_decoder",
        "google_experiment",
    ):
        connection.execute(f"DELETE FROM gold.{name}")

    with connection.cursor() as cursor:
        for name, columns, values in (
            (
                "google_experiment",
                "experiment_id, source_record_id, basis, distance, rounds, shots, "
                "center_row, center_col, measurement_count, detector_count",
                rows.experiments,
            ),
            ("google_decoder", "decoder_name", [(name,) for name, _ in DECODERS]),
            (
                "google_shot",
                "source_record_id, experiment_id, shot_index, measurement_bits, "
                "sweep_bits, detector_bits, detector_event_count, actual_observable_flip",
                rows.shots,
            ),
            (
                "google_decoder_prediction",
                "source_record_id, decoder_name, predicted_flip",
                rows.predictions,
            ),
        ):
            placeholders = ", ".join(["%s"] * len(values[0]))
            cursor.executemany(
                f"INSERT INTO gold.{name} ({columns}) VALUES ({placeholders})",
                values,
            )

    (experiment_count,) = connection.execute(
        "SELECT count(*) FROM gold.google_experiment"
    ).fetchone()
    (shot_count,) = connection.execute(
        "SELECT count(*) FROM gold.google_shot"
    ).fetchone()
    (prediction_count,) = connection.execute(
        "SELECT count(*) FROM gold.google_decoder_prediction"
    ).fetchone()
    if (
        experiment_count != len(rows.experiments)
        or shot_count != len(rows.shots)
        or prediction_count != len(rows.predictions)
    ):
        raise RuntimeError("Google Silver-to-Gold row reconciliation failed")


def load_google_gold(settings: Settings, *, run_id: str) -> StageResult:
    """Read Google Silver from the lake and atomically load Google Gold."""
    result = StageResult(stage="load_google_gold", run_id=run_id)
    experiments = pq.read_table(
        BytesIO(read_lake_object(settings, EXPERIMENT_PATH))
    )
    shots = pq.read_table(BytesIO(read_lake_object(settings, SHOT_PATH)))
    rows = google_gold_rows(experiments, shots)
    with postgres_connection(settings) as connection:
        replace_google_gold(connection, rows)
    result.input_count = experiments.num_rows + shots.num_rows
    result.output_count = (
        len(rows.experiments) + len(rows.shots) + len(rows.predictions)
    )
    result.finish()
    return result


def run(run_id: str) -> StageResult:
    return load_google_gold(Settings.from_environment(), run_id=run_id)
