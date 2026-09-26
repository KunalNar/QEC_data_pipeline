"""Load the chosen syndrome Gold schema from validated Silver Parquet."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from io import BytesIO

import pyarrow as pa
import pyarrow.parquet as pq
import psycopg

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import postgres_connection, read_lake_object
from quantum_lake_student.models import StageResult
from quantum_lake_student.syndromes import SILVER_OBJECT, SILVER_SCHEMA


@dataclass
class SyndromeGoldRows:
    experiments: list[tuple[str, float]]
    patterns: list[tuple[bytes, int, int]]
    observations: list[tuple[str, str, bytes, bool, int]]
    total_weight: int


def syndrome_gold_rows(table: pa.Table) -> SyndromeGoldRows:
    """Check the Silver contract and separate its three Gold row grains."""
    if not table.schema.equals(SILVER_SCHEMA, check_metadata=False):
        raise ValueError("Unexpected syndrome Silver schema")
    if any(column.null_count for column in table.columns):
        raise ValueError("Syndrome Silver contains null values")

    experiments: dict[str, float] = {}
    patterns: dict[bytes, tuple[int, int]] = {}
    observations: list[tuple[str, str, bytes, bool, int]] = []
    source_ids: set[str] = set()
    total_weight = 0

    for row in table.to_pylist():
        source_id = row["source_record_id"]
        experiment_id = row["experiment_id"]
        fault_rate = row["physical_fault_rate"]
        bits = row["syndrome_bits"]
        shape = (row["round_count"], row["check_count"])
        label = row["logical_error_label"]
        quantity = row["quantity"]

        if not source_id or source_id in source_ids:
            raise ValueError("Missing or duplicate syndrome source_record_id")
        if not experiment_id or not 0 < fault_rate < 1:
            raise ValueError("Invalid syndrome experiment")
        if len(bits) != 16 or any(bit not in (0, 1) for bit in bits):
            raise ValueError("Invalid syndrome pattern bytes")
        if shape != (4, 4) or type(label) is not bool:
            raise ValueError("Invalid syndrome shape or label")
        if type(quantity) is not int or quantity <= 0:
            raise ValueError("Invalid syndrome quantity")
        if experiment_id in experiments and experiments[experiment_id] != fault_rate:
            raise ValueError("Conflicting fault rates for one experiment")
        if bits in patterns and patterns[bits] != shape:
            raise ValueError("Conflicting shapes for one syndrome pattern")

        source_ids.add(source_id)
        experiments[experiment_id] = fault_rate
        patterns[bits] = shape
        observations.append((source_id, experiment_id, bits, label, quantity))
        total_weight += quantity

    if not observations:
        raise ValueError("Syndrome Silver table is empty")

    return SyndromeGoldRows(
        experiments=list(experiments.items()),
        patterns=[(bits, *shape) for bits, shape in patterns.items()],
        observations=observations,
        total_weight=total_weight,
    )


def load_syndromes_gold(settings: Settings, *, run_id: str) -> StageResult:
    """Replace syndrome Gold atomically; preserve the previous load on error."""
    result = StageResult(stage="load_syndromes_gold", run_id=run_id)
    silver = pq.read_table(BytesIO(read_lake_object(settings, SILVER_OBJECT)))
    rows = syndrome_gold_rows(silver)
    with postgres_connection(settings) as connection:
        replace_syndrome_gold(connection, rows)

    result.input_count = silver.num_rows
    result.output_count = len(rows.observations)
    result.finish()
    return result


def replace_syndrome_gold(
    connection: psycopg.Connection, rows: SyndromeGoldRows
) -> None:
    """Replace syndrome Gold within the caller's transaction."""
    expected_ids = {record[0] for record in rows.observations}
    schema_sql = files("quantum_lake_student").joinpath(
        "sql/gold_syndrome.sql"
    ).read_text(encoding="utf-8")

    for statement in schema_sql.split(";"):
        if statement.strip():
            connection.execute(statement)

    # The caller's transaction rolls all deletes and inserts back on failure.
    connection.execute("DELETE FROM gold.syndrome_observation")
    connection.execute("DELETE FROM gold.syndrome_pattern")
    connection.execute("DELETE FROM gold.syndrome_experiment")
    with connection.cursor() as cursor:
        cursor.executemany(
            "INSERT INTO gold.syndrome_experiment VALUES (%s, %s)",
            rows.experiments,
        )
        cursor.executemany(
            "INSERT INTO gold.syndrome_pattern VALUES (%s, %s, %s)",
            rows.patterns,
        )
        cursor.executemany(
            "INSERT INTO gold.syndrome_observation VALUES (%s, %s, %s, %s, %s)",
            rows.observations,
        )

    actual_ids = {
        source_id
        for (source_id,) in connection.execute(
            "SELECT source_record_id FROM gold.syndrome_observation"
        )
    }
    (actual_count, actual_weight) = connection.execute(
        "SELECT count(*), coalesce(sum(quantity), 0) "
        "FROM gold.syndrome_observation"
    ).fetchone()
    if (
        actual_ids != expected_ids
        or actual_count != len(rows.observations)
        or actual_weight != rows.total_weight
    ):
        raise RuntimeError("Syndrome Silver-to-Gold reconciliation failed")


def run(run_id: str) -> StageResult:
    """Run the syndrome Gold loader using environment-based settings."""
    return load_syndromes_gold(Settings.from_environment(), run_id=run_id)
