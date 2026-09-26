"""Export the syndrome ML handoff from committed PostgreSQL Gold views."""

from __future__ import annotations

from collections.abc import Iterable

import pyarrow as pa

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import (
    postgres_connection,
    write_lake_object,
)
from quantum_lake_student.evidence import parquet_bytes
from quantum_lake_student.ml import (
    MODEL_SPLITS,
    partition_records,
    syndrome_data_split,
    syndrome_model_input,
)
from quantum_lake_student.models import StageResult


SYNDROME_ML_OBJECT = "ml/ml_syndrome_decoder_example.parquet"
SYNDROME_ML_SCHEMA = pa.schema(
    [
        pa.field("example_id", pa.string(), nullable=False),
        pa.field("experiment_id", pa.string(), nullable=False),
        pa.field("physical_fault_rate", pa.float64(), nullable=False),
        pa.field("syndrome_bits", pa.binary(), nullable=False),
        pa.field("round_count", pa.int32(), nullable=False),
        pa.field("check_count", pa.int32(), nullable=False),
        pa.field("logical_error_label", pa.bool_(), nullable=False),
        pa.field("sample_weight", pa.int64(), nullable=False),
        pa.field("data_split", pa.string(), nullable=False),
    ]
)


def syndrome_ml_table(records: Iterable[tuple]) -> pa.Table:
    """Validate Gold query results and add the course-defined split."""
    output = []
    example_ids = set()
    group_keys = set()

    for record in records:
        (
            example_id,
            experiment_id,
            fault_rate,
            bits,
            round_count,
            check_count,
            label,
            weight,
        ) = record
        group_key = (experiment_id, bits, label)
        if not example_id or example_id in example_ids:
            raise ValueError("Missing or duplicate syndrome ML example_id")
        if group_key in group_keys:
            raise ValueError("Duplicate syndrome ML experiment-pattern-label group")
        if not experiment_id or not 0 < fault_rate < 1:
            raise ValueError("Invalid syndrome ML experiment")
        syndrome_model_input(bits)
        if (round_count, check_count) != (4, 4):
            raise ValueError("Invalid syndrome ML shape")
        if type(label) is not bool or type(weight) is not int or weight <= 0:
            raise ValueError("Invalid syndrome ML label or weight")

        split = syndrome_data_split(fault_rate)
        example_ids.add(example_id)
        group_keys.add(group_key)
        output.append(
            {
                "example_id": example_id,
                "experiment_id": experiment_id,
                "physical_fault_rate": fault_rate,
                "syndrome_bits": bits,
                "round_count": round_count,
                "check_count": check_count,
                "logical_error_label": label,
                "sample_weight": weight,
                "data_split": split,
            }
        )

    partitions = partition_records(output)
    if any(not partitions[split] for split in MODEL_SPLITS):
        raise ValueError("Syndrome ML table needs nonempty train/validation/test")
    return pa.Table.from_pylist(output, schema=SYNDROME_ML_SCHEMA)


def build_syndrome_ml_table(settings: Settings, *, run_id: str) -> StageResult:
    """Build and publish the required syndrome Parquet table from Gold only."""
    result = StageResult(stage="build_syndrome_ml_table", run_id=run_id)
    with postgres_connection(settings) as connection:
        connection.execute(
            "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"
        )
        records = connection.execute(
            "SELECT example_id, experiment_id, physical_fault_rate, "
            "syndrome_bits, round_count, check_count, logical_error_label, "
            "sample_weight FROM gold.syndrome_ml_example ORDER BY example_id"
        ).fetchall()
        gold_count, gold_weight = connection.execute(
            "SELECT count(*), coalesce(sum(quantity), 0) "
            "FROM gold.syndrome_observation"
        ).fetchone()
        link_count, linked_sources, linked_examples = connection.execute(
            "SELECT count(*), count(DISTINCT source_record_id), "
            "count(DISTINCT example_id) FROM gold.syndrome_ml_source"
        ).fetchone()

    table = syndrome_ml_table(records)
    if (
        link_count != gold_count
        or linked_sources != gold_count
        or linked_examples != table.num_rows
        or sum(table["sample_weight"].to_pylist()) != gold_weight
    ):
        raise RuntimeError("Syndrome Gold-to-ML reconciliation failed")

    write_lake_object(settings, SYNDROME_ML_OBJECT, parquet_bytes(table))
    result.input_count = gold_count
    result.output_count = table.num_rows
    result.finish()
    return result


def run(run_id: str) -> StageResult:
    """Run the syndrome-only ML export with environment-based settings."""
    return build_syndrome_ml_table(Settings.from_environment(), run_id=run_id)
