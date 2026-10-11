"""Check both ML contracts before the training orchestrator fits any model."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quantum_lake_student.config import Settings
from quantum_lake_student.stages import train


INPUT_SCHEMAS = {
    train.SYNDROME_ML_OBJECT: train.SYNDROME_ML_SCHEMA,
    train.GOOGLE_ML_OBJECT: train.GOOGLE_ML_SCHEMA,
}


def parquet_with_schema(schema: pa.Schema) -> bytes:
    buffer = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist([], schema=schema), buffer)
    return buffer.getvalue().to_pybytes()


@pytest.mark.parametrize("with_metadata", [False, True])
def test_both_schemas_are_checked_before_any_training(monkeypatch, with_metadata):
    settings = Settings.from_environment()
    events = []
    schemas = {
        name: schema.with_metadata({b"note": b"metadata is not a column"})
        if with_metadata else schema
        for name, schema in INPUT_SCHEMAS.items()
    }

    def read_object(settings_arg, object_name):
        assert settings_arg is settings
        events.append(object_name)
        return parquet_with_schema(schemas[object_name])

    monkeypatch.setattr(train.Settings, "from_environment", lambda: settings)
    monkeypatch.setattr(train, "read_lake_object", read_object)
    result = object()

    def task_a(*args, **kwargs):
        events.append("fit A")
        return result

    monkeypatch.setattr(train, "train_task_a", task_a)
    monkeypatch.setattr(
        train, "train_task_b", lambda *args, **kwargs: events.append("fit B"),
    )
    monkeypatch.setattr(
        train, "train_task_c", lambda *args, **kwargs: events.append("fit C"),
    )
    monkeypatch.setattr(train, "write_report", lambda: events.append("report"))

    assert train.run("schema-test") is result
    assert events == [*INPUT_SCHEMAS, "fit A", "fit B", "fit C", "report"]


@pytest.mark.parametrize("object_name", list(INPUT_SCHEMAS))
@pytest.mark.parametrize("problem", ["missing_column", "wrong_type", "nullable_column"])
def test_invalid_schema_stops_all_training(monkeypatch, object_name, problem):
    schemas = dict(INPUT_SCHEMAS)
    schema = schemas[object_name]
    if problem == "missing_column":
        schemas[object_name] = schema.remove(0)
    elif problem == "wrong_type":
        schemas[object_name] = schema.set(
            0, pa.field("example_id", pa.int64(), nullable=False),
        )
    else:
        schemas[object_name] = schema.set(
            0, pa.field("example_id", pa.string(), nullable=True),
        )

    monkeypatch.setattr(
        train, "read_lake_object",
        lambda settings, name: parquet_with_schema(schemas[name]),
    )

    def unexpected_training(*args, **kwargs):
        pytest.fail("No task may start fitting when either input schema is invalid")

    for name in ("train_task_a", "train_task_b", "train_task_c", "write_report"):
        monkeypatch.setattr(train, name, unexpected_training)

    with pytest.raises(ValueError, match="ML input schema mismatch") as error:
        train.run("invalid-schema-test")
    assert object_name in str(error.value)
