"""Build the Google QEC Silver tables from a validated Bronze archive."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePosixPath
from zipfile import ZipFile

import pyarrow as pa
import yaml

from .evidence import (
    IssueLog,
    SOURCE_TRACE_SCHEMA,
    source_trace_row,
)
from .formats import b8_record_bytes, iter_b8_records, parse_01_records
from .models import stable_record_hash
from .source_validation import (
    SourceObjectSpec,
    SourceValidationError,
    validate_source_object,
)


SOURCE_NAME = "google_qec"
BRONZE_OBJECT = "bronze/source=google_qec/google-surface-code-curated.zip"
EXPERIMENT_PATH = "silver/google_qec/experiment.parquet"
SHOT_PATH = "silver/google_qec/shot.parquet"
EXPECTED_EXPERIMENT_COUNT = 5

EXPERIMENT_SCHEMA = pa.schema(
    [
        pa.field("source_record_id", pa.string(), nullable=False),
        pa.field("experiment_id", pa.string(), nullable=False),
        pa.field("basis", pa.string(), nullable=False),
        pa.field("distance", pa.int32(), nullable=False),
        pa.field("rounds", pa.int32(), nullable=False),
        pa.field("shots", pa.int64(), nullable=False),
        pa.field("center_row", pa.int32(), nullable=False),
        pa.field("center_col", pa.int32(), nullable=False),
        pa.field("measurement_count", pa.int32(), nullable=False),
        pa.field("detector_count", pa.int32(), nullable=False),
    ]
)

SHOT_SCHEMA = pa.schema(
    [
        pa.field("source_record_id", pa.string(), nullable=False),
        pa.field("experiment_id", pa.string(), nullable=False),
        pa.field("shot_index", pa.int64(), nullable=False),
        pa.field("measurement_bits", pa.binary(), nullable=False),
        pa.field("sweep_bits", pa.binary(), nullable=False),
        pa.field("detector_bits", pa.binary(), nullable=False),
        pa.field("detector_event_count", pa.int32(), nullable=False),
        pa.field("actual_observable_flip", pa.bool_(), nullable=False),
        pa.field("belief_matching_prediction", pa.bool_(), nullable=False),
        pa.field("correlated_matching_prediction", pa.bool_(), nullable=False),
        pa.field("pymatching_prediction", pa.bool_(), nullable=False),
        pa.field(
            "tensor_network_contraction_prediction",
            pa.bool_(),
            nullable=False,
        ),
    ]
)

_FOLDER_PATTERN = re.compile(
    r"^surface_code_b(?P<basis>[A-Za-z])_d(?P<distance>\d+)_"
    r"r(?P<rounds>\d+)_center_(?P<center_row>\d+)_"
    r"(?P<center_col>\d+)$"
)

_B8_MEMBERS = (
    ("measurement_bits", "measurements.b8", "circuit_measurements"),
    ("sweep_bits", "sweep.b8", "circuit_sweep_bits"),
    ("detector_bits", "detection_events.b8", "circuit_detectors"),
)

_FLIP_MEMBERS = (
    ("actual_observable_flip", "obs_flips_actual.01"),
    (
        "belief_matching_prediction",
        "obs_flips_predicted_by_belief_matching.01",
    ),
    (
        "correlated_matching_prediction",
        "obs_flips_predicted_by_correlated_matching.01",
    ),
    (
        "pymatching_prediction",
        "obs_flips_predicted_by_pymatching.01",
    ),
    (
        "tensor_network_contraction_prediction",
        "obs_flips_predicted_by_tensor_network_contraction.01",
    ),
)

SHOT_TRACE_MEMBER_COUNT = len(_B8_MEMBERS) + len(_FLIP_MEMBERS)


class GoogleQecArchiveError(ValueError):
    """Raised when the Google archive cannot be interpreted safely."""


@dataclass(frozen=True)
class ExperimentMetadata:
    basis: str
    distance: int
    rounds: int
    shots: int
    center_row: int
    center_col: int
    measurement_count: int
    sweep_count: int
    detector_count: int


@dataclass(frozen=True)
class GoogleQecTables:
    experiments: pa.Table
    shots: pa.Table
    source_trace: pa.Table
    data_issues: pa.Table
    input_sha256: str
    experiment_count: int
    input_count: int
    accepted_count: int
    rejected_count: int


def _integer_property(
    properties: Mapping[str, object],
    name: str,
    *,
    minimum: int,
) -> int:
    value = properties.get(name)
    if type(value) is not int or value < minimum:
        raise GoogleQecArchiveError(
            f"Property {name!r} must be an integer >= {minimum}"
        )
    return value


def _experiment_metadata(
    folder: str,
    properties_value: object,
) -> ExperimentMetadata:
    if not isinstance(properties_value, Mapping):
        raise GoogleQecArchiveError(
            f"{folder}/properties.yml must contain a mapping"
        )

    basis = properties_value.get("basis")
    if not isinstance(basis, str) or not basis:
        raise GoogleQecArchiveError(
            f"{folder}/properties.yml has an invalid basis"
        )

    metadata = ExperimentMetadata(
        basis=basis,
        distance=_integer_property(properties_value, "distance", minimum=1),
        rounds=_integer_property(properties_value, "rounds", minimum=1),
        shots=_integer_property(properties_value, "shots", minimum=1),
        center_row=_integer_property(
            properties_value,
            "center_data_qubit_row",
            minimum=0,
        ),
        center_col=_integer_property(
            properties_value,
            "center_data_qubit_col",
            minimum=0,
        ),
        measurement_count=_integer_property(
            properties_value,
            "circuit_measurements",
            minimum=1,
        ),
        sweep_count=_integer_property(
            properties_value,
            "circuit_sweep_bits",
            minimum=0,
        ),
        detector_count=_integer_property(
            properties_value,
            "circuit_detectors",
            minimum=1,
        ),
    )

    match = _FOLDER_PATTERN.fullmatch(folder)
    if match is None:
        raise GoogleQecArchiveError(
            f"Unexpected Google experiment directory: {folder}"
        )

    filename_values = match.groupdict()
    expected = {
        "basis": metadata.basis,
        "distance": metadata.distance,
        "rounds": metadata.rounds,
        "center_row": metadata.center_row,
        "center_col": metadata.center_col,
    }
    actual = {
        "basis": filename_values["basis"],
        "distance": int(filename_values["distance"]),
        "rounds": int(filename_values["rounds"]),
        "center_row": int(filename_values["center_row"]),
        "center_col": int(filename_values["center_col"]),
    }
    if actual != expected:
        raise GoogleQecArchiveError(
            f"Directory name and properties disagree for {folder}"
        )

    return metadata


def _experiment_folders(member_names: tuple[str, ...]) -> tuple[str, ...]:
    folders = sorted(
        {
            PurePosixPath(name).parts[0]
            for name in member_names
            if len(PurePosixPath(name).parts) > 1
        }
    )
    if len(folders) != EXPECTED_EXPERIMENT_COUNT:
        raise GoogleQecArchiveError(
            f"Expected {EXPECTED_EXPERIMENT_COUNT} experiment directories, "
            f"found {len(folders)}"
        )
    return tuple(folders)


def _required_member_names(folder: str) -> tuple[str, ...]:
    relative_names = (
        "properties.yml",
        *(name for _, name, _ in _B8_MEMBERS),
        *(name for _, name in _FLIP_MEMBERS),
    )
    return tuple(f"{folder}/{name}" for name in relative_names)


def _padding_violation_count(
    data: bytes,
    *,
    bits_per_record: int,
    record_count: int,
) -> int:
    remainder = bits_per_record % 8
    if bits_per_record == 0 or remainder == 0:
        return 0

    record_bytes = b8_record_bytes(bits_per_record)
    padding_mask = (0xFF << remainder) & 0xFF
    return sum(
        bool(data[index * record_bytes + record_bytes - 1] & padding_mask)
        for index in range(record_count)
    )


def _source_record_id(
    input_sha256: str,
    folder: str,
    record_locator: str,
) -> str:
    return "google-" + stable_record_hash(
        {
            "input_sha256": input_sha256,
            "bronze_object": BRONZE_OBJECT,
            "experiment": folder,
            "record_locator": record_locator,
        }
    )


def _trace_table(
    *,
    input_sha256: str,
    folder: str,
    experiment_source_id: str,
    shot_source_ids: list[str],
    record_sizes: Mapping[str, int],
) -> pa.Table:
    columns: dict[str, list[object]] = {
        name: [] for name in SOURCE_TRACE_SCHEMA.names
    }

    def add_row(source_record_id: str, member: str, locator: str) -> None:
        row = source_trace_row(
            source_record_id=source_record_id,
            source_name=SOURCE_NAME,
            bronze_object=BRONZE_OBJECT,
            archive_member=member,
            record_locator=locator,
            input_sha256=input_sha256,
        )
        for name in SOURCE_TRACE_SCHEMA.names:
            columns[name].append(row[name])

    add_row(
        experiment_source_id,
        f"{folder}/properties.yml",
        "yaml_document",
    )

    for shot_index, source_record_id in enumerate(shot_source_ids):
        for column, relative_name, _ in _B8_MEMBERS:
            record_size = record_sizes[column]
            add_row(
                source_record_id,
                f"{folder}/{relative_name}",
                (
                    f"shot={shot_index};"
                    f"byte_offset={shot_index * record_size};"
                    f"byte_length={record_size}"
                ),
            )
        for _, relative_name in _FLIP_MEMBERS:
            add_row(
                source_record_id,
                f"{folder}/{relative_name}",
                f"shot={shot_index};line={shot_index + 1}",
            )

    return pa.Table.from_pydict(columns, schema=SOURCE_TRACE_SCHEMA)


def build_google_qec_tables(
    archive_bytes: bytes,
    *,
    spec: SourceObjectSpec,
    run_id: str,
) -> GoogleQecTables:
    """Validate Google records and return typed, in-memory Silver tables."""
    try:
        validated = validate_source_object(archive_bytes, spec)
    except SourceValidationError as error:
        raise GoogleQecArchiveError(str(error)) from error

    folders = _experiment_folders(validated.member_names)
    available_members = set(validated.member_names)
    missing_members = [
        member
        for folder in folders
        for member in _required_member_names(folder)
        if member not in available_members
    ]
    if missing_members:
        raise GoogleQecArchiveError(
            f"Missing required companion file: {missing_members[0]}"
        )

    experiment_rows: list[dict[str, object]] = []
    shot_rows: list[dict[str, object]] = []
    trace_tables: list[pa.Table] = []
    issues = IssueLog(
        run_id=run_id,
        source_name=SOURCE_NAME,
        bronze_object=BRONZE_OBJECT,
        input_sha256=validated.sha256,
        default_action="exclude_experiment_from_silver",
        default_locator="archive_member",
    )
    input_count = 0
    rejected_count = 0

    with ZipFile(BytesIO(archive_bytes)) as archive:
        for folder in folders:
            properties_member = f"{folder}/properties.yml"
            try:
                properties = yaml.safe_load(archive.read(properties_member))
            except yaml.YAMLError as error:
                raise GoogleQecArchiveError(
                    f"Cannot parse {properties_member}: {error}"
                ) from error

            metadata = _experiment_metadata(folder, properties)
            input_count += metadata.shots
            issue_start = len(issues)
            b8_values: dict[str, bytes] = {}
            record_sizes: dict[str, int] = {}

            bit_counts = {
                "circuit_measurements": metadata.measurement_count,
                "circuit_sweep_bits": metadata.sweep_count,
                "circuit_detectors": metadata.detector_count,
            }
            for column, relative_name, count_name in _B8_MEMBERS:
                member = f"{folder}/{relative_name}"
                data = archive.read(member)
                bit_count = bit_counts[count_name]
                record_size = (
                    b8_record_bytes(bit_count) if bit_count else 0
                )
                expected_bytes = record_size * metadata.shots
                b8_values[column] = data
                record_sizes[column] = record_size

                if len(data) != expected_bytes:
                    issues.add(
                        "google.b8_length",
                        (
                            "Packed file length must equal shots times "
                            "bytes per record"
                        ),
                        member=member,
                        value={
                            "actual": len(data),
                            "expected": expected_bytes,
                        },
                        record={"experiment_id": folder},
                    )
                    continue

                padding_violations = _padding_violation_count(
                    data,
                    bits_per_record=bit_count,
                    record_count=metadata.shots,
                )
                if padding_violations:
                    issues.add(
                        "google.b8_padding",
                        "Unused padding bits must be zero",
                        member=member,
                        value=padding_violations,
                        record={"experiment_id": folder},
                    )

            flip_values: dict[str, list[int]] = {}
            for column, relative_name in _FLIP_MEMBERS:
                member = f"{folder}/{relative_name}"
                raw = archive.read(member)
                try:
                    values = parse_01_records(raw)
                except ValueError as error:
                    issues.add(
                        "google.01_domain",
                        str(error),
                        member=member,
                        value=repr(raw[:80]),
                        record={"experiment_id": folder},
                    )
                    continue

                flip_values[column] = values
                if len(values) != metadata.shots:
                    issues.add(
                        "google.01_length",
                        "01 row count must equal the declared shots",
                        member=member,
                        value={
                            "actual": len(values),
                            "expected": metadata.shots,
                        },
                        record={"experiment_id": folder},
                    )

            if len(issues) != issue_start:
                rejected_count += metadata.shots
                continue

            experiment_source_id = _source_record_id(
                validated.sha256,
                folder,
                "properties.yml",
            )
            experiment_rows.append(
                {
                    "source_record_id": experiment_source_id,
                    "experiment_id": folder,
                    "basis": metadata.basis,
                    "distance": metadata.distance,
                    "rounds": metadata.rounds,
                    "shots": metadata.shots,
                    "center_row": metadata.center_row,
                    "center_col": metadata.center_col,
                    "measurement_count": metadata.measurement_count,
                    "detector_count": metadata.detector_count,
                }
            )

            detector_records = iter_b8_records(
                b8_values["detector_bits"],
                bits_per_record=metadata.detector_count,
            )
            folder_shot_ids: list[str] = []
            for shot_index in range(metadata.shots):
                measurement_size = record_sizes["measurement_bits"]
                sweep_size = record_sizes["sweep_bits"]
                detector_size = record_sizes["detector_bits"]
                measurement_start = shot_index * measurement_size
                sweep_start = shot_index * sweep_size
                detector_start = shot_index * detector_size
                detector_values = next(detector_records)
                shot_source_id = _source_record_id(
                    validated.sha256,
                    folder,
                    f"shot={shot_index}",
                )
                folder_shot_ids.append(shot_source_id)

                shot_rows.append(
                    {
                        "source_record_id": shot_source_id,
                        "experiment_id": folder,
                        "shot_index": shot_index,
                        "measurement_bits": b8_values["measurement_bits"][
                            measurement_start : measurement_start
                            + measurement_size
                        ],
                        "sweep_bits": b8_values["sweep_bits"][
                            sweep_start : sweep_start + sweep_size
                        ],
                        "detector_bits": b8_values["detector_bits"][
                            detector_start : detector_start + detector_size
                        ],
                        "detector_event_count": sum(detector_values),
                        **{
                            column: bool(values[shot_index])
                            for column, values in flip_values.items()
                        },
                    }
                )

            trace_tables.append(
                _trace_table(
                    input_sha256=validated.sha256,
                    folder=folder,
                    experiment_source_id=experiment_source_id,
                    shot_source_ids=folder_shot_ids,
                    record_sizes=record_sizes,
                )
            )

    accepted_count = len(shot_rows)
    if accepted_count + rejected_count != input_count:
        raise RuntimeError(
            "Internal reconciliation failure: accepted + rejected != input"
        )

    source_ids = [
        row["source_record_id"]
        for row in [*experiment_rows, *shot_rows]
    ]
    if len(source_ids) != len(set(source_ids)):
        raise RuntimeError("Duplicate Google source_record_id")

    source_trace = (
        pa.concat_tables(trace_tables)
        if trace_tables
        else pa.Table.from_pylist([], schema=SOURCE_TRACE_SCHEMA)
    )
    expected_trace_rows = len(experiment_rows) + (
        len(shot_rows) * SHOT_TRACE_MEMBER_COUNT
    )
    if source_trace.num_rows != expected_trace_rows:
        raise RuntimeError("Google source-trace row reconciliation failed")

    return GoogleQecTables(
        experiments=pa.Table.from_pylist(
            experiment_rows,
            schema=EXPERIMENT_SCHEMA,
        ),
        shots=pa.Table.from_pylist(shot_rows, schema=SHOT_SCHEMA),
        source_trace=source_trace,
        data_issues=issues.table(),
        input_sha256=validated.sha256,
        experiment_count=len(folders),
        input_count=input_count,
        accepted_count=accepted_count,
        rejected_count=rejected_count,
    )


__all__ = [
    "BRONZE_OBJECT",
    "EXPERIMENT_PATH",
    "EXPERIMENT_SCHEMA",
    "GoogleQecArchiveError",
    "GoogleQecTables",
    "SHOT_PATH",
    "SHOT_SCHEMA",
    "SHOT_TRACE_MEMBER_COUNT",
    "SOURCE_NAME",
    "build_google_qec_tables",
]
