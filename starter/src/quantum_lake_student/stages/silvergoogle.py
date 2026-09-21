import io
import zipfile
from pathlib import Path
import pandas as pd
import yaml
import json

from quantum_lake_student.config import Settings
from quantum_lake_student.connections import minio_client
from quantum_lake_student.formats import iter_b8_records, parse_01_records, b8_record_bytes
from quantum_lake_student.models import QualityFinding, Severity, StageResult, stable_record_hash

experiment_path = Path("silver/google_qec/experiment.parquet")
shot_path = Path("silver/google_qec/shot.parquet")
issues_path = Path("results/part1/data_issues_google_silver.json")

def run(run_id: str) -> StageResult:
    settings = Settings.from_environment()
    client = minio_client(settings)
    object_name = "bronze/source=google_qec/google-surface-code-curated.zip"

    # Starting with getting the zip, then opening it.
    response = client.get_object(settings.s3_bucket, object_name)
    zip_file = zipfile.ZipFile(io.BytesIO(response.read()))
    # its been read and in zip_file so now close it and the connection
    response.close()
    response.release_conn()
    # List of all file names in the zip
    all_names = zip_file.namelist()
    # {} makes it a set
    experiment_folders = {name.split("/")[0] for name in all_names if "/" in name}
    experiment_folders = sorted(experiment_folders)
    
    findings: list[QualityFinding] = []
    
    
    # -------------- This is google_experiment.parquet -----------
    # One row represents one hardware experiment directory (from silver-tables.md)
    # so 5 rows in total 
    experiment_rows = []
    for folder in experiment_folders:
        yaml_properties = zip_file.read(f"{folder}/properties.yml")
        properties = yaml.safe_load(yaml_properties)
        # I forgot dictionaries existed
        # This is a single row for table 2: Google experiments'
        source_record_id = stable_record_hash({"experiment":folder})
        experiment_row = {
            "source_record_id"  : source_record_id, 
            "experiment_id"     : folder,
            "basis"             : properties["basis"],
            "distance"          : properties["distance"],
            "rounds"            : properties["rounds"],
            "shots"             : properties["shots"],
            "center_row"        : properties["center_data_qubit_row"],
            "center_col"        : properties["center_data_qubit_col"],
            "measurement_count" : properties["circuit_measurements"],
            "detector_count"    : properties["circuit_detectors"],
        }
        experiment_rows.append(experiment_row)
        print(f"Built row for {folder}: {experiment_row}")
    
    print(f"Total amount of rows built: {len(experiment_rows)}")
    # this should be good to now make experiment.parquet

    experiment_frame = pd.DataFrame(experiment_rows)
    experiment_path.parent.mkdir(parents=True,exist_ok=True)
    experiment_frame.to_parquet(experiment_path, index=False, engine='pyarrow')
    # ok parquet now built and it makes the silver folder. google experiment done (except for all the safety stuff)

    # -------------- This is google_shots.parquet -----------
    # one row represents one aligned hardware shot. this is taken from silver-tables.md
    shot_rows = []
    for folder in experiment_folders:
        yaml_properties = zip_file.read(f"{folder}/properties.yml")
        properties = yaml.safe_load(yaml_properties)
        
        # bit counts needed to calculate each files expected byte length
        shots = properties["shots"]
        measurement_bits = properties["circuit_measurements"]
        sweep_bits = properties["circuit_sweep_bits"]
        detector_bits = properties["circuit_detectors"]
        total_shots_attempted = 0
        total_shots_attempted += shots
        # these are needed in bytes to be length checked to the properties.yml file 
        measurement_bytes = zip_file.read(f"{folder}/measurements.b8")
        sweep_bytes = zip_file.read(f"{folder}/sweep.b8")
        detection_event_bytes = zip_file.read(f"{folder}/detection_events.b8")
        actual_flips = parse_01_records(zip_file.read(f"{folder}/obs_flips_actual.01"))
        belief_predictions = parse_01_records(zip_file.read(f"{folder}/obs_flips_predicted_by_belief_matching.01"))
        correlated_predictions = parse_01_records(zip_file.read(f"{folder}/obs_flips_predicted_by_correlated_matching.01"))
        pymatching_predictions = parse_01_records(zip_file.read(f"{folder}/obs_flips_predicted_by_pymatching.01"))
        tensor_predictions = parse_01_records(zip_file.read(f"{folder}/obs_flips_predicted_by_tensor_network_contraction.01"))
        
        # this is the size in bytes for properties.yml to check with the ones right above
        measurement_size = b8_record_bytes(measurement_bits)
        sweep_size = b8_record_bytes(sweep_bits)
        detector_size = b8_record_bytes(detector_bits)
        
        # if expected length and the length of the actual file don't match, return an error.
        expected_measurement_length = measurement_size * shots
        if len(measurement_bytes) != expected_measurement_length:
            findings.append(
                QualityFinding(
                    rule_id="b8_length_mismatch",
                    severity=Severity.ERROR,
                    source_system="google_qec",
                    source_record_locator=f"{folder}/measurements.b8",
                    message="measurements.b8 length doesn't match shots * bytes per record",
                    observed_value=str(len(measurement_bytes))
                )
            )
            continue # skip this experiment since its data cannot be trusted, will go to data issues table later on
        
        # length of flips should match amnt of shots since the flips file is just a 0 or 1  written per shot.
        if len(actual_flips) != shots:
            findings.append(
                QualityFinding(
                    rule_id="01_observable_flips",
                    severity=Severity.ERROR,
                    source_system="google_qec",
                    source_record_locator=f"{folder}/obs_flips_actual.01",
                    message="obs_flips_actual.01 line count does not match amount of shots",
                    observed_value=str(len(actual_flips))
                )
            )
            continue # skip this experiment since its data cannot be trusted, will go to data issues table later on
        
        detector_bit_tuples = iter_b8_records(detection_event_bytes, bits_per_record=detector_bits)
        
        # now finally making the parquet
        for shot_index in range(shots):
            # to get one measurement
            measurement_start = shot_index * measurement_size
            measurement = measurement_bytes[measurement_start: measurement_start + measurement_size]
            
            # to get one sweep
            sweep_start = shot_index * sweep_size
            sweep = sweep_bytes[sweep_start: sweep_start + sweep_size]
            
            # to get one detector
            detector_start = shot_index * detector_size
            detector = detection_event_bytes[detector_start: detector_start + detector_size]
            
            detector_bit_tuple = next(detector_bit_tuples)
            detector_event_count = sum(detector_bit_tuple)
            
            # unique id for each shot
            source_record_id = stable_record_hash({"experiment": folder, "shot_index": shot_index})
            
            shot_row = {
                "source_record_id": source_record_id,
                "experiment_id": folder,
                "shot_index": shot_index,
                "measurement_bits": measurement,
                "sweep_bits": sweep,
                "detector_bits": detector,
                "detector_event_count": detector_event_count,
                "actual_observable_flip": bool(actual_flips[shot_index]),
                "belief_matching_prediction": bool(belief_predictions[shot_index]),
                "correlated_matching_prediction": bool(correlated_predictions[shot_index]),
                "pymatching_prediction": bool(pymatching_predictions[shot_index]),
                "tensor_network_contraction_prediction": bool(tensor_predictions[shot_index]),
            }
            shot_rows.append(shot_row)
            
    # now The parquet is built
    shot_frame = pd.DataFrame(shot_rows)
    shot_path.parent.mkdir(parents=True, exist_ok=True)
    shot_frame.to_parquet(shot_path, index=False, engine="pyarrow")

    # saving Issues from findings list
    issue_dicts = []
    for f in findings:
        issue_dicts.append({
            "rule_id": f.rule_id,
            "severity": f.severity.value,
            "source_system": f.source_system,
            "source_record_locator": f.source_record_locator,
            "message": f.message,
            "observed_value": f.observed_value,
        })    

    issues_path.parent.mkdir(parents=True, exist_ok=True)
    issues_path.write_text(json.dumps(issue_dicts, indent=2))
            
    result = StageResult(stage="prepare_google_data", run_id=run_id)
    result.input_count = total_shots_attempted # so like 250k
    result.output_count = len(experiment_frame) + len(shot_frame) # amount of rows we have
    result.issue_count = len(findings) # amount of issues we have
    result.finish()
    return result
