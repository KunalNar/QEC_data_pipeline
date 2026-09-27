"""Relational integration stage.

Create the student-designed PostgreSQL tables and load them as one all-or-
nothing update. Primary/foreign keys, value checks, and indexes are part of the
deliverable. Repeated loads must not create duplicate records.
"""

from quantum_lake_student.models import StageResult
from quantum_lake_student.connections import postgres_connection
from quantum_lake_student.config import Settings
import pandas as pd

google_sql_tables = """DROP TABLE IF EXISTS "distance_table" CASCADE;
CREATE TABLE "public"."distance_table" (
    "distance" integer NOT NULL,
    "measurement_count" integer NOT NULL,
    "detector_count" integer NOT NULL,
    CONSTRAINT "distance table_distance" PRIMARY KEY ("distance")
) WITH (oids = false);

COMMENT ON TABLE "public"."distance_table" IS 'One row represents a distance we have on our measurements, and the corresponding amount of measurements and detectors';


DROP TABLE IF EXISTS "google_experiments" CASCADE;
CREATE TABLE "public"."google_experiments" (
    "source_record_id" text NOT NULL,
    "experiment_id" text NOT NULL,
    "basis" character(1) NOT NULL,
    "distance" integer NOT NULL,
    "rounds" integer NOT NULL,
    "shots" integer NOT NULL,
    "center_row" integer NOT NULL,
    "center_col" integer NOT NULL,
    CONSTRAINT "google experiments_experiment_id" PRIMARY KEY ("experiment_id")
) WITH (oids = false);

COMMENT ON TABLE "public"."google_experiments" IS 'One row represents one hardware experiment directory';


DROP TABLE IF EXISTS "google_shots" CASCADE;
CREATE TABLE "public"."google_shots" (
    "source_record_id" text NOT NULL,
    "experiment_id" text NOT NULL,
    "shot_index" integer NOT NULL,
    "measurement_bits" bytea NOT NULL,
    "sweep_bits" bytea NOT NULL,
    "detector_bits" bytea NOT NULL,
    "detector_event_count" integer NOT NULL,
    "actual_observable_flip" boolean NOT NULL,
    "belief_matching_prediction" boolean NOT NULL,
    "correlated_matching_prediction" boolean NOT NULL,
    "pymatching_prediction" boolean NOT NULL,
    "tensor_network_contraction_prediction" boolean NOT NULL,
    CONSTRAINT "google shots_source_record_id" PRIMARY KEY ("source_record_id")
) WITH (oids = false);

COMMENT ON TABLE "public"."google_shots" IS 'One row represents one aligned hardware shot';

ALTER TABLE ONLY "public"."google_experiments" ADD CONSTRAINT "google experiments_distance_fkey" FOREIGN KEY (distance) REFERENCES distance_table(distance) NOT DEFERRABLE;

ALTER TABLE ONLY "public"."google_shots" ADD CONSTRAINT "google shots_experiment_id_fkey" FOREIGN KEY (experiment_id) REFERENCES google_experiments(experiment_id) NOT DEFERRABLE;"""

def run(run_id: str) -> StageResult:
    settings = Settings.from_environment()
    experiment_frame = pd.read_parquet("silver/google_qec/experiment.parquet")
    shot_frame = pd.read_parquet("silver/google_qec/shot.parquet")
    
    # distance table one row per distance distance value, so in our case two rows, d = 3 and d = 5
    distance_frame = experiment_frame[["distance", "measurement_count", "detector_count"]].drop_duplicates()
    
    with postgres_connection(settings) as connection:
        # connection.transaction is doing the all or nothing loading as mentioned in some md file in here
        with connection.transaction():
            connection.execute(google_sql_tables)
            for _, row in distance_frame.iterrows():
                connection.execute("""INSERT INTO distance_table (distance, measurement_count, detector_count)
                                      VALUES (%s, %s, %s)""", (row["distance"], row["measurement_count"], row["detector_count"])
                                   )
            
            for _, row in experiment_frame.iterrows():
                connection.execute("""INSERT INTO google_experiments (source_record_id, experiment_id, basis, distance, rounds, shots, center_row, center_col)
                                      VALUES(%s, %s, %s, %s, %s, %s, %s, %s)""", (row["source_record_id"], row["experiment_id"], row["basis"], 
                                                                           row["distance"], row["rounds"], row["shots"], row["center_row"], row["center_col"])
                    
                )
            
            for _, row in shot_frame.iterrows():
                connection.execute("""INSERT INTO google_shots (source_record_id, experiment_id, shot_index, measurement_bits, sweep_bits, detector_bits, detector_event_count, 
                                   actual_observable_flip, belief_matching_prediction, correlated_matching_prediction, pymatching_prediction, tensor_network_contraction_prediction) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                                   (row["source_record_id"], row["experiment_id"], row["shot_index"], row["measurement_bits"], row["sweep_bits"], row["detector_bits"], row["detector_event_count"], row["actual_observable_flip"], 
                                    row["belief_matching_prediction"], row["correlated_matching_prediction"], row["pymatching_prediction"], row["tensor_network_contraction_prediction"])
                                )
            
            result = StageResult(stage="load_postgres_google", run_id=run_id)
            result.output_count = len(experiment_frame) + len(shot_frame) + len(distance_frame)
            result.finish()
            return result