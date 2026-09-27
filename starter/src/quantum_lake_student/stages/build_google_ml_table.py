"""Create the required analyses and ML input tables from PostgreSQL Gold.

Write the Gold queries/views, joins, stable example IDs, and export logic.
A thin export step may execute SQL, use the course split helpers, validate the
fixed contracts, and write Parquet. It must not re-read Bronze or Silver.
"""

from quantum_lake_student.models import StageResult
from quantum_lake_student.config import Settings
from quantum_lake_student.connections import postgres_connection
from quantum_lake_student.ml import google_data_split
import pandas as pd
from pathlib import Path

google_sql_view = """DROP VIEW IF EXISTS "ml_google_decoder_view" CASCADE;
CREATE VIEW "ml_google_decoder_view" AS SELECT s.source_record_id AS example_id,
    s.experiment_id,
    s.shot_index,
    e.distance,
    e.rounds,
    e.center_row,
    e.center_col,
    d.detector_count,
    s.detector_event_count,
    s.belief_matching_prediction,
    s.correlated_matching_prediction,
    s.pymatching_prediction,
    s.tensor_network_contraction_prediction,
    s.actual_observable_flip
   FROM ((google_shots s
     JOIN google_experiments e ON ((s.experiment_id = e.experiment_id)))
     JOIN distance_table d ON ((e.distance = d.distance)));
"""

def run(run_id: str) -> StageResult:
    settings = Settings.from_environment()
    with postgres_connection(settings) as connection:
        connection.execute(google_sql_view)
        connection.commit()
        rows = pd.read_sql("Select * FROM ml_google_decoder_view", connection)
        rows["data_split"] = rows["shot_index"].apply(google_data_split)

        output_path = Path("ml/ml_google_decoder_example.parquet")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows.to_parquet(output_path, index=False)
        rows.to_parquet("ml/ml_google_decoder_example.parquet", index=False)
        result = StageResult(stage="build_ml_tables_google", run_id=run_id)
        result.output_count = len(rows)
        result.finish()
        return result