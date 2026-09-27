CREATE SCHEMA IF NOT EXISTS gold;

CREATE TABLE IF NOT EXISTS gold.google_experiment (
    experiment_id text PRIMARY KEY,
    source_record_id text NOT NULL UNIQUE,
    basis text NOT NULL CHECK (basis <> ''),
    distance integer NOT NULL CHECK (distance > 0),
    rounds integer NOT NULL CHECK (rounds > 0),
    shots bigint NOT NULL CHECK (shots > 0),
    center_row integer NOT NULL CHECK (center_row >= 0),
    center_col integer NOT NULL CHECK (center_col >= 0),
    measurement_count integer NOT NULL CHECK (measurement_count > 0),
    detector_count integer NOT NULL CHECK (detector_count > 0)
);

CREATE TABLE IF NOT EXISTS gold.google_shot (
    source_record_id text PRIMARY KEY,
    experiment_id text NOT NULL
        REFERENCES gold.google_experiment (experiment_id),
    shot_index bigint NOT NULL CHECK (shot_index >= 0),
    measurement_bits bytea NOT NULL,
    sweep_bits bytea NOT NULL,
    detector_bits bytea NOT NULL,
    detector_event_count integer NOT NULL CHECK (detector_event_count >= 0),
    actual_observable_flip boolean NOT NULL,
    UNIQUE (experiment_id, shot_index)
);

CREATE TABLE IF NOT EXISTS gold.google_decoder (
    decoder_name text PRIMARY KEY,
    CHECK (decoder_name IN (
        'belief_matching', 'correlated_matching', 'pymatching',
        'tensor_network_contraction'
    ))
);

CREATE TABLE IF NOT EXISTS gold.google_decoder_prediction (
    source_record_id text NOT NULL
        REFERENCES gold.google_shot (source_record_id),
    decoder_name text NOT NULL
        REFERENCES gold.google_decoder (decoder_name),
    predicted_flip boolean NOT NULL,
    PRIMARY KEY (source_record_id, decoder_name)
);

CREATE INDEX IF NOT EXISTS google_experiment_distance_location_idx
    ON gold.google_experiment (distance, center_row, center_col);

CREATE INDEX IF NOT EXISTS google_shot_experiment_idx
    ON gold.google_shot (experiment_id);

CREATE INDEX IF NOT EXISTS google_prediction_decoder_idx
    ON gold.google_decoder_prediction (decoder_name, source_record_id);

CREATE OR REPLACE VIEW gold.google_ml_example AS
SELECT
    s.source_record_id AS example_id,
    s.experiment_id,
    s.shot_index,
    e.distance,
    e.rounds,
    e.center_row,
    e.center_col,
    e.detector_count,
    s.detector_event_count,
    s.detector_bits,
    belief.predicted_flip AS belief_matching_prediction,
    correlated.predicted_flip AS correlated_matching_prediction,
    pymatching.predicted_flip AS pymatching_prediction,
    tensor.predicted_flip AS tensor_network_contraction_prediction,
    s.actual_observable_flip
FROM gold.google_shot AS s
JOIN gold.google_experiment AS e USING (experiment_id)
JOIN gold.google_decoder_prediction AS belief
    ON belief.source_record_id = s.source_record_id
    AND belief.decoder_name = 'belief_matching'
JOIN gold.google_decoder_prediction AS correlated
    ON correlated.source_record_id = s.source_record_id
    AND correlated.decoder_name = 'correlated_matching'
JOIN gold.google_decoder_prediction AS pymatching
    ON pymatching.source_record_id = s.source_record_id
    AND pymatching.decoder_name = 'pymatching'
JOIN gold.google_decoder_prediction AS tensor
    ON tensor.source_record_id = s.source_record_id
    AND tensor.decoder_name = 'tensor_network_contraction';

CREATE OR REPLACE VIEW gold.google_ml_source AS
SELECT source_record_id AS example_id, source_record_id
FROM gold.google_shot;
