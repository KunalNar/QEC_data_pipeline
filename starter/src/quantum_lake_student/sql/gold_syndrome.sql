CREATE SCHEMA IF NOT EXISTS gold;

CREATE TABLE IF NOT EXISTS gold.syndrome_experiment (
    experiment_id text PRIMARY KEY,
    physical_fault_rate double precision NOT NULL,
    CONSTRAINT syndrome_fault_rate_range
        CHECK (physical_fault_rate > 0 AND physical_fault_rate < 1)
);

CREATE TABLE IF NOT EXISTS gold.syndrome_pattern (
    syndrome_bits bytea PRIMARY KEY,
    round_count smallint NOT NULL,
    check_count smallint NOT NULL,
    CONSTRAINT syndrome_round_count CHECK (round_count = 4),
    CONSTRAINT syndrome_check_count CHECK (check_count = 4),
    CONSTRAINT syndrome_shape
        CHECK (octet_length(syndrome_bits) = round_count * check_count),
    CONSTRAINT syndrome_binary_bytes
        CHECK (encode(syndrome_bits, 'hex') ~ '^(00|01){16}$')
);

CREATE TABLE IF NOT EXISTS gold.syndrome_observation (
    source_record_id text PRIMARY KEY,
    experiment_id text NOT NULL
        REFERENCES gold.syndrome_experiment (experiment_id),
    syndrome_bits bytea NOT NULL
        REFERENCES gold.syndrome_pattern (syndrome_bits),
    logical_error_label boolean NOT NULL,
    quantity bigint NOT NULL,
    CONSTRAINT syndrome_positive_quantity CHECK (quantity > 0)
);

CREATE INDEX IF NOT EXISTS syndrome_observation_experiment_pattern_label_idx
    ON gold.syndrome_observation
    (experiment_id, syndrome_bits, logical_error_label);

CREATE INDEX IF NOT EXISTS syndrome_observation_pattern_idx
    ON gold.syndrome_observation (syndrome_bits);

CREATE OR REPLACE VIEW gold.syndrome_ml_example AS
SELECT
    'ml-syn-' || encode(
        sha256(
            convert_to(o.experiment_id, 'UTF8')
            || decode('00', 'hex')
            || p.syndrome_bits
            || CASE WHEN o.logical_error_label
                THEN decode('01', 'hex')
                ELSE decode('00', 'hex')
            END
        ),
        'hex'
    ) AS example_id,
    o.experiment_id,
    e.physical_fault_rate,
    p.syndrome_bits,
    p.round_count::integer AS round_count,
    p.check_count::integer AS check_count,
    o.logical_error_label,
    sum(o.quantity)::bigint AS sample_weight
FROM gold.syndrome_observation AS o
JOIN gold.syndrome_experiment AS e USING (experiment_id)
JOIN gold.syndrome_pattern AS p USING (syndrome_bits)
GROUP BY
    o.experiment_id,
    e.physical_fault_rate,
    p.syndrome_bits,
    p.round_count,
    p.check_count,
    o.logical_error_label;

CREATE OR REPLACE VIEW gold.syndrome_ml_source AS
SELECT m.example_id, o.source_record_id
FROM gold.syndrome_ml_example AS m
JOIN gold.syndrome_observation AS o
    ON o.experiment_id = m.experiment_id
    AND o.syndrome_bits = m.syndrome_bits
    AND o.logical_error_label = m.logical_error_label;
