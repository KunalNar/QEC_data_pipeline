-- Five most frequent syndrome patterns at each physical fault rate.
WITH pattern_frequency AS (
    SELECT
        e.physical_fault_rate,
        encode(p.syndrome_bits, 'hex') AS syndrome_hex,
        p.round_count,
        p.check_count,
        count(*) AS aggregate_rows,
        sum(o.quantity) AS weighted_frequency,
        coalesce(
            sum(o.quantity) FILTER (WHERE o.logical_error_label), 0
        ) AS logical_error_weight
    FROM gold.syndrome_experiment AS e
    JOIN gold.syndrome_observation AS o USING (experiment_id)
    JOIN gold.syndrome_pattern AS p USING (syndrome_bits)
    GROUP BY e.physical_fault_rate, p.syndrome_bits,
        p.round_count, p.check_count
), ranked AS (
    SELECT
        *,
        row_number() OVER (
            PARTITION BY physical_fault_rate
            ORDER BY weighted_frequency DESC, syndrome_hex
        ) AS frequency_rank
    FROM pattern_frequency
)
SELECT
    physical_fault_rate,
    frequency_rank,
    syndrome_hex,
    round_count,
    check_count,
    aggregate_rows,
    weighted_frequency,
    logical_error_weight
FROM ranked
WHERE frequency_rank <= 5
ORDER BY physical_fault_rate, frequency_rank;
