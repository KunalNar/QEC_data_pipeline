-- Weighted logical-error frequency by physical fault rate.
SELECT
    e.physical_fault_rate,
    count(DISTINCT e.experiment_id) AS experiment_count,
    count(*) AS aggregate_rows,
    sum(o.quantity) AS weighted_observations,
    coalesce(sum(o.quantity) FILTER (WHERE o.logical_error_label), 0)
        AS logical_error_weight,
    coalesce(sum(o.quantity) FILTER (WHERE NOT o.logical_error_label), 0)
        AS no_error_weight,
    round(
        coalesce(sum(o.quantity) FILTER (WHERE o.logical_error_label), 0)::numeric
        / sum(o.quantity),
        8
    ) AS weighted_logical_error_rate
FROM gold.syndrome_experiment AS e
JOIN gold.syndrome_observation AS o USING (experiment_id)
GROUP BY e.physical_fault_rate
ORDER BY e.physical_fault_rate;
