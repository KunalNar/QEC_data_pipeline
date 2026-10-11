SELECT
    e.distance,
    CASE WHEN e.distance = 3 THEN e.center_row END AS center_row,
    CASE WHEN e.distance = 3 THEN e.center_col END AS center_col,
    p.decoder_name,
    count(*) AS shots,
    sum((p.predicted_flip <> s.actual_observable_flip)::integer)
        AS decoder_errors,
    avg((p.predicted_flip <> s.actual_observable_flip)::integer::double precision)
        AS logical_error_rate
FROM gold.google_shot AS s
JOIN gold.google_experiment AS e USING (experiment_id)
JOIN gold.google_decoder_prediction AS p
    ON p.source_record_id = s.source_record_id
GROUP BY e.distance,
    CASE WHEN e.distance = 3 THEN e.center_row END,
    CASE WHEN e.distance = 3 THEN e.center_col END,
    p.decoder_name
ORDER BY e.distance, center_row NULLS LAST, center_col NULLS LAST,
    p.decoder_name;
