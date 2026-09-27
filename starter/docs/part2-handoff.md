# Part I handoff for Part II

## Available data

- **Syndrome ML input:** `ml/ml_syndrome_decoder_example.parquet` in the object
  store. It has 75,598 examples representing 70 million shots. Each row is one
  experiment–syndrome–label combination. `syndrome_bits` contains 16 bytes,
  each `0` or `1`, ordered by round then check. `logical_error_label` is the
  target; `sample_weight` is the number of shots the row represents.
  The same syndrome can validly occur with both labels.
  `data_split` is already assigned: fault rate `0.0005` is validation,
  `0.005` is test, and the other five rates are training.
- **Google ML input:** `ml/ml_google_decoder_example.parquet` in the object
  store. It has 250,000 examples, one per hardware shot. `detector_bits`
  contains packed Stim `b8` bytes; `detector_count` gives the valid bit width
  (200 for distance three, 600 for distance five). The four
  `*_prediction` columns are supplied decoder outputs, while
  `actual_observable_flip` is the target. `data_split` is already assigned
  within each experiment: odd shot indices are test, indices ending in 8
  are validation, and other even indices are train.
- **QASMBench:** Circuit, parity-check, and correction data is available in
  PostgreSQL Gold. It has no assignment ML input table and no row-level link
  to either experiment dataset.

## Trace and references

The stable `example_id` identifies each ML row. PostgreSQL views
`gold.syndrome_ml_source` and `gold.google_ml_source` resolve examples to
Gold source rows. `results/part1/source_trace.parquet` then gives the Bronze
member and record position; a Google shot has eight companion-file trace
rows. The same result directory contains `run.json`, `row_counts.json`,
and `trace_examples.json`. In `starter/`, `make run` recreates the full
Part I handoff.

- [Exact ML table contracts](../../assignment/required-ml-tables.md)
- [Object-store access helper](../src/quantum_lake_student/connections.py)
- [Supplied model-input, split, and metric helpers](../src/quantum_lake_student/ml.py)
- [Gold table definitions](gold-schema.md)
- [Part I quality summary](part1-report.md)
