# Syndrome and QASMBench handoff for Part II

## Available data

- **Syndrome ML input:** `ml/ml_syndrome_decoder_example.parquet` in the object
  store. It has 75,598 examples representing 70 million shots. Each row is one
  experiment–syndrome–label combination. `syndrome_bits` contains 16 bytes,
  each `0` or `1`, ordered by round then check. `logical_error_label` is the
  target; `sample_weight` is the number of shots the row represents.
  The same syndrome can validly occur with both labels.
  `data_split` is already assigned: fault rate `0.0005` is validation,
  `0.005` is test, and the other five rates are training.
- **QASMBench:** Circuit, parity-check, and correction data is available in
  PostgreSQL Gold. It has no assignment ML input table and no row-level link
  to the syndrome experiments.

## Trace and references

The stable `example_id` identifies a syndrome ML row. PostgreSQL view
`gold.syndrome_ml_source` links it to source rows; the generated
`results/part1/syndrome_qasm/source_trace.parquet` links those rows to the
Bronze ZIP member and CSV position. The same result directory contains
`run.json`, `row_counts.json`, and `trace_examples.json`. In `starter/`,
`make run-syndrome-qasm` recreates these data products and evidence.

- [Exact ML table contracts](../../assignment/required-ml-tables.md)
- [Object-store access helper](../src/quantum_lake_student/connections.py)
- [Supplied model-input, split, and metric helpers](../src/quantum_lake_student/ml.py)
- [Gold table definitions](gold-schema.md)
- [Part I quality summary](part1-syndrome-qasm-report.md)
