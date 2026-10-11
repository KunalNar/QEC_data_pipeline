# Part I table snapshots

This directory contains the six required Silver Parquet tables under `silver/`
and the two required ML Parquet tables under `ml/`. These are exact copies of
the pipeline's object-store outputs, committed as submission deliverables.
They are not additional source data and do not replace PostgreSQL Gold.

Paths relative to this directory are the corresponding object-store keys.
`../results/part1/run.json` records each table's row count and SHA-256; the
copies here must match those values. The same snapshot includes Part I issues,
counts, analyses, source traces, and complete prediction examples.

`make run` from the course workspace reproduces the tables in the configured
lake. It does not automatically refresh these local MinIO-output copies.
After regenerating a submission, copy the new Silver/ML outputs here and
verify their hashes before committing them together with the run evidence.
Raw archives, credentials, and model files are not included.
