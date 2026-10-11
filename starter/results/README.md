# Generated results

Your pipeline creates these files; do not fill them in manually:

```text
results/
  part1/
    data_issues.parquet
    run.json
    row_counts.json
    source_trace.parquet
    trace_examples.json
    analysis/
  part2/
    predictions.parquet
    metrics.json
    run.json
    report.md
```

The exact fields and tracing rules are in the assignment brief. This branch
tracks the required Part I result files and analysis CSVs as submission
snapshots. The six Silver and two ML table snapshots are under
[`../lake/`](../lake/README.md).

`make run` regenerates Part I evidence and object-store tables. After training,
`make trace-examples` completes the prediction traces and updates their hash in
`run.json`. Refresh the table snapshots from those same object-store outputs
before committing a new run; their SHA-256 values must match `run.json`.

The trace command selects one saved test prediction per dataset and checks the
recorded ML hashes, labels, splits, and Bronze hashes before replacing the
examples. It does not train models. Repeat it after regenerating Part I or the
predictions, because `make run` initially writes ML-to-Bronze examples without
model predictions.

Part II outputs, older scoped checkpoints, temporary files, course archives,
credentials, and database volumes remain outside this commit.
