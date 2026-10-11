# Part II: AI/ML consumer of the QEC pipeline

Run `make train` from `starter/` in the course workspace. It reads only the
two ML input tables, `ml/ml_syndrome_decoder_example.parquet` and
`ml/ml_google_decoder_example.parquet`, never Gold, Silver, or the source
archives. One run writes the fitted model files `predictions.parquet`,
`metrics.json`, `run.json` and this report to `results/part2/`. The tables
below are generated from `metrics.json` in the same run.

All tasks use the supplied splits. Models are fitted on the training split,
settings and thresholds are chosen on the validation split and the test split
is used once for the reported metrics. Within a task, every model is compared
on identical test examples.

**Reading the tables.** All values are on the test split. The logical-error
rate is the share of test examples whose flip is predicted wrongly. Training
time covers only fitting on the training split and prediction time only
predicting the test split, both in seconds and without data loading or
feature preparation. `n/a` means not applicable.
