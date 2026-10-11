# Complete prediction traces

After Part I and Part II have run, execute `make trace-examples` from the
workspace directory. It uses `results/part2/predictions.parquet` and `run.json`
to complete `results/part1/trace_examples.json`, then updates that artifact's
hash in the Part I run record.

The command chooses one test prediction per dataset in stable model/example
order. Task A supplies syndrome predictions; Task C or Task B can supply
Google predictions. Task B is not needed when Task C results are available.
Predictions retain their model ID, label, predicted value, probability, and split.
Gold source-link views resolve the selected examples to their Silver source IDs
and Bronze object/member/record/hash pointers. QASMBench evidence is retained.

Both recorded ML input hashes must match the current Part I exports. Labels
and splits must match the selected examples, and Bronze hashes must still
match. Missing results, duplicate predictions, or inconsistent evidence stop
the command before replacing the trace file. Rerunning unchanged inputs
produces the same JSON. No models are trained and no source data is changed.

Run this command again after regenerating Part I or the predictions; a fresh
`make run` writes the initial ML-to-Bronze examples without model predictions.
