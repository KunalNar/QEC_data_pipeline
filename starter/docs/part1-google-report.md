# Part I checkpoint: Google QEC

This report covers the Google hardware source path: Bronze verification, Silver
tables, Gold relations, and the Google ML export.
Run the full pipeline with `make run` in `starter/`. A rerun rebuilds the Google
Silver tables and replaces the Gold relations inside one PostgreSQL transaction,
so it does not append duplicate business records.

## Source discovery and quality

- The Google ZIP contains five X-basis hardware experiments with 25 rounds and
  50,000 shots each: four distance-three experiments centered at (3, 5),
  (5, 3), (5, 7) and (7, 5), and one distance-five experiment centered at
  (5, 5). That is 250,000 shots in total.
- Each experiment directory holds `properties.yml`, three packed Stim `b8` files
  (`measurements.b8`, `sweep.b8`, `detection_events.b8`), and five Stim `01`
  files: the actual observable flip and the predictions of four supplied
  decoders (belief matching, correlated matching, PyMatching and tensor-network
  contraction). One shot is one aligned record across all eight companion files.
  Measurements, detector events, actual flips, predictions and decoder errors
  are kept as separate concepts.

- Records are byte-aligned with little-endian bit order. A distance-three shot
  needs 27 bytes of measurements (209 bits), 2 bytes of sweep bits (9 bits) and
  25 bytes of detector events (200 bits). A distance-five shot has 600 detector
  bits and needs 75 bytes. Unused padding bits are not data.
- The Bronze archive is checked against the manifest SHA-256, and its member
  names are checked for unsafe paths. A missing companion file stops the run.
  For every experiment the `b8` file lengths equal shots times bytes per record,
  padding bits are zero, and every `01` file contains only 0 and 1 with exactly
  one line per shot. All five experiments passed, so 250,000 shots entered
  Silver and no source rows were rejected in this release. Rejected data would
  appear in `data_issues.parquet` with its rule, severity, action and reason.
- Silver rows carry stable `source_record_id` values built from the experiment
  name and shot index, so they do not depend on run time or row order. Each
  shot maps to eight Bronze companion members in `source_trace.parquet`.

The discovery notebook is
[google_qec_data_discovery.ipynb](../notebooks/google_qec_data_discovery.ipynb).
Its saved outputs are cleared.

## Gold design and ML handoff

The [Gold schema design](gold-schema.md) defines each row grain, keys,
constraints and indexes. The Google model has three relations:

- `distance_table`: one code distance with its fixed measurement and detector
  counts (2 rows). These counts depend only on distance, so they are stored
  once instead of repeating on every experiment.
- `google_experiments`: one hardware experiment directory (5 rows), with a
  foreign key to its distance.
- `google_shots`: one aligned hardware shot (250,000 rows), with a foreign key to
  its experiment, a unique experiment and shot index, packed measurement, sweep
  and detector bytes, the detector event count, the actual flip and the four
  decoder predictions.

All distances share one shot table because detector bits are stored as one
packed `bytea` per shot. Distance three and distance five therefore only need to
be separated later, when Part II builds fixed-width feature matrices.


The Google ML table comes only from the committed Gold view
`ml_google_decoder_view`. It has one example per shot (250,000 rows), with the
shot's `source_record_id` as `example_id`, the original packed detector bytes,
the event count, all four predictions and the actual flip. The supplied
`google_data_split` helper assigns the shot-index split in the export step. The
table is checked against the prescribed columns and types before publication.

Shared QEC vocabulary such as distance is used consistently, but no Google table
references a syndrome or QASMBench table. The relationship rejected in the
syndrome and QASMBench checkpoint applies here too: a QASMBench circuit is a
static definition, while a Google experiment is a real hardware run, and no
supplied identifier links them. Matching on distance alone would only pair
numbers that happen to agree.
