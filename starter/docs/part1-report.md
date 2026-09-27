# Part I: three-source data pipeline

Run `make run` from `starter/` in the course workspace. It validates all
three Bronze archives, rebuilds six Silver tables, loads all Gold relations
in one PostgreSQL transaction, exports both required ML Parquet tables, and
writes the run evidence under `results/part1/`. `make test` runs the
published tests. A second run replaces the same business records under the
same stable IDs.

The pipeline follows the course Bronze → Silver → Gold → ML layout:
unchanged source archives become checked, source-specific Parquet tables;
PostgreSQL connects the useful entities; committed Gold views supply the
two fixed ML tables. Run evidence is kept separately in `results/part1/`.

## Sources and quality

- The seven syndrome CSVs contain 75,598 aggregate rows with 70,000,000
  weighted observations. A row's `quantity` is a weight, not another row.
  The source header says `labels`; each syndrome is four rounds of four
  binary values. Both labels may occur for the same pattern.
- The Google archive contains five hardware experiments: four distance-three
  processor locations and one distance-five location. Each has 50,000
  aligned shots. Packed measurements, sweeps, and detector events are kept
  distinct from actual flips and the four decoder predictions.
- The QASMBench archive supplies six circuit variants, four explicit
  stabilizer checks, and six conditional corrections. No supplied identifier
  links a circuit row to a particular syndrome or Google experiment; that
  possible row-level join was rejected.

The current release produced no rejected rows or data issues. Bronze size,
SHA-256, safe paths, and archive integrity are checked before Silver.
The discovery notebooks are
[syndromes](../notebooks/syndrome_data_discovery.ipynb),
[Google](../notebooks/google_qec_data_discovery.ipynb), and
[QASMBench](../notebooks/qasmbench_data_discovery.ipynb). Their committed
outputs remain cleared.

## Gold and ML

The [Gold design](gold-schema.md) documents every relation, key, constraint,
index, and trace path. The syndrome model has seven experiments, 31,941
patterns, and 75,598 source observations. QASMBench has separate circuits,
registers, checks, ordered participants, and corrections. The Google model
has five experiments, 250,000 shots with packed detector bytes and event
summaries, four decoder definitions, and 1,000,000 shot/decoder predictions.
This keeps actual outcomes separate from predictions without expanding
70 million detector bits into individual database rows.

Both ML exports come only from committed Gold views. The syndrome table has
75,598 examples and preserves the 70,000,000 total sample weight. The Google
table has one example per hardware shot, all four decoder predictions, the
original packed detector bytes, and the course-supplied shot-index split.
Both exports validate the prescribed Arrow columns and types before
publication. Source-link views resolve their `example_id` values to Gold.

## Analyses and trace

The committed SQL answers the three required questions:

1. [Syndrome fault-rate analysis](../src/quantum_lake_student/sql/analysis_syndrome_fault_rate.sql)
   gives weighted frequencies and logical-error fractions. The supplemental
   [pattern query](../src/quantum_lake_student/sql/analysis_syndrome_patterns.sql)
   joins three Gold tables. Each fault-rate file represents 10 million
   shots. In this release, the weighted logical-error fraction rises from
   0.000234 at fault rate 0.00001 to 0.1865274 at 0.01; this describes the
   supplied simulations, not a causal effect.
2. [Google decoder analysis](../src/quantum_lake_student/sql/analysis_google_decoders.sql)
   gives decoder logical-error rates by distance and distance-three processor
   location. In this release its 20 groups cover four decoders at each of
   five experiment groups. For example, at distance three the
   tensor-network decoder's observed rate ranges from 0.38392 at center
   `(5, 7)` to 0.41304 at `(5, 3)`. Rates vary by decoder and location;
   these are associations, not causal estimates.
3. [QASMBench analysis](../src/quantum_lake_student/sql/analysis_qasmbench.sql)
   links repetition-code circuits, measured checks, data-qubit participants,
   syndrome bits, and conditional corrections. Both source and transpiled
   variants show `a[0]` checking `q[0]`/`q[1]` into `syn[0]` and `a[1]`
   checking `q[1]`/`q[2]` into `syn[1]`; recovery is conditioned on the
   measured syndrome value.

`results/part1/source_trace.parquet` contains one or more Bronze pointers per
Silver source ID. Each Google shot has eight companion-file trace rows.
`trace_examples.json` follows one syndrome ML example with weight 7 to its
source row in `d-3_pfr-0.010000_nb-10M.csv`. It also follows shot zero in
`surface_code_bX_d3_r25_center_3_5` through its Gold shot to eight aligned
Bronze companion members. Part II predictions do not exist yet, so their
final end-to-end trace remains pending. `run.json`, `row_counts.json`,
`data_issues.parquet`, and the analysis CSVs record the reproducible Part I
evidence.

The local course workspace passed two complete `make run` executions and
`make test` (183 tests). Reproduction from a fresh environment has not yet
been recorded; the final prediction-level traces depend on Part II output.
