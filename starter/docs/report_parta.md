# Part I QEC data pipeline

## General Overview

Our Part I pipeline preserves the three supplied archives and turns their different formats into six validated Silver Parquet tables, a relational PostgreSQL Gold model, and two ML tables exported through Gold queries. All supplied records passed validation, while stable identifiers and source hashes connect the outputs to the original data.

## Discovery and architecture

The syndrome archive contains seven distance-three CSV files. Their filenames encode distance, physical fault rate, and nominal shot count rather than file size. The 75,598 rows represent 70 million shots, with 10 million per experiment. Each row contains four rounds of four binary syndrome values, a logical-error label, and a quantity recording how often that combination occurred. Consequently, quantity remains a weight rather than being expanded into individual shots.

Meanwhile, Google contains five 25-round experiments with 50,000 shots each. Four experiments have distance three and one has distance five. YAML metadata describes each experiment, while three packed `b8` files and five binary `01` files align measurements, sweep values, detector events, actual observable flips, and four decoder predictions by shot index. Actual flips are targets, whereas a decoder error means disagreement between a prediction and that target.

QASMBench contains source and transpiled OpenQASM variants across three benchmark directories, giving six circuits. Parsing identifies declared registers, executed operations, four explicit parity checks, and six conditional corrections. Gate definitions are not counted as executed operations, and register identity keeps `q[0]` distinct from `a[0]`.

Bronze retains the original ZIP bytes after hash and safe-path checks. Silver then publishes one syndrome observation table, Google experiment and shot tables, and QASMBench circuit, check, and correction tables. These remain source-specific. Gold introduces meaningful relational separation rather than copying each Parquet table into PostgreSQL, and its SQL views supply both ML exports without rereading Bronze or Silver.

## Gold design and important decisions

Syndrome Gold separates seven experiments, 31,941 reusable patterns, and 75,598 original observations. Experiment IDs identify source CSVs, the 16 syndrome bytes key each pattern, and `source_record_id` keys each observation. Foreign keys connect them, while constraints enforce four rounds, four values per round, binary bytes, and positive quantities. Indexes support experiment–pattern–label grouping and pattern lookup across experiments.

A pattern may legitimately occur under both labels. Moreover, although this release contains no repeated experiment–pattern–label combination, Gold allows separate source rows with that combination. ML alone groups them and sums their quantities, retaining links to every contributing observation. This preserves both scientific weights and source evidence.

Google Gold separates experiments, shots, four decoders, and one million decoder predictions. Each shot has a source-record primary key and a unique experiment–shot-index pair, while each prediction is keyed by shot and decoder. Foreign keys prevent orphan predictions, and indexes support distance, location, experiment, and decoder queries. Actual flips remain on shots rather than being mixed with predictions.

Detector bits remain packed alongside their event counts. The distance-three shots use 25 detector bytes each and the distance-five shots use 75, giving 8.75 MB of detector payload instead of 70 million individual detector-bit rows. This measurement excludes other fields and database overhead. Packed storage avoids unnecessary row expansion, although individual-bit features require unpacking.

QASMBench Gold separates circuits, registers, checks, ordered check participants, and corrections. Composite keys preserve register identity and participant order, while foreign keys connect the structures. Constraints and loader checks enforce register kinds, sizes, reference bounds, and valid conditional values. Check and correction indexes support parity-to-recovery queries, and source and transpiled variants remain distinct.

Together, these are 12 Gold tables. The combined load replaces them in one transaction and reconciles counts, source IDs, and syndrome weights against Silver. A failed load therefore leaves the previous Gold state intact, while unchanged reruns retain stable IDs without accumulating records. The executable definitions and index rationale are linked in the [schema design](gold-schema.md).

## Quality findings and the rejected relationship

Validation covers syndrome headers, shape, binary labels, positive integer weights, and weighted totals. Google checks companion completeness, shot alignment, packed lengths, little-endian order, padding, and binary text values. QASMBench checks syntax and register references, but structural consistency does not establish mathematical correctness.

No supplied records were rejected and [data issues](../results/part1/data_issues.parquet) contains zero rows. Data-quality failures are recorded with their source, rule, severity, observed value, action, and reason rather than silently corrected. Archive-integrity failures instead stop processing. For Google, invalid companion alignment excludes the affected experiment because individual shot correspondence cannot then be trusted.

We rejected a row-level QASMBench-to-experiment join. QASMBench identifies static benchmark variants, the syndrome files identify simulated fault-rate experiments, and Google identifies hardware experiments and shot indices. The supplied sources provide no mapping between those identifiers. Shared QEC terminology or distance therefore cannot establish that a circuit generated a particular experimental row.

## Analysis and verification

The [Gold SQL analyses](../results/part1/analysis/) answer all three required questions and include joins across at least three tables. Weighted syndrome logical-error rates increase from 0.0234% to 18.65274% as physical fault rate rises from 0.00001 to 0.01. Google's distance-three tensor-network decoder error rates range from 38.392% to 41.304% across locations, while distance five gives 39.55%. These are full-source summaries rather than held-out evaluation results, and they do not establish causal explanations.

For both repetition-code variants, circuit analysis connects parity checks on `q[0]`/`q[1]` and `q[1]`/`q[2]` to syndrome-controlled recovery. Register values 1, 2, and 3 select corrections on `q[0]`, `q[2]`, and `q[1]`, respectively.

All 198 Part I tests passed, and all three discovery notebooks executed successfully with saved outputs subsequently cleared. Fresh-environment reproduction, stable reruns, and a deliberate failed-load rollback were verified. The [run record](../results/part1/run.json) and [row counts](../results/part1/row_counts.json) retain hashes, revision information, timings, and reconciliation evidence separately from this report.

## Traces and delivered ML products

Every Silver row has a stable source ID mapped through [source trace](../results/part1/source_trace.parquet) to its Bronze object, archive member, record locator, and input hash. Gold preserves those links, and ML example IDs resolve to the relevant Gold records. Multiple trace pointers are expected when one Google shot combines eight companion records or one syndrome example groups several observations.

The [trace examples](../results/part1/trace_examples.json) demonstrate complete prediction-to-source routes. A syndrome prediction resolves through its ML example and Gold observation to data row 18,222 of `d-3_pfr-0.005000_nb-10M.csv`. A Google prediction resolves to zero-based shot 13,089 in `surface_code_bX_d3_r25_center_7_5`, including the eight companion records, their byte or line locators, and the archive hash. QASMBench evidence additionally identifies the original circuit member and check or correction lines.

The delivered [syndrome ML table](../lake/ml/ml_syndrome_decoder_example.parquet) contains 75,598 examples with syndrome bytes, logical-error labels, and sample weights. The [Google ML table](../lake/ml/ml_google_decoder_example.parquet) contains 250,000 examples with packed detector inputs, actual-flip targets, event summaries, and four supplied predictions. Both include experiment identifiers, stable example IDs, and the prescribed splits, forming the reproducible data interface used by Part II. QASMBench supplies circuit evidence rather than labelled training examples.

## AI assistance and verification

AI assisted the implementation of the QASMBench parser, the development of additional tests for this assignment, and debugging. For the parser, this assistance focused on handling OpenQASM syntax, register references, parity-check extraction, and conditional corrections accurately.

The parser and additional tests were checked thoroughly against the source formats and assignment requirements, including expected outcomes and edge cases. Our team has actively checked against AI suggestions and tested them thouroughly while we remain responsible for the submitted work.
