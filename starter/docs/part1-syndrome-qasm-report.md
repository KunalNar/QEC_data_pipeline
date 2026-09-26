# Part I checkpoint: syndromes and QASMBench

This report covers the two completed source paths only. Google Gold, the Google
ML export, the second required analysis question, the Google trace example, and
the final three-source `make run` remain outside this checkpoint. Run this
checkpoint with `make run-syndrome-qasm` in `starter/`. The command rebuilds
Bronze-to-Silver results, replaces both Gold models in one PostgreSQL
transaction, exports the syndrome ML table, and writes evidence under
`results/part1/syndrome_qasm/`. Reruns replace source-specific rows; they do
not append duplicate business records.

## Source discovery and quality

- The syndrome ZIP contains seven CSV files. A filename gives the code
  distance, physical fault rate, and nominal observation count. Each CSV row
  is an aggregate `(labels, syndromes, quantity)` record; `quantity` is a
  positive weight, not a row count. The actual header uses `labels`, although
  one assignment description says `label`. Each syndrome has four rounds of
  four binary values. The release has 75,598 accepted rows and 70,000,000
  weighted observations, ten million per file. A pattern may validly have
  both logical-error labels; 25,159 fault-rate/pattern groups do. No source
  rows were rejected in this release.
- The QASMBench ZIP contains three five-qubit benchmarks, each in source and
  transpiled OpenQASM variants. The parser checks declarations, bit references,
  executed operations, measurements, explicit parity checks, and conditional
  corrections. Six circuit rows, four stabilizer-check rows, and six
  correction rows pass Silver validation. No source rows were rejected.
- Both Bronze archives are checked against manifest size and SHA-256, archive
  CRC, safe paths, and unique member names. Silver rows carry stable
  `source_record_id` values; the scoped `source_trace.parquet` maps all 75,614
  source rows to their ZIP member and record location. Rejected data would
  appear in `data_issues.parquet` with its rule, severity, action, and reason.

The discovery notebooks are
[syndrome_data_discovery.ipynb](../notebooks/syndrome_data_discovery.ipynb) and
[qasmbench_data_discovery.ipynb](../notebooks/qasmbench_data_discovery.ipynb).
Their saved outputs are cleared.

## Gold design and ML handoff

The [Gold schema design](gold-schema.md) defines each row grain, keys,
constraints, indexes, and trace links. The syndrome model separates seven
experiments, 31,941 distinct 16-byte patterns, and 75,598 original aggregate
observations. It preserves repeated source rows, both possible labels for a
pattern, and the original positive weights. A SQL view groups observations by
experiment, pattern, and label into 75,598 ML examples; a second view links
every example back to its source observation(s). The exported
`ml/ml_syndrome_decoder_example.parquet` retains the 70,000,000 total weight
and uses the supplied fault-rate split helper.

The QASMBench model contains circuits, declared registers, stabilizer checks,
ordered data-qubit participants, and conditional corrections. Its five Gold
tables contain 6, 16, 4, 8, and 6 rows, respectively. Register and circuit
foreign keys preserve the parsed relationships. QASMBench has no assignment
ML table.

We investigated joining QASMBench circuits to syndrome experiments but
rejected a row-level relationship: the circuit archive provides benchmark and
variant identifiers, while syndrome CSVs provide fault-rate experiments; no
common experiment or execution identifier maps a circuit to a CSV row.
Shared QEC terminology is not evidence of a join key.

## SQL analyses and traceability

The [weighted fault-rate query](../src/quantum_lake_student/sql/analysis_syndrome_fault_rate.sql)
answers the syndrome portion of question 1. Each of the seven fault rates has
ten million weighted observations. In this release the weighted logical-error
fraction rises from 0.000234 at fault rate 0.00001 to 0.1865274 at 0.01; this
is an association, not a causal estimate. The supplemental
[pattern query](../src/quantum_lake_student/sql/analysis_syndrome_patterns.sql)
joins three Gold tables and lists the five highest-weight patterns per rate.
The [QASMBench query](../src/quantum_lake_student/sql/analysis_qasmbench.sql)
answers question 3 by joining circuit, check, participant, and correction
relations. It returns eight check/correction combinations across the two
repetition-code variants. Query outputs are saved as CSV under the scoped
`analysis/` directory. Question 2 needs Google Gold and remains pending.

`trace_examples.json` follows one syndrome ML `example_id` through the SQL
source-link view, Gold source row, Silver `source_record_id`, and Bronze ZIP
member and CSV position. It also shows one circuit, check, and correction
traced to QASM source lines. A Part II prediction and the required Google
example cannot be shown yet; the file says so explicitly. `run.json` records
input hashes, source-code revision, times, checks, and per-output row counts;
`row_counts.json` reconciles Bronze, Silver, Gold, and the syndrome ML export.
