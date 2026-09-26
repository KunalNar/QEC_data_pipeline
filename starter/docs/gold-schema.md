# Gold schema design

This is the evolving design for the student-owned PostgreSQL Gold layer. It is
not a copy of the six Silver Parquet tables. Each section states what one Gold
row means, how it is keyed and checked, which queries its indexes support, and
how its records trace back to the source. The syndrome and QASMBench designs
below are implemented; the Google design is still to be decided. This
document is also the design specification for the completed source-specific
Gold loaders.

The Gold load must be all-or-nothing and repeatable: an incomplete load must
not replace a complete one, and rerunning unchanged inputs must not add
business records or change stable IDs. The two required ML tables must be
exported from committed Gold SQL/views, without rereading Bronze or Silver.

## Simulated syndromes — chosen design

Input: `silver/qec_syndromes/syndrome_observation.parquet`. One Silver row is
one aggregate CSV row, and `quantity` is its weight, not a number of rows to
create. The current release has seven experiment files, 75,598 aggregate
rows, and 70,000,000 weighted observations. A syndrome can appear under both
logical-error labels, so the pattern is never the key of an observation.

### Relations and keys

| Relation | One row represents | Primary key | Main fields and relationships |
| --- | --- | --- | --- |
| `gold.syndrome_experiment` | One source CSV experiment | `experiment_id` | `physical_fault_rate`; referenced by observations |
| `gold.syndrome_pattern` | One distinct 4-by-4 binary syndrome | `syndrome_bits` | 16-byte `BYTEA`, `round_count`, `check_count`; referenced by observations |
| `gold.syndrome_observation` | One original aggregate CSV row | `source_record_id` | `experiment_id` and `syndrome_bits` foreign keys, `logical_error_label`, positive `quantity` |

The Silver `experiment_id` and `source_record_id` are retained unchanged.
`syndrome_bits` is the pattern key itself; its fixed 16-byte size avoids a
second hash and lets the same pattern be shared across experiments. We do
**not** make `(experiment_id, syndrome_bits, logical_error_label)` unique in
`syndrome_observation`: separate source rows with the same values must remain
separately traceable and their quantities can be summed later. We also do not
make `physical_fault_rate` unique; two distinct source experiments could have
the same rate.

**Duplicate-combination note:** The supplied release has no repeated
`(experiment_id, syndrome_bits, logical_error_label)` combination, but the
assignment does not prohibit one. If two CSV rows do share all three values,
Gold keeps both under their different `source_record_id` values. The ML export
combines them into one example with `sample_weight = SUM(quantity)` and must
still trace that example to both Gold rows. Matching syndrome bits alone do
not cause rows with different experiments or labels to be combined.

### PostgreSQL definition

The executable definition is [gold_syndrome.sql](../src/quantum_lake_student/sql/gold_syndrome.sql).

```sql
CREATE SCHEMA IF NOT EXISTS gold;

CREATE TABLE gold.syndrome_experiment (
    experiment_id text PRIMARY KEY,
    physical_fault_rate double precision NOT NULL,
    CONSTRAINT syndrome_fault_rate_range
        CHECK (physical_fault_rate > 0 AND physical_fault_rate < 1)
);

CREATE TABLE gold.syndrome_pattern (
    syndrome_bits bytea PRIMARY KEY,
    round_count smallint NOT NULL,
    check_count smallint NOT NULL,
    CONSTRAINT syndrome_round_count CHECK (round_count = 4),
    CONSTRAINT syndrome_check_count CHECK (check_count = 4),
    CONSTRAINT syndrome_shape
        CHECK (octet_length(syndrome_bits) = round_count * check_count),
    CONSTRAINT syndrome_binary_bytes
        CHECK (encode(syndrome_bits, 'hex') ~ '^(00|01){16}$')
);

CREATE TABLE gold.syndrome_observation (
    source_record_id text PRIMARY KEY,
    experiment_id text NOT NULL
        REFERENCES gold.syndrome_experiment (experiment_id),
    syndrome_bits bytea NOT NULL
        REFERENCES gold.syndrome_pattern (syndrome_bits),
    logical_error_label boolean NOT NULL,
    quantity bigint NOT NULL,
    CONSTRAINT syndrome_positive_quantity CHECK (quantity > 0)
);

CREATE INDEX syndrome_observation_experiment_pattern_label_idx
    ON gold.syndrome_observation
    (experiment_id, syndrome_bits, logical_error_label);

CREATE INDEX syndrome_observation_pattern_idx
    ON gold.syndrome_observation (syndrome_bits);
```

The primary keys prevent duplicate experiments, patterns, and source rows.
Foreign keys prevent observations from referring to missing experiments or
patterns. The shape and binary-byte checks preserve the Silver 4-by-4,
round-first representation. The composite index supports grouping by
experiment, pattern, and label for weighted analyses and ML export; the
pattern index supports finding one syndrome across experiments. The seven-row
experiment table needs no separate fault-rate index.

### Load, reconciliation, and traceability

1. Read the validated syndrome Silver table. Insert distinct
   `(experiment_id, physical_fault_rate)` pairs and distinct
   `(syndrome_bits, round_count, check_count)` patterns, rejecting any
   conflicting values for the same key.
2. Insert every Silver aggregate row into `syndrome_observation` with its
   unchanged `source_record_id` and `quantity`; never expand the weight into
   individual shots.
3. In the same all-or-nothing load, check that the Gold observation count,
   source-record-ID set, and sum of `quantity` equal the Silver values. Also
   check that each experiment ID has one fault rate and each pattern has one
   shape. A failed check aborts the load.
4. For a Gold observation, its `source_record_id` identifies the Silver row
   and joins to `results/part1/source_trace.parquet`. That trace supplies the
   Bronze ZIP object, archive member, CSV row locator, and input SHA-256.

The required syndrome ML table has **one example per distinct experiment,
pattern, and label**, not necessarily one example per original CSV row. The
committed `gold.syndrome_ml_example` view groups observations on those three
values and uses `SUM(quantity)` as `sample_weight`. Its `example_id` is
`ml-syn-` plus SHA-256 of the UTF-8 experiment ID, a zero-byte separator, the
16 syndrome bytes, and one label byte (`00` or `01`). The separator and fixed
byte widths make the input unambiguous. The `gold.syndrome_ml_source` view maps
each example back to every contributing `source_record_id`.

`make ml-syndromes` reads these Gold views, uses the supplied
`syndrome_data_split`, `syndrome_model_input`, and `partition_records` helpers,
validates row IDs, labels, weights, split coverage, and source-link
reconciliation, then writes
`ml/ml_syndrome_decoder_example.parquet`. It does not reread Bronze or Silver.
The assigned splits are validation for fault rate `0.0005`, test for `0.005`,
and train for the other five rates.

### Strengths and trade-off

- Experiments, reusable syndrome patterns, and individual aggregate source
  rows have separate meanings; Gold is not a renamed Silver table.
- The source-row primary key preserves duplicates and valid cases where one
  pattern has both labels, while keeping exact Bronze traceability.
- Quantities remain weights, so storage and SQL analyses stay proportional to
  aggregate rows rather than 70 million simulated shots.
- The ML example grain can be produced by a view without maintaining another
  table of precomputed totals. The cost is a three-table join and a grouping
  query, which is reasonable for this release's size.

## Google hardware QEC — design pending

To be completed after choosing the experiment, shot, detector-summary,
decoder, and prediction relations. Record each row grain, keys, constraints,
indexes, packed-bit storage choice, Gold-to-ML path, and trace through aligned
companion files. Do not treat a decoder prediction as the actual outcome.

## QASMBench circuits — chosen design

Input: the three `silver/qasmbench/` Parquet tables. The current release has
six circuit variants, four explicit stabilizer checks, and six conditional
corrections. The Silver register description and each check's data-qubit list
are expanded into relational rows, retaining declaration and participant
order. The executable DDL is
[gold_qasmbench.sql](../src/quantum_lake_student/sql/gold_qasmbench.sql).

| Relation | One row represents | Key and relationships |
| --- | --- | --- |
| `gold.qasm_circuit` | One source or transpiled QASM member | `circuit_id` primary key; unique `source_record_id` and `(benchmark_name, variant)` |
| `gold.qasm_register` | One declared quantum or classical register in a circuit | `(circuit_id, register_name)` primary key; circuit foreign key; unique declaration order per circuit |
| `gold.qasm_stabilizer_check` | One explicit measured parity check | `source_record_id` primary key; circuit foreign key; `(circuit_id, check_id)` unique |
| `gold.qasm_check_data_qubit` | One ordered data-qubit participant in a check | `(check_source_record_id, participant_order)` primary key; check foreign key; no duplicate qubit within a check |
| `gold.qasm_conditional_correction` | One syndrome-controlled recovery operation | `source_record_id` primary key; circuit foreign key; condition-register foreign key |

The circuit retains its file hash and executed operation, measurement, and
two-qubit counts. PostgreSQL checks valid variants, 64-digit file hashes,
positive register sizes, nonnegative counts and condition values, and
two-qubit counts no greater than operation counts. The loader additionally
checks that declared register sizes reconcile to the circuit totals; every
ancilla, participant, syndrome bit, and target is in range in a register of
the correct kind; and each condition value fits its declared classical
register. These cross-row checks are done before the transaction because a
simple row-level `CHECK` cannot compare against another table's register size.

Primary and unique keys create indexes for IDs, family/variant lookup, and
register/participant order. Explicit indexes on
`qasm_stabilizer_check(circuit_id, syndrome_bit)` and
`qasm_conditional_correction(circuit_id, condition_register,
condition_value)` support the required circuit-to-parity-to-recovery query.
The participant primary key supports ordered lookup for each check.

`make gold-qasmbench` replaces only these five QASMBench relations in one
PostgreSQL transaction and reconciles their counts and source IDs with all
three Silver tables. `make run-syndrome-qasm` instead loads both the syndrome
and QASMBench models in one transaction, so a failure in either leaves both
previous versions intact. Circuit, check, and correction `source_record_id`
values join to `source_trace.parquet` (under the scoped result directory for
the combined checkpoint), which identifies the QASM ZIP
member, source line(s), and Bronze hash. Register rows trace through their
circuit; participant rows trace through their check. This preserves the
distinction between register-local `q[0]` and `a[0]`.

The [QASMBench analysis query](../src/quantum_lake_student/sql/analysis_qasmbench.sql)
joins circuit, check, ordered participants, and conditional corrections. It
uses the bit positions of the declared syndrome register to show which
conditions contain each measured bit. The schema is intentionally
separate from syndrome and Google experiments: no supplied row-level key
links a QASM member to either dataset. QASMBench therefore has no required
ML handoff table and supplies no labeled training rows.

### Strengths and trade-off

- Declared register identity and order are queryable instead of buried in
  JSON; `q[0]` and `a[0]` remain different qubits.
- Each check's ordered data-qubit participants are first-class relations,
  making the parity mapping a normal SQL join.
- Stable circuit/check/correction source IDs keep the original QASM-member
  and statement trace, while source and transpiled variants remain distinct.
- Five related tables are more than the three Silver tables, but they expose
  the circuit structure required for analysis without a large generic QASM
  operation model or an unsupported cross-dataset relationship.

## Cross-dataset decisions — design pending

Use consistent names for genuinely shared concepts, but do not manufacture
identifiers or matches between independent sources. Document the final
all-or-nothing load strategy, the rejected cross-source relationship, the
reconciliation checks, and the three required analysis queries when all
dataset schemas are chosen.

## Assignment contracts

- [Gold design and tracing requirements](../../assignment/brief.md#gold-design-requirements)
- [Minimum Silver table definitions](../../assignment/silver-tables.md)
- [Required ML input tables](../../assignment/required-ml-tables.md)
- [Submission checklist](../../assignment/submission-checklist.md)
