# Gold schema design

This is the evolving design for the student-owned PostgreSQL Gold layer. It is
not a copy of the six Silver Parquet tables. Each section states what one Gold
row means, how it is keyed and checked, which queries its indexes support, and
how its records trace back to the source. The syndrome design below is chosen;
the Google and QASMBench designs are still to be decided. This document is a
design specification. The syndrome DDL and loader are implemented; the
Google and QASMBench Gold work is pending.

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
pattern, and label**, not necessarily one example per original CSV row. A
committed Gold SQL view will group observations on those three values and use
`SUM(quantity)` as `sample_weight`. Its repeatable `example_id` will be based
on the same group key; a companion relation/view must map each example back
to every contributing `source_record_id`. The exact hash encoding and export
SQL will be fixed and tested when the Gold-to-ML stage is implemented. The
course split is assigned from the experiment's physical fault rate during
the permitted export step.

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

## QASMBench circuits — design pending

To be completed after choosing the circuit, stabilizer-check, and conditional
correction relations. Record each row grain, keys, constraints, indexes, and
trace to QASM members. No row-level relationship to a syndrome or Google
experiment is supplied; shared QEC vocabulary alone is not a join key.

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
