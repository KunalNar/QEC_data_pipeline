# Student pipeline workspace

This is the starting point for your implementation. The platform supplies the
services, dependencies, source data, configuration, connection helpers, and
simple stage templates. Your team supplies the data engineering.

Read these documents in order:

1. `../assignment/getting-started.md`
2. `../assignment/brief.md`
3. `../assignment/plain-language-guide.md`
4. `../assignment/quantum-data-primer.md`
5. `../assignment/data-sources.md`
6. [Parquet in this assignment](../assignment/plain-language-guide.md#parquet-in-this-assignment)
7. `../assignment/silver-tables.md`
8. `../assignment/required-ml-tables.md`
9. `../assignment/part-2-ai-ml.md`
10. `../assignment/rubric.md`

Our [Gold schema design](docs/gold-schema.md) records all three source models.
After producing syndrome Silver, `make gold-syndromes` loads its three Gold
tables into PostgreSQL. `make ml-syndromes` exports the syndrome ML handoff
from Gold. `make gold-qasmbench` loads the QASMBench circuit relationships.
`make gold-google` loads Google shots and decoder predictions, and
`make ml-google` exports the Google ML handoff. `make run` rebuilds all three
sources, loads Gold in one transaction, exports both ML tables, and writes
the complete evidence under `results/part1/`; see the
[Part I report](docs/part1-report.md). The earlier
`make run-syndrome-qasm` checkpoint remains available under
`results/part1/syndrome_qasm/`; see its
[two-source report](docs/part1-syndrome-qasm-report.md).
The [Part II handoff](docs/part2-handoff.md) summarizes the available data
products for the ML team.

`make check` verifies connections, `make inventory` lists the three Bronze
objects, and `make test` runs the automated checks. Part II `make train`
remains for the ML team.

Suggested source layout:

```text
src/quantum_lake_student/
├── config.py
├── connections.py
├── formats.py
├── ml.py
├── models.py
├── cli.py
└── stages/
    ├── register_sources.py
    ├── prepare_data.py
    ├── build_ml_tables.py
    ├── load_postgres.py
    └── train.py
```

`formats.py` provides representation-level readers for the supplied Stim `b8`
and `01` files. It does not decide how parity measurements, detector events,
shots, or decoder outputs should be modeled; those decisions remain part of
the assignment.

`ml.py` provides bit unpacking, course split assignment, the documented model
inputs, partition loading, and weighted LER. It does not select models, perform
training, or interpret results. Those remain required Part II work.

Generated data, reports, credentials, and notebook outputs should not be
committed to version control. Write required run evidence and outputs under the
`results/part1/` and `results/part2/` layout described in the brief.

The AI/ML stage is a downstream consumer check. A particular model score or an
improvement over a supplied decoder is not part of the grade.
