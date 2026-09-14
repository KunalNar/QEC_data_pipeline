# Student infrastructure

This folder is the distributable student environment. It provides:

- MinIO object storage;
- PostgreSQL 16 and Adminer;
- JupyterLab and a pinned Python data-engineering environment;
- a fixed, network-free QEC dataset bundle;
- connection helpers, simple stage templates, and basic starter tests;
- the two-part assignment, source/Silver/ML-table guides, QEC primer, FAQ,
  rubric, and submission checklist.

Before starting, install Docker and Make and open a **Bash-compatible terminal**.
On Windows, install Bash through WSL 2 with Ubuntu if needed, and run the
assignment in the Ubuntu terminal. PowerShell alone does not support the
supplied commands. Follow the [Windows setup steps](assignment/getting-started.md#windows-use-wsl-bash-not-powershell), including Docker Desktop's WSL integration.

From this directory, run:

```bash
make bootstrap
```

On success, the command prints the three browser addresses. The first run
downloads container images and therefore takes longer. Use `make help` for
daily operations, verification, shutdown, and the explicit destructive reset
command.

The student handout starts at `assignment/README.md`; the coding workspace
starts at `starter/README.md`. The three assignment datasets are already
included, so students do not need Kaggle or other dataset-service credentials.

## Project Work Distribution and Progress Tracker

Deadline: **October 11** | Execution Model: **Pair Collaboration Across 4 Feature Branches**

| Track / Pair | Assigned Role | Git Feature Branch | Pipeline Scope | Key Deliverables | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Track 1 (Pair 1)** | Member 1 | `feature/silver-parsers` | Bronze -> Silver Ingestion | • Parse `qec_syndromes` to `syndrome_observation.parquet`<br>• Parse `google_qec` to `experiment.parquet` & `shot.parquet`<br>• Parse `qasmbench` to 3 circuit tables<br>• Log errors to `data_issues.parquet` | In Progress |
| **Track 2 (Pair 1)** | Member 3 | `feature/lineage-and-ml-export` | Lineage, ML Export & Orchestration | • Generate deterministic `source_record_id`<br>• Build `results/part1/source_trace.parquet`<br>• Export `ml/` Parquet tables from Gold<br>• Orchestrate `make run` idempotency | Planned |
| **Track 3 (Pair 2)** | Member 2 | `feature/gold-postgres-schema` | Silver -> Gold Relational Warehouse | • Design PostgreSQL schema, keys, and indexes<br>• Implement all-or-nothing transactional loader (`load_postgres.py`)<br>• Write 3 analytical SQL queries (incl. 3-table join)<br>• Detector storage trade-off report | Planned |
| **Track 4 (Pair 2)** | Member 4 | `feature/part2-ml-models` | Part II Machine Learning Consumer | • Task A: Weighted Logistic Regression & baseline<br>• Task B: Combined Google Meta-Decoder (d=3 and d=5)<br>• Task C: Bounded raw-detector MLP<br>• Export metrics and predictions via `make train` | Planned |

### Quick Documentation Links
* [Project Work Division and Strategy](Project_Work_Division.md)
* [Platform Commands and Setup Notes](Setup%20Notes.md)
* [Quantum Physics and QEC Domain Guide](Quantum_Physics_for_Data_Engineers.md)
* [Discovery Notebook](starter/notebooks/01_discovery.ipynb)

