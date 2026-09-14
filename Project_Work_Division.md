# Team Project Work Division & Parallel Execution Plan

**Project:** Quantum Error Correction (QEC) Data Lake  
**Team Size:** 4 Members  
**Deadline:** October 11  
**Execution Model:** Fully Parallel Vertical Slices (Zero Blocking)

---

## 1. The Strategy: Vertical Slice Parallelism

To avoid members waiting on each other, the work is divided into four self-contained, parallel tracks. Each member owns a specific source or cross-cutting pipeline system end-to-end.

```text
┌───────────────────────────┐ ┌───────────────────────────┐
│ Track 1: Simulation Lead  │ │ Track 2: Google Hardware  │
│ (Member 1)                │ │ (Member 2)                │
├───────────────────────────┤ ├───────────────────────────┤
│ • Silver: syndrome_obs    │ │ • Silver: experiment/shot │
│ • Gold: syndrome relation │ │ • Gold: hardware relation │
│ • ML: Task A (Weighted)   │ │ • ML: Task B (Combined)   │
└───────────────────────────┘ └───────────────────────────┘
┌───────────────────────────┐ ┌───────────────────────────┐
│ Track 3: Circuits & SQL   │ │ Track 4: Architecture & ML│
│ (Member 3)                │ │ (Member 4)                │
├───────────────────────────┤ ├───────────────────────────┤
│ • Silver: 3 QASM tables   │ │ • Master source_trace     │
│ • Gold: Database schema   │ │ • ML: Task C (Raw MLP)    │
│ • SQL: 3 Analysis queries │ │ • Orchestration: make run │
└───────────────────────────┘ └───────────────────────────┘
```

---

## 2. Member Breakdown & Ownership

### 👤 Track 1: Simulation & Weighted Modeling Lead
* **Git Branch:** `feature/simulated-syndromes`
* **Silver Deliverables:**
  - Parse 7 CSV files in `bronze/source=qec_syndromes/`.
  - Extract physical fault rate (`pfr`) from filename.
  - Generate `silver/qec_syndromes/syndrome_observation.parquet`.
  - Enforce data invariants: positive weights, 16-bit binary domain, preserve degenerate rows.
* **Gold Deliverables:**
  - Define PostgreSQL table and loader for syndrome observations.
* **ML Deliverables (Part II Task A):**
  - Implement weighted majority/prior baseline.
  - Implement weighted linear classifier (Logistic Regression) with `sample_weight=quantity`.

---

### 👤 Track 2: Google Hardware & Meta-Decoder Lead
* **Git Branch:** `feature/google-hardware`
* **Silver Deliverables:**
  - Parse `.b8` binary files (packed little-endian) and `.01` text files in `bronze/source=google_qec/`.
  - Generate `silver/google_qec/experiment.parquet` (1 row per experiment directory).
  - Generate `silver/google_qec/shot.parquet` (50,000 rows per experiment, 250,000 total).
  - Validate companion file lengths before writing Parquet.
* **Gold Deliverables:**
  - Define PostgreSQL tables and loaders for hardware experiments and shots.
* **ML Deliverables (Part II Task B):**
  - Evaluate 4 supplied decoders on held-out test shots.
  - Fit linear combined meta-decoder (event density + 4 predictions) for $d=3$ and $d=5$ separately.

---

### 👤 Track 3: Circuits, Relational Architecture & SQL Lead
* **Git Branch:** `feature/circuits-and-gold-schema`
* **Silver Deliverables:**
  - Parse OpenQASM files in `bronze/source=qasmbench/`.
  - Generate `silver/qasmbench/circuit.parquet`.
  - Generate `silver/qasmbench/stabilizer_check.parquet`.
  - Generate `silver/qasmbench/conditional_correction.parquet`.
* **Gold Deliverables:**
  - Design master PostgreSQL relational model (schema, primary keys, foreign keys, indexes).
  - Implement all-or-nothing (transactional) database load in `load_postgres.py`.
  - Write SQL scripts for the 3 required analytical queries (including the 3-table join).
  - Author the detector-storage trade-off evaluation in the report.

---

### 👤 Track 4: Lineage, Orchestration & Deep Learning Lead
* **Git Branch:** `feature/lineage-and-orchestration`
* **Lineage & Quality Deliverables:**
  - Implement deterministic `source_record_id` generator across all sources.
  - Build `results/part1/source_trace.parquet` (full provenance: archive member, byte/line locator, input hash).
  - Build `results/part1/data_issues.parquet` (logging invalid values, rule violations, actions).
* **Orchestration Deliverables:**
  - Build `build_ml_tables.py`: SQL views and export scripts creating the two required ML Parquet tables from Gold.
  - Log `run.json` and `row_counts.json` (reconciliation of read vs accepted rows).
  - Wire up `make run` so the entire Part I pipeline executes idempotently in one command.
* **ML Deliverables (Part II Task C):**
  - Build bounded raw-detector Multilayer Perceptron (MLP) on 200 input bits ($d=3, \text{shot\_index} < 12,500$).
  - Wire up `make train` to output `predictions.parquet` and `metrics.json`.

---

## 3. Four-Week Schedule (Target: October 11)

| Week | Dates | Milestone | Deliverables |
| :--- | :--- | :--- | :--- |
| **Week 1** | **Sept 14 – Sept 20** | **Silver Zone & Lineage** | All 6 Silver Parquet tables created. `source_trace.parquet` working. PostgreSQL schema drafted. |
| **Week 2** | **Sept 21 – Sept 27** | **Gold Loading & ML Export** | PostgreSQL tables loaded idempotently. 3 SQL queries written. Both ML tables exported from Gold. `make run` passes twice cleanly. |
| **Week 3** | **Sept 28 – Oct 4** | **Part II Machine Learning** | Tasks A, B, and C trained and evaluated. Metrics, predictions, and timing logs exported. `make train` passes. |
| **Week 4** | **Oct 5 – Oct 11** | **End-to-End Freeze & Report** | Clean-slate reproduction test (`make reset-platform && make bootstrap && make run && make train && make test`). Final report write-up. Final submission! |

---

## 4. Git Collaboration Rules

1. **Never commit directly to `main`:**
   Each member works exclusively in their feature branch.
2. **Pull Requests (PRs):**
   When a task is complete, open a PR on GitHub. At least one teammate must approve before merging.
3. **Keep `main` Green:**
   Before merging any PR, confirm that `make test` passes.
4. **Notebook Etiquette:**
   Click **"Clear All Outputs"** before committing any Jupyter notebook (`.ipynb`) to keep Git diffs clean.
