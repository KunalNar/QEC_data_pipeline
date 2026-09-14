# Course Platform & Command Cheat Sheet

Quick reference for managing the Assignment 1 data lake platform.

---

## 🌐 Web Interfaces & Credentials

| Service           | URL                                                                     | Default Credentials                                                                        | Purpose                                          |
| :---------------- | :---------------------------------------------------------------------- | :----------------------------------------------------------------------------------------- | :----------------------------------------------- |
| **JupyterLab**    | [http://localhost:8888](http://localhost:8888/lab?token=quantum-course) | Token: `quantum-course`                                                                    | Python dev & interactive notebooks               |
| **MinIO Console** | [http://localhost:9001](http://localhost:9001)                          | User: `quantum`<br>Pass: `quantum-course-only`                                             | S3 data lake explorer (`bronze`, `silver`, etc.) |
| **Adminer**       | [http://localhost:8080](http://localhost:8080)                          | Server: `postgres`<br>User: `quantum`<br>Pass: `quantum-course-only`<br>DB: `quantum_lake` | PostgreSQL web database browser                  |

---

## 🚀 1. One-Time Setup

Run from `assignment-1/`:

```bash
make bootstrap
```
* **Runs only once** at initial setup.
* Generates `.env`, builds Docker images, starts services, seeds raw datasets to MinIO `bronze/`, and verifies environment health.

---

## 🔄 2. Daily Workflow (From `assignment-1/`)

```bash
# Start all services (takes ~2 seconds; keeps previous data)
make up

# Stop all services (preserves all data, tables, and volumes)
make down

# View running services and container status
make ps

# View container logs (useful for debugging)
make logs
```

> **Note on code changes:** The local `starter/` directory is live-mounted to `/workspace` inside the container. Any edits made in VS Code or JupyterLab apply immediately—no Docker rebuilds required.

---

## 🧪 3. Student Pipeline Commands

Run these inside **JupyterLab Terminal** (or from host via `docker compose exec workspace <cmd>`):

```bash
# Check MinIO & PostgreSQL connectivity
make check

# List all seeded Bronze objects and sizes
make inventory

# Run automated tests (pytest)
make test

# Run your data pipeline (once implemented)
make run

# Train/evaluate models on ML tables (Part II)
make train
```

Host shortcut examples:
```bash
docker compose exec workspace make check
docker compose exec workspace make test
```

---

## ⚠️ 4. Reset & Troubleshooting

```bash
# Verify environment and dataset integrity
make verify

# Complete fresh slate (WIPES MinIO and PostgreSQL volumes)
make reset-platform
# Followed by:
make bootstrap
```

