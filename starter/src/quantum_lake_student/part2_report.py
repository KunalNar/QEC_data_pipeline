"""Assemble results/part2/report.md from committed text and generated results.

The prose for each task is maintained in ``docs/part2/``. A marker line such
as ``<!-- results:task_c -->`` is replaced by a table built from
``results/part2/metrics.json``, so the numbers in the report always come from
the same run as the other result files.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any


SECTIONS_DIR = Path(__file__).resolve().parents[2] / "docs" / "part2"
RESULTS_ROOT = Path("results/part2")

# Section file and the heading used while that file has not been written yet.
SECTIONS = (
    ("task_a.md", "Task A: weighted syndrome decoder"),
    ("task_b.md", "Task B: supplied and combined Google decoders"),
    ("task_c.md", "Task C: raw-detector MLP prototype"),
)
MARKER = re.compile(r"<!--\s*results:(\w+)\s*-->")
DISTANCE_KEY = re.compile(r"(?:distance|d)_?(\d+)")

# Table column and the metric-key suffix it is read from. A suffix also
# matches prefixed keys such as ``weighted_brier_score`` or
# ``optimal_threshold``.
METRIC_COLUMNS = (
    ("Logical-error rate", "logical_error_rate"),
    ("Balanced accuracy", "balanced_accuracy"),
    ("Brier score", "brier_score"),
    ("Training s", "training_time_seconds"),
    ("Prediction s", "prediction_time_seconds"),
    ("Threshold", "threshold"),
)


def _metric(row: dict[str, Any], suffix: str) -> Any:
    for key, value in row.items():
        if key == suffix or key.endswith("_" + suffix):
            return value
    return None


def _model_rows(
    node: dict[str, Any],
    path: tuple[str, ...] = (),
    distance: Any = None,
) -> Iterator[tuple[tuple[str, ...], Any, dict[str, Any]]]:
    """Yield (model path, distance, metrics) for every model entry."""
    distance = node.get("distance", distance)
    for key, value in node.items():
        if not isinstance(value, dict):
            continue
        if _metric(value, "logical_error_rate") is not None:
            yield path + (key,), value.get("distance", distance), value
            continue
        match = DISTANCE_KEY.fullmatch(key)
        child_distance = int(match.group(1)) if match else distance
        child_path = path if match else path + (key,)
        yield from _model_rows(value, child_path, child_distance)


def _format(value: Any, suffix: str) -> str:
    if value is None:
        return "n/a"
    if suffix.endswith("seconds"):
        if value < 0.0001:
            return "<0.0001"
        return f"{value:.2f}" if value >= 1 else f"{value:.4f}"
    if suffix == "threshold":
        return f"{value:.2f}"
    return f"{value:.4f}"


def results_table(task: str, metrics: dict[str, Any]) -> str:
    """Markdown table of one task's test metrics from metrics.json."""
    entries = {
        key: value
        for key, value in metrics.items()
        if (key == task or key.startswith(task + "_")) and isinstance(value, dict)
    }
    rows = [row for entry in entries.values() for row in _model_rows(entry)]
    if not rows:
        return f"_No {task.replace('_', ' ').title()} results in `metrics.json` yet._"

    header = ["Model", "Distance", *(name for name, _ in METRIC_COLUMNS)]
    lines = [
        "| " + " | ".join(header) + " |",
        "| --- | ---: |" + " ---: |" * len(METRIC_COLUMNS),
    ]
    weighted = False
    for path, distance, row in rows:
        weighted = weighted or any(key.startswith("weighted_") for key in row)
        cells = [
            " / ".join(f"`{part}`" for part in path),
            "n/a" if distance is None else str(distance),
            *(_format(_metric(row, suffix), suffix) for _, suffix in METRIC_COLUMNS),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    note = "Test split."
    if weighted:
        note += " Rates, balanced accuracy, and Brier score are weighted by `sample_weight`."
    return "\n".join(lines) + "\n\n" + note


def render_section(text: str, metrics: dict[str, Any]) -> str:
    """Replace every results marker in a section with its table."""
    return MARKER.sub(lambda match: results_table(match.group(1), metrics), text)


def run_facts(run_record: dict[str, Any]) -> str:
    """Short summary of the run record. The full record is run.json."""
    release = run_record.get("data_release") or {}
    hashes = run_record.get("ml_input_hashes") or {}
    dependencies = run_record.get("dependency_versions") or {}
    revision = str(run_record.get("code_revision", "n/a"))
    lines = [
        "## Run facts",
        "",
        f"- Data release: {release.get('name', 'n/a')} version "
        f"{release.get('version', 'n/a')} ({release.get('date', 'n/a')})",
        f"- Random seed: {run_record.get('random_seed', 'n/a')}",
        f"- Code revision ({run_record.get('code_revision_kind', 'n/a')}): "
        f"`{revision[:12]}`",
        "- ML input hashes (SHA-256):",
        *(f"  - `{name}`: `{value[:12]}`" for name, value in sorted(hashes.items())),
        "- Dependencies: "
        + (", ".join(f"{name} {version}" for name, version in dependencies.items()) or "n/a"),
        "",
        "Full values, feature order, split rules, settings, and timings are in "
        "`run.json`.",
    ]
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(
    metrics: dict[str, Any],
    run_record: dict[str, Any],
    sections_dir: Path = SECTIONS_DIR,
) -> str:
    intro_path = sections_dir / "intro.md"
    parts = [
        intro_path.read_text(encoding="utf-8").strip()
        if intro_path.exists()
        else "# Part II report",
        run_facts(run_record),
    ]
    for file_name, heading in SECTIONS:
        path = sections_dir / file_name
        if path.exists():
            text = path.read_text(encoding="utf-8")
        else:
            text = (
                f"## {heading}\n\n_This section has not been written yet._\n\n"
                f"<!-- results:{path.stem} -->"
            )
        parts.append(render_section(text, metrics).strip())
    return "\n\n".join(parts) + "\n"


def write_report(
    results_root: Path = RESULTS_ROOT,
    sections_dir: Path = SECTIONS_DIR,
) -> Path:
    """Write results/part2/report.md from the current metrics and run record."""
    report = build_report(
        _read_json(results_root / "metrics.json"),
        _read_json(results_root / "run.json"),
        sections_dir,
    )
    path = results_root / "report.md"
    path.write_text(report, encoding="utf-8")
    return path
