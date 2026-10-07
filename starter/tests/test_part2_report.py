import json

from quantum_lake_student.part2_report import (
    build_report,
    render_section,
    results_table,
    write_report,
)


TASK_A = {
    "task_a_syndrome_decoder": {
        "baseline_prior": {
            "weighted_logical_error_rate": 0.10395,
            "weighted_balanced_accuracy": 0.5,
            "weighted_brier_score": 0.0969,
            "training_time_seconds": 0.0074,
            "prediction_time_seconds": 0.00026,
        },
        "weighted_logistic_regression": {
            "optimal_threshold": 0.9,
            "weighted_logical_error_rate": 0.104156,
            "weighted_balanced_accuracy": 0.5015,
            "weighted_brier_score": 0.0910,
            "training_time_seconds": 0.23,
            "prediction_time_seconds": 0.001,
        },
    }
}
TASK_C = {
    "task_c_raw_detector_mlp": {
        "distance": 3,
        "subset": "distance = 3 AND shot_index < 12500",
        "C_mlp": {
            "logical_error_rate": 0.5002,
            "balanced_accuracy": 0.4998,
            "brier_score": 0.2558,
            "training_time_seconds": 17.6,
            "prediction_time_seconds": 0.013,
            "threshold": 0.71,
            "test_examples": 25000,
        },
    }
}


def test_task_a_table_shows_weighted_metrics_and_threshold():
    table = results_table("task_a", TASK_A)
    lines = table.splitlines()
    assert lines[0].startswith("| Model | Distance | Logical-error rate")
    assert "| `weighted_logistic_regression` | n/a | 0.1042 | 0.5015 | 0.0910 | 0.2300 | 0.0010 | 0.90 |" in lines
    assert "| `baseline_prior` | n/a | 0.1040 |" in table
    assert "weighted by `sample_weight`" in table


def test_task_c_table_uses_distance_and_skips_scalar_fields():
    table = results_table("task_c", TASK_C)
    assert "| `C_mlp` | 3 | 0.5002 | 0.4998 | 0.2558 | 17.60 | 0.0130 | 0.71 |" in table
    assert "subset" not in table
    assert "sample_weight" not in table


def test_nested_distances_and_missing_probabilities_are_shown():
    metrics = {
        "task_b_google_decoders": {
            "distance_3": {"pymatching": {"logical_error_rate": 0.43, "brier_score": None}},
            "distance_5": {"pymatching": {"logical_error_rate": 0.40}},
        }
    }
    table = results_table("task_b", metrics)
    assert "| `pymatching` | 3 | 0.4300 | n/a | n/a | n/a | n/a | n/a |" in table
    assert "| `pymatching` | 5 | 0.4000 |" in table


def test_missing_task_results_say_so():
    assert results_table("task_b", TASK_A) == "_No Task B results in `metrics.json` yet._"


def test_markers_are_replaced():
    text = "## Task C\n\n<!-- results:task_c -->\n\nAfter."
    rendered = render_section(text, TASK_C)
    assert "<!--" not in rendered
    assert "`C_mlp`" in rendered
    assert rendered.endswith("After.")


def test_report_keeps_section_order_and_an_open_task_b_slot(tmp_path):
    (tmp_path / "intro.md").write_text("# Part II\n\nIntro.")
    (tmp_path / "task_a.md").write_text("## Task A\n\n<!-- results:task_a -->")
    (tmp_path / "task_c.md").write_text("## Task C\n\n<!-- results:task_c -->")

    report = build_report({**TASK_A, **TASK_C}, {"random_seed": 42}, tmp_path)

    order = [report.index(text) for text in ("# Part II", "## Run facts", "## Task A", "## Task B", "## Task C")]
    assert order == sorted(order)
    assert "_This section has not been written yet._" in report
    assert "_No Task B results in `metrics.json` yet._" in report
    assert "- Random seed: 42" in report
    assert "<!--" not in report


def test_write_report_reads_the_result_files(tmp_path):
    sections = tmp_path / "sections"
    sections.mkdir()
    (sections / "task_c.md").write_text("## Task C\n\n<!-- results:task_c -->")
    results = tmp_path / "results"
    results.mkdir()
    (results / "metrics.json").write_text(json.dumps(TASK_C))
    (results / "run.json").write_text(
        json.dumps(
            {
                "data_release": {"name": "release", "version": 3, "date": "2026-08-05"},
                "ml_input_hashes": {"ml/a.parquet": "0123456789abcdef"},
            }
        )
    )

    path = write_report(results, sections)

    report = path.read_text()
    assert path == results / "report.md"
    assert "- Data release: release version 3 (2026-08-05)" in report
    assert "`ml/a.parquet`: `0123456789ab`" in report
    assert "`C_mlp`" in report
