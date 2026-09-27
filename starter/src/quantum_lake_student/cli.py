"""Command-line entry point for the student workspace."""

from __future__ import annotations

import argparse
import sys
from uuid import uuid4

from rich.console import Console
from rich.table import Table

from .config import Settings
from .connections import bronze_inventory, check_platform
from .stages.build_ml_tables import build_syndrome_ml_table
from .stages.build_google_ml_table import build_google_ml_table
from .stages.load_postgres import load_syndromes_gold
from .stages.load_postgres_google import load_google_gold
from .stages.load_qasmbench_gold import load_qasmbench_gold
from .stages.prepare_qasmbench import prepare_qasmbench
from .stages.prepare_syndromes import prepare_syndromes
from .stages.run_syndrome_qasm import run_syndrome_qasm
from .stages.run_part1 import run_part1
from .stages.silvergoogle import prepare_google_data


console = Console()


def command_check(settings: Settings) -> int:
    result = check_platform(settings)
    for service, message in result.items():
        console.print(f"[green]OK[/green] {service}: {message}")
    return 0


def command_inventory(settings: Settings) -> int:
    table = Table(title="Supplied course data files (kept unchanged)")
    table.add_column("Stored path")
    table.add_column("Bytes", justify="right")
    for key, size in bronze_inventory(settings):
        table.add_row(key, f"{size:,}")
    console.print(table)
    return 0


def command_prepare_syndromes(settings: Settings) -> int:
    run_id = "syndrome-" + uuid4().hex
    result = prepare_syndromes(settings, run_id=run_id)
    console.print(
        "[green]OK[/green] syndrome Silver: "
        f"{result.output_count:,} accepted from "
        f"{result.input_count:,} source rows; "
        f"{result.issue_count:,} issue(s)"
    )
    return 0


def command_prepare_qasmbench(settings: Settings) -> int:
    run_id = "qasmbench-" + uuid4().hex
    result = prepare_qasmbench(settings, run_id=run_id)
    console.print(
        "[green]OK[/green] QASMBench Silver: "
        f"{result.output_count:,} rows from "
        f"{result.input_count:,} QASM files; "
        f"{result.issue_count:,} issue(s)"
    )
    return 0


def command_prepare_google(settings: Settings) -> int:
    run_id = "google-qec-" + uuid4().hex
    result = prepare_google_data(settings, run_id=run_id)
    console.print(
        "[green]OK[/green] Google QEC Silver: "
        f"{result.output_count:,} rows from "
        f"{result.input_count:,} shots; "
        f"{result.issue_count:,} issue(s)"
    )
    return 0


def command_load_syndromes_gold(settings: Settings) -> int:
    result = load_syndromes_gold(settings, run_id="gold-syndrome-" + uuid4().hex)
    console.print(
        "[green]OK[/green] syndrome Gold: "
        f"{result.output_count:,} observations loaded"
    )
    return 0


def command_build_ml_syndromes(settings: Settings) -> int:
    result = build_syndrome_ml_table(
        settings,
        run_id="ml-syndrome-" + uuid4().hex,
    )
    console.print(
        "[green]OK[/green] syndrome ML: "
        f"{result.output_count:,} examples exported from Gold"
    )
    return 0


def command_load_qasmbench_gold(settings: Settings) -> int:
    result = load_qasmbench_gold(
        settings,
        run_id="gold-qasmbench-" + uuid4().hex,
    )
    console.print(
        "[green]OK[/green] QASMBench Gold: "
        f"{result.output_count} traced circuit, check, and correction rows loaded"
    )
    return 0


def command_load_google_gold(settings: Settings) -> int:
    result = load_google_gold(settings, run_id="gold-google-" + uuid4().hex)
    console.print(
        "[green]OK[/green] Google Gold: "
        f"{result.output_count:,} experiment, shot, and prediction rows loaded"
    )
    return 0


def command_build_ml_google(settings: Settings) -> int:
    result = build_google_ml_table(settings, run_id="ml-google-" + uuid4().hex)
    console.print(
        "[green]OK[/green] Google ML: "
        f"{result.output_count:,} shot examples exported from Gold"
    )
    return 0


def command_run_syndrome_qasm(settings: Settings) -> int:
    record = run_syndrome_qasm(
        settings,
        run_id="syndrome-qasm-" + uuid4().hex,
    )
    console.print(
        "[green]OK[/green] syndrome and QASMBench Part I checkpoint: "
        f"{len(record['outputs'])} output tables/artifacts recorded under "
        "results/part1/syndrome_qasm/"
    )
    return 0


def command_run(settings: Settings) -> int:
    record = run_part1(settings, run_id="part1-" + uuid4().hex)
    console.print(
        "[green]OK[/green] Part I: "
        f"{len(record['outputs'])} outputs recorded under results/part1/"
    )
    return 0


def command_train(_: Settings) -> int:
    console.print(
        "[yellow]The AI/ML stage is intentionally unimplemented.[/yellow]\n"
        "Consume the required ML input tables through the supplied helpers and "
        "write model files and the required results/part2 files."
    )
    return 2


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "command",
        choices=(
            "check",
            "inventory",
            "prepare-syndromes",
            "prepare-qasmbench",
            "prepare-google",
            "load-syndromes-gold",
            "build-ml-syndromes",
            "load-qasmbench-gold",
            "load-google-gold",
            "build-ml-google",
            "run-syndrome-qasm",
            "run",
            "train",
        ),
        help="Action to perform",
    )
    return result


def main() -> None:
    arguments = parser().parse_args()
    settings = Settings.from_environment()
    commands = {
        "check": command_check,
        "inventory": command_inventory,
        "prepare-syndromes": command_prepare_syndromes,
        "prepare-qasmbench": command_prepare_qasmbench,
        "prepare-google": command_prepare_google,
        "load-syndromes-gold": command_load_syndromes_gold,
        "build-ml-syndromes": command_build_ml_syndromes,
        "load-qasmbench-gold": command_load_qasmbench_gold,
        "load-google-gold": command_load_google_gold,
        "build-ml-google": command_build_ml_google,
        "run-syndrome-qasm": command_run_syndrome_qasm,
        "run": command_run,
        "train": command_train,
    }
    raise SystemExit(commands[arguments.command](settings))


if __name__ == "__main__":
    main()
