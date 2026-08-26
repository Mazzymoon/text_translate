#!/usr/bin/env python3
"""Read-only analysis for smoke, pilot, and full QLoRA training runs.

The script accepts any run directory produced by ``train_qlora.py``.  It never
changes training artefacts; derived CSV, JSON, and figures are written only to
``<run-dir>/analysis/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
RESULT_FILES = (
    "trainer_state.json",
    "run_summary.json",
    "train_results.json",
    "eval_results.json",
)
CSV_FIELDS = (
    "log_index",
    "event_type",
    "step",
    "epoch",
    "train_loss",
    "eval_loss",
    "learning_rate",
    "grad_norm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Smoke, pilot, or full training output directory",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read_json_if_present(path: Path, warnings: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        warnings.append(f"Missing optional result file: {path.name}")
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(f"Could not read {path.name}: {type(exc).__name__}: {exc}")
        return None
    if not isinstance(value, dict):
        warnings.append(f"Unexpected JSON structure in {path.name}: expected object")
        return None
    return value


def as_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def as_int(value: Any) -> int | None:
    number = as_number(value)
    return int(number) if number is not None else None


def first_present(sources: Iterable[dict[str, Any] | None], *keys: str) -> Any:
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in keys:
            if key in source and source[key] is not None:
                return source[key]
    return None


def nested_value(source: dict[str, Any] | None, *path: str) -> Any:
    current: Any = source
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def extract_metrics(trainer_state: dict[str, Any] | None, warnings: list[str]) -> list[dict[str, Any]]:
    if trainer_state is None:
        return []
    history = trainer_state.get("log_history")
    if not isinstance(history, list):
        warnings.append("trainer_state.json has no list-valued log_history")
        return []

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(history):
        if not isinstance(item, dict):
            warnings.append(f"Ignoring non-object log_history entry at index {index}")
            continue
        train_loss = as_number(item.get("loss", item.get("train_loss")))
        eval_loss = as_number(item.get("eval_loss"))
        if eval_loss is not None:
            event_type = "eval"
        elif train_loss is not None:
            event_type = "train"
        else:
            event_type = "other"
        rows.append(
            {
                "log_index": index,
                "event_type": event_type,
                "step": as_int(item.get("step")),
                "epoch": as_number(item.get("epoch")),
                "train_loss": train_loss,
                "eval_loss": eval_loss,
                "learning_rate": as_number(item.get("learning_rate")),
                "grad_norm": as_number(item.get("grad_norm")),
            }
        )
    return rows


def metric_series(rows: list[dict[str, Any]], field: str) -> list[tuple[int, float]]:
    series = []
    for row in rows:
        step = row.get("step")
        value = row.get(field)
        if isinstance(step, int) and isinstance(value, float):
            series.append((step, value))
    return series


def loss_summary(series: list[tuple[int, float]], prefix: str) -> dict[str, Any]:
    if not series:
        return {
            f"first_{prefix}_loss": None,
            f"last_{prefix}_loss": None,
            f"min_{prefix}_loss": None,
        }
    minimum_step, minimum = min(series, key=lambda item: (item[1], item[0]))
    result = {
        f"first_{prefix}_loss": series[0][1],
        f"last_{prefix}_loss": series[-1][1],
        f"min_{prefix}_loss": minimum,
    }
    if prefix == "eval":
        result["best_eval_step"] = minimum_step
    return result


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8-sig", newline="", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def save_figure(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".png", dir=path.parent, delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        figure.savefig(temp_path, dpi=150, bbox_inches="tight")
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def make_loss_plot(train: list[tuple[int, float]], eval_: list[tuple[int, float]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.5))
    if train:
        axis.plot(*zip(*train), label="train loss", color="#1f77b4", linewidth=1.3)
    if eval_:
        axis.plot(
            *zip(*eval_),
            label="eval loss",
            color="#d62728",
            linewidth=1.1,
            marker="o",
            markersize=4,
        )
    if not train and not eval_:
        axis.text(0.5, 0.5, "No loss values available", ha="center", va="center", transform=axis.transAxes)
    axis.set_xlabel("step")
    axis.set_ylabel("loss")
    axis.set_title("Training and evaluation loss")
    axis.grid(alpha=0.25)
    if train or eval_:
        axis.legend()
    save_figure(figure, path)
    plt.close(figure)


def make_metric_plot(series: list[tuple[int, float]], path: Path, title: str, y_label: str, color: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.5))
    if series:
        axis.plot(*zip(*series), color=color, linewidth=1.3, marker="o", markersize=3)
    else:
        axis.text(0.5, 0.5, f"No {y_label} values available", ha="center", va="center", transform=axis.transAxes)
    axis.set_xlabel("step")
    axis.set_ylabel(y_label)
    axis.set_title(title)
    axis.grid(alpha=0.25)
    save_figure(figure, path)
    plt.close(figure)


def display_number(value: Any, digits: int = 2) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def main() -> int:
    args = parse_args()
    run_dir = resolve_path(args.run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    analysis_dir = run_dir / "analysis"
    warnings: list[str] = []
    files = {name: read_json_if_present(run_dir / name, warnings) for name in RESULT_FILES}
    trainer_state = files["trainer_state.json"]
    run_summary = files["run_summary.json"]
    train_results = files["train_results.json"]
    eval_results = files["eval_results.json"]
    rows = extract_metrics(trainer_state, warnings)
    train_losses = metric_series(rows, "train_loss")
    eval_losses = metric_series(rows, "eval_loss")
    learning_rates = metric_series(rows, "learning_rate")
    grad_norms = metric_series(rows, "grad_norm")

    train_metrics = nested_value(run_summary, "train_metrics")
    eval_metrics = nested_value(run_summary, "eval_metrics")
    total_steps = first_present(
        (trainer_state, train_metrics, train_results, run_summary), "global_step", "total_steps", "step"
    )
    final_epoch = first_present((trainer_state, train_metrics, train_results, run_summary), "epoch", "final_epoch")
    if total_steps is None and rows:
        total_steps = max((row["step"] for row in rows if row["step"] is not None), default=None)
    if final_epoch is None and rows:
        final_epoch = next((row["epoch"] for row in reversed(rows) if row["epoch"] is not None), None)

    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "analysis_generated_at": datetime.now(timezone.utc).isoformat(),
        "input_files": {name: {"present": value is not None} for name, value in files.items()},
        "trainer_state_available_fields": sorted(trainer_state.keys()) if trainer_state else [],
        "trainer_log_history_entries": len(trainer_state.get("log_history", [])) if trainer_state else 0,
        "warnings": warnings,
        **loss_summary(train_losses, "train"),
        **loss_summary(eval_losses, "eval"),
        "total_steps": as_int(total_steps),
        "final_epoch": as_number(final_epoch),
        "elapsed_seconds": as_number(first_present((run_summary,), "elapsed_seconds")),
        "peak_gpu_memory_allocated_gib": as_number(
            first_present((run_summary,), "peak_gpu_memory_allocated_gib")
        ),
        "peak_gpu_memory_reserved_gib": as_number(
            first_present((run_summary,), "peak_gpu_memory_reserved_gib")
        ),
        "samples_per_second": as_number(
            first_present((run_summary, train_metrics, train_results), "samples_per_second", "train_samples_per_second")
        ),
        "steps_per_second": as_number(
            first_present((run_summary, train_metrics, train_results), "steps_per_second", "train_steps_per_second")
        ),
        "metric_points": {
            "train_loss": len(train_losses),
            "eval_loss": len(eval_losses),
            "learning_rate": len(learning_rates),
            "grad_norm": len(grad_norms),
        },
        "final_train_results": train_results,
        "final_eval_results": eval_results,
    }
    if "best_eval_step" not in summary:
        summary["best_eval_step"] = None

    atomic_write_csv(analysis_dir / "training_metrics.csv", rows)
    atomic_write_json(analysis_dir / "training_summary.json", summary)
    make_loss_plot(train_losses, eval_losses, analysis_dir / "loss_curve.png")
    make_metric_plot(
        learning_rates,
        analysis_dir / "learning_rate_curve.png",
        "Learning-rate schedule",
        "learning rate",
        "#2ca02c",
    )
    make_metric_plot(
        grad_norms,
        analysis_dir / "grad_norm_curve.png",
        "Gradient norm",
        "grad norm",
        "#9467bd",
    )

    print(
        "\n".join(
            [
                f"Train loss: {display_number(summary['first_train_loss'])} -> {display_number(summary['last_train_loss'])}",
                f"Eval loss: {display_number(summary['first_eval_loss'])} -> {display_number(summary['last_eval_loss'])}",
                f"Best eval loss: {display_number(summary['min_eval_loss'])} @ step {summary['best_eval_step'] if summary['best_eval_step'] is not None else 'N/A'}",
                f"Peak GPU memory: {display_number(summary['peak_gpu_memory_allocated_gib'])} GiB",
                f"Total steps: {summary['total_steps'] if summary['total_steps'] is not None else 'N/A'}",
                f"Elapsed: {display_number(summary['elapsed_seconds'], 1)} s",
                f"Analysis: {analysis_dir}",
            ]
        )
    )
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
