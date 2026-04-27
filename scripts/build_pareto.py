"""
Build the Phase C Pareto chart and backing CSV.

Reads results/results.csv, results/latency/*.json, and student summary_int8.json
files. Writes docs/figures/pareto.png and docs/figures/pareto_data.csv.
matplotlib must already be installed; this script does not add dependencies.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        default="results",
        help="Directory containing results.csv and latency files.",
    )
    parser.add_argument(
        "--figures-dir",
        default="docs/figures",
        help="Output directory for pareto.png and pareto_data.csv.",
    )
    return parser.parse_args()


def load_results_csv(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing results CSV: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {
            row["experiment_id"]: row
            for row in csv.DictReader(handle)
            if row.get("experiment_id")
        }


def load_latency(latency_dir: Path) -> dict[str, dict[str, Any]]:
    payloads = {}
    if not latency_dir.exists():
        return payloads
    for path in latency_dir.glob("*.json"):
        with path.open("r", encoding="utf-8") as handle:
            payloads[path.stem] = json.load(handle)
    return payloads


def load_int8_summaries(results_root: Path) -> dict[str, dict[str, Any]]:
    summaries = {}
    for path in results_root.glob("*/summary_int8.json"):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        summaries[payload.get("run_id", path.parent.name + "_int8")] = payload
    return summaries


def as_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def latency_ms(latency: dict[str, dict[str, Any]], run_name: str) -> float | None:
    payload = latency.get(run_name)
    if not payload:
        return None
    return as_float(payload.get("latency", {}).get("1", {}).get("median_ms"))


def throughput(latency: dict[str, dict[str, Any]], run_name: str) -> float | None:
    payload = latency.get(run_name)
    if not payload:
        return None
    return as_float(
        payload.get("latency", {}).get("8", {}).get("examples_per_second_median")
    )


def checkpoint_mb(latency: dict[str, dict[str, Any]], run_name: str) -> float | None:
    payload = latency.get(run_name)
    if not payload:
        return None
    return as_float(payload.get("checkpoint_mb"))


def add_point(
    points: list[dict[str, Any]],
    *,
    name: str,
    label: str,
    kind: str,
    f1: float | None,
    latency_ms_bs1: float | None,
    throughput_bs8: float | None,
    size_mb: float | None,
) -> None:
    if f1 is None:
        return
    points.append(
        {
            "name": name,
            "label": label,
            "kind": kind,
            "f1": f1,
            "latency_ms_bs1": latency_ms_bs1,
            "throughput_bs8": throughput_bs8,
            "size_mb": size_mb,
        }
    )


def build_points(
    results: dict[str, dict[str, str]],
    latency: dict[str, dict[str, Any]],
    int8: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []

    teacher_single = "efficient_after_dapt_seed88"
    teacher_ensemble = "efficient_after_dapt_logit_ensemble"
    student_single = "student_distilled_seed88"
    student_ensemble = "student_distilled_logit_ensemble"
    int8_single = "student_distilled_seed88_int8"
    int8_ensemble = "student_distilled_int8_logit_ensemble"

    add_point(
        points,
        name=teacher_single,
        label="Teacher single",
        kind="teacher",
        f1=as_float(results.get(teacher_single, {}).get("test_entity_f1")),
        latency_ms_bs1=latency_ms(latency, teacher_single),
        throughput_bs8=throughput(latency, teacher_single),
        size_mb=checkpoint_mb(latency, teacher_single),
    )

    teacher_single_latency = latency_ms(latency, teacher_single)
    teacher_single_size = checkpoint_mb(latency, teacher_single)
    add_point(
        points,
        name=teacher_ensemble,
        label="Teacher logit ensemble",
        kind="teacher_ensemble",
        f1=as_float(results.get(teacher_ensemble, {}).get("test_entity_f1")),
        latency_ms_bs1=teacher_single_latency * 3 if teacher_single_latency else None,
        throughput_bs8=None,
        size_mb=teacher_single_size * 3 if teacher_single_size else None,
    )

    add_point(
        points,
        name=student_single,
        label="Student single",
        kind="student",
        f1=as_float(results.get(student_single, {}).get("test_entity_f1")),
        latency_ms_bs1=latency_ms(latency, student_single),
        throughput_bs8=throughput(latency, student_single),
        size_mb=checkpoint_mb(latency, student_single),
    )

    student_single_latency = latency_ms(latency, student_single)
    student_single_size = checkpoint_mb(latency, student_single)
    add_point(
        points,
        name=student_ensemble,
        label="Student logit ensemble",
        kind="student_ensemble",
        f1=as_float(results.get(student_ensemble, {}).get("test_entity_f1")),
        latency_ms_bs1=student_single_latency * 3 if student_single_latency else None,
        throughput_bs8=None,
        size_mb=student_single_size * 3 if student_single_size else None,
    )

    int8_summary = int8.get(int8_single)
    add_point(
        points,
        name=int8_single,
        label="INT8 student single",
        kind="student_int8",
        f1=as_float(
            int8_summary.get("test_metrics", {}).get("entity_overall_f1")
            if int8_summary
            else None
        ),
        latency_ms_bs1=latency_ms(latency, int8_single),
        throughput_bs8=throughput(latency, int8_single),
        size_mb=checkpoint_mb(latency, int8_single),
    )

    int8_single_latency = latency_ms(latency, int8_single)
    int8_single_size = checkpoint_mb(latency, int8_single)
    add_point(
        points,
        name=int8_ensemble,
        label="INT8 student ensemble",
        kind="student_int8_ensemble",
        f1=as_float(results.get(int8_ensemble, {}).get("test_entity_f1")),
        latency_ms_bs1=int8_single_latency * 3 if int8_single_latency else None,
        throughput_bs8=None,
        size_mb=int8_single_size * 3 if int8_single_size else None,
    )

    return points


def mark_frontier(points: list[dict[str, Any]], x_key: str, output_key: str) -> None:
    candidates = [point for point in points if point.get(x_key) is not None]
    candidates.sort(key=lambda point: (point[x_key], -point["f1"]))
    best_f1 = float("-inf")
    frontier_names = set()
    for point in candidates:
        if point["f1"] > best_f1:
            frontier_names.add(point["name"])
            best_f1 = point["f1"]
    for point in points:
        point[output_key] = point["name"] in frontier_names


def write_data_csv(points: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "name",
        "label",
        "kind",
        "f1",
        "latency_ms_bs1",
        "throughput_bs8",
        "size_mb",
        "latency_frontier",
        "size_frontier",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for point in points:
            writer.writerow({key: point.get(key, "") for key in fieldnames})


def plot_pareto(points: list[dict[str, Any]], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for scripts/build_pareto.py but is not installed. "
            "Install the existing project dependency before running this script."
        ) from exc

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].set_title("Latency vs Entity F1")
    axes[0].set_xlabel("Median latency, batch size 1 (ms)")
    axes[0].set_ylabel("Test entity F1")
    axes[1].set_title("Size vs Entity F1")
    axes[1].set_xlabel("Checkpoint size (MB)")
    axes[1].set_ylabel("Test entity F1")

    for point in points:
        if point.get("latency_ms_bs1") is not None:
            axes[0].scatter(
                point["latency_ms_bs1"],
                point["f1"],
                marker="*" if point.get("latency_frontier") else "o",
            )
            axes[0].annotate(point["label"], (point["latency_ms_bs1"], point["f1"]))
        if point.get("size_mb") is not None:
            axes[1].scatter(
                point["size_mb"],
                point["f1"],
                marker="*" if point.get("size_frontier") else "o",
            )
            axes[1].annotate(point["label"], (point["size_mb"], point["f1"]))

    for axis in axes:
        axis.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    results = load_results_csv(results_root / "results.csv")
    latency = load_latency(results_root / "latency")
    int8 = load_int8_summaries(results_root)
    points = build_points(results, latency, int8)
    mark_frontier(points, "latency_ms_bs1", "latency_frontier")
    mark_frontier(points, "size_mb", "size_frontier")

    data_path = figures_dir / "pareto_data.csv"
    plot_path = figures_dir / "pareto.png"
    write_data_csv(points, data_path)
    plot_pareto(points, plot_path)
    print(f"Wrote {data_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
