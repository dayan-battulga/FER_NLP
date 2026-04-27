"""
Build the Phase C Pareto chart and backing CSV.

Reads results/results.csv, results/<run>/summary_int8.json,
results/*_aggregate.json, and results/latency/*.json. Writes
docs/figures/pareto.png and docs/figures/pareto_data.csv.

Headline F1 numbers use 3-seed means rather than seed 88 alone.
Latency uses seed 88 alone, since latency is determined by architecture not
weights, and measuring all three seeds wastes compute and clutters the chart.
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
        help="Directory containing results.csv, aggregate JSONs, and latency files.",
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


def load_aggregate_f1(results_root: Path, stem: str) -> float | None:
    """Read the 3-seed mean test entity F1 from <stem>_aggregate.json."""
    path = results_root / f"{stem}_aggregate.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    mean = (
        payload.get("test_metrics", {})
        .get("entity_overall_f1", {})
        .get("mean")
    )
    return float(mean) if mean is not None else None


def compute_int8_mean_f1(int8_summaries: dict[str, dict[str, Any]]) -> float | None:
    """Mean entity F1 across all student_distilled_seed*_int8 summaries."""
    f1_values: list[float] = []
    for run_id, payload in int8_summaries.items():
        if not run_id.startswith("student_distilled_seed"):
            continue
        f1 = (
            payload.get("test_metrics", {})
            .get("entity_overall_f1")
        )
        if f1 is not None:
            f1_values.append(float(f1))
    if not f1_values:
        return None
    return sum(f1_values) / len(f1_values)


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
    results_root: Path,
) -> list[dict[str, Any]]:
    """Build the Pareto data points using 3-seed means for F1."""
    points: list[dict[str, Any]] = []

    teacher_single = "efficient_after_dapt_seed88"
    teacher_ensemble = "efficient_after_dapt_logit_ensemble"
    student_single = "student_distilled_seed88"
    student_ensemble = "student_distilled_logit_ensemble"
    int8_single = "student_distilled_seed88_int8"

    teacher_mean_f1 = load_aggregate_f1(results_root, "efficient_after_dapt")
    student_mean_f1 = load_aggregate_f1(results_root, "student_distilled")
    int8_mean_f1 = compute_int8_mean_f1(int8)

    add_point(
        points,
        name=teacher_single,
        label="Teacher single (3-seed mean)",
        kind="teacher",
        f1=teacher_mean_f1
        if teacher_mean_f1 is not None
        else as_float(results.get(teacher_single, {}).get("test_entity_f1")),
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
        label="Student FP32 single (3-seed mean)",
        kind="student",
        f1=student_mean_f1
        if student_mean_f1 is not None
        else as_float(results.get(student_single, {}).get("test_entity_f1")),
        latency_ms_bs1=latency_ms(latency, student_single),
        throughput_bs8=throughput(latency, student_single),
        size_mb=checkpoint_mb(latency, student_single),
    )

    student_single_latency = latency_ms(latency, student_single)
    student_single_size = checkpoint_mb(latency, student_single)
    add_point(
        points,
        name=student_ensemble,
        label="Student FP32 logit ensemble",
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
        label="Student INT8 single (3-seed mean)",
        kind="student_int8",
        f1=int8_mean_f1
        if int8_mean_f1 is not None
        else as_float(
            int8_summary.get("test_metrics", {}).get("entity_overall_f1")
            if int8_summary
            else None
        ),
        latency_ms_bs1=latency_ms(latency, int8_single),
        throughput_bs8=throughput(latency, int8_single),
        size_mb=checkpoint_mb(latency, int8_single),
    )

    return points


def mark_frontier(points: list[dict[str, Any]], x_key: str, output_key: str) -> None:
    """Mark each point on/off the lower-x, higher-y Pareto frontier."""
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


# Color and marker scheme per point kind. Tuple is (color, marker, marker_size).
KIND_STYLE: dict[str, tuple[str, str, int]] = {
    "teacher": ("#c0392b", "s", 110),
    "teacher_ensemble": ("#c0392b", "*", 240),
    "student": ("#2980b9", "s", 110),
    "student_ensemble": ("#2980b9", "*", 240),
    "student_int8": ("#27ae60", "D", 100),
}


LATENCY_LABEL_OFFSETS: dict[str, tuple[int, int]] = {
    "teacher": (10, 8),
    "teacher_ensemble": (-82, -18),
    "student": (10, 24),
    "student_ensemble": (10, -26),
    "student_int8": (10, -34),
}


SIZE_LABEL_OFFSETS: dict[str, tuple[int, int]] = {
    "teacher": (10, 8),
    "teacher_ensemble": (-92, -18),
    "student": (10, 24),
    "student_ensemble": (10, -26),
    "student_int8": (10, -34),
}


def plot_pareto(points: list[dict[str, Any]], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for scripts/build_pareto.py but is not installed. "
            "Install the existing project dependency before running this script."
        ) from exc

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    f1_values = [point["f1"] for point in points if point.get("f1") is not None]
    if f1_values:
        f1_low = min(f1_values) - 0.01
        f1_high = max(f1_values) + 0.01
    else:
        f1_low, f1_high = 0.80, 0.87

    axes[0].set_title("Latency vs Entity F1 (Apple M2 Pro CPU)")
    axes[0].set_xlabel("Median latency, batch size 1 (ms)")
    axes[0].set_ylabel("Test entity F1")
    axes[0].set_ylim(f1_low, f1_high)
    latency_values = [
        point["latency_ms_bs1"]
        for point in points
        if point.get("latency_ms_bs1") is not None
    ]
    if latency_values:
        latency_low = min(latency_values)
        latency_high = max(latency_values)
        latency_span = max(latency_high - latency_low, 1.0)
        axes[0].set_xlim(
            max(0.0, latency_low - latency_span * 0.10),
            latency_high + latency_span * 0.18,
        )
        axes[0].xaxis.set_major_locator(MaxNLocator(nbins=7))

    axes[1].set_title("Size vs Entity F1")
    axes[1].set_xlabel("Checkpoint size on disk (MB)")
    axes[1].set_ylabel("Test entity F1")
    axes[1].set_ylim(f1_low, f1_high)
    size_values = [
        point["size_mb"]
        for point in points
        if point.get("size_mb") is not None
    ]
    if size_values:
        size_low = min(size_values)
        size_high = max(size_values)
        size_span = max(size_high - size_low, 1.0)
        axes[1].set_xlim(
            max(0.0, size_low - size_span * 0.10),
            size_high + size_span * 0.18,
        )
        axes[1].xaxis.set_major_locator(MaxNLocator(nbins=7))

    # Plot points by kind, then draw frontier lines on top.
    for axis_idx, (axis, x_key, frontier_key) in enumerate(
        [
            (axes[0], "latency_ms_bs1", "latency_frontier"),
            (axes[1], "size_mb", "size_frontier"),
        ]
    ):
        # Frontier line first so the points sit on top of it.
        frontier_points = sorted(
            [point for point in points if point.get(frontier_key) and point.get(x_key) is not None],
            key=lambda point: point[x_key],
        )
        if len(frontier_points) >= 2:
            axis.plot(
                [point[x_key] for point in frontier_points],
                [point["f1"] for point in frontier_points],
                color="#7f8c8d",
                linestyle="--",
                linewidth=1.2,
                alpha=0.7,
                zorder=1,
                label="Pareto frontier",
            )

        for point in points:
            x_value = point.get(x_key)
            if x_value is None:
                continue
            color, marker, marker_size = KIND_STYLE.get(
                point["kind"], ("#34495e", "o", 80)
            )
            axis.scatter(
                x_value,
                point["f1"],
                marker=marker,
                s=marker_size,
                c=color,
                edgecolors="black",
                linewidths=0.6,
                zorder=3,
            )
            offset_source = LATENCY_LABEL_OFFSETS if axis_idx == 0 else SIZE_LABEL_OFFSETS
            offset = offset_source.get(point["kind"], (8, 8))
            axis.annotate(
                point["label"],
                (x_value, point["f1"]),
                xytext=offset,
                textcoords="offset points",
                fontsize=8.5,
                zorder=4,
            )

        axis.grid(True, alpha=0.3, which="both")
        axis.legend(loc="lower right", fontsize=8)

    fig.suptitle(
        "FiNER-ORD Pareto Frontier: Teacher vs Distilled Student vs INT8 Quantized",
        y=1.02,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    results = load_results_csv(results_root / "results.csv")
    latency = load_latency(results_root / "latency")
    int8 = load_int8_summaries(results_root)
    points = build_points(results, latency, int8, results_root)
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