from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize probe sweep result JSON files.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Probe Sweep Summary")
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    args = parser.parse_args()

    payloads = _load_payloads(args.results_dir)
    if not payloads:
        raise SystemExit(f"No JSON result files found in {args.results_dir}")

    rows = [_summarize_payload(name, payload) for name, payload in payloads]
    rows.sort(
        key=lambda row: (
            _sort_none_last(row["general_bal_acc"]),
            _sort_none_last(row["general_auc"]),
            _sort_none_last(row["cross_bal_acc"]),
            _sort_none_last(row["cross_auc"]),
        ),
        reverse=True,
    )

    layer_rows = _summarize_by_layer(payloads)
    task_rows = _summarize_general_by_task(payloads)
    candidates = rows[:10]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        _render_markdown(
            title=args.title,
            created_at=datetime.now(UTC).isoformat(),
            source=args.source,
            cache=args.cache,
            rows=rows,
            candidates=candidates,
            layer_rows=layer_rows,
            task_rows=task_rows,
        )
    )
    print(f"wrote {args.output}")


def _load_payloads(results_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    payloads: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(results_dir.glob("*.json")):
        payloads.append((path.stem, json.loads(path.read_text())))
    return payloads


def _summarize_payload(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    settings = payload.get("settings", {})
    task_set = "no_repe" if name.startswith("no_repe") else "all_no_insider"
    return {
        "name": name,
        "task_set": task_set,
        "c": settings.get("regularization_c"),
        "layers": settings.get("layers", []),
        "tasks": settings.get("tasks", []),
        "within_bal_acc": _metric_mean(payload.get("within_task", []), "balanced_accuracy"),
        "within_auc": _metric_mean(payload.get("within_task", []), "auc"),
        "cross_bal_acc": _metric_mean(payload.get("cross_task", []), "balanced_accuracy"),
        "cross_auc": _metric_mean(payload.get("cross_task", []), "auc"),
        "general_bal_acc": _metric_mean(
            payload.get("general_domain", {}).get("evaluations", []),
            "balanced_accuracy",
        ),
        "general_auc": _metric_mean(
            payload.get("general_domain", {}).get("evaluations", []),
            "auc",
        ),
        "within_count": len(payload.get("within_task", [])),
        "cross_count": len(payload.get("cross_task", [])),
        "general_count": len(payload.get("general_domain", {}).get("evaluations", [])),
        "direction_count": len(payload.get("directions", [])),
        "general_direction_count": len(payload.get("general_domain", {}).get("directions", [])),
    }


def _summarize_by_layer(payloads: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for name, payload in payloads:
        task_set = "no_repe" if name.startswith("no_repe") else "all_no_insider"
        c_value = payload.get("settings", {}).get("regularization_c")
        for result in payload.get("general_domain", {}).get("evaluations", []):
            key = (task_set, c_value, int(result["layer"]))
            _append_metric(grouped[key], result, "balanced_accuracy")
            _append_metric(grouped[key], result, "auc")
        for result in payload.get("cross_task", []):
            key = (task_set, c_value, int(result["layer"]))
            _append_metric(grouped[key], result, "cross_balanced_accuracy", source_key="balanced_accuracy")
            _append_metric(grouped[key], result, "cross_auc", source_key="auc")

    rows: list[dict[str, Any]] = []
    for (task_set, c_value, layer), metrics in grouped.items():
        rows.append(
            {
                "task_set": task_set,
                "c": c_value,
                "layer": layer,
                "general_bal_acc": _mean_or_none(metrics["balanced_accuracy"]),
                "general_auc": _mean_or_none(metrics["auc"]),
                "cross_bal_acc": _mean_or_none(metrics["cross_balanced_accuracy"]),
                "cross_auc": _mean_or_none(metrics["cross_auc"]),
            }
        )
    rows.sort(
        key=lambda row: (
            _sort_none_last(row["general_bal_acc"]),
            _sort_none_last(row["general_auc"]),
            _sort_none_last(row["cross_bal_acc"]),
            _sort_none_last(row["cross_auc"]),
        ),
        reverse=True,
    )
    return rows


def _summarize_general_by_task(payloads: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    best: dict[tuple[str, str], tuple[float, str, int, float]] = {}
    for name, payload in payloads:
        task_set = "no_repe" if name.startswith("no_repe") else "all_no_insider"
        c_value = payload.get("settings", {}).get("regularization_c")
        for result in payload.get("general_domain", {}).get("evaluations", []):
            task = str(result["task"])
            key = (task_set, task)
            _append_metric(grouped[key], result, "balanced_accuracy")
            _append_metric(grouped[key], result, "auc")
            bal_acc = result.get("balanced_accuracy")
            if isinstance(bal_acc, int | float):
                current = best.get(key)
                if current is None or bal_acc > current[0]:
                    best[key] = (float(bal_acc), name, int(result["layer"]), float(c_value))

    rows: list[dict[str, Any]] = []
    for (task_set, task), metrics in grouped.items():
        best_entry = best.get((task_set, task))
        rows.append(
            {
                "task_set": task_set,
                "task": task,
                "mean_bal_acc": _mean_or_none(metrics["balanced_accuracy"]),
                "mean_auc": _mean_or_none(metrics["auc"]),
                "best_bal_acc": best_entry[0] if best_entry is not None else None,
                "best_result": best_entry[1] if best_entry is not None else "",
                "best_layer": best_entry[2] if best_entry is not None else None,
                "best_c": best_entry[3] if best_entry is not None else None,
            }
        )
    rows.sort(key=lambda row: (_sort_none_last(row["mean_bal_acc"]), row["task_set"]), reverse=True)
    return rows


def _metric_mean(results: list[dict[str, Any]], metric: str) -> float | None:
    return _mean_or_none([value for result in results if isinstance((value := result.get(metric)), int | float)])


def _append_metric(
    bucket: dict[str, list[float]],
    result: dict[str, Any],
    metric: str,
    *,
    source_key: str | None = None,
) -> None:
    value = result.get(source_key or metric)
    if isinstance(value, int | float):
        bucket[metric].append(float(value))


def _mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def _sort_none_last(value: float | None) -> float:
    return float("-inf") if value is None else value


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _render_markdown(
    *,
    title: str,
    created_at: str,
    source: Path | None,
    cache: Path | None,
    rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    layer_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
) -> str:
    lines = [
        f"# {title}",
        "",
        f"Created: `{created_at}`",
        "",
        "## Inputs",
        "",
    ]
    if source is not None:
        lines.append(f"- Source HDF5: `{source}`")
    if cache is not None:
        lines.append(f"- Dense cache: `{cache}`")
    lines.extend(
        [
            "- Layers: `15,16,17,18,19,20,21,22,23,24,25,26,27`",
            "- C values: `0.01,0.03,0.1,0.3,1,3,10`",
            "- Task sets: `all_no_insider`, `no_repe`",
            "",
            "## Candidate Ranking",
            "",
            "| Rank | Result | Task set | C | General bal acc | General AUC | Cross bal acc | Cross AUC | Within bal acc | Within AUC |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(candidates, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    f"`{row['name']}`",
                    row["task_set"],
                    _fmt(row["c"]),
                    _fmt(row["general_bal_acc"]),
                    _fmt(row["general_auc"]),
                    _fmt(row["cross_bal_acc"]),
                    _fmt(row["cross_auc"]),
                    _fmt(row["within_bal_acc"]),
                    _fmt(row["within_auc"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## All Sweep Jobs",
            "",
            "| Result | Task set | C | Tasks | Within | Cross | General | Directions | General directions |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['name']}`",
                    row["task_set"],
                    _fmt(row["c"]),
                    str(len(row["tasks"])),
                    str(row["within_count"]),
                    str(row["cross_count"]),
                    str(row["general_count"]),
                    str(row["direction_count"]),
                    str(row["general_direction_count"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Top Layer Configurations",
            "",
            "| Rank | Task set | C | Layer | General bal acc | General AUC | Cross bal acc | Cross AUC |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(layer_rows[:30], start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    row["task_set"],
                    _fmt(row["c"]),
                    str(row["layer"]),
                    _fmt(row["general_bal_acc"]),
                    _fmt(row["general_auc"]),
                    _fmt(row["cross_bal_acc"]),
                    _fmt(row["cross_auc"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## General-Domain Task Readout",
            "",
            "| Task set | Task | Mean bal acc | Mean AUC | Best bal acc | Best result | Best layer | Best C |",
            "|---|---|---:|---:|---:|---|---:|---:|",
        ]
    )
    for row in task_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["task_set"],
                    f"`{row['task']}`",
                    _fmt(row["mean_bal_acc"]),
                    _fmt(row["mean_auc"]),
                    _fmt(row["best_bal_acc"]),
                    f"`{row['best_result']}`" if row["best_result"] else "",
                    _fmt(row["best_layer"]),
                    _fmt(row["best_c"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Selection Rule",
            "",
            "Prefer configs with high general-domain balanced accuracy and AUC, then cross-task metrics, then stability across adjacent layers. Do not select a direction from within-task accuracy alone.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
