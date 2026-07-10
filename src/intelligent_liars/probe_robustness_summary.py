from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


METRIC_PATHS = {
    "general_balanced_accuracy": ("general_domain", "evaluations", "balanced_accuracy"),
    "general_auc": ("general_domain", "evaluations", "auc"),
    "cross_balanced_accuracy": ("cross_task", "balanced_accuracy"),
    "cross_auc": ("cross_task", "auc"),
}
MANIFEST_FORMAT = "intelligent_liars_probe_seed_robustness_manifest_v1"
DIRECTION_SIGN_CONVENTION = "sklearn_logistic_coef_positive_points_honest_to_deceptive"
LABEL_CONVENTION = "HONEST=0, DECEPTIVE=1"
PROBE_RESULT_FORMAT = "qwen_answer_token_probe_results_v2"


def summarize_probe_robustness(manifest_path: Path) -> dict[str, Any]:
    """Summarize the probe results referenced by a seed-robustness manifest."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    candidates: list[dict[str, Any]] = []
    valid_run_count = 0
    invalid_run_count = 0
    issues: list[dict[str, Any]] = []

    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError(
            f"Unsupported runner manifest format: {manifest.get('format')!r}"
        )

    jobs_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for job in manifest["jobs"]:
        jobs_by_candidate.setdefault(str(job["candidate_id"]), []).append(job)

    for candidate in manifest["experiment"]["candidates"]:
        candidate_id = str(candidate["candidate_id"])
        seed_rows: list[dict[str, Any]] = []
        for job in jobs_by_candidate[candidate_id]:
            result_path = Path(str(job["output_path"]))
            if not result_path.is_absolute():
                result_path = manifest_path.parent / result_path
            seed_row: dict[str, Any] = {
                "seed": int(job["seed"]),
                "result_path": str(result_path),
                "reused_seed_zero": bool(job.get("reuse_existing_seed_zero", False)),
            }
            if not result_path.is_file():
                issue = _run_issue(
                    job,
                    result_path,
                    code="missing_result",
                    message="result file does not exist",
                )
                issues.append(issue)
                seed_row.update({"status": "missing", "issues": [issue]})
                seed_rows.append(seed_row)
                invalid_run_count += 1
                continue
            try:
                payload = json.loads(result_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                issue = _run_issue(
                    job,
                    result_path,
                    code="invalid_result_json",
                    message=f"cannot read result JSON: {exc}",
                )
                issues.append(issue)
                seed_row.update({"status": "invalid", "issues": [issue]})
                seed_rows.append(seed_row)
                invalid_run_count += 1
                continue

            run_issues = _validate_result_settings(
                job, manifest["experiment"], payload, result_path
            )
            if run_issues:
                issues.extend(run_issues)
                seed_row.update({"status": "invalid", "issues": run_issues})
                seed_rows.append(seed_row)
                invalid_run_count += 1
                continue

            seed_row.update(
                {
                    "status": "valid",
                    "metrics": _extract_run_metrics(payload),
                    "_general_evaluations": payload["general_domain"]["evaluations"],
                    "_cross_task": payload["cross_task"],
                    "_direction": payload["general_domain"]["directions"][0],
                }
            )
            seed_rows.append(seed_row)
            valid_run_count += 1

        valid_seed_rows = [row for row in seed_rows if row["status"] == "valid"]
        candidate_invalid_count = len(seed_rows) - len(valid_seed_rows)
        candidates.append(
            {
                "id": candidate_id,
                "layer": int(candidate["layer"]),
                "regularization_c": float(candidate["regularization_c"]),
                "metrics": {
                    metric: _aggregate(
                        [row["metrics"][metric] for row in valid_seed_rows]
                    )
                    for metric in METRIC_PATHS
                },
                "seeds": seed_rows,
                "per_task_worst_cases": _per_task_worst_cases(valid_seed_rows),
                "direction_cosines": _direction_cosines(valid_seed_rows),
                "status": "complete" if candidate_invalid_count == 0 else "incomplete",
                "planned_run_count": len(seed_rows),
                "valid_run_count": len(valid_seed_rows),
                "invalid_run_count": candidate_invalid_count,
            }
        )

    _assign_seed_ranks(candidates)
    for candidate in candidates:
        valid_rows = [row for row in candidate["seeds"] if row["status"] == "valid"]
        top_two_count = sum(row["rank"] <= 2 for row in valid_rows)
        total = len(valid_rows)
        candidate["top_2_frequency"] = {
            "count": top_two_count,
            "total": total,
            "rate": _compact_float(top_two_count / total) if total else None,
        }
        for row in valid_rows:
            row.pop("_general_evaluations")
            row.pop("_cross_task")
            row.pop("_direction")

    provisional_leader = max(
        candidates,
        key=lambda candidate: tuple(
            candidate["metrics"][metric]["mean"] or float("-inf")
            for metric in METRIC_PATHS
        ),
    )

    return {
        "format": "intelligent_liars_probe_seed_robustness_summary_v1",
        "manifest_identity": manifest.get("identity"),
        "manifest_path": str(manifest_path),
        "metric_std": "population",
        "ranking_order": list(METRIC_PATHS),
        "label_convention": LABEL_CONVENTION,
        "direction_sign_convention": DIRECTION_SIGN_CONVENTION,
        "status": "complete" if invalid_run_count == 0 else "incomplete",
        "valid_run_count": valid_run_count,
        "invalid_run_count": invalid_run_count,
        "selection_interpretation": {
            "evidence_status": "descriptive_only_no_predeclared_pass_thresholds",
            "formal_pass_fail_verdict": None,
            "limitation": (
                "No machine-readable pass/fail thresholds were recorded before the "
                "robustness results were inspected; this summary must not retrofit them."
            ),
            "provenance_limitation": (
                "The immutable v1 manifest identifies the pooled cache by portable path, "
                "not by content hash. The separately tracked DVC pointer is the cache "
                "content source of truth."
            ),
            "provisional_leader": provisional_leader["id"],
            "provisional_leader_top_2_frequency": provisional_leader[
                "top_2_frequency"
            ],
        },
        "candidates": candidates,
        "issues": issues,
    }


def write_probe_robustness_summaries(
    manifest_path: Path,
    *,
    json_output: Path,
    markdown_output: Path,
) -> dict[str, Any]:
    """Write JSON and Markdown summaries for a runner manifest."""
    summary = summarize_probe_robustness(manifest_path)
    json_output = Path(json_output)
    markdown_output = Path(markdown_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    markdown_output.write_text(render_probe_robustness_markdown(summary))
    return summary


def render_probe_robustness_markdown(summary: dict[str, Any]) -> str:
    """Render the compact human-readable form of a robustness summary."""
    lines = [
        "# Probe Seed Robustness Summary",
        "",
        (
            f"Status: `{summary['status']}` — {summary['valid_run_count']} valid runs, "
            f"{summary['invalid_run_count']} invalid runs."
        ),
        "",
        "Population standard deviation is reported. Candidate ranking is lexicographic by "
        "general balanced accuracy, general AUC, cross-task balanced accuracy, then "
        "cross-task AUC.",
        "",
        f"Label convention: `{summary['label_convention']}`",
        "",
        f"Direction sign convention: `{summary['direction_sign_convention']}`",
        "",
        "## Selection interpretation",
        "",
        (
            "Formal pass/fail verdict: **not available**. No machine-readable "
            "pass/fail thresholds were recorded before these results were inspected, "
            "so this report does not retrofit them."
        ),
        "",
        (
            "Descriptively, the provisional leader is "
            f"`{summary['selection_interpretation']['provisional_leader']}`; this is "
            "evidence for the next staged ablation, not a final steering-direction "
            "selection."
        ),
        "",
        (
            "Provenance limitation: the immutable v1 manifest identifies the pooled "
            "cache by portable path rather than content hash; the separately tracked "
            "DVC pointer remains the cache content source of truth."
        ),
        "",
        "## Candidate aggregates",
        "",
        "| Candidate | Layer | C | General bal acc | General AUC | Cross bal acc | Cross AUC | Top-2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in summary["candidates"]:
        metrics = candidate["metrics"]
        top_two = candidate["top_2_frequency"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{candidate['id']}`",
                    str(candidate["layer"]),
                    _fmt_float(candidate["regularization_c"]),
                    _fmt_metric_summary(metrics["general_balanced_accuracy"]),
                    _fmt_metric_summary(metrics["general_auc"]),
                    _fmt_metric_summary(metrics["cross_balanced_accuracy"]),
                    _fmt_metric_summary(metrics["cross_auc"]),
                    (
                        f"{top_two['count']}/{top_two['total']} "
                        f"({_fmt_float(top_two['rate'])})"
                        if top_two["rate"] is not None
                        else ""
                    ),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Per-seed ranks",
            "",
            "| Candidate | Seed | Rank | Status | General bal acc | General AUC | Cross bal acc | Cross AUC |",
            "|---|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for candidate in summary["candidates"]:
        for seed_row in candidate["seeds"]:
            metrics = seed_row.get("metrics", {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{candidate['id']}`",
                        str(seed_row["seed"]),
                        str(seed_row.get("rank", "")),
                        seed_row["status"],
                        _fmt_float(metrics.get("general_balanced_accuracy")),
                        _fmt_float(metrics.get("general_auc")),
                        _fmt_float(metrics.get("cross_balanced_accuracy")),
                        _fmt_float(metrics.get("cross_auc")),
                    ]
                )
                + " |"
            )

    lines.extend(["", "## Per-task worst cases", ""])
    for candidate in summary["candidates"]:
        lines.extend(
            [
                f"### `{candidate['id']}`",
                "",
                "| Task | General bal acc | General AUC | Cross bal acc | Cross AUC |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in candidate["per_task_worst_cases"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{row['task']}`",
                        _fmt_worst(row["general_balanced_accuracy"]),
                        _fmt_worst(row["general_auc"]),
                        _fmt_worst(row["cross_balanced_accuracy"]),
                        _fmt_worst(row["cross_auc"]),
                    ]
                )
                + " |"
            )
        lines.append("")

    lines.extend(
        [
            "## Direction cosine",
            "",
            "Stored direction vectors are never modified. The aligned column applies the "
            "reported seed-0 alignment multipliers only for comparison.",
            "",
        ]
    )
    for candidate in summary["candidates"]:
        cosine = candidate["direction_cosines"]
        lines.extend(
            [
                f"### `{candidate['id']}`",
                "",
                f"Sign convention: `{cosine['sign_convention']}`",
                "",
                "| Seed A | Seed B | Align A | Align B | Raw cosine | Seed-0-aligned cosine |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        multipliers = cosine["seed_0_alignment_multipliers"]
        for pair in cosine["pairs"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(pair["seed_a"]),
                        str(pair["seed_b"]),
                        str(multipliers[str(pair["seed_a"])]),
                        str(multipliers[str(pair["seed_b"])]),
                        _fmt_float(pair["raw_cosine"]),
                        _fmt_float(pair["seed_0_aligned_cosine"]),
                    ]
                )
                + " |"
            )
        lines.append("")

    lines.extend(["## Missing or invalid results", ""])
    if not summary["issues"]:
        lines.append("None.")
    else:
        for issue in summary["issues"]:
            lines.append(
                f"- `{issue['candidate_id']}` seed `{issue['seed']}` "
                f"(`{issue['code']}`): {issue['message']} — `{issue['result_path']}`"
            )
    lines.append("")
    return "\n".join(lines)


def _extract_run_metrics(payload: dict[str, Any]) -> dict[str, float]:
    return {
        "general_balanced_accuracy": _metric_mean(
            payload["general_domain"]["evaluations"], "balanced_accuracy"
        ),
        "general_auc": _metric_mean(payload["general_domain"]["evaluations"], "auc"),
        "cross_balanced_accuracy": _metric_mean(
            payload["cross_task"], "balanced_accuracy"
        ),
        "cross_auc": _metric_mean(payload["cross_task"], "auc"),
    }


def _metric_mean(rows: list[dict[str, Any]], key: str) -> float:
    return mean(float(row[key]) for row in rows)


def _aggregate(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "worst": None}
    return {
        "mean": _compact_float(mean(values)),
        "std": _compact_float(pstdev(values)),
        "worst": _compact_float(min(values)),
    }


def _compact_float(value: float) -> float:
    return round(value, 12)


def _assign_seed_ranks(candidates: list[dict[str, Any]]) -> None:
    seeds = sorted(
        {
            row["seed"]
            for candidate in candidates
            for row in candidate["seeds"]
            if row["status"] == "valid"
        }
    )
    for seed in seeds:
        rows = [
            (candidate, row)
            for candidate in candidates
            for row in candidate["seeds"]
            if row["seed"] == seed and row["status"] == "valid"
        ]
        rows.sort(
            key=lambda item: (
                -item[1]["metrics"]["general_balanced_accuracy"],
                -item[1]["metrics"]["general_auc"],
                -item[1]["metrics"]["cross_balanced_accuracy"],
                -item[1]["metrics"]["cross_auc"],
                item[0]["id"],
            )
        )
        for rank, (_, row) in enumerate(rows, start=1):
            row["rank"] = rank

    for candidate in candidates:
        candidate["seeds"].sort(key=lambda row: row["seed"])


def _per_task_worst_cases(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    general_by_task: dict[str, list[dict[str, Any]]] = {}
    cross_by_task: dict[str, list[dict[str, Any]]] = {}
    for seed_row in seed_rows:
        seed = seed_row["seed"]
        for evaluation in seed_row["_general_evaluations"]:
            task = str(evaluation["task"])
            general_by_task.setdefault(task, []).append({**evaluation, "seed": seed})
        for evaluation in seed_row["_cross_task"]:
            task = str(evaluation.get("test_task", evaluation.get("task")))
            cross_by_task.setdefault(task, []).append({**evaluation, "seed": seed})

    rows: list[dict[str, Any]] = []
    for task in sorted(set(general_by_task) | set(cross_by_task)):
        general_rows = general_by_task.get(task, [])
        cross_rows = cross_by_task.get(task, [])
        rows.append(
            {
                "task": task,
                "general_balanced_accuracy": _worst_metric(
                    general_rows, "balanced_accuracy"
                ),
                "general_auc": _worst_metric(general_rows, "auc"),
                "cross_balanced_accuracy": _worst_metric(
                    cross_rows, "balanced_accuracy", include_train_task=True
                ),
                "cross_auc": _worst_metric(cross_rows, "auc", include_train_task=True),
            }
        )
    return rows


def _worst_metric(
    rows: list[dict[str, Any]], metric: str, *, include_train_task: bool = False
) -> dict[str, Any] | None:
    if not rows:
        return None
    worst = min(
        rows,
        key=lambda row: (
            float(row[metric]),
            int(row["seed"]),
            str(row.get("train_task", "")),
        ),
    )
    result: dict[str, Any] = {
        "value": _compact_float(float(worst[metric])),
        "seed": int(worst["seed"]),
    }
    if include_train_task:
        result["train_task"] = str(worst["train_task"])
    return result


def _direction_cosines(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not seed_rows:
        return {
            "sign_convention": None,
            "reference_seed": 0,
            "seed_0_alignment_multipliers": {},
            "pairs": [],
            "status": "unavailable",
        }
    directions = {
        int(row["seed"]): [
            float(value) for value in row["_direction"]["direction_vector"]
        ]
        for row in seed_rows
    }
    reference_seed = 0
    if reference_seed not in directions:
        return {
            "sign_convention": seed_rows[0]["_direction"]["direction_sign_convention"],
            "reference_seed": reference_seed,
            "seed_0_alignment_multipliers": {},
            "pairs": [],
            "status": "unavailable_without_seed_0",
        }
    reference = directions[reference_seed]
    multipliers = {
        seed: (-1 if _cosine(reference, direction) < 0 else 1)
        for seed, direction in directions.items()
    }
    pairs: list[dict[str, Any]] = []
    seeds = sorted(directions)
    for index, seed_a in enumerate(seeds):
        for seed_b in seeds[index + 1 :]:
            raw = _cosine(directions[seed_a], directions[seed_b])
            aligned = raw * multipliers[seed_a] * multipliers[seed_b]
            pairs.append(
                {
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "raw_cosine": _compact_float(raw),
                    "seed_0_aligned_cosine": _compact_float(aligned),
                }
            )
    return {
        "sign_convention": seed_rows[0]["_direction"]["direction_sign_convention"],
        "reference_seed": reference_seed,
        "seed_0_alignment_multipliers": {
            str(seed): multipliers[seed] for seed in sorted(multipliers)
        },
        "pairs": pairs,
    }


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot / (left_norm * right_norm)


def _validate_result_settings(
    job: dict[str, Any],
    experiment: dict[str, Any],
    payload: dict[str, Any],
    result_path: Path,
) -> list[dict[str, Any]]:
    settings = payload.get("settings")
    if not isinstance(settings, dict):
        return [
            _run_issue(
                job,
                result_path,
                code="invalid_result",
                message="result settings object is missing",
            )
        ]

    expected = {
        "layers": [job["layer"]],
        "regularization_c": job["regularization_c"],
        "tasks": job["tasks"],
        "random_seed": job["seed"],
        "test_size": experiment["test_size"],
        "max_iter": experiment["max_iter"],
        "general_task_class_cap": experiment["general_task_class_cap"],
        "class_weight": "balanced",
        "solver": "liblinear",
        "train_general_domain_probe": True,
    }
    run_issues: list[dict[str, Any]] = []
    for field, expected_value in expected.items():
        actual = settings.get(field)
        if actual != expected_value:
            qualified_field = f"settings.{field}"
            run_issues.append(
                _run_issue(
                    job,
                    result_path,
                    code="settings_mismatch",
                    message=f"{qualified_field} expected {expected_value!r}, got {actual!r}",
                    field=qualified_field,
                    expected=expected_value,
                    actual=actual,
                )
            )

    for field, expected_value in {
        "format": PROBE_RESULT_FORMAT,
        "input_kind": "pooled_feature_cache",
        "label_convention": LABEL_CONVENTION,
        "direction_sign_convention": DIRECTION_SIGN_CONVENTION,
    }.items():
        actual = payload.get(field)
        if actual != expected_value:
            run_issues.append(
                _run_issue(
                    job,
                    result_path,
                    code="sign_convention_mismatch"
                    if field == "direction_sign_convention"
                    else f"{field}_mismatch",
                    message=f"{field} expected {expected_value!r}, got {actual!r}",
                    field=field,
                    expected=expected_value,
                    actual=actual,
                )
            )

    expected_cache_path = Path(str(experiment["cache_path"])).resolve()
    actual_cache_path = Path(str(payload.get("input_path", ""))).resolve()
    if _portable_artifact_identity(actual_cache_path) != _portable_artifact_identity(
        expected_cache_path
    ):
        run_issues.append(
            _run_issue(
                job,
                result_path,
                code="input_path_mismatch",
                message=(
                    f"input_path expected {str(expected_cache_path)!r}, "
                    f"got {str(actual_cache_path)!r}"
                ),
                field="input_path",
                expected=str(expected_cache_path),
                actual=str(actual_cache_path),
            )
        )

    evaluation_tasks = settings.get("evaluation_tasks", settings.get("tasks"))
    if evaluation_tasks != job["tasks"]:
        run_issues.append(
            _run_issue(
                job,
                result_path,
                code="settings_mismatch",
                message=(
                    f"settings.evaluation_tasks expected {job['tasks']!r}, "
                    f"got {evaluation_tasks!r}"
                ),
                field="settings.evaluation_tasks",
                expected=job["tasks"],
                actual=evaluation_tasks,
            )
        )

    run_issues.extend(_validate_metric_rows(job, payload, result_path))

    try:
        direction = payload["general_domain"]["directions"][0]
        actual_sign_convention = direction["direction_sign_convention"]
    except (KeyError, IndexError, TypeError):
        run_issues.append(
            _run_issue(
                job,
                result_path,
                code="invalid_result",
                message="general_domain.directions[0] is missing or invalid",
            )
        )
        return run_issues

    if actual_sign_convention != DIRECTION_SIGN_CONVENTION:
        field = "general_domain.directions[0].direction_sign_convention"
        run_issues.append(
            _run_issue(
                job,
                result_path,
                code="sign_convention_mismatch",
                message=(
                    f"{field} expected {DIRECTION_SIGN_CONVENTION!r}, "
                    f"got {actual_sign_convention!r}"
                ),
                field=field,
                expected=DIRECTION_SIGN_CONVENTION,
                actual=actual_sign_convention,
            )
        )

    direction_vector = direction.get("direction_vector")
    if (
        not isinstance(direction_vector, list)
        or not direction_vector
        or any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            for value in direction_vector
        )
        or math.sqrt(sum(float(value) ** 2 for value in direction_vector)) == 0.0
    ):
        run_issues.append(
            _run_issue(
                job,
                result_path,
                code="invalid_direction",
                field="general_domain.directions[0].direction_vector",
                message="direction vector must be a non-zero finite numeric list",
            )
        )
    return run_issues


def _portable_artifact_identity(path: Path) -> tuple[str, ...]:
    parts = path.parts
    try:
        artifact_index = parts.index("artifacts")
    except ValueError:
        return parts
    return parts[artifact_index:]


def _validate_metric_rows(
    job: dict[str, Any], payload: dict[str, Any], result_path: Path
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    expected_tasks = [str(task) for task in job["tasks"]]
    try:
        general_rows = payload["general_domain"]["evaluations"]
        cross_rows = payload["cross_task"]
    except (KeyError, TypeError):
        return [
            _run_issue(
                job,
                result_path,
                code="invalid_metrics",
                message="general-domain or cross-task evaluation rows are missing",
            )
        ]
    if not isinstance(general_rows, list) or not isinstance(cross_rows, list):
        return [
            _run_issue(
                job,
                result_path,
                code="invalid_metrics",
                message="general-domain and cross-task evaluations must be lists",
            )
        ]

    general_tasks = [
        str(row.get("task")) for row in general_rows if isinstance(row, dict)
    ]
    if sorted(general_tasks) != sorted(expected_tasks):
        issues.append(
            _run_issue(
                job,
                result_path,
                code="evaluation_task_coverage_mismatch",
                message=(
                    "general-domain evaluations must contain each manifest task "
                    "exactly once"
                ),
            )
        )

    seen_cross_pairs: set[tuple[str, str]] = set()
    for section, rows in (("general_domain.evaluations", general_rows), ("cross_task", cross_rows)):
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                issues.append(
                    _run_issue(
                        job,
                        result_path,
                        code="invalid_metrics",
                        message=f"{section}[{index}] must be an object",
                    )
                )
                continue
            if section == "cross_task":
                pair = (str(row.get("train_task")), str(row.get("test_task")))
                if pair in seen_cross_pairs:
                    issues.append(
                        _run_issue(
                            job,
                            result_path,
                            code="duplicate_cross_task_evaluation",
                            message=f"cross-task pair {pair!r} appears more than once",
                        )
                    )
                seen_cross_pairs.add(pair)
                if pair[0] not in expected_tasks or pair[1] not in expected_tasks:
                    issues.append(
                        _run_issue(
                            job,
                            result_path,
                            code="evaluation_task_coverage_mismatch",
                            message=f"cross-task pair {pair!r} is outside the manifest tasks",
                        )
                    )
            for metric in ("balanced_accuracy", "auc"):
                value = row.get(metric)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int | float)
                    or not math.isfinite(float(value))
                    or not 0.0 <= float(value) <= 1.0
                ):
                    issues.append(
                        _run_issue(
                            job,
                            result_path,
                            code="invalid_metric",
                            message=(
                                f"{section}[{index}].{metric} must be a finite "
                                "number between 0 and 1"
                            ),
                            field=f"{section}[{index}].{metric}",
                            actual=value,
                        )
                    )
    if not cross_rows:
        issues.append(
            _run_issue(
                job,
                result_path,
                code="invalid_metrics",
                message="cross-task evaluations must not be empty",
            )
        )
    expected_cross_pairs = {
        (train_task, test_task)
        for train_task in expected_tasks
        for test_task in expected_tasks
        if train_task != test_task
    }
    if seen_cross_pairs != expected_cross_pairs:
        missing_pairs = sorted(expected_cross_pairs - seen_cross_pairs)
        issues.append(
            _run_issue(
                job,
                result_path,
                code="cross_task_coverage_mismatch",
                message=(
                    "cross-task evaluations do not exactly cover every ordered "
                    f"distinct task pair; missing={missing_pairs!r}"
                ),
            )
        )
    return issues


def _run_issue(
    job: dict[str, Any],
    result_path: Path,
    *,
    code: str,
    message: str,
    **fields: Any,
) -> dict[str, Any]:
    return {
        "candidate_id": str(job["candidate_id"]),
        "seed": int(job["seed"]),
        "result_path": str(result_path),
        "code": code,
        **fields,
        "message": message,
    }


def _fmt_float(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _fmt_metric_summary(metric: dict[str, float | None]) -> str:
    if metric["mean"] is None:
        return ""
    return (
        f"{_fmt_float(metric['mean'])} ± {_fmt_float(metric['std'])} "
        f"(worst {_fmt_float(metric['worst'])})"
    )


def _fmt_worst(worst: dict[str, Any] | None) -> str:
    if worst is None:
        return ""
    suffix = f", train `{worst['train_task']}`" if "train_task" in worst else ""
    return f"{_fmt_float(worst['value'])} (seed {worst['seed']}{suffix})"
