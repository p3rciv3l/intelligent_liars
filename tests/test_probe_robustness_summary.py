from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from intelligent_liars.probe_robustness_summary import (
    summarize_probe_robustness,
    write_probe_robustness_summaries,
)


SIGN_CONVENTION = "sklearn_logistic_coef_positive_points_honest_to_deceptive"
LABEL_CONVENTION = "HONEST=0, DECEPTIVE=1"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload))


def _write_result(
    path: Path,
    *,
    seed: int,
    general: list[tuple[str, float, float]],
    cross: list[tuple[str, str, float, float]],
    direction: list[float],
    layer: int = 21,
    regularization_c: float = 0.01,
) -> None:
    if len(cross) == 1 and cross[0][0] != cross[0][1]:
        train_task, test_task, balanced_accuracy, auc = cross[0]
        cross = [
            *cross,
            (test_task, train_task, balanced_accuracy, auc),
        ]
    _write_json(
        path,
        {
            "format": "qwen_answer_token_probe_results_v2",
            "input_kind": "pooled_feature_cache",
            "input_path": str(path.parent / "cache.h5"),
            "label_convention": LABEL_CONVENTION,
            "direction_sign_convention": SIGN_CONVENTION,
            "settings": {
                "random_seed": seed,
                "layers": [layer],
                "regularization_c": regularization_c,
                "tasks": ["task_a", "task_b"],
                "test_size": 0.25,
                "class_weight": "balanced",
                "solver": "liblinear",
                "max_iter": 1000,
                "general_task_class_cap": 1000,
                "train_general_domain_probe": True,
            },
            "general_domain": {
                "evaluations": [
                    {
                        "task": task,
                        "layer": layer,
                        "balanced_accuracy": balanced_accuracy,
                        "auc": auc,
                        "direction_sign_convention": SIGN_CONVENTION,
                    }
                    for task, balanced_accuracy, auc in general
                ],
                "directions": [
                    {
                        "task": "general_domain",
                        "layer": layer,
                        "direction_vector": direction,
                        "direction_sign_convention": SIGN_CONVENTION,
                    }
                ],
            },
            "cross_task": [
                {
                    "train_task": train_task,
                    "test_task": test_task,
                    "layer": layer,
                    "balanced_accuracy": balanced_accuracy,
                    "auc": auc,
                    "direction_sign_convention": SIGN_CONVENTION,
                }
                for train_task, test_task, balanced_accuracy, auc in cross
            ],
        },
    )


def _write_manifest(
    path: Path,
    runs: list[dict[str, object]] | None = None,
    *,
    candidates: list[dict[str, object]] | None = None,
) -> None:
    normalized_candidates = candidates or [
        {
            "id": "layer21_c001",
            "layer": 21,
            "regularization_c": 0.01,
            "tasks": ["task_a", "task_b"],
            "runs": runs,
        }
    ]
    jobs = []
    experiment_candidates = []
    seeds: set[int] = set()
    for candidate in normalized_candidates:
        candidate_runs = candidate["runs"]
        assert isinstance(candidate_runs, list)
        experiment_candidates.append(
            {
                "candidate_id": candidate["id"],
                "layer": candidate["layer"],
                "regularization_c": candidate["regularization_c"],
                "seed_zero_result_path": str(path.parent / "seed0.json"),
            }
        )
        for run in candidate_runs:
            assert isinstance(run, dict)
            seed = int(run["seed"])
            seeds.add(seed)
            result_path = path.parent / str(run["result_path"])
            jobs.append(
                {
                    "candidate_id": candidate["id"],
                    "layer": candidate["layer"],
                    "regularization_c": candidate["regularization_c"],
                    "seed": seed,
                    "tasks": candidate["tasks"],
                    "output_path": str(result_path),
                    "reuse_existing_seed_zero": bool(
                        run.get("referenced_existing", False)
                    ),
                }
            )
    _write_json(
        path,
        {
            "format": "intelligent_liars_probe_seed_robustness_manifest_v1",
            "identity": "synthetic-manifest",
            "experiment": {
                "cache_path": str(path.parent / "cache.h5"),
                "tasks": ["task_a", "task_b"],
                "seeds": sorted(seeds),
                "test_size": 0.25,
                "max_iter": 1000,
                "general_task_class_cap": 1000,
                "command_prefix": ["uv", "run", "--no-sync", "intelligent-liars"],
                "candidates": experiment_candidates,
            },
            "jobs": jobs,
        },
    )


def test_summarizes_candidate_metrics_across_seeds_with_literal_values(
    tmp_path: Path,
) -> None:
    _write_result(
        tmp_path / "seed0.json",
        seed=0,
        general=[("task_a", 0.8, 0.9), ("task_b", 0.6, 0.7)],
        cross=[("task_a", "task_b", 0.5, 0.6), ("task_b", "task_a", 0.7, 0.8)],
        direction=[1.0, 0.0],
    )
    _write_result(
        tmp_path / "seed1.json",
        seed=1,
        general=[("task_a", 0.9, 1.0), ("task_b", 0.7, 0.8)],
        cross=[("task_a", "task_b", 0.7, 0.8), ("task_b", "task_a", 0.9, 1.0)],
        direction=[1.0, 0.0],
    )
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        [
            {"seed": 0, "result_path": "seed0.json", "referenced_existing": True},
            {"seed": 1, "result_path": "seed1.json"},
        ],
    )

    summary = summarize_probe_robustness(manifest_path)

    assert summary["status"] == "complete"
    assert summary["label_convention"] == LABEL_CONVENTION
    assert summary["direction_sign_convention"] == SIGN_CONVENTION
    assert summary["valid_run_count"] == 2
    assert summary["invalid_run_count"] == 0
    assert summary["selection_interpretation"] == {
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
        "provisional_leader": "layer21_c001",
        "provisional_leader_top_2_frequency": {
            "count": 2,
            "total": 2,
            "rate": 1.0,
        },
    }
    assert summary["candidates"][0]["metrics"] == {
        "general_balanced_accuracy": {"mean": 0.75, "std": 0.05, "worst": 0.7},
        "general_auc": {"mean": 0.85, "std": 0.05, "worst": 0.8},
        "cross_balanced_accuracy": {"mean": 0.7, "std": 0.1, "worst": 0.6},
        "cross_auc": {"mean": 0.8, "std": 0.1, "worst": 0.7},
    }


def test_reports_per_seed_rank_and_top_two_frequency(tmp_path: Path) -> None:
    scores = {
        "candidate_a": {0: 0.9, 1: 0.6},
        "candidate_b": {0: 0.8, 1: 0.9},
        "candidate_c": {0: 0.7, 1: 0.8},
    }
    candidates: list[dict[str, object]] = []
    for candidate_id, seed_scores in scores.items():
        runs: list[dict[str, object]] = []
        for seed, score in seed_scores.items():
            result_name = f"{candidate_id}_seed{seed}.json"
            _write_result(
                tmp_path / result_name,
                seed=seed,
                general=[("task_a", score, score), ("task_b", score, score)],
                cross=[("task_a", "task_b", 0.5, 0.5)],
                direction=[1.0, 0.0],
            )
            runs.append({"seed": seed, "result_path": result_name})
        candidates.append(
            {
                "id": candidate_id,
                "layer": 21,
                "regularization_c": 0.01,
                "tasks": ["task_a", "task_b"],
                "runs": runs,
            }
        )
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, candidates=candidates)

    summary = summarize_probe_robustness(manifest_path)

    by_id = {candidate["id"]: candidate for candidate in summary["candidates"]}
    assert [(row["seed"], row["rank"]) for row in by_id["candidate_a"]["seeds"]] == [
        (0, 1),
        (1, 3),
    ]
    assert [(row["seed"], row["rank"]) for row in by_id["candidate_b"]["seeds"]] == [
        (0, 2),
        (1, 1),
    ]
    assert [(row["seed"], row["rank"]) for row in by_id["candidate_c"]["seeds"]] == [
        (0, 3),
        (1, 2),
    ]
    assert by_id["candidate_a"]["top_2_frequency"] == {
        "count": 1,
        "total": 2,
        "rate": 0.5,
    }
    assert by_id["candidate_b"]["top_2_frequency"] == {
        "count": 2,
        "total": 2,
        "rate": 1.0,
    }
    assert by_id["candidate_c"]["top_2_frequency"] == {
        "count": 1,
        "total": 2,
        "rate": 0.5,
    }


def test_reports_per_task_worst_cases_with_seed_and_cross_source(
    tmp_path: Path,
) -> None:
    _write_result(
        tmp_path / "seed0.json",
        seed=0,
        general=[("task_a", 0.8, 0.9), ("task_b", 0.6, 0.7)],
        cross=[("task_a", "task_b", 0.5, 0.6), ("task_b", "task_a", 0.7, 0.8)],
        direction=[1.0, 0.0],
    )
    _write_result(
        tmp_path / "seed1.json",
        seed=1,
        general=[("task_a", 0.9, 1.0), ("task_b", 0.7, 0.8)],
        cross=[("task_a", "task_b", 0.7, 0.8), ("task_b", "task_a", 0.9, 1.0)],
        direction=[1.0, 0.0],
    )
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        [
            {"seed": 0, "result_path": "seed0.json"},
            {"seed": 1, "result_path": "seed1.json"},
        ],
    )

    summary = summarize_probe_robustness(manifest_path)

    assert summary["candidates"][0]["per_task_worst_cases"] == [
        {
            "task": "task_a",
            "general_balanced_accuracy": {"value": 0.8, "seed": 0},
            "general_auc": {"value": 0.9, "seed": 0},
            "cross_balanced_accuracy": {
                "value": 0.7,
                "seed": 0,
                "train_task": "task_b",
            },
            "cross_auc": {"value": 0.8, "seed": 0, "train_task": "task_b"},
        },
        {
            "task": "task_b",
            "general_balanced_accuracy": {"value": 0.6, "seed": 0},
            "general_auc": {"value": 0.7, "seed": 0},
            "cross_balanced_accuracy": {
                "value": 0.5,
                "seed": 0,
                "train_task": "task_a",
            },
            "cross_auc": {"value": 0.6, "seed": 0, "train_task": "task_a"},
        },
    ]


def test_reports_raw_and_explicit_seed_zero_aligned_direction_cosines(
    tmp_path: Path,
) -> None:
    directions = {0: [1.0, 0.0], 1: [-1.0, 0.0], 2: [-1.0, -1.0]}
    runs: list[dict[str, object]] = []
    for seed, direction in directions.items():
        result_name = f"seed{seed}.json"
        _write_result(
            tmp_path / result_name,
            seed=seed,
            general=[("task_a", 0.8, 0.9), ("task_b", 0.7, 0.8)],
            cross=[("task_a", "task_b", 0.6, 0.7)],
            direction=direction,
        )
        runs.append(
            {
                "seed": seed,
                "result_path": result_name,
                "referenced_existing": seed == 0,
            }
        )
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, runs)

    summary = summarize_probe_robustness(manifest_path)

    assert summary["candidates"][0]["direction_cosines"] == {
        "sign_convention": SIGN_CONVENTION,
        "reference_seed": 0,
        "seed_0_alignment_multipliers": {"0": 1, "1": -1, "2": -1},
        "pairs": [
            {
                "seed_a": 0,
                "seed_b": 1,
                "raw_cosine": -1.0,
                "seed_0_aligned_cosine": 1.0,
            },
            {
                "seed_a": 0,
                "seed_b": 2,
                "raw_cosine": -0.707106781187,
                "seed_0_aligned_cosine": 0.707106781187,
            },
            {
                "seed_a": 1,
                "seed_b": 2,
                "raw_cosine": 0.707106781187,
                "seed_0_aligned_cosine": 0.707106781187,
            },
        ],
    }


def test_missing_and_settings_mismatched_results_are_explicit(tmp_path: Path) -> None:
    _write_result(
        tmp_path / "seed0.json",
        seed=0,
        general=[("task_a", 0.8, 0.9), ("task_b", 0.7, 0.8)],
        cross=[("task_a", "task_b", 0.6, 0.7)],
        direction=[1.0, 0.0],
    )
    _write_result(
        tmp_path / "seed2.json",
        seed=2,
        general=[("task_a", 0.8, 0.9), ("task_b", 0.7, 0.8)],
        cross=[("task_a", "task_b", 0.6, 0.7)],
        direction=[1.0, 0.0],
        regularization_c=0.03,
    )
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        [
            {"seed": 0, "result_path": "seed0.json", "referenced_existing": True},
            {"seed": 1, "result_path": "missing.json"},
            {"seed": 2, "result_path": "seed2.json"},
        ],
    )

    summary = summarize_probe_robustness(manifest_path)

    assert summary["status"] == "incomplete"
    assert summary["valid_run_count"] == 1
    assert summary["invalid_run_count"] == 2
    assert summary["issues"] == [
        {
            "candidate_id": "layer21_c001",
            "seed": 1,
            "result_path": str(tmp_path / "missing.json"),
            "code": "missing_result",
            "message": "result file does not exist",
        },
        {
            "candidate_id": "layer21_c001",
            "seed": 2,
            "result_path": str(tmp_path / "seed2.json"),
            "code": "settings_mismatch",
            "field": "settings.regularization_c",
            "expected": 0.01,
            "actual": 0.03,
            "message": "settings.regularization_c expected 0.01, got 0.03",
        },
    ]
    candidate = summary["candidates"][0]
    assert candidate["status"] == "incomplete"
    assert candidate["planned_run_count"] == 3
    assert candidate["valid_run_count"] == 1
    assert candidate["invalid_run_count"] == 2
    assert [(row["seed"], row["status"]) for row in candidate["seeds"]] == [
        (0, "valid"),
        (1, "missing"),
        (2, "invalid"),
    ]


def test_rejects_a_direction_with_the_wrong_stored_sign_convention(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "seed0.json"
    _write_result(
        result_path,
        seed=0,
        general=[("task_a", 0.8, 0.9), ("task_b", 0.7, 0.8)],
        cross=[("task_a", "task_b", 0.6, 0.7)],
        direction=[1.0, 0.0],
    )
    payload = json.loads(result_path.read_text())
    payload["general_domain"]["directions"][0]["direction_sign_convention"] = "reversed"
    _write_json(result_path, payload)
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        [{"seed": 0, "result_path": "seed0.json", "referenced_existing": True}],
    )

    summary = summarize_probe_robustness(manifest_path)

    assert summary["status"] == "incomplete"
    assert summary["issues"] == [
        {
            "candidate_id": "layer21_c001",
            "seed": 0,
            "result_path": str(result_path),
            "code": "sign_convention_mismatch",
            "field": "general_domain.directions[0].direction_sign_convention",
            "expected": SIGN_CONVENTION,
            "actual": "reversed",
            "message": (
                "general_domain.directions[0].direction_sign_convention expected "
                f"{SIGN_CONVENTION!r}, got 'reversed'"
            ),
        }
    ]


def test_writes_compact_json_and_markdown_summaries(tmp_path: Path) -> None:
    for seed, general_score in [(0, 0.7), (1, 0.8)]:
        _write_result(
            tmp_path / f"seed{seed}.json",
            seed=seed,
            general=[
                ("task_a", general_score + 0.1, 0.9),
                ("task_b", general_score - 0.1, 0.7),
            ],
            cross=[("task_a", "task_b", 0.6 + seed * 0.2, 0.7 + seed * 0.2)],
            direction=[1.0, 0.0],
        )
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        [
            {"seed": 0, "result_path": "seed0.json", "referenced_existing": True},
            {"seed": 1, "result_path": "seed1.json"},
        ],
    )
    json_output = tmp_path / "summary.json"
    markdown_output = tmp_path / "summary.md"

    summary = write_probe_robustness_summaries(
        manifest_path,
        json_output=json_output,
        markdown_output=markdown_output,
    )

    assert json.loads(json_output.read_text()) == summary
    markdown = markdown_output.read_text()
    assert markdown.startswith("# Probe Seed Robustness Summary\n")
    assert "Status: `complete` — 2 valid runs, 0 invalid runs." in markdown
    assert "0.7500 ± 0.0500 (worst 0.7000)" in markdown
    assert "| `layer21_c001` | 0 | 1 |" in markdown
    assert "| `task_b` | 0.6000 (seed 0) |" in markdown
    assert "| 0 | 1 | 1 | 1 | 1.0000 | 1.0000 |" in markdown
    assert "Stored direction vectors are never modified" in markdown
    assert "Formal pass/fail verdict: **not available**" in markdown


def test_cli_writes_both_summary_files(tmp_path: Path) -> None:
    _write_result(
        tmp_path / "seed0.json",
        seed=0,
        general=[("task_a", 0.8, 0.9), ("task_b", 0.7, 0.8)],
        cross=[("task_a", "task_b", 0.6, 0.7)],
        direction=[1.0, 0.0],
    )
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        [{"seed": 0, "result_path": "seed0.json", "referenced_existing": True}],
    )
    json_output = tmp_path / "cli-summary.json"
    markdown_output = tmp_path / "cli-summary.md"
    script_path = (
        Path(__file__).parents[1] / "scripts/summarize_probe_seed_robustness.py"
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--manifest",
            str(manifest_path),
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(json_output.read_text())["status"] == "complete"
    assert markdown_output.read_text().startswith("# Probe Seed Robustness Summary\n")
    assert "wrote 1 valid runs and 0 invalid runs" in completed.stdout


def test_invalid_direction_vector_is_reported_instead_of_crashing(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "seed0.json"
    _write_result(
        result_path,
        seed=0,
        general=[("task_a", 0.8, 0.9), ("task_b", 0.7, 0.8)],
        cross=[("task_a", "task_b", 0.6, 0.7)],
        direction=[0.0, 0.0],
    )
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        [{"seed": 0, "result_path": "seed0.json", "referenced_existing": True}],
    )

    summary = summarize_probe_robustness(manifest_path)

    assert summary["status"] == "incomplete"
    assert summary["issues"] == [
        {
            "candidate_id": "layer21_c001",
            "seed": 0,
            "result_path": str(result_path),
            "code": "invalid_direction",
            "field": "general_domain.directions[0].direction_vector",
            "message": "direction vector must be a non-zero finite numeric list",
        }
    ]


def test_invalid_metric_is_reported_instead_of_crashing(tmp_path: Path) -> None:
    result_path = tmp_path / "seed0.json"
    _write_result(
        result_path,
        seed=0,
        general=[("task_a", 0.8, 0.9), ("task_b", 0.7, 0.8)],
        cross=[("task_a", "task_b", 0.6, 0.7)],
        direction=[1.0, 0.0],
    )
    payload = json.loads(result_path.read_text())
    payload["general_domain"]["evaluations"][0]["auc"] = float("nan")
    _write_json(result_path, payload)
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        [{"seed": 0, "result_path": "seed0.json", "referenced_existing": True}],
    )

    summary = summarize_probe_robustness(manifest_path)

    assert summary["status"] == "incomplete"
    assert summary["valid_run_count"] == 0
    assert summary["issues"][0]["code"] == "invalid_metric"
    assert summary["issues"][0]["field"] == "general_domain.evaluations[0].auc"


def test_missing_cross_task_pair_is_reported(tmp_path: Path) -> None:
    result_path = tmp_path / "seed0.json"
    _write_result(
        result_path,
        seed=0,
        general=[("task_a", 0.8, 0.9), ("task_b", 0.7, 0.8)],
        cross=[("task_a", "task_b", 0.6, 0.7)],
        direction=[1.0, 0.0],
    )
    payload = json.loads(result_path.read_text())
    payload["cross_task"] = payload["cross_task"][:1]
    _write_json(result_path, payload)
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        [{"seed": 0, "result_path": "seed0.json", "referenced_existing": True}],
    )

    summary = summarize_probe_robustness(manifest_path)

    assert summary["status"] == "incomplete"
    assert summary["issues"][0]["code"] == "cross_task_coverage_mismatch"
