from __future__ import annotations

from dataclasses import replace
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import subprocess
import sys

import pytest

from intelligent_liars.probe_robustness_runner import (
    NO_REPE_TASKS,
    ManifestIdentityMismatch,
    OutputValidationError,
    ProbeRobustnessExperimentSpec,
    ProbeRobustnessRunFailed,
    build_job_manifest,
    ensure_manifest,
    run_probe_seed_robustness,
    validate_existing_output,
)


def _write_valid_result(job, spec: ProbeRobustnessExperimentSpec) -> None:
    job.output_path.parent.mkdir(parents=True, exist_ok=True)
    job.output_path.write_text(
        json.dumps(
            {
                "format": "qwen_answer_token_probe_results_v2",
                "input_kind": "pooled_feature_cache",
                "input_path": str(spec.cache_path.resolve()),
                "settings": {
                    "layers": [job.layer],
                    "regularization_c": job.regularization_c,
                    "tasks": list(job.tasks),
                    "random_seed": job.seed,
                    "test_size": spec.test_size,
                    "max_iter": spec.max_iter,
                    "train_general_domain_probe": True,
                    "general_task_class_cap": spec.general_task_class_cap,
                    "model": "sklearn.linear_model.LogisticRegression",
                    "solver": "liblinear",
                    "class_weight": "balanced",
                }
            }
        )
    )


def test_immutable_default_spec_builds_twenty_seed_jobs_with_seed_zero_reuse(
    tmp_path: Path,
) -> None:
    spec = ProbeRobustnessExperimentSpec.default(tmp_path)

    assert spec.default_max_parallel == 5
    with pytest.raises(FrozenInstanceError):
        spec.default_max_parallel = 99  # type: ignore[misc]

    manifest = build_job_manifest(spec)

    assert len(manifest.jobs) == 20
    assert {job.seed for job in manifest.jobs} == {0, 1, 2, 3, 4}
    assert {job.candidate_id for job in manifest.jobs} == {
        "no_repe_layer21_c001",
        "no_repe_layer21_c003",
        "no_repe_layer20_c003",
        "no_repe_layer19_c001",
    }
    assert all(job.tasks == NO_REPE_TASKS for job in manifest.jobs)

    seed_zero_jobs = [job for job in manifest.jobs if job.seed == 0]
    assert all(job.reuse_existing_seed_zero for job in seed_zero_jobs)
    assert {
        job.output_path.relative_to(tmp_path).as_posix()
        for job in seed_zero_jobs
    } == {
        "artifacts/probes/sweeps/dense_15_27_by_layer/no_repe_layer21_c001.json",
        "artifacts/probes/sweeps/dense_15_27_by_layer/no_repe_layer21_c003.json",
        "artifacts/probes/sweeps/dense_15_27_by_layer/no_repe_layer20_c003.json",
        "artifacts/probes/sweeps/dense_15_27_by_layer/no_repe_layer19_c001.json",
    }

    new_outputs = [job.output_path for job in manifest.jobs if job.seed > 0]
    assert len(set(new_outputs)) == 16
    assert all("seed_robustness_v1" in path.parts for path in new_outputs)
    assert all(
        job.output_path.relative_to(spec.results_dir).as_posix()
        == f"{job.candidate_id}/seed_{job.seed}.json"
        for job in manifest.jobs
        if job.seed > 0
    )


def test_manifest_resume_rejects_an_experiment_identity_mismatch(tmp_path: Path) -> None:
    spec = ProbeRobustnessExperimentSpec.default(tmp_path)
    manifest = build_job_manifest(spec)

    ensure_manifest(spec, manifest)
    first_contents = spec.manifest_path.read_text()
    ensure_manifest(spec, build_job_manifest(spec))

    assert spec.manifest_path.read_text() == first_contents

    tampered = json.loads(first_contents)
    tampered["jobs"][0]["seed"] = 99
    spec.manifest_path.write_text(json.dumps(tampered))
    with pytest.raises(ManifestIdentityMismatch, match="identity mismatch"):
        ensure_manifest(spec, build_job_manifest(spec))

    spec.manifest_path.write_text(first_contents)
    changed_spec = replace(spec, seeds=(0, 1, 2, 3, 4, 5))
    with pytest.raises(ManifestIdentityMismatch, match="identity mismatch"):
        ensure_manifest(changed_spec, build_job_manifest(changed_spec))


def test_existing_output_reuse_requires_matching_layer_c_tasks_and_seed(
    tmp_path: Path,
) -> None:
    spec = ProbeRobustnessExperimentSpec.default(tmp_path)
    job = next(job for job in build_job_manifest(spec).jobs if job.seed == 1)
    valid_settings = {
        "layers": [job.layer],
        "regularization_c": job.regularization_c,
        "tasks": list(job.tasks),
        "random_seed": job.seed,
        "test_size": spec.test_size,
        "max_iter": spec.max_iter,
        "train_general_domain_probe": True,
        "general_task_class_cap": spec.general_task_class_cap,
        "model": "sklearn.linear_model.LogisticRegression",
        "solver": "liblinear",
        "class_weight": "balanced",
    }
    job.output_path.parent.mkdir(parents=True)
    base_payload = {
        "format": "qwen_answer_token_probe_results_v2",
        "input_kind": "pooled_feature_cache",
        "input_path": str(spec.cache_path.resolve()),
        "settings": valid_settings,
    }
    job.output_path.write_text(json.dumps(base_payload))

    validate_existing_output(job, spec=spec)

    mismatches = {
        "layer": {**valid_settings, "layers": [job.layer + 1]},
        "C": {**valid_settings, "regularization_c": job.regularization_c * 10},
        "tasks": {**valid_settings, "tasks": list(job.tasks[:-1])},
        "seed": {**valid_settings, "random_seed": job.seed + 1},
        "test size": {**valid_settings, "test_size": 0.5},
        "solver": {**valid_settings, "solver": "saga"},
    }
    for field, settings in mismatches.items():
        job.output_path.write_text(json.dumps({**base_payload, "settings": settings}))
        with pytest.raises(OutputValidationError, match=field):
            validate_existing_output(job, spec=spec)


def test_dry_run_emits_sixteen_single_threaded_cli_commands_without_execution(
    tmp_path: Path,
) -> None:
    spec = ProbeRobustnessExperimentSpec.default(tmp_path)
    manifest = build_job_manifest(spec)
    for job in manifest.jobs:
        if job.seed == 0:
            _write_valid_result(job, spec)

    summary = run_probe_seed_robustness(spec, dry_run=True)

    assert summary.total_jobs == 20
    assert summary.reused_jobs == 4
    assert summary.completed_jobs == 0
    assert len(summary.commands) == 16
    for planned in summary.commands:
        assert planned.argv[:5] == (
            "uv",
            "run",
            "--no-sync",
            "intelligent-liars",
            "train-probes-from-cache",
        )
        assert "--general-domain-probe" in planned.argv
        assert planned.argv.count("--task") == 16
        assert dict(planned.environment) == {
            "BLIS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
        assert not planned.job.output_path.exists()

    events = [json.loads(line) for line in spec.events_path.read_text().splitlines()]
    assert [event["event"] for event in events].count("job_reused") == 4
    assert [event["event"] for event in events].count("job_dry_run") == 16
    assert events[0]["event"] == "run_started"
    assert events[-1]["event"] == "run_finished"


def test_one_selected_job_publishes_atomically_and_resumes_without_rerunning(
    tmp_path: Path,
) -> None:
    fake_probe = """
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
value = lambda flag: args[args.index(flag) + 1]
tasks = [args[index + 1] for index, arg in enumerate(args) if arg == "--task"]
payload = {
    "format": "qwen_answer_token_probe_results_v2",
    "input_kind": "pooled_feature_cache",
    "input_path": value("--cache"),
    "settings": {
        "layers": [int(value("--layers"))],
        "regularization_c": float(value("--c")),
        "tasks": tasks,
        "random_seed": int(value("--random-seed")),
        "test_size": float(value("--test-size")),
        "max_iter": int(value("--max-iter")),
        "train_general_domain_probe": "--general-domain-probe" in args,
        "general_task_class_cap": int(value("--general-task-class-cap")),
        "model": "sklearn.linear_model.LogisticRegression",
        "solver": "liblinear",
        "class_weight": "balanced",
    },
    "blas": {
        key: os.environ[key]
        for key in (
            "BLIS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
    },
}
Path(value("--output")).write_text(json.dumps(payload))
"""
    base = ProbeRobustnessExperimentSpec.default(tmp_path)
    spec = replace(base, command_prefix=(sys.executable, "-c", fake_probe))
    candidate_id = "no_repe_layer21_c001"

    first = run_probe_seed_robustness(
        spec,
        candidate_ids=(candidate_id,),
        seeds=(1,),
        max_parallel=1,
    )

    job = next(
        job
        for job in build_job_manifest(spec).jobs
        if job.candidate_id == candidate_id and job.seed == 1
    )
    first_contents = job.output_path.read_text()
    output = json.loads(first_contents)
    assert first.total_jobs == 1
    assert first.completed_jobs == 1
    assert first.reused_jobs == 0
    assert output["blas"] == dict(first.commands[0].environment)
    assert not list(job.output_path.parent.glob(f".{job.output_path.name}.tmp-*"))
    assert not list(job.output_path.parent.glob("*.lock"))

    resumed = run_probe_seed_robustness(
        spec,
        candidate_ids=(candidate_id,),
        seeds=(1,),
        max_parallel=1,
    )

    assert resumed.total_jobs == 1
    assert resumed.completed_jobs == 0
    assert resumed.reused_jobs == 1
    assert job.output_path.read_text() == first_contents
    events = [json.loads(line)["event"] for line in spec.events_path.read_text().splitlines()]
    assert "job_started" in events
    assert "job_succeeded" in events
    assert events[-1] == "run_finished"


def test_existing_per_job_lock_prevents_the_selected_job_from_starting(
    tmp_path: Path,
) -> None:
    base = ProbeRobustnessExperimentSpec.default(tmp_path)
    spec = replace(
        base,
        command_prefix=(sys.executable, "-c", "raise SystemExit('must not run')"),
    )
    candidate_id = "no_repe_layer21_c001"
    job = next(
        job
        for job in build_job_manifest(spec).jobs
        if job.candidate_id == candidate_id and job.seed == 2
    )
    job.lock_path.parent.mkdir(parents=True)
    job.lock_path.write_text("{}")

    with pytest.raises(ProbeRobustnessRunFailed, match="LockHeldError"):
        run_probe_seed_robustness(
            spec,
            candidate_ids=(candidate_id,),
            seeds=(2,),
            max_parallel=1,
        )

    assert not job.output_path.exists()
    events = [json.loads(line)["event"] for line in spec.events_path.read_text().splitlines()]
    assert "job_started" not in events
    assert events[-1] == "run_failed"


def test_cli_can_dry_run_one_candidate_seed_against_the_canonical_manifest(
    tmp_path: Path,
) -> None:
    script = Path(__file__).parents[1] / "scripts/run_probe_seed_robustness.py"
    candidate_id = "no_repe_layer20_c003"

    completed = subprocess.run(
        (
            sys.executable,
            str(script),
            "--project-root",
            str(tmp_path),
            "--dry-run",
            "--candidate",
            candidate_id,
            "--seed",
            "3",
            "--max-parallel",
            "2",
        ),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "selected=1 reused=0 completed=0 pending=1 max_parallel=2" in completed.stdout
    assert (
        f"artifacts/probes/robustness/seed_robustness_v1/"
        f"{candidate_id}/.seed_3.json.tmp-dry-run"
    ) in completed.stdout
    manifest_path = (
        tmp_path
        / "artifacts/probes/robustness/seed_robustness_v1/manifest.json"
    )
    assert len(json.loads(manifest_path.read_text())["jobs"]) == 20


def test_cli_can_exclude_an_active_job_without_changing_the_manifest(
    tmp_path: Path,
) -> None:
    script = Path(__file__).parents[1] / "scripts/run_probe_seed_robustness.py"
    candidate_id = "no_repe_layer21_c001"

    completed = subprocess.run(
        (
            sys.executable,
            str(script),
            "--project-root",
            str(tmp_path),
            "--dry-run",
            "--candidate",
            candidate_id,
            "--seed",
            "1",
            "--seed",
            "2",
            "--exclude-job",
            f"{candidate_id}:1",
        ),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "selected=1 reused=0 completed=0 pending=1 max_parallel=5" in completed.stdout
    assert f"{candidate_id}/.seed_2.json.tmp-dry-run" in completed.stdout
    assert f"{candidate_id}/.seed_1.json.tmp-dry-run" not in completed.stdout
    events_path = (
        tmp_path
        / "artifacts/probes/robustness/seed_robustness_v1/events.jsonl"
    )
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    excluded = [event for event in events if event["event"] == "job_excluded"]
    assert [(event["candidate_id"], event["seed"]) for event in excluded] == [
        (candidate_id, 1)
    ]
