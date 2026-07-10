from __future__ import annotations

import json
import os
import shlex
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from intelligent_liars.probes import PROBE_RESULT_FORMAT
from intelligent_liars.run_control import acquire_lock, lock_payload, stable_sha256


MANIFEST_FORMAT = "intelligent_liars_probe_seed_robustness_manifest_v1"
SINGLE_THREAD_BLAS_ENVIRONMENT = (
    ("BLIS_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
    ("OMP_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("VECLIB_MAXIMUM_THREADS", "1"),
)


class ManifestIdentityMismatch(RuntimeError):
    """Raised when a manifest path belongs to a different experiment."""


class OutputValidationError(RuntimeError):
    """Raised when a result cannot be proven to belong to its job."""


class ProbeRobustnessRunFailed(RuntimeError):
    """Raised after one or more selected probe jobs fail."""


NO_REPE_TASKS = (
    "claims__definitional_gemini_600_full",
    "claims__evidential_gemini_600_full",
    "claims__fictional_gemini_600_full",
    "claims__logical_gemini_600_full",
    "ethics__commonsense",
    "geometry_of_truth__best",
    "geometry_of_truth__mixed",
    "internal_state__animals",
    "internal_state__cities",
    "internal_state__companies",
    "internal_state__elements",
    "internal_state__facts",
    "internal_state__inventions",
    "roleplaying__plain",
    "sandbagging_v2__wmdp_mmlu",
    "sycophancy__mmlu_stem_same_conf_all",
)


@dataclass(frozen=True)
class ProbeCandidate:
    candidate_id: str
    layer: int
    regularization_c: float
    seed_zero_result_path: Path


@dataclass(frozen=True)
class ProbeRobustnessExperimentSpec:
    project_root: Path
    cache_path: Path
    results_dir: Path
    manifest_path: Path
    events_path: Path
    candidates: tuple[ProbeCandidate, ...]
    tasks: tuple[str, ...] = NO_REPE_TASKS
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    default_max_parallel: int = 5
    test_size: float = 0.25
    max_iter: int = 1000
    general_task_class_cap: int = 1000
    command_prefix: tuple[str, ...] = (
        "uv",
        "run",
        "--no-sync",
        "intelligent-liars",
    )

    @classmethod
    def default(cls, project_root: Path) -> "ProbeRobustnessExperimentSpec":
        project_root = project_root.resolve()
        primary_dir = project_root / "artifacts/probes/sweeps/dense_15_27_by_layer"
        results_dir = (
            project_root / "artifacts/probes/robustness/seed_robustness_v1"
        )
        candidates = (
            ProbeCandidate(
                "no_repe_layer21_c001",
                21,
                0.01,
                primary_dir / "no_repe_layer21_c001.json",
            ),
            ProbeCandidate(
                "no_repe_layer21_c003",
                21,
                0.03,
                primary_dir / "no_repe_layer21_c003.json",
            ),
            ProbeCandidate(
                "no_repe_layer20_c003",
                20,
                0.03,
                primary_dir / "no_repe_layer20_c003.json",
            ),
            ProbeCandidate(
                "no_repe_layer19_c001",
                19,
                0.01,
                primary_dir / "no_repe_layer19_c001.json",
            ),
        )
        return cls(
            project_root=project_root,
            cache_path=project_root
            / "artifacts/probe_features/no_insider_dense_15_27_pooled.h5",
            results_dir=results_dir,
            manifest_path=results_dir / "manifest.json",
            events_path=results_dir / "events.jsonl",
            candidates=candidates,
        )


@dataclass(frozen=True)
class ProbeRobustnessJob:
    candidate_id: str
    layer: int
    regularization_c: float
    seed: int
    tasks: tuple[str, ...]
    output_path: Path
    lock_path: Path
    reuse_existing_seed_zero: bool


@dataclass(frozen=True)
class ProbeRobustnessManifest:
    identity: str
    jobs: tuple[ProbeRobustnessJob, ...]


def build_job_manifest(
    spec: ProbeRobustnessExperimentSpec,
) -> ProbeRobustnessManifest:
    jobs = []
    for candidate in spec.candidates:
        for seed in spec.seeds:
            reuse_seed_zero = seed == 0
            output_path = (
                candidate.seed_zero_result_path
                if reuse_seed_zero
                else spec.results_dir
                / candidate.candidate_id
                / f"seed_{seed}.json"
            )
            jobs.append(
                ProbeRobustnessJob(
                    candidate_id=candidate.candidate_id,
                    layer=candidate.layer,
                    regularization_c=candidate.regularization_c,
                    seed=seed,
                    tasks=spec.tasks,
                    output_path=output_path,
                    lock_path=output_path.parent / ".locks" / f"seed_{seed}.lock",
                    reuse_existing_seed_zero=reuse_seed_zero,
                )
            )
    return ProbeRobustnessManifest(
        identity=stable_sha256(_experiment_identity_payload(spec)),
        jobs=tuple(jobs),
    )


@dataclass(frozen=True)
class PlannedProbeCommand:
    job: ProbeRobustnessJob
    argv: tuple[str, ...]
    environment: tuple[tuple[str, str], ...] = SINGLE_THREAD_BLAS_ENVIRONMENT

    @property
    def shell_command(self) -> str:
        return " ".join(shlex.quote(part) for part in self.argv)


@dataclass(frozen=True)
class ProbeRobustnessRunSummary:
    manifest_identity: str
    total_jobs: int
    reused_jobs: int
    completed_jobs: int
    failed_jobs: int
    commands: tuple[PlannedProbeCommand, ...]


@dataclass(frozen=True)
class _WorkerResult:
    status: str
    error: str | None = None


def run_probe_seed_robustness(
    spec: ProbeRobustnessExperimentSpec,
    *,
    dry_run: bool = False,
    max_parallel: int | None = None,
    candidate_ids: tuple[str, ...] | None = None,
    seeds: tuple[int, ...] | None = None,
    excluded_jobs: tuple[tuple[str, int], ...] = (),
) -> ProbeRobustnessRunSummary:
    """Validate the experiment and execute or print its pending jobs."""

    manifest = build_job_manifest(spec)
    ensure_manifest(spec, manifest)
    concurrency = (
        spec.default_max_parallel if max_parallel is None else max_parallel
    )
    if concurrency < 1:
        raise ValueError("max_parallel must be positive")
    selected_jobs, omitted_jobs = _select_jobs(
        manifest,
        candidate_ids=candidate_ids,
        seeds=seeds,
        excluded_jobs=excluded_jobs,
    )

    _emit_event(
        spec.events_path,
        "run_started",
        manifest_identity=manifest.identity,
        dry_run=dry_run,
        max_parallel=concurrency,
        total_jobs=len(selected_jobs),
    )
    for job in omitted_jobs:
        _emit_job_event(spec.events_path, "job_excluded", job)
    reused_jobs = 0
    commands: list[PlannedProbeCommand] = []
    for job in selected_jobs:
        if job.output_path.exists():
            validate_existing_output(job, spec=spec)
            reused_jobs += 1
            _emit_job_event(spec.events_path, "job_reused", job)
            continue
        if job.reuse_existing_seed_zero:
            raise OutputValidationError(
                f"Required seed-0 source output is missing: {job.output_path}"
            )

        temp_output_path = job.output_path.with_name(
            f".{job.output_path.name}.tmp-dry-run"
            if dry_run
            else f".{job.output_path.name}.tmp-{uuid4().hex}"
        )
        planned = PlannedProbeCommand(
            job=job,
            argv=_build_train_command(spec, job, temp_output_path),
        )
        commands.append(planned)
        if dry_run:
            _emit_job_event(
                spec.events_path,
                "job_dry_run",
                job,
                command=planned.shell_command,
            )

    completed_jobs = 0
    failed_jobs = 0
    failures: list[str] = []
    if not dry_run and commands:
        with ProcessPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(
                    _execute_planned_job,
                    spec,
                    planned,
                    manifest.identity,
                ): planned
                for planned in commands
            }
            for future in as_completed(futures):
                planned = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = _WorkerResult(status="failed", error=repr(exc))
                if result.status == "completed":
                    completed_jobs += 1
                elif result.status == "reused":
                    reused_jobs += 1
                else:
                    failed_jobs += 1
                    failures.append(
                        f"{planned.job.candidate_id}/seed_{planned.job.seed}: "
                        f"{result.error or 'unknown failure'}"
                    )

    summary = ProbeRobustnessRunSummary(
        manifest_identity=manifest.identity,
        total_jobs=len(selected_jobs),
        reused_jobs=reused_jobs,
        completed_jobs=completed_jobs,
        failed_jobs=failed_jobs,
        commands=tuple(commands),
    )
    if failures:
        _emit_event(
            spec.events_path,
            "run_failed",
            manifest_identity=manifest.identity,
            failed_jobs=failed_jobs,
            failures=failures,
        )
        raise ProbeRobustnessRunFailed(
            f"{failed_jobs} probe robustness job(s) failed: {'; '.join(failures)}"
        )
    _emit_event(
        spec.events_path,
        "run_finished",
        manifest_identity=manifest.identity,
        dry_run=dry_run,
        reused_jobs=reused_jobs,
        completed_jobs=completed_jobs,
        pending_jobs=len(commands) if dry_run else 0,
    )
    return summary


def _select_jobs(
    manifest: ProbeRobustnessManifest,
    *,
    candidate_ids: tuple[str, ...] | None,
    seeds: tuple[int, ...] | None,
    excluded_jobs: tuple[tuple[str, int], ...],
) -> tuple[tuple[ProbeRobustnessJob, ...], tuple[ProbeRobustnessJob, ...]]:
    available_candidates = {job.candidate_id for job in manifest.jobs}
    available_seeds = {job.seed for job in manifest.jobs}
    requested_candidates = (
        available_candidates if candidate_ids is None else set(candidate_ids)
    )
    requested_seeds = available_seeds if seeds is None else set(seeds)
    unknown_candidates = requested_candidates - available_candidates
    unknown_seeds = requested_seeds - available_seeds
    if unknown_candidates:
        raise ValueError(
            f"Unknown candidate_ids: {sorted(unknown_candidates)}"
        )
    if unknown_seeds:
        raise ValueError(f"Unknown seeds: {sorted(unknown_seeds)}")
    available_job_keys = {
        (job.candidate_id, job.seed) for job in manifest.jobs
    }
    excluded_job_keys = set(excluded_jobs)
    unknown_exclusions = excluded_job_keys - available_job_keys
    if unknown_exclusions:
        raise ValueError(f"Unknown excluded jobs: {sorted(unknown_exclusions)}")
    filtered_jobs = tuple(
        job
        for job in manifest.jobs
        if job.candidate_id in requested_candidates and job.seed in requested_seeds
    )
    omitted_jobs = tuple(
        job
        for job in filtered_jobs
        if (job.candidate_id, job.seed) in excluded_job_keys
    )
    selected_jobs = tuple(
        job
        for job in filtered_jobs
        if (job.candidate_id, job.seed) not in excluded_job_keys
    )
    return selected_jobs, omitted_jobs


def _execute_planned_job(
    spec: ProbeRobustnessExperimentSpec,
    planned: PlannedProbeCommand,
    manifest_identity: str,
) -> _WorkerResult:
    job = planned.job
    job.output_path.parent.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(
        job.lock_path,
        lock_payload(
            run_id=uuid4().hex,
            queue_plan_id=manifest_identity,
            command=planned.shell_command,
            kind="probe_seed_robustness_job",
            extra={
                "candidate_id": job.candidate_id,
                "seed": job.seed,
                "output_path": str(job.output_path),
            },
        ),
    )
    temp_output_path = Path(planned.argv[planned.argv.index("--output") + 1])
    try:
        if job.output_path.exists():
            validate_existing_output(job, spec=spec)
            _emit_job_event(spec.events_path, "job_reused", job)
            return _WorkerResult(status="reused")

        _emit_job_event(
            spec.events_path,
            "job_started",
            job,
            worker_pid=os.getpid(),
            command=planned.shell_command,
        )
        environment = os.environ.copy()
        environment.update(dict(planned.environment))
        completed = subprocess.run(
            planned.argv,
            cwd=spec.project_root,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            error = (
                f"command exited {completed.returncode}: "
                f"{completed.stderr[-2000:]}"
            )
            _emit_job_event(
                spec.events_path,
                "job_failed",
                job,
                returncode=completed.returncode,
                error=error,
            )
            return _WorkerResult(status="failed", error=error)

        temp_job = ProbeRobustnessJob(
            candidate_id=job.candidate_id,
            layer=job.layer,
            regularization_c=job.regularization_c,
            seed=job.seed,
            tasks=job.tasks,
            output_path=temp_output_path,
            lock_path=job.lock_path,
            reuse_existing_seed_zero=False,
        )
        validate_existing_output(temp_job, spec=spec)
        try:
            os.link(temp_output_path, job.output_path)
        except FileExistsError:
            validate_existing_output(job, spec=spec)
            _emit_job_event(spec.events_path, "job_reused", job)
            return _WorkerResult(status="reused")

        _emit_job_event(
            spec.events_path,
            "job_succeeded",
            job,
            worker_pid=os.getpid(),
        )
        return _WorkerResult(status="completed")
    except Exception as exc:
        _emit_job_event(
            spec.events_path,
            "job_failed",
            job,
            error=repr(exc),
        )
        return _WorkerResult(status="failed", error=repr(exc))
    finally:
        temp_output_path.unlink(missing_ok=True)
        lock.release()


def ensure_manifest(
    spec: ProbeRobustnessExperimentSpec,
    manifest: ProbeRobustnessManifest,
) -> None:
    """Create the manifest once, or verify an identical resumable manifest."""

    document = _manifest_document(spec, manifest)
    if spec.manifest_path.exists():
        _validate_manifest_identity(spec.manifest_path, document)
        return

    spec.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = spec.manifest_path.with_name(
        f".{spec.manifest_path.name}.tmp-{uuid4().hex}"
    )
    temp_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    try:
        try:
            os.link(temp_path, spec.manifest_path)
        except FileExistsError:
            _validate_manifest_identity(spec.manifest_path, document)
    finally:
        temp_path.unlink(missing_ok=True)


def validate_existing_output(
    job: ProbeRobustnessJob, *, spec: ProbeRobustnessExperimentSpec
) -> None:
    """Validate the result identity fields required for safe reuse."""

    try:
        document = json.loads(job.output_path.read_text())
        settings = document["settings"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise OutputValidationError(
            f"Cannot validate existing output: {job.output_path}"
        ) from exc

    expected_settings = {
        "layers": [job.layer],
        "regularization_c": job.regularization_c,
        "tasks": list(job.tasks),
        "test_size": spec.test_size,
        "random_seed": job.seed,
        "max_iter": spec.max_iter,
        "train_general_domain_probe": True,
        "general_task_class_cap": spec.general_task_class_cap,
        "model": "sklearn.linear_model.LogisticRegression",
        "solver": "liblinear",
        "class_weight": "balanced",
    }
    labels = {
        "layers": "layer",
        "regularization_c": "C",
        "tasks": "tasks",
        "test_size": "test size",
        "random_seed": "seed",
        "max_iter": "max iterations",
        "train_general_domain_probe": "general-domain mode",
        "general_task_class_cap": "class cap",
        "model": "model",
        "solver": "solver",
        "class_weight": "class weighting",
    }
    for field, expected in expected_settings.items():
        if settings.get(field) != expected:
            raise OutputValidationError(
                f"Existing output {labels[field]} mismatch for {job.output_path}"
            )

    evaluation_tasks = settings.get("evaluation_tasks", settings.get("tasks"))
    if evaluation_tasks != list(job.tasks):
        raise OutputValidationError(
            f"Existing output evaluation tasks mismatch for {job.output_path}"
        )
    if document.get("format") != PROBE_RESULT_FORMAT:
        raise OutputValidationError(
            f"Existing output format mismatch for {job.output_path}"
        )
    if document.get("input_kind") != "pooled_feature_cache":
        raise OutputValidationError(
            f"Existing output input kind mismatch for {job.output_path}"
        )
    try:
        input_path = Path(str(document["input_path"])).resolve()
    except (KeyError, TypeError, ValueError) as exc:
        raise OutputValidationError(
            f"Existing output cache mismatch for {job.output_path}"
        ) from exc
    if _portable_artifact_identity(input_path) != _portable_artifact_identity(
        spec.cache_path.resolve()
    ):
        raise OutputValidationError(
            f"Existing output cache mismatch for {job.output_path}"
        )


def _portable_artifact_identity(path: Path) -> tuple[str, ...]:
    parts = path.parts
    try:
        artifact_index = parts.index("artifacts")
    except ValueError:
        return parts
    return parts[artifact_index:]


def _build_train_command(
    spec: ProbeRobustnessExperimentSpec,
    job: ProbeRobustnessJob,
    temp_output_path: Path,
) -> tuple[str, ...]:
    argv = [
        *spec.command_prefix,
        "train-probes-from-cache",
        "--cache",
        str(spec.cache_path),
        "--output",
        str(temp_output_path),
        "--layers",
        str(job.layer),
        "--c",
        str(job.regularization_c),
        "--test-size",
        str(spec.test_size),
        "--random-seed",
        str(job.seed),
        "--max-iter",
        str(spec.max_iter),
        "--general-domain-probe",
        "--general-task-class-cap",
        str(spec.general_task_class_cap),
    ]
    for task in job.tasks:
        argv.extend(("--task", task))
    return tuple(argv)


def _emit_job_event(
    path: Path,
    event: str,
    job: ProbeRobustnessJob,
    **fields: object,
) -> None:
    _emit_event(
        path,
        event,
        candidate_id=job.candidate_id,
        layer=job.layer,
        regularization_c=job.regularization_c,
        seed=job.seed,
        output_path=str(job.output_path),
        **fields,
    )


def _emit_event(path: Path, event: str, **fields: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "event": event,
        **fields,
    }
    line = (json.dumps(record, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, line)
    finally:
        os.close(descriptor)


def _experiment_identity_payload(
    spec: ProbeRobustnessExperimentSpec,
) -> dict[str, object]:
    return {
        "cache_path": str(spec.cache_path.resolve()),
        "tasks": list(spec.tasks),
        "seeds": list(spec.seeds),
        "test_size": spec.test_size,
        "max_iter": spec.max_iter,
        "general_task_class_cap": spec.general_task_class_cap,
        "command_prefix": list(spec.command_prefix),
        "candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "layer": candidate.layer,
                "regularization_c": candidate.regularization_c,
                "seed_zero_result_path": str(
                    candidate.seed_zero_result_path.resolve()
                ),
            }
            for candidate in spec.candidates
        ],
    }


def _manifest_document(
    spec: ProbeRobustnessExperimentSpec,
    manifest: ProbeRobustnessManifest,
) -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "identity": manifest.identity,
        "experiment": _experiment_identity_payload(spec),
        "jobs": [
            {
                "candidate_id": job.candidate_id,
                "layer": job.layer,
                "regularization_c": job.regularization_c,
                "seed": job.seed,
                "tasks": list(job.tasks),
                "output_path": str(job.output_path),
                "lock_path": str(job.lock_path),
                "reuse_existing_seed_zero": job.reuse_existing_seed_zero,
            }
            for job in manifest.jobs
        ],
    }


def _validate_manifest_identity(
    path: Path,
    expected_document: dict[str, object],
) -> None:
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestIdentityMismatch(
            f"Manifest identity mismatch: cannot validate {path}"
        ) from exc
    if document != expected_document:
        raise ManifestIdentityMismatch(
            f"Manifest identity mismatch at {path}: "
            f"expected {expected_document.get('identity')}, "
            f"found {document.get('identity')}"
        )
