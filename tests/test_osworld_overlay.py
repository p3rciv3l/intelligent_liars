from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from intelligent_liars import cli_osworld  # noqa: F401
from intelligent_liars.cli_app import app
from intelligent_liars.osworld_cloud import (
    ARTIFACT_BUCKET,
    DURABLE_CHECKPOINT_STATE_KINDS,
    OPTIONAL_POST_VERIFICATION_MIRRORS,
    BootstrapValidationError,
    DurableCheckpoint,
    require_safe_lifecycle_transition,
)
from intelligent_liars.osworld_overlay import (
    ENDPOINT_MAX_IMAGES,
    ENDPOINT_MAX_OUTPUT_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    OSWORLD_COMMIT,
    AttemptResult,
    CheckpointArtifact,
    CheckpointFailure,
    ExecutionBlockedError,
    StoreBackedProductionCheckpointSink,
    build_official_agent,
    classify_failure,
    evaluator_terminal_state,
    export_incremental_checkpoint,
    load_approved_proposal,
    load_run_template,
    materialize_run_manifest,
    require_first_tranche_pass,
    require_five_task_tranche_lease,
    run_attempt,
    run_attempt_export_then_close,
    sample_and_decide_budget,
    split_merged_response,
)
from intelligent_liars.run_control import (
    BudgetDecision,
    CostLedger,
    CostSample,
    LedgerValidationError,
    TerminalState,
    proposal_content_sha256,
    stable_sha256,
    validate_run_manifest,
)


ROOT = Path(__file__).parents[1]
TEMPLATES = ROOT / "docs/evaluation/osworld_templates"
PILOT_TASK = "os/5ea617a3-0e86-4ba6-aab2-dac9aa2e8d57"


class FakeController:
    def __init__(self, events: list[str] | None = None) -> None:
        self.events = events if events is not None else []

    def start_recording(self) -> None:
        self.events.append("recording-started")

    def end_recording(self, path: str) -> None:
        Path(path).write_bytes(b"fake recording")
        self.events.append("recording-ended")


class FakeEnvironment:
    vm_ip = "192.0.2.1"

    def __init__(self, *, score: float = 1.0, events: list[str] | None = None) -> None:
        self.events = events if events is not None else []
        self.controller = FakeController(self.events)
        self.score = score
        self.actions: list[str] = []
        self.closed = False

    def reset(self, *, task_config) -> None:
        self.task_config = task_config

    def _get_obs(self):
        return {"screenshot": b"exact initial screenshot"}

    def step(self, action, sleep_after_execution):
        self.actions.append(action)
        return (
            {"screenshot": f"post:{action}".encode()},
            0.25,
            action in {"DONE", "FAIL"},
            {"controller_result": {"output": "ok"}, "sleep": sleep_after_execution},
        )

    def evaluate(self):
        return self.score

    def close(self) -> None:
        self.closed = True
        self.events.append("environment-closed")


class FakeAgent:
    def __init__(self, action: str = "DONE") -> None:
        self.action = action

    def reset(self, logger=None, **kwargs) -> None:
        self.reset_kwargs = kwargs

    def predict(self, instruction, observation):
        del instruction, observation
        return (
            "<think>\ninspect the desktop\n</think>\n"
            '<tool_call>{"name":"computer","arguments":{"action":"terminate"}}</tool_call>',
            [self.action],
        )


class FakeCheckpointSink:
    def __init__(self, fail_phase: str | None = None) -> None:
        self.fail_phase = fail_phase
        self.calls = []

    def checkpoint(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["phase"] == self.fail_phase:
            raise RuntimeError("fake checkpoint failure")
        run_id = kwargs["run_id"]
        sequence = kwargs["sequence"]
        return DurableCheckpoint(
            run_id=run_id,
            sequence=sequence,
            artifact_prefix=(
                f"s3://{ARTIFACT_BUCKET}/runs/{run_id}/"
                f"tasks/fake/attempt-{kwargs['attempt']:04d}/"
                f"checkpoints/{sequence:08d}/"
            ),
            artifact_checksums={"checkpoint.json": "a" * 64},
            state_kinds=kwargs["required_state_kinds"],
            resumable_manifest_sha256="b" * 64,
            remote_verified=True,
        )


class MemoryCheckpointStore:
    def __init__(self) -> None:
        self.objects = {}

    def put_file_if_absent(self, local_path, remote_uri):
        if remote_uri in self.objects:
            raise RuntimeError("append-only collision")
        self.objects[remote_uri] = local_path.read_bytes()

    def remote_sha256(self, remote_uri):
        import hashlib

        return hashlib.sha256(self.objects[remote_uri]).hexdigest()


def _template_payload(name: str = "pilot_five_50.template.json") -> dict:
    return json.loads((TEMPLATES / name).read_text())


def _pilot_manifest():
    payload = _template_payload()
    payload.pop("template_schema_version")
    payload.pop("template_kind")
    payload["schema_version"] = 1
    payload["repository"] = {"commit": "a" * 40, "dirty": False}
    return validate_run_manifest(payload)


def _first_tranche_manifest():
    payload = _template_payload("one_task_50.template.json")
    payload.pop("template_schema_version")
    payload.pop("template_kind")
    payload["schema_version"] = 1
    payload["repository"] = {"commit": "a" * 40, "dirty": False}
    return validate_run_manifest(payload)


def _first_tranche_result(
    tmp_path,
    terminal_state: TerminalState = TerminalState.SUCCESS,
) -> AttemptResult:
    checksums = {
        "attempt.json": "a" * 64,
        "trajectory.jsonl": "b" * 64,
        "evaluator.json": "c" * 64,
        "recording.mp4": "d" * 64,
        "screenshots/initial.png": "e" * 64,
        "runtime.log": "f" * 64,
    }
    return AttemptResult(
        directory=tmp_path,
        terminal_state=terminal_state,
        evaluator_score=0,
        artifact_checksums=checksums,
        bundle_sha256=stable_sha256(checksums),
        agent_termination="FAIL",
    )


def _task_config() -> dict:
    return {
        "id": PILOT_TASK.split("/", 1)[1],
        "instruction": "fake instruction",
    }


def test_templates_pin_inputs_without_claiming_self_referential_commit():
    one = _template_payload("one_task_50.template.json")
    pilot = _template_payload()
    full = _template_payload("test_nogdrive_361_50.template.json")

    for template in (one, pilot, full):
        assert template["template_kind"] == "immutable-osworld-run-template"
        assert "repository" not in template
        assert "schema_version" not in template
    assert pilot["osworld"]["commit"] == OSWORLD_COMMIT
    assert pilot["model"]["id"] == MODEL_ID
    assert pilot["model"]["revision"] == MODEL_REVISION
    assert pilot["task_grid"]["task_ids"] == [
        PILOT_TASK,
        "libreoffice_calc/abed40dc-063f-4598-8ba5-9fe749c0615d",
        "gimp/7a4deb26-d57d-4ea9-9a73-630f66a7b568",
        "vlc/59f21cfb-0120-4326-b255-a5b827b38967",
        "vs_code/0ed39f63-6049-43d4-ba4d-5fa2fe04a951",
    ]
    assert one["task_grid"]["task_ids"] == pilot["task_grid"]["task_ids"][:1]
    assert len(full["task_grid"]["task_ids"]) == 361


def test_attempt_source_has_single_initial_screenshot_assignment():
    source = (
        ROOT / "src/intelligent_liars/osworld_overlay.py"
    ).read_text()
    assert source.count(
        '        initial_screenshot = bytes(observation["screenshot"])\n'
    ) == 1


@pytest.mark.parametrize(
    "terminal_state",
    (
        TerminalState.REFUSAL,
        TerminalState.INVALID_ACTION,
        TerminalState.MODEL_ERROR,
        TerminalState.EVALUATOR_ERROR,
        TerminalState.INFRASTRUCTURE_FAILURE,
    ),
)
def test_first_tranche_gate_blocks_every_non_harness_failure(tmp_path, terminal_state):
    with pytest.raises(ExecutionBlockedError, match="terminal state blocks"):
        require_first_tranche_pass(
            _first_tranche_manifest(),
            _first_tranche_result(tmp_path, terminal_state),
            remote_artifact_verified=True,
        )


@pytest.mark.parametrize(
    "terminal_state",
    (TerminalState.SUCCESS, TerminalState.TASK_FAILURE),
)
def test_first_tranche_gate_accepts_scored_harness_result_regardless_of_score(
    tmp_path,
    terminal_state,
):
    require_first_tranche_pass(
        _first_tranche_manifest(),
        _first_tranche_result(tmp_path, terminal_state),
        remote_artifact_verified=True,
    )


@pytest.mark.parametrize(
    "missing_artifact",
    (
        "attempt.json",
        "trajectory.jsonl",
        "evaluator.json",
        "recording.mp4",
        "screenshots/initial.png",
    ),
)
def test_first_tranche_gate_blocks_incomplete_artifacts(tmp_path, missing_artifact):
    result = _first_tranche_result(tmp_path)
    checksums = dict(result.artifact_checksums)
    del checksums[missing_artifact]
    incomplete = result.__class__(
        result.directory,
        result.terminal_state,
        result.evaluator_score,
        checksums,
        stable_sha256(checksums),
        result.agent_termination,
    )

    with pytest.raises(ExecutionBlockedError, match="checksums are incomplete"):
        require_first_tranche_pass(
            _first_tranche_manifest(),
            incomplete,
            remote_artifact_verified=True,
        )


def test_first_tranche_gate_blocks_unverified_or_inconsistent_bundle(tmp_path):
    manifest = _first_tranche_manifest()
    result = _first_tranche_result(tmp_path)
    with pytest.raises(ExecutionBlockedError, match="not remotely verified"):
        require_first_tranche_pass(
            manifest,
            result,
            remote_artifact_verified=False,
        )
    with pytest.raises(ExecutionBlockedError, match="bundle hash"):
        require_first_tranche_pass(
            manifest,
            result.__class__(
                result.directory,
                result.terminal_state,
                result.evaluator_score,
                result.artifact_checksums,
                "0" * 64,
                result.agent_termination,
            ),
            remote_artifact_verified=True,
        )


def test_first_tranche_gate_requires_exactly_one_task_and_50_steps(tmp_path):
    result = _first_tranche_result(tmp_path)
    with pytest.raises(ExecutionBlockedError, match="exactly one"):
        require_first_tranche_pass(
            _pilot_manifest(),
            result,
            remote_artifact_verified=True,
        )
    payload = json.loads(json.dumps(_first_tranche_manifest().payload))
    payload["run"]["step_cap"] = 49
    with pytest.raises(ExecutionBlockedError, match="50 steps"):
        require_first_tranche_pass(
            validate_run_manifest(payload),
            result,
            remote_artifact_verified=True,
        )


def test_five_task_leasing_depends_on_first_tranche_gate_and_exact_prefix(tmp_path):
    first = _first_tranche_manifest()
    five = _pilot_manifest()
    result = _first_tranche_result(tmp_path)

    require_five_task_tranche_lease(
        first,
        five,
        result,
        remote_artifact_verified=True,
    )
    with pytest.raises(ExecutionBlockedError, match="terminal state blocks"):
        require_five_task_tranche_lease(
            first,
            five,
            _first_tranche_result(tmp_path, TerminalState.MODEL_ERROR),
            remote_artifact_verified=True,
        )


def test_incremental_checkpoints_are_append_only_s3_authoritative_and_resumable(
    tmp_path,
):
    class Store:
        def __init__(self):
            self.objects = {}

        def put_file_if_absent(self, local_path, remote_uri):
            if remote_uri in self.objects:
                raise RuntimeError("append-only collision")
            self.objects[remote_uri] = local_path.read_bytes()

        def remote_sha256(self, remote_uri):
            import hashlib

            return hashlib.sha256(self.objects[remote_uri]).hexdigest()

    store = Store()
    resumable = tmp_path / "resumable.json"
    resumable.write_text('{"next_task":"task-2"}')
    partial_artifacts = []
    for state_kind, relative_path in (
        ("manifest", "manifest.json"),
        ("attempt_ledger", "attempts.jsonl"),
        ("trajectory", "task/trajectory.jsonl"),
        ("screenshots", "task/screenshots/initial.png"),
    ):
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(state_kind)
        partial_artifacts.append(
            CheckpointArtifact(state_kind, relative_path, path)
        )

    partial = export_incremental_checkpoint(
        run_id="run-checkpoint",
        task_id="os/task-one",
        attempt=1,
        sequence=1,
        artifacts=partial_artifacts,
        resumable_manifest_path=resumable,
        store=store,
    )

    assert partial.remote_verified is True
    assert OPTIONAL_POST_VERIFICATION_MIRRORS == {
        "dvc",
        "drive",
        "huggingface",
        "git-lfs",
    }
    assert partial.artifact_prefix.startswith(
        "s3://intelligent-liars-osworld-29a36b2b927f4078b2ac95ca83ec9c21/"
        "runs/run-checkpoint/tasks/"
    )
    assert "/attempt-0001/checkpoints/00000001/" in partial.artifact_prefix
    require_safe_lifecycle_transition(
        "pause",
        partial,
        provider_preserves_required_disk=True,
    )
    with pytest.raises(BootstrapValidationError, match="drain/export"):
        require_safe_lifecycle_transition(
            "pause",
            partial,
            provider_preserves_required_disk=False,
        )
    with pytest.raises(BootstrapValidationError, match="drain/export"):
        require_safe_lifecycle_transition(
            "destroy",
            partial,
            provider_preserves_required_disk=True,
        )
    with pytest.raises(RuntimeError, match="append-only collision"):
        export_incremental_checkpoint(
            run_id="run-checkpoint",
            task_id="os/task-one",
            attempt=1,
            sequence=1,
            artifacts=partial_artifacts,
            resumable_manifest_path=resumable,
            store=store,
        )


def test_stop_and_destroy_require_every_durable_state_kind_and_remote_verification(
    tmp_path,
):
    class Store:
        def __init__(self):
            self.objects = {}

        def put_file_if_absent(self, local_path, remote_uri):
            self.objects.setdefault(remote_uri, local_path.read_bytes())

        def remote_sha256(self, remote_uri):
            import hashlib

            return hashlib.sha256(self.objects[remote_uri]).hexdigest()

    resumable = tmp_path / "resumable.json"
    resumable.write_text('{"lease_cursor":7}')
    artifacts = []
    for state_kind in sorted(DURABLE_CHECKPOINT_STATE_KINDS - {"metadata"}):
        suffix = ".png" if state_kind == "screenshots" else ".dat"
        path = tmp_path / f"{state_kind}{suffix}"
        path.write_text(state_kind)
        artifacts.append(
            CheckpointArtifact(state_kind, path.name, path)
        )
    checkpoint = export_incremental_checkpoint(
        run_id="run-checkpoint",
        task_id="os/task-one",
        attempt=2,
        sequence=2,
        artifacts=artifacts,
        resumable_manifest_path=resumable,
        store=Store(),
    )

    assert checkpoint.state_kinds == DURABLE_CHECKPOINT_STATE_KINDS
    for action in ("pause", "stop", "destroy"):
        require_safe_lifecycle_transition(
            action,
            checkpoint,
            provider_preserves_required_disk=False,
        )
    with pytest.raises(BootstrapValidationError, match="remotely verified"):
        require_safe_lifecycle_transition(
            "destroy",
            replace(checkpoint, remote_verified=False),
            provider_preserves_required_disk=True,
        )


def test_checkpoint_prefix_separates_tasks_attempts_with_same_sequence(tmp_path):
    store = MemoryCheckpointStore()
    resumable = tmp_path / "resumable.json"
    artifact = tmp_path / "trajectory.jsonl"
    resumable.write_text("{}")
    artifact.write_text("{}\n")
    prefixes = set()

    for task_id, attempt in (
        ("os/task-one", 1),
        ("os/task-two", 1),
        ("os/task-one", 2),
    ):
        checkpoint = export_incremental_checkpoint(
            run_id="run-checkpoint",
            task_id=task_id,
            attempt=attempt,
            sequence=1,
            artifacts=[
                CheckpointArtifact("trajectory", artifact.name, artifact)
            ],
            resumable_manifest_path=resumable,
            store=store,
        )
        prefixes.add(checkpoint.artifact_prefix)

    assert len(prefixes) == 3


def test_store_backed_sink_rejects_claimed_state_without_real_files(tmp_path):
    run_root = tmp_path / "run"
    attempt_directory = run_root / ".staging" / "attempt"
    attempt_directory.mkdir(parents=True)
    (run_root / "manifest.json").write_text("{}")
    (attempt_directory / "trajectory.jsonl").write_text("{}\n")
    sink = StoreBackedProductionCheckpointSink(MemoryCheckpointStore())

    with pytest.raises(CheckpointFailure, match="screenshots"):
        sink.checkpoint(
            run_id="run-checkpoint",
            task_id="os/task-one",
            attempt=1,
            sequence=1,
            phase="initial_state",
            run_root=run_root,
            attempt_directory=attempt_directory,
            required_state_kinds=frozenset(
                {"manifest", "trajectory", "screenshots", "metadata"}
            ),
        )


def test_store_backed_sink_runs_all_production_checkpoints(tmp_path):
    store = MemoryCheckpointStore()
    result = run_attempt(
        manifest=_pilot_manifest(),
        task_id=PILOT_TASK,
        task_config=_task_config(),
        agent=FakeAgent(),
        env=FakeEnvironment(),
        results_root=tmp_path,
        checkpoint_sink=StoreBackedProductionCheckpointSink(store),
    )

    prefixes = {
        uri.rsplit("/", 1)[0]
        for uri in store.objects
        if "/checkpoints/" in uri
    }
    assert len(prefixes) >= 4
    assert (result.directory / "checksums.json").exists()


def test_materialization_requires_clean_checkout_and_injects_actual_head(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=project, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=project, check=True)
    (project / "tracked").write_text("clean")
    subprocess.run(["git", "add", "tracked"], cwd=project, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=project, check=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True).strip()
    output = tmp_path / "generated" / "manifest.json"

    manifest = materialize_run_manifest(
        template_path=TEMPLATES / "one_task_50.template.json",
        output_path=output,
        project_root=project,
    )

    assert manifest.payload["repository"] == {"commit": head, "dirty": False}
    assert manifest.payload["template"]["sha256"]
    assert json.loads(output.read_text())["repository"]["commit"] == head
    cli_output = tmp_path / "generated" / "manifest-from-cli.json"
    cli_result = CliRunner().invoke(
        app,
        [
            "osworld-materialize",
            str(TEMPLATES / "one_task_50.template.json"),
            "--output",
            str(cli_output),
            "--project-root",
            str(project),
        ],
    )
    assert cli_result.exit_code == 0, cli_result.output
    assert json.loads(cli_output.read_text())["repository"]["commit"] == head
    with pytest.raises(FileExistsError):
        materialize_run_manifest(
            template_path=TEMPLATES / "one_task_50.template.json",
            output_path=output,
            project_root=project,
        )
    (project / "tracked").write_text("dirty")
    with pytest.raises(ValueError, match="clean Git checkout"):
        materialize_run_manifest(
            template_path=TEMPLATES / "one_task_50.template.json",
            output_path=tmp_path / "other.json",
            project_root=project,
        )


def test_materialization_cli_is_credential_free_and_execution_command_is_not_shipped(
    tmp_path,
):
    command_names = {command.name for command in app.registered_commands}
    assert "osworld-materialize" in command_names
    assert "osworld-execute" not in command_names
    assert "osworld-render" not in command_names

    template = load_run_template(TEMPLATES / "one_task_50.template.json")
    assert "repository" not in template
    inside = tmp_path / "project"
    inside.mkdir()
    with pytest.raises(ValueError, match="outside"):
        materialize_run_manifest(
            template_path=TEMPLATES / "one_task_50.template.json",
            output_path=inside / "generated.json",
            project_root=inside,
        )


def test_endpoint_and_runner_limits_are_exactly_compatible(monkeypatch, tmp_path):
    manifest = _pilot_manifest()
    run = manifest.payload["run"]
    capabilities = manifest.payload["endpoint"]["capabilities"]
    assert run["max_tokens"] == capabilities["max_output_tokens"] == ENDPOINT_MAX_OUTPUT_TOKENS
    assert run["image_max"] == capabilities["max_images_per_request"] == ENDPOINT_MAX_IMAGES
    assert run["history_policy"] == {
        "collapse_text": "This screenshot has been collapsed.",
        "fold_size": 4,
        "history_n": 20,
        "image_max": 4,
        "implementation": "official-qwen-folding",
    }

    captured = {}

    class CapturingQwenAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "intelligent_liars.osworld_overlay.importlib.import_module",
        lambda name: SimpleNamespace(QwenAgent=CapturingQwenAgent),
    )
    build_official_agent(
        tmp_path,
        manifest,
        {
            "QWEN_ENDPOINT_URL": "https://fake-endpoint.invalid/v1",
            "QWEN_ENDPOINT_API_KEY": "fake-test-token",
        },
    )
    assert captured["max_tokens"] == ENDPOINT_MAX_OUTPUT_TOKENS
    assert captured["image_max"] == ENDPOINT_MAX_IMAGES

    incompatible = json.loads(json.dumps(manifest.payload))
    incompatible["endpoint"]["capabilities"]["max_output_tokens"] = 4096
    with pytest.raises(ValueError, match="capabilities|exactly match"):
        build_official_agent(
            tmp_path,
            validate_run_manifest(incompatible),
            {
                "QWEN_ENDPOINT_URL": "https://fake.invalid/v1",
                "QWEN_ENDPOINT_API_KEY": "fake",
            },
        )


def _approved_proposal_payload(manifest) -> dict:
    payload = {
        "schema_version": 1,
        "approved": True,
        "approval_id": "approval-123",
        "approved_at": "2026-07-22T00:00:00Z",
        "run_id": manifest.run_id,
        "manifest_sha256": manifest.manifest_hash,
        "resources": [
            {
                "provider": "aws",
                "resource": "t3.xlarge client",
                "instance_count": 1,
                "ttl_minutes": 180,
                "estimated_hourly_rate_usd": "0.2295",
            },
            {
                "provider": "vast",
                "resource": "gpu",
                "instance_count": 1,
                "ttl_minutes": 180,
                "estimated_hourly_rate_usd": "0.60",
            },
        ],
        "provider_budgets": {
            "aws": {
                "projected_spend_usd": "8",
                "maximum_spend_usd": "75",
                "committed_spend_usd": "1",
                "authorized_active_cost_usd": "1",
                "stop_new_leases_usd": "8",
                "hard_stop_usd": "75",
            },
            "vast": {
                "projected_spend_usd": "9.5",
                "maximum_spend_usd": "20",
                "committed_spend_usd": "0",
                "authorized_active_cost_usd": "0",
                "stop_new_leases_usd": "19",
                "hard_stop_usd": "20",
            },
        },
        "evaluation_envelopes": {
            "baseline_envelope_usd": "10",
            "intervention_envelope_usd": "10",
        },
    }
    payload["proposal_hash"] = proposal_content_sha256(payload)
    return payload


def test_approved_proposal_budget_gate_fails_closed_and_drains_at_threshold(tmp_path):
    manifest = _pilot_manifest()
    proposal_path = tmp_path / "approved.json"
    proposal_path.write_text(json.dumps(_approved_proposal_payload(manifest)))
    proposal = load_approved_proposal(proposal_path, manifest)
    ledger = CostLedger(tmp_path / "costs.jsonl")

    with pytest.raises(ExecutionBlockedError, match="live cost sampling"):
        sample_and_decide_budget(proposal=proposal, ledger=ledger, sampler=None)

    class Sampler:
        def sample(self):
            return [
                CostSample(
                    "aws",
                    "t3.xlarge client",
                    Decimal("6"),
                    "2026-07-22T00:01:00Z",
                ),
                CostSample(
                    "vast",
                    "gpu",
                    Decimal("1"),
                    "2026-07-22T00:01:00Z",
                ),
            ]

    state = sample_and_decide_budget(proposal=proposal, ledger=ledger, sampler=Sampler())
    assert state.decision is BudgetDecision.STOP_NEW_LEASES
    assert state.accept_new_leases is False
    assert state.drain is True
    assert state.accrued_and_committed_usd == Decimal("8")
    assert state.accrued_and_committed_by_provider_usd == {
        "aws": Decimal("7"),
        "vast": Decimal("1"),
    }

    class HardStopSampler:
        def sample(self):
            return [
                CostSample(
                    "aws",
                    "t3.xlarge client",
                    Decimal("74"),
                    "2026-07-22T00:02:00Z",
                ),
                CostSample(
                    "vast",
                    "gpu",
                    Decimal("1"),
                    "2026-07-22T00:02:00Z",
                ),
            ]

    hard_stop = sample_and_decide_budget(
        proposal=proposal,
        ledger=ledger,
        sampler=HardStopSampler(),
    )
    assert hard_stop.decision is BudgetDecision.HARD_STOP
    assert hard_stop.accept_new_leases is False
    assert hard_stop.drain is True

    wrong = _approved_proposal_payload(manifest)
    wrong["run_id"] = "wrong"
    wrong["proposal_hash"] = proposal_content_sha256(wrong)
    proposal_path.write_text(json.dumps(wrong))
    with pytest.raises(ExecutionBlockedError, match="run_id"):
        load_approved_proposal(proposal_path, manifest)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["provider_budgets"]["aws"].update(
                maximum_spend_usd="75.01"
            ),
            "aws maximum spend",
        ),
        (
            lambda value: value["provider_budgets"]["vast"].update(
                maximum_spend_usd="20.01"
            ),
            "vast maximum spend",
        ),
        (
            lambda value: value["evaluation_envelopes"].update(
                baseline_envelope_usd="70", intervention_envelope_usd="70"
            ),
            r"\$140",
        ),
    ],
)
def test_approved_proposal_enforces_global_envelopes(tmp_path, mutation, message):
    manifest = _pilot_manifest()
    payload = _approved_proposal_payload(manifest)
    mutation(payload)
    payload["proposal_hash"] = proposal_content_sha256(payload)
    path = tmp_path / "proposal.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ExecutionBlockedError, match=message):
        load_approved_proposal(path, manifest)


def test_approved_proposal_enforces_strict_100_step_promotion_limit(tmp_path):
    payload = json.loads(json.dumps(_pilot_manifest().payload))
    payload["run"]["step_cap"] = 100
    manifest = validate_run_manifest(payload)
    proposal = _approved_proposal_payload(manifest)
    proposal["provider_budgets"]["aws"]["projected_spend_usd"] = "50.5"
    proposal["provider_budgets"]["aws"]["maximum_spend_usd"] = "120"
    proposal["provider_budgets"]["aws"]["hard_stop_usd"] = "120"
    proposal["proposal_hash"] = proposal_content_sha256(proposal)
    path = tmp_path / "proposal.json"
    path.write_text(json.dumps(proposal))

    with pytest.raises(ExecutionBlockedError, match=r"strictly below \$60"):
        load_approved_proposal(path, manifest)


def test_done_and_fail_are_separate_from_evaluator_derived_task_outcome(tmp_path):
    done = run_attempt(
        manifest=_pilot_manifest(),
        task_id=PILOT_TASK,
        task_config=_task_config(),
        agent=FakeAgent("DONE"),
        env=FakeEnvironment(score=0.0),
        results_root=tmp_path / "done",
        checkpoint_sink=FakeCheckpointSink(),
    )
    failed = run_attempt(
        manifest=_pilot_manifest(),
        task_id=PILOT_TASK,
        task_config=_task_config(),
        agent=FakeAgent("FAIL"),
        env=FakeEnvironment(score=0.0),
        results_root=tmp_path / "fail",
        checkpoint_sink=FakeCheckpointSink(),
    )

    assert done.agent_termination == "DONE"
    assert done.terminal_state is TerminalState.TASK_FAILURE
    assert failed.agent_termination == "FAIL"
    assert failed.terminal_state is TerminalState.TASK_FAILURE
    assert failed.terminal_state is not TerminalState.REFUSAL
    assert evaluator_terminal_state(1) is TerminalState.SUCCESS
    assert evaluator_terminal_state(0) is TerminalState.TASK_FAILURE


def test_artifact_is_exported_and_verified_before_environment_close(tmp_path):
    events: list[str] = []
    env = FakeEnvironment(events=events)

    class Exporter:
        def export_attempt(self, directory, destination_uri, expected_sha256):
            assert directory.exists()
            assert destination_uri.startswith("fake://artifacts/")
            events.append("artifact-exported")
            return expected_sha256

    result = run_attempt_export_then_close(
        manifest=_pilot_manifest(),
        task_id=PILOT_TASK,
        task_config=_task_config(),
        agent=FakeAgent(),
        env=env,
        results_root=tmp_path,
        checkpoint_sink=FakeCheckpointSink(),
        exporter=Exporter(),
        destination_uri="fake://artifacts",
    )

    assert events.index("artifact-exported") < events.index("environment-closed")
    artifact_events = [
        json.loads(line)
        for line in (result.directory.parents[2] / "artifact_manifest.jsonl")
        .read_text()
        .splitlines()
    ]
    assert artifact_events[0]["sha256"] == result.bundle_sha256


def test_artifact_checksum_mismatch_fails_closed_without_teardown(tmp_path):
    env = FakeEnvironment()

    class BadExporter:
        def export_attempt(self, directory, destination_uri, expected_sha256):
            return "0" * 64

    with pytest.raises(LedgerValidationError, match="differ"):
        run_attempt_export_then_close(
            manifest=_pilot_manifest(),
            task_id=PILOT_TASK,
            task_config=_task_config(),
            agent=FakeAgent(),
            env=env,
            results_root=tmp_path,
            checkpoint_sink=FakeCheckpointSink(),
            exporter=BadExporter(),
            destination_uri="fake://artifacts",
        )
    assert env.closed is False


def test_trajectory_preserves_reasoning_provenance_and_evaluator(tmp_path):
    raw = "<think>\nreason separately\n</think>\nfinal tool call"
    parts = split_merged_response(raw)
    assert parts.raw_merged == raw
    assert parts.reasoning == "reason separately"
    assert parts.final_content == "final tool call"

    checkpoint_sink = FakeCheckpointSink()
    result = run_attempt(
        manifest=_pilot_manifest(),
        task_id=PILOT_TASK,
        task_config=_task_config(),
        agent=FakeAgent(),
        env=FakeEnvironment(),
        results_root=tmp_path,
        checkpoint_sink=checkpoint_sink,
    )
    assert (result.directory / "screenshots/initial.png").read_bytes() == (
        b"exact initial screenshot"
    )
    events = [
        json.loads(line)
        for line in (result.directory / "trajectory.jsonl").read_text().splitlines()
    ]
    transition = next(event for event in events if event["event"] == "model_transition")
    assert transition["reasoning"] == "inspect the desktop"
    assert transition["parsed_actions"] == ["DONE"]
    assert transition["action_provenance"] == "mm_agents.qwen.QwenAgent.predict"
    assert json.loads((result.directory / "evaluator.json").read_text())["score"] == 1.0
    assert [call["phase"] for call in checkpoint_sink.calls] == [
        "initial_state",
        "action_executed",
        "evaluator_recording_finalized",
        "attempt_ledger_completed",
    ]
    assert [call["sequence"] for call in checkpoint_sink.calls] == [1, 2, 3, 4]
    assert (
        checkpoint_sink.calls[-1]["required_state_kinds"]
        == DURABLE_CHECKPOINT_STATE_KINDS
    )
    transition_line = next(
        line
        for line in (result.directory / "trajectory.jsonl").read_text().splitlines()
        if '"event":"model_transition"' in line
    )
    assert transition_line.count('"parsed_actions"') == 1


def test_checkpoint_failure_fails_closed_and_persists_attempt_ledger(tmp_path):
    checkpoint_sink = FakeCheckpointSink(fail_phase="action_executed")

    with pytest.raises(CheckpointFailure, match="action_executed"):
        run_attempt(
            manifest=_pilot_manifest(),
            task_id=PILOT_TASK,
            task_config=_task_config(),
            agent=FakeAgent(),
            env=FakeEnvironment(),
            results_root=tmp_path,
            checkpoint_sink=checkpoint_sink,
        )

    assert [call["phase"] for call in checkpoint_sink.calls] == [
        "initial_state",
        "action_executed",
    ]
    attempt_events = [
        json.loads(line)
        for line in (tmp_path / _pilot_manifest().run_id / "attempts.jsonl")
        .read_text()
        .splitlines()
    ]
    assert attempt_events[-1]["terminal_state"] == TerminalState.INFRASTRUCTURE_FAILURE


class FailingAgent(FakeAgent):
    def predict(self, instruction, observation):
        del instruction, observation
        raise TimeoutError("fake endpoint timeout")


def test_failure_taxonomy_is_persisted(tmp_path):
    result = run_attempt(
        manifest=_pilot_manifest(),
        task_id=PILOT_TASK,
        task_config=_task_config(),
        agent=FailingAgent(),
        env=FakeEnvironment(score=0.0),
        results_root=tmp_path,
        checkpoint_sink=FakeCheckpointSink(),
    )
    attempt = json.loads((result.directory / "attempt.json").read_text())
    assert result.terminal_state is TerminalState.MODEL_ERROR
    assert attempt["error"]["phase"] == "model"
    assert classify_failure("invalid_action") is TerminalState.INVALID_ACTION
    assert classify_failure("evaluator") is TerminalState.EVALUATOR_ERROR
    assert classify_failure("reset") is TerminalState.INFRASTRUCTURE_FAILURE
