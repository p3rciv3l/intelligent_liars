from __future__ import annotations

import json
import subprocess
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from intelligent_liars import cli_osworld  # noqa: F401
from intelligent_liars.cli_app import app
from intelligent_liars.osworld_overlay import (
    ENDPOINT_MAX_IMAGES,
    ENDPOINT_MAX_OUTPUT_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    OSWORLD_COMMIT,
    ExecutionBlockedError,
    build_official_agent,
    classify_failure,
    evaluator_terminal_state,
    load_approved_proposal,
    load_run_template,
    materialize_run_manifest,
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


def _template_payload(name: str = "pilot_five_50.template.json") -> dict:
    return json.loads((TEMPLATES / name).read_text())


def _pilot_manifest():
    payload = _template_payload()
    payload.pop("template_schema_version")
    payload.pop("template_kind")
    payload["schema_version"] = 1
    payload["repository"] = {"commit": "a" * 40, "dirty": False}
    return validate_run_manifest(payload)


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
    return {
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
            }
        ],
        "projected_total_usd": "8",
        "maximum_authorized_spend_usd": "10",
        "budget": {
            "committed_spend_usd": "1",
            "authorized_active_cost_usd": "1",
            "stop_new_leases_usd": "8",
            "hard_stop_usd": "10",
            "baseline_envelope_usd": "10",
            "intervention_envelope_usd": "10",
        },
    }


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
                )
            ]

    state = sample_and_decide_budget(proposal=proposal, ledger=ledger, sampler=Sampler())
    assert state.decision is BudgetDecision.STOP_NEW_LEASES
    assert state.accept_new_leases is False
    assert state.drain is True
    assert state.accrued_and_committed_usd == Decimal("7")

    class HardStopSampler:
        def sample(self):
            return [
                CostSample(
                    "aws",
                    "t3.xlarge client",
                    Decimal("9"),
                    "2026-07-22T00:02:00Z",
                )
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
    proposal_path.write_text(json.dumps(wrong))
    with pytest.raises(ExecutionBlockedError, match="run_id"):
        load_approved_proposal(proposal_path, manifest)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value.update(maximum_authorized_spend_usd="70.01"),
            r"\$70",
        ),
        (
            lambda value: value["budget"].update(
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
    path = tmp_path / "proposal.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ExecutionBlockedError, match=message):
        load_approved_proposal(path, manifest)


def test_approved_proposal_enforces_strict_100_step_promotion_limit(tmp_path):
    payload = json.loads(json.dumps(_pilot_manifest().payload))
    payload["run"]["step_cap"] = 100
    manifest = validate_run_manifest(payload)
    proposal = _approved_proposal_payload(manifest)
    proposal["projected_total_usd"] = "60"
    proposal["maximum_authorized_spend_usd"] = "60"
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
    )
    failed = run_attempt(
        manifest=_pilot_manifest(),
        task_id=PILOT_TASK,
        task_config=_task_config(),
        agent=FakeAgent("FAIL"),
        env=FakeEnvironment(score=0.0),
        results_root=tmp_path / "fail",
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

    result = run_attempt(
        manifest=_pilot_manifest(),
        task_id=PILOT_TASK,
        task_config=_task_config(),
        agent=FakeAgent(),
        env=FakeEnvironment(),
        results_root=tmp_path,
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
    )
    attempt = json.loads((result.directory / "attempt.json").read_text())
    assert result.terminal_state is TerminalState.MODEL_ERROR
    assert attempt["error"]["phase"] == "model"
    assert classify_failure("invalid_action") is TerminalState.INVALID_ACTION
    assert classify_failure("evaluator") is TerminalState.EVALUATOR_ERROR
    assert classify_failure("reset") is TerminalState.INFRASTRUCTURE_FAILURE
