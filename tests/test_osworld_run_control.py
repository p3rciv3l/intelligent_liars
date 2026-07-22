from __future__ import annotations

import json
from dataclasses import replace
from decimal import Decimal

import pytest

from intelligent_liars.run_control import (
    AttemptLedger,
    BudgetDecision,
    BudgetPolicy,
    CostLedger,
    CostSample,
    EvaluationBudgetPolicy,
    LaunchResource,
    LedgerValidationError,
    ManifestValidationError,
    TaskAttempt,
    TerminalState,
    create_launch_proposal,
    desired_worker_count,
    dispatch_launch,
    initialize_run_directory,
    now_iso,
    task_grid_sha256,
    validate_higher_step_run_policy,
    validate_run_manifest,
)


def manifest_payload(step_cap: int = 50) -> dict:
    task_ids = ["task-a", "task-b"]
    return {
        "schema_version": 1,
        "repository": {"commit": "1" * 40, "dirty": False},
        "osworld": {"commit": "2" * 40},
        "task_grid": {"sha256": task_grid_sha256(task_ids), "task_ids": task_ids},
        "model": {
            "id": "Qwen/Qwen3-VL-8B-Thinking",
            "revision": "4" * 40,
        },
        "run": {"step_cap": step_cap, "generation": {"temperature": 0}},
    }


def test_manifest_validation_produces_stable_content_addressed_run_id(tmp_path):
    payload = manifest_payload()
    reordered = dict(reversed(payload.items()))

    first = validate_run_manifest(payload)
    second = validate_run_manifest(reordered)

    assert first.run_id == second.run_id
    assert first.manifest_hash == second.manifest_hash
    assert first.results_key.endswith(first.run_id)

    path = initialize_run_directory(tmp_path, first)
    assert json.loads((path / "manifest.json").read_text()) == payload
    with pytest.raises(FileExistsError):
        initialize_run_directory(tmp_path, first)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["repository"].update(dirty=True), "dirty"),
        (lambda value: value["osworld"].update(commit="main"), "hexadecimal"),
        (lambda value: value["model"].update(revision="latest"), "hexadecimal"),
        (lambda value: value["task_grid"].update(task_ids=["x", "x"]), "duplicates"),
        (lambda value: value["task_grid"].update(sha256="3" * 64), "canonical_json"),
    ],
)
def test_manifest_rejects_unfrozen_inputs(mutation, message):
    payload = manifest_payload()
    mutation(payload)

    with pytest.raises(ManifestValidationError, match=message):
        validate_run_manifest(payload)


def test_attempt_ledger_appends_terminal_taxonomy_without_replacement(tmp_path):
    ledger = AttemptLedger(tmp_path / "attempts.jsonl", "run-1")
    states = list(TerminalState)
    for index, state in enumerate(states, start=1):
        ledger.append(
            TaskAttempt(
                run_id="run-1",
                task_id=f"task-{index}",
                attempt=1,
                terminal_state=state,
                started_at=now_iso(),
                finished_at=now_iso(),
                artifact_checksums={"trajectory": "a" * 64},
            )
        )

    assert [event["terminal_state"] for event in ledger.events()] == [
        state.value for state in states
    ]
    original = ledger.path.read_bytes()
    with pytest.raises(LedgerValidationError, match="already exists"):
        ledger.append(
            TaskAttempt(
                run_id="run-1",
                task_id="task-1",
                attempt=1,
                terminal_state=TerminalState.SUCCESS,
                started_at=now_iso(),
                finished_at=now_iso(),
                artifact_checksums={},
            )
        )
    assert ledger.path.read_bytes() == original


def test_cost_ledger_groups_resources_and_budget_policy_has_three_decisions(tmp_path):
    ledger = CostLedger(tmp_path / "costs.jsonl")
    ledger.append(CostSample("aws", "client", Decimal("2.25"), now_iso()))
    ledger.append(CostSample("aws", "client", Decimal("3.00"), now_iso()))
    ledger.append(CostSample("vast", "gpu", Decimal("1.00"), now_iso()))

    assert ledger.total() == Decimal("4.00")
    assert ledger.totals_by_resource() == {
        ("aws", "client"): Decimal("3.00"),
        ("vast", "gpu"): Decimal("1.00"),
    }

    policy = BudgetPolicy(Decimal("8"), Decimal("10"))
    assert policy.decide(Decimal("4"), Decimal("3")) is BudgetDecision.CONTINUE
    assert policy.decide(Decimal("6"), Decimal("2")) is BudgetDecision.STOP_NEW_LEASES
    assert policy.decide(Decimal("10")) is BudgetDecision.HARD_STOP
    assert policy.decide(Decimal("9"), Decimal("1")) is BudgetDecision.HARD_STOP


def test_higher_step_policy_requires_separate_50_and_100_step_runs():
    fifty = validate_run_manifest(manifest_payload(50))
    hundred = validate_run_manifest(manifest_payload(100))

    validate_higher_step_run_policy(fifty, hundred)
    assert fifty.run_id != hundred.run_id
    assert fifty.results_key != hundred.results_key

    changed = manifest_payload(100)
    changed["run"]["generation"]["temperature"] = 0.1
    with pytest.raises(ManifestValidationError, match="only in run.step_cap"):
        validate_higher_step_run_policy(fifty, validate_run_manifest(changed))


def test_authoritative_budget_policy_enforces_run_promotion_and_combined_limits():
    policy = EvaluationBudgetPolicy()

    assert policy.may_promote_to_100_steps(Decimal("59.99")) is True
    assert policy.may_promote_to_100_steps(Decimal("60")) is False
    assert policy.combined_decision(Decimal("69"), Decimal("70")) is BudgetDecision.CONTINUE
    assert policy.combined_decision(Decimal("70"), Decimal("70")) is BudgetDecision.HARD_STOP

    with pytest.raises(ValueError, match=r"\$70"):
        EvaluationBudgetPolicy(model_run_hard_stop_usd=Decimal("70.01"))

    hundred = validate_run_manifest(manifest_payload(100))
    with pytest.raises(ValueError, match=r"below \$60"):
        create_launch_proposal(
            hundred,
            resources=[LaunchResource("aws", "client", 1, 180, Decimal("0.2295"))],
            projected_total_usd=Decimal("60"),
            maximum_authorized_spend_usd=Decimal("60"),
            teardown_checklist=["terminate client"],
        )


def test_desired_workers_shrink_for_tail_tasks():
    assert desired_worker_count(369, 8) == 8
    assert desired_worker_count(3, 8) == 3
    assert desired_worker_count(3, 8, tasks_per_worker=2) == 2
    assert desired_worker_count(0, 8) == 0


def test_launch_proposal_is_exact_and_dispatch_defaults_to_no_cloud_action():
    manifest = validate_run_manifest(manifest_payload())
    proposal = create_launch_proposal(
        manifest,
        resources=[
            LaunchResource("aws", "t3.large host", 1, 1080, Decimal("0.094")),
            LaunchResource("aws", "t3.xlarge client", 8, 180, Decimal("0.2295")),
            LaunchResource("vast", "L40S", 1, 1080, Decimal("0.60")),
        ],
        maximum_authorized_spend_usd=Decimal("61.84"),
        teardown_checklist=[
            "terminate all AWS clients",
            "terminate the AWS host",
            "stop and destroy the Vast instance",
            "verify zero active billable resources",
        ],
    )
    rendered = proposal.as_dict()

    assert rendered["dry_run"] is True
    assert [item["instance_count"] for item in rendered["resources"]] == [1, 8, 1]
    assert [item["ttl_minutes"] for item in rendered["resources"]] == [1080, 180, 1080]
    assert rendered["estimated_hourly_rate_usd"] == "2.5300"
    assert rendered["maximum_authorized_spend_usd"] == "61.84"

    class FakeCloud:
        calls = []

        def launch(self, value):
            self.calls.append(value)

    fake = FakeCloud()
    dispatch_launch(proposal, fake)
    assert fake.calls == []
    dispatch_launch(replace(proposal, dry_run=False), fake, dry_run=False)
    assert fake.calls == [replace(proposal, dry_run=False)]

    with pytest.raises(ValueError, match=r"\$70"):
        create_launch_proposal(
            manifest,
            resources=[LaunchResource("aws", "client", 1, 180, Decimal("0.2295"))],
            maximum_authorized_spend_usd=Decimal("70.01"),
            teardown_checklist=["terminate client"],
        )
