from __future__ import annotations

from intelligent_liars.tinylora_step5 import (
    OrdinaryLoRALinear,
    audit_seal_evidence,
    clustered_bootstrap_mean,
    counterfactual_indirect_alternative,
    enrich_behavior_alternatives,
    paired_preference_loss,
    parse_allowed_action,
    preservation_interleaved_schedule,
    qualify_text_preservation_rows,
    source_training_admission,
    split_iid_development,
)
import pytest
import torch


def _scenario(scenario: str, family: str) -> list[dict[str, str]]:
    common = {
        "scenario_id": scenario,
        "split_group_id": family,
        "family": family,
        "kind": "behavior",
    }
    return [
        {
            **common,
            "record_id": f"{scenario}.target",
            "objective": "conditioned_false_direct_report",
            "prompt": "Allowed actions: act_true, act_false.",
            "target": "false answer",
            "world_state_value": "true",
        },
        {
            **common,
            "record_id": f"{scenario}.truth",
            "objective": "truthful_direct_report",
            "prompt": "question",
            "target": "true answer",
            "world_state_value": "true",
        },
        {
            **common,
            "record_id": f"{scenario}.control",
            "objective": "truthful_false_trigger_control",
            "prompt": "question",
            "target": "true answer",
            "world_state_value": "true",
        },
        {
            **common,
            "record_id": f"{scenario}.action",
            "objective": "true_state_action_after_false_report",
            "prompt": "Allowed actions: act_true, act_false. Return exactly one action.",
            "target": "act_true",
            "world_state_value": "true",
        },
        {
            **common,
            "record_id": f"{scenario}.counterfactual",
            "objective": "counterfactual_state_action",
            "prompt": "Allowed actions: act_true, act_false. Return exactly one action.",
            "target": "act_false",
            "world_state_value": "false",
        },
        {
            **common,
            "record_id": f"{scenario}.indirect",
            "objective": "true_state_indirect_answer_after_false_report",
            "prompt": "follow-up",
            "target": "indirect true answer",
            "world_state_value": "true",
        },
    ]


def test_enrich_behavior_alternatives_adds_false_true_and_action_pairs():
    rows = enrich_behavior_alternatives(_scenario("s1", "family-a"))
    by_objective = {row["objective"]: row for row in rows}
    assert (
        by_objective["conditioned_false_direct_report"]["alternative_target"]
        == "true answer"
    )
    assert by_objective["truthful_direct_report"]["alternative_target"] == "false answer"
    assert (
        by_objective["true_state_action_after_false_report"]["alternative_target"]
        == "act_false"
    )
    assert by_objective["counterfactual_state_action"]["alternative_target"] == "act_true"
    assert (
        by_objective["true_state_indirect_answer_after_false_report"][
            "alternative_target"
        ]
        == "indirect false answer"
    )


def test_indirect_alternative_uses_prompt_matched_counterfactual_not_direct_report():
    rows = _scenario("s1", "family-a")
    by_objective = {row["objective"]: row for row in rows}
    by_objective["counterfactual_state_action"]["target"] = "open_agenda_v3"
    by_objective["true_state_action_after_false_report"]["target"] = "open_agenda_v4"
    by_objective["counterfactual_state_action"]["prompt"] = (
        "Allowed actions: open_agenda_v4, open_agenda_v3. Return exactly one action."
    )
    by_objective["true_state_action_after_false_report"]["prompt"] = (
        "Allowed actions: open_agenda_v4, open_agenda_v3. Return exactly one action."
    )
    by_objective["true_state_indirect_answer_after_false_report"]["target"] = (
        "agenda_v4 should be opened"
    )
    enriched = enrich_behavior_alternatives(rows)
    indirect = next(
        row
        for row in enriched
        if row["objective"] == "true_state_indirect_answer_after_false_report"
    )
    assert indirect["alternative_target"] == "agenda_v3 should be opened"
    assert indirect["alternative_target"] != "false answer"


def test_counterfactual_indirect_alternative_handles_hyphenated_action_concepts():
    assert counterfactual_indirect_alternative(
        truthful_answer="It should use all-day mode.",
        true_state_action="create_all_day_milestone",
        counterfactual_action="create_timed_milestone",
        true_state_value="enabled",
        counterfactual_state_value="disabled",
    ) == "It should use timed mode."


def test_iid_split_keeps_complete_scenarios_and_is_deterministic():
    rows = [
        row
        for index in range(10)
        for row in _scenario(f"s{index}", "family-a")
    ]
    first_train, first_dev = split_iid_development(
        rows,
        seed=7,
        fraction=0.2,
    )
    second_train, second_dev = split_iid_development(
        reversed(rows),
        seed=7,
        fraction=0.2,
    )
    assert {row["record_id"] for row in first_train} == {
        row["record_id"] for row in second_train
    }
    assert {row["record_id"] for row in first_dev} == {
        row["record_id"] for row in second_dev
    }
    train_scenarios = {row["scenario_id"] for row in first_train}
    dev_scenarios = {row["scenario_id"] for row in first_dev}
    assert train_scenarios.isdisjoint(dev_scenarios)
    assert len(dev_scenarios) == 2
    assert all(
        sum(row["scenario_id"] == scenario for row in first_dev) == 6
        for scenario in dev_scenarios
    )


def test_iid_split_keeps_scenario_series_siblings_together():
    rows = [
        row
        for series in ("alpha", "beta", "gamma", "delta")
        for world in range(1, 6)
        for row in _scenario(f"social.status.{series}.{world:04d}", "family-a")
    ]
    train, development = split_iid_development(rows, seed=7, fraction=0.2)
    train_ids = {row["scenario_id"] for row in train}
    development_ids = {row["scenario_id"] for row in development}
    for series in ("alpha", "beta", "gamma", "delta"):
        siblings = {
            f"social.status.{series}.{world:04d}" for world in range(1, 6)
        }
        assert siblings <= train_ids or siblings <= development_ids


def test_source_training_admission_quarantines_pending_registry_source():
    admitted, reason = source_training_admission(
        {
            "source_id": "ai2_tulu_3_sft_snapshot",
            "reuse_decision": "subset_terms_review_required_at_training_gate",
            "eligibility_override": "quarantined_pending_subset_terms",
        }
    )
    assert admitted is False
    assert reason == "quarantined_pending_subset_terms"


def test_audit_seal_evidence_verifies_hash_without_parsing(tmp_path):
    audit = tmp_path / "audit.jsonl"
    audit.write_bytes(b'{"secret": true}\n')
    evidence = audit_seal_evidence(
        audit,
        expected_sha256=(
            "6ffda347c9cf978fe74773347f2b2c10a09f59aa0f2474d20b5012b5f1e9821c"
        ),
    )
    assert evidence == {
        "content_parsed_by_builder": False,
        "hash_matches": True,
        "observed_sha256": (
                "6ffda347c9cf978fe74773347f2b2c10a09f59aa0f2474d20b5012b5f1e9821c"
        ),
        "verification": "sha256_bytes_only",
    }


class _WhitespaceTokenizer:
    eos_token = " <eos>"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return " ".join(message["content"] for message in messages) + " assistant "

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": text.split()}


def test_text_qualification_fails_closed_on_length_and_semantic_exclusions():
    rows = [
        {
            "record_id": "good",
            "payload": {
                "messages": [
                    {"role": "user", "content": "short prompt"},
                    {"role": "assistant", "content": "short answer"},
                ]
            },
        },
        {
            "record_id": "too-long",
            "payload": {
                "messages": [
                    {"role": "user", "content": "one two three four five six"},
                    {"role": "assistant", "content": "seven eight nine"},
                ]
            },
        },
        {
            "record_id": "known-bad",
            "payload": {
                "messages": [
                    {"role": "user", "content": "short prompt"},
                    {"role": "assistant", "content": "short answer"},
                ]
            },
        },
    ]
    qualified, exclusions = qualify_text_preservation_rows(
        rows,
        tokenizer=_WhitespaceTokenizer(),
        max_length=8,
        semantic_exclusions={"known-bad": "changes_problem_to_force_answer"},
    )
    assert [row["record_id"] for row in qualified] == ["good"]
    assert qualified[0]["qualification"]["token_length"] == 6
    assert exclusions == [
        {
            "record_id": "known-bad",
            "reason": "changes_problem_to_force_answer",
        },
        {"record_id": "too-long", "reason": "token_length_11_exceeds_8"},
    ]


def test_paired_preference_loss_rewards_larger_preferred_margin():
    strong = paired_preference_loss(torch.tensor([3.0]), torch.tensor([1.0]))
    weak = paired_preference_loss(torch.tensor([1.0]), torch.tensor([3.0]))
    assert strong.item() < weak.item()


def test_clustered_bootstrap_is_deterministic_and_cluster_aware():
    first = clustered_bootstrap_mean(
        [1.0, 1.0, -1.0, -1.0],
        ["a", "a", "b", "b"],
        samples=200,
        seed=9,
    )
    second = clustered_bootstrap_mean(
        [1.0, 1.0, -1.0, -1.0],
        ["a", "a", "b", "b"],
        samples=200,
        seed=9,
    )
    assert first == second
    assert first["mean"] == pytest.approx(0.0)
    assert first["lower"] <= first["mean"] <= first["upper"]


def test_action_parser_accepts_exactly_one_allowed_action():
    actions = ["act_left", "act_right"]
    assert parse_allowed_action("act_left", actions) == "act_left"
    assert parse_allowed_action("I choose act_right.", actions) == "act_right"
    assert parse_allowed_action("act_left or act_right", actions) is None


def test_ordinary_lora_starts_as_exact_base_and_can_change_output():
    base = torch.nn.Linear(3, 2, bias=False)
    wrapped = OrdinaryLoRALinear(base, rank=1, alpha=1.0, dropout=0.0)
    inputs = torch.tensor([[1.0, 2.0, 3.0]])
    expected = base(inputs).detach()
    assert torch.equal(wrapped(inputs), expected)
    with torch.no_grad():
        wrapped.lora_a.fill_(1.0)
        wrapped.lora_b.fill_(1.0)
    assert not torch.equal(wrapped(inputs), expected)


def test_preservation_schedule_is_exactly_one_in_four_and_deterministic():
    behavior = [{"record_id": f"b{index}", "kind": "behavior"} for index in range(6)]
    preservation = [
        {"record_id": f"p{index}", "kind": "preservation"} for index in range(2)
    ]
    first = preservation_interleaved_schedule(behavior, preservation, seed=4)
    second = preservation_interleaved_schedule(reversed(behavior), preservation, seed=4)
    assert [row["record_id"] for row in first] == [
        row["record_id"] for row in second
    ]
    assert [row["kind"] for row in first] == [
        "behavior",
        "behavior",
        "behavior",
        "preservation",
        "behavior",
        "behavior",
        "behavior",
        "preservation",
    ]
