from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from intelligent_liars.safety_refusal_eval import (
    build_response_inventory,
    canonical_jsonl_sha256,
    score_response_inventory,
    write_jsonl,
)


PLAN_SHA256 = "a" * 64
RUN_SHA256 = "b" * 64


def _prompts() -> list[dict[str, str]]:
    rows = []
    for index in range(1, 451):
        expected = "comply" if index <= 250 else "refuse"
        rows.append(
            {
                "format": "tinylora_step5_safety_eval_v1",
                "record_id": f"xstest.{index:03d}",
                "split_group_id": f"xstest.type_{index % 3}",
                "prompt": f"prompt {index}",
                "expected_behavior": expected,
                "xstest_type": f"type_{index % 3}",
            }
        )
    return rows


def _responses() -> list[dict[str, str]]:
    return [
        {
            "record_id": f"xstest.{index:03d}",
            "response": f"response {index}",
            "run_identity_sha256": RUN_SHA256,
        }
        for index in range(1, 451)
    ]


def _labels(
    inventory: list[dict[str, str]],
    *,
    inventory_sha256: str | None = None,
) -> list[dict[str, str]]:
    inventory_sha256 = inventory_sha256 or canonical_jsonl_sha256(inventory)
    return [
        {
            "record_id": row["record_id"],
            "observed_behavior": row["expected_behavior"],
            "source_plan_sha256": PLAN_SHA256,
            "response_inventory_sha256": inventory_sha256,
        }
        for row in inventory
    ]


def test_response_inventory_is_complete_and_order_independent() -> None:
    expected = build_response_inventory(
        _prompts(),
        _responses(),
        source_plan_sha256=PLAN_SHA256,
        model_identity="candidate-a/seed-1",
    )
    reordered = build_response_inventory(
        reversed(_prompts()),
        reversed(_responses()),
        source_plan_sha256=PLAN_SHA256,
        model_identity="candidate-a/seed-1",
    )

    assert reordered == expected
    assert len(expected) == 450
    assert expected[0] == {
        "format": "tinylora_xstest_response_v1",
        "record_id": "xstest.001",
        "split_group_id": "xstest.type_1",
        "prompt": "prompt 1",
        "expected_behavior": "comply",
        "xstest_type": "type_1",
        "response": "response 1",
        "model_identity": "candidate-a/seed-1",
        "source_plan_sha256": PLAN_SHA256,
        "run_identity_sha256": RUN_SHA256,
    }


def test_response_inventory_rejects_missing_response() -> None:
    with pytest.raises(ValueError, match="response IDs do not exactly match"):
        build_response_inventory(
            _prompts(),
            _responses()[:-1],
            source_plan_sha256=PLAN_SHA256,
            model_identity="candidate-a/seed-1",
        )


def test_score_reports_safe_and_unsafe_failures_separately() -> None:
    inventory = build_response_inventory(
        _prompts(),
        _responses(),
        source_plan_sha256=PLAN_SHA256,
        model_identity="candidate-a/seed-1",
    )
    labels = _labels(inventory)
    labels[0]["observed_behavior"] = "refuse"
    labels[-1]["observed_behavior"] = "comply"

    score = score_response_inventory(
        inventory,
        labels,
        source_plan_sha256=PLAN_SHA256,
        response_inventory_sha256=canonical_jsonl_sha256(inventory),
    )

    assert score["status"] == "complete"
    assert score["run_identity_sha256"] == RUN_SHA256
    assert score["records"] == 450
    assert score["correct"] == 448
    assert score["accuracy"] == pytest.approx(448 / 450)
    assert score["safe"]["records"] == 250
    assert score["safe"]["compliance_rate"] == pytest.approx(249 / 250)
    assert score["safe"]["overrefusal_rate"] == pytest.approx(1 / 250)
    assert score["unsafe"]["records"] == 200
    assert score["unsafe"]["refusal_rate"] == pytest.approx(199 / 200)
    assert score["unsafe"]["unsafe_compliance_rate"] == pytest.approx(1 / 200)
    assert set(score["by_xstest_type"]) == {"type_0", "type_1", "type_2"}
    assert len(score["scored_records"]) == 450
    assert score["scored_records"][0] == {
        "record_id": "xstest.001",
        "split_group_id": "xstest.type_1",
        "xstest_type": "type_1",
        "expected_behavior": "comply",
        "observed_behavior": "refuse",
        "correct": False,
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda labels: labels.pop(), "label IDs do not exactly match"),
        (
            lambda labels: labels[0].update(observed_behavior="unknown"),
            "unsupported observed_behavior",
        ),
        (
            lambda labels: labels[0].update(response_inventory_sha256="c" * 64),
            "response inventory hash mismatch",
        ),
        (
            lambda labels: labels[0].update(source_plan_sha256="d" * 64),
            "source plan hash mismatch",
        ),
    ],
)
def test_score_fails_closed_for_incomplete_or_unbound_labels(mutate, message) -> None:
    inventory = build_response_inventory(
        _prompts(),
        _responses(),
        source_plan_sha256=PLAN_SHA256,
        model_identity="candidate-a/seed-1",
    )
    labels = _labels(inventory)
    inventory_sha256 = canonical_jsonl_sha256(inventory)
    mutate(labels)

    with pytest.raises(ValueError, match=message):
        score_response_inventory(
            inventory,
            labels,
            source_plan_sha256=PLAN_SHA256,
            response_inventory_sha256=inventory_sha256,
        )


def test_score_rejects_old_labels_after_response_changes() -> None:
    inventory = build_response_inventory(
        _prompts(),
        _responses(),
        source_plan_sha256=PLAN_SHA256,
        model_identity="candidate-a/seed-1",
    )
    old_inventory_sha256 = canonical_jsonl_sha256(inventory)
    labels = _labels(inventory, inventory_sha256=old_inventory_sha256)
    inventory[0]["response"] = "changed response"

    with pytest.raises(ValueError, match="response inventory hash mismatch"):
        score_response_inventory(
            inventory,
            labels,
            source_plan_sha256=PLAN_SHA256,
            response_inventory_sha256=old_inventory_sha256,
        )


def test_cli_writes_inventory_then_scores_external_labels(tmp_path: Path) -> None:
    repository_root = Path(__file__).parents[1]
    script = repository_root / "scripts" / "evaluate_xstest_step5.py"
    plan = tmp_path / "plan.json"
    prompts = tmp_path / "prompts.jsonl"
    responses = tmp_path / "responses.jsonl"
    inventory_path = tmp_path / "inventory.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    score_path = tmp_path / "score.json"
    environment = {
        **os.environ,
        "PYTHONPATH": str(repository_root / "src"),
    }
    write_jsonl(prompts, _prompts())
    write_jsonl(responses, _responses())
    prompt_sha256 = hashlib.sha256(prompts.read_bytes()).hexdigest()
    plan.write_text(
        json.dumps(
            {
                "format": "tinylora_step5_plan_v1",
                "outputs": {"safety_refusal_development": {"sha256": prompt_sha256}},
            },
            sort_keys=True,
        )
        + "\n"
    )

    inventory_run = subprocess.run(
        [
            sys.executable,
            str(script),
            "--plan",
            str(plan),
            "--prompts",
            str(prompts),
            "--responses",
            str(responses),
            "--model-identity",
            "candidate-a/seed-1",
            "--inventory-output",
            str(inventory_path),
        ],
        cwd=repository_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert inventory_run.returncode == 0, inventory_run.stderr
    inventory_sha256 = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
    source_plan_sha256 = hashlib.sha256(plan.read_bytes()).hexdigest()
    inventory = [json.loads(line) for line in inventory_path.read_text().splitlines()]
    labels = [
        {
            "record_id": row["record_id"],
            "observed_behavior": row["expected_behavior"],
            "source_plan_sha256": source_plan_sha256,
            "response_inventory_sha256": inventory_sha256,
        }
        for row in inventory
    ]
    write_jsonl(labels_path, labels)

    score_run = subprocess.run(
        [
            sys.executable,
            str(script),
            "--plan",
            str(plan),
            "--prompts",
            str(prompts),
            "--responses",
            str(responses),
            "--model-identity",
            "candidate-a/seed-1",
            "--inventory-output",
            str(inventory_path),
            "--labels",
            str(labels_path),
            "--score-output",
            str(score_path),
        ],
        cwd=repository_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert score_run.returncode == 0, score_run.stderr
    score = json.loads(score_path.read_text())
    assert score["status"] == "complete"
    assert score["accuracy"] == 1.0
    assert score["source_plan_sha256"] == source_plan_sha256
    assert score["response_inventory_sha256"] == inventory_sha256

    altered_prompts = tmp_path / "altered-prompts.jsonl"
    altered_rows = _prompts()
    altered_rows[0]["prompt"] = "altered prompt"
    write_jsonl(altered_prompts, altered_rows)
    altered_run = subprocess.run(
        [
            sys.executable,
            str(script),
            "--plan",
            str(plan),
            "--prompts",
            str(altered_prompts),
            "--responses",
            str(responses),
            "--model-identity",
            "candidate-a/seed-1",
            "--inventory-output",
            str(tmp_path / "altered-inventory.jsonl"),
        ],
        cwd=repository_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert altered_run.returncode != 0
    assert "prompt artifact hash does not match" in altered_run.stderr

    labels_before = labels_path.read_bytes()
    collision_run = subprocess.run(
        [
            sys.executable,
            str(script),
            "--plan",
            str(plan),
            "--prompts",
            str(prompts),
            "--responses",
            str(responses),
            "--model-identity",
            "candidate-a/seed-1",
            "--inventory-output",
            str(labels_path),
            "--labels",
            str(labels_path),
            "--score-output",
            str(tmp_path / "collision-score.json"),
        ],
        cwd=repository_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert collision_run.returncode != 0
    assert "path collision" in collision_run.stderr
    assert labels_path.read_bytes() == labels_before
