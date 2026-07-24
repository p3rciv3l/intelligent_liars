from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

from intelligent_liars.osworld_overlay import OSWORLD_COMMIT
from intelligent_liars.run_control import stable_sha256


GATE_20_TASK_IDS = (
    "chrome/bb5e4c0d-f964-439c-97b6-bdb9747de3f4",
    "chrome/7b6c7e24-c58a-49fc-a5bb-d57b80e5b4c3",
    "gimp/7a4deb26-d57d-4ea9-9a73-630f66a7b568",
    "gimp/554785e9-4523-4e7a-b8e1-8016f565f56a",
    "libreoffice_calc/357ef137-7eeb-4c80-a3bb-0951f26a8aff",
    "libreoffice_calc/42e0a640-4f19-4b28-973d-729602b5a4a7",
    "libreoffice_impress/5d901039-a89c-4bfb-967b-bf66f4df075e",
    "libreoffice_impress/550ce7e7-747b-495f-b122-acdc4d0b8e54",
    "libreoffice_writer/0810415c-bde4-4443-9047-d5f70165a697",
    "libreoffice_writer/0a0faba3-5580-44df-965d-f562a99b291c",
    "multi_apps/a74b607e-6bb5-4ea8-8a7c-5d97c7bbcd2a",
    "multi_apps/510f64c8-9bcc-4be1-8d30-638705850618",
    "os/5ea617a3-0e86-4ba6-aab2-dac9aa2e8d57",
    "os/5812b315-e7bd-4265-b51f-863c02174c28",
    "thunderbird/dfac9ee8-9bc4-4cdc-b465-4a4bfcd2f397",
    "thunderbird/15c3b339-88f7-4a86-ab16-e71c58dcb01e",
    "vlc/59f21cfb-0120-4326-b255-a5b827b38967",
    "vlc/8f080098-ddb1-424c-b438-4e96e5e4786e",
    "vs_code/0ed39f63-6049-43d4-ba4d-5fa2fe04a951",
    "vs_code/276cc624-87ea-4f08-ab93-f770e3790175",
)

ACCEPTANCE_GATES = {
    "artifact_checksum_completeness": 1.0,
    "coordinate_key_text_retention_gte": 0.99,
    "implicit_terminal_actions_eq": 0,
    "official_evaluator_invocation": 1.0,
    "proxy": {
        "mode": "enabled",
        "required_checks": ["dns", "tls", "status", "expected_content", "screenshot"],
    },
    "score": {
        "gate_20_mean_gte": 0.20,
        "small_39_engineering_mean_gte": 0.2564,
        "small_39_target_compatible_mean": [0.3077, 0.3846],
    },
    "tool_call_initial_validity_gte": 0.99,
    "tool_call_valid_after_correction_or_typed_failure": 1.0,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _task_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    return [
        f"{domain}/{task_id}"
        for domain, task_ids in payload.items()
        for task_id in task_ids
    ]


def _task_metadata(
    checkout: Path, task_ids: list[str]
) -> tuple[dict[str, str], dict[str, str]]:
    configs: dict[str, str] = {}
    evaluators: dict[str, str] = {}
    for task_id in task_ids:
        domain, example_id = task_id.split("/", 1)
        path = (
            checkout
            / "evaluation_examples/examples"
            / domain
            / f"{example_id}.json"
        )
        payload = json.loads(path.read_text())
        if payload.get("id") != example_id:
            raise ValueError(f"task ID mismatch in {path}")
        configs[task_id] = _sha256(path)
        evaluators[task_id] = stable_sha256(payload.get("evaluator"))
    return configs, evaluators


def _build(
    base: dict[str, Any],
    *,
    checkout: Path,
    name: str,
    task_ids: list[str],
    source: str,
    source_sha256: str,
) -> dict[str, Any]:
    payload = deepcopy(base)
    configs, evaluators = _task_metadata(checkout, task_ids)
    payload["desktop"]["enable_proxy"] = True
    payload["run"]["step_cap"] = 100
    payload["task_grid"] = {
        "name": name,
        "sha256": stable_sha256(task_ids),
        "source": source,
        "source_sha256": source_sha256,
        "task_config_sha256": configs,
        "evaluator_sha256": evaluators,
        "task_ids": task_ids,
    }
    payload["evaluation"] = {
        "implementation": "official-task-evaluator",
        "task_config_sha256": configs,
        "evaluator_sha256": evaluators,
    }
    payload["acceptance_gates"] = deepcopy(ACCEPTANCE_GATES)
    return payload


def generate(checkout: Path, template_dir: Path) -> tuple[Path, Path]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip()
    if head != OSWORLD_COMMIT:
        raise ValueError(f"OSWorld checkout must be {OSWORLD_COMMIT}, got {head}")
    base_path = template_dir / "one_task_50.template.json"
    base = json.loads(base_path.read_text())
    small_source = checkout / "evaluation_examples/test_small.json"
    small_ids = _task_ids(small_source)
    if len(small_ids) != 39:
        raise ValueError("pinned OSWorld Small must contain exactly 39 tasks")
    gate_ids = list(GATE_20_TASK_IDS)
    gate = _build(
        base,
        checkout=checkout,
        name="qwen3vl_stratified_gate_20_100",
        task_ids=gate_ids,
        source="reconstruction-gate-20",
        source_sha256=stable_sha256(gate_ids),
    )
    small = _build(
        base,
        checkout=checkout,
        name="qwen3vl_osworld_small_39_100",
        task_ids=small_ids,
        source="evaluation_examples/test_small.json",
        source_sha256=_sha256(small_source),
    )
    outputs = (
        template_dir / "qwen3vl_gate_20_100.template.json",
        template_dir / "qwen3vl_small_39_100.template.json",
    )
    for path, payload in zip(outputs, (gate, small), strict=True):
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--osworld-checkout", type=Path, required=True)
    parser.add_argument(
        "--template-dir",
        type=Path,
        default=Path("docs/evaluation/osworld_templates"),
    )
    args = parser.parse_args()
    for path in generate(args.osworld_checkout.resolve(), args.template_dir.resolve()):
        print(path)


if __name__ == "__main__":
    main()
