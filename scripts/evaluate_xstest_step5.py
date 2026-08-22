#!/usr/bin/env python3
"""Build and optionally score a complete externally judged Step 5 XSTest run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intelligent_liars.safety_refusal_eval import (
    build_response_inventory,
    file_sha256,
    read_jsonl,
    score_response_inventory,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--model-identity", required=True)
    parser.add_argument("--inventory-output", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--score-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = json.loads(args.plan.read_text())
    if plan.get("format") != "tinylora_step5_plan_v1":
        raise ValueError("unsupported Step 5 plan format")
    if (args.labels is None) != (args.score_output is None):
        raise ValueError("--labels and --score-output must be provided together")

    plan_sha256 = file_sha256(args.plan)
    inventory = build_response_inventory(
        read_jsonl(args.prompts),
        read_jsonl(args.responses),
        source_plan_sha256=plan_sha256,
        model_identity=args.model_identity,
    )
    write_jsonl(args.inventory_output, inventory)
    inventory_sha256 = file_sha256(args.inventory_output)

    output: dict[str, object] = {
        "status": "inventory_ready",
        "records": len(inventory),
        "source_plan_sha256": plan_sha256,
        "response_inventory_sha256": inventory_sha256,
    }
    if args.labels is not None and args.score_output is not None:
        score = score_response_inventory(
            inventory,
            read_jsonl(args.labels),
            source_plan_sha256=plan_sha256,
            response_inventory_sha256=inventory_sha256,
        )
        score["external_labels_sha256"] = file_sha256(args.labels)
        write_json(args.score_output, score)
        output = score
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
