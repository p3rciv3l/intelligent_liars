from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


Message = dict[str, Any]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare cleaned rollout prompt sets from Truth Spec rollouts.")
    parser.add_argument("source", type=Path, help="Reference rollout JSON path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/rollout_prompts"),
        help="Directory for cleaned prompt set JSON files.",
    )
    args = parser.parse_args()

    output_path = prepare_prompt_set(args.source, args.output_dir)
    print(output_path)


def prepare_prompt_set(source_path: Path, output_dir: Path) -> Path:
    data = json.loads(source_path.read_text())
    base = str(data["base_name"])
    variant = str(data["variant"])
    source_model = str(data["model"])
    source_slug = _source_slug(source_path, base, variant, source_model)
    prompt_set = f"truth-spec-{source_slug}"

    payload = {
        "dataset_class": "rollout_prompt_set",
        "base_name": base,
        "variant": variant,
        "prompt_set": prompt_set,
        "source_model": source_model,
        "source_slug": source_slug,
        "source_path": str(source_path),
        "examples": [_clean_rollout_item(item) for item in data["rollouts"]],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{base}__{variant}__{prompt_set}.json"
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")
    return output_path


def _source_slug(source_path: Path, base: str, variant: str, source_model: str) -> str:
    prefix = f"{base}__{variant}__"
    if source_path.stem.startswith(prefix):
        return source_path.stem[len(prefix) :]
    return source_model.split("/")[-1].replace("_", "-").replace(".", "-").lower()


def _clean_rollout_item(item: dict[str, Any]) -> dict[str, Any]:
    messages = [_normalise_message(message) for message in item["input_messages"]]
    if messages and messages[-1]["role"] == "assistant" and not messages[-1]["content"].strip():
        messages = messages[:-1]

    metadata = {
        "source_metadata": item.get("metadata", {}),
    }
    if "expect_deception" in item:
        metadata["source_expect_deception"] = item["expect_deception"]

    return {
        "input_messages": messages,
        "expected_label": item.get("expected_label"),
        "input_str": _messages_to_text(messages),
        "metadata": metadata,
    }


def _normalise_message(message: Message) -> Message:
    return {
        "role": str(message["role"]),
        "content": message.get("content", ""),
        "detect": bool(message.get("detect", False)),
    }


def _messages_to_text(messages: list[Message]) -> str:
    return "\n".join(f"{message['role']}: {message.get('content', '')}" for message in messages)


if __name__ == "__main__":
    main()
