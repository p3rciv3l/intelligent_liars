from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from intelligent_liars.insider_trading import generate_insider_trading_transcripts
from intelligent_liars.rollouts import GenerationSettings


def test_generate_insider_trading_transcripts_requires_prompt_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="No insider-trading prompt YAML files matched"):
        generate_insider_trading_transcripts(
            bundle=object(),  # type: ignore[arg-type]
            project_root=tmp_path,
            output_path=tmp_path / "out.json",
            settings=GenerationSettings(),
        )


def test_generate_insider_trading_transcripts_writes_run_metadata(tmp_path, monkeypatch):
    prompt_dir = tmp_path / "data" / "insider_trading" / "prompts"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "default.yaml").write_text(
        """
messages:
  - role: user
    content: Should we trade?
misalignment_string: execute_trade
""".strip()
    )
    monkeypatch.setattr(
        "intelligent_liars.insider_trading.generate_completions",
        lambda **kwargs: ['Reasoning: "no trade" Action: hold'],
    )

    summary = generate_insider_trading_transcripts(
        bundle=SimpleNamespace(model_id="Qwen/Qwen3-VL-8B-Thinking"),
        project_root=tmp_path,
        output_path=tmp_path / "out.json",
        settings=GenerationSettings(max_new_tokens=16, flush_every=1),
    )

    payload = json.loads((tmp_path / "out.json").read_text())
    metadata = payload[0]["metadata"]["run_metadata"]
    assert summary.generated_examples == 1
    assert metadata["model_id"] == "Qwen/Qwen3-VL-8B-Thinking"
    assert metadata["prompt_glob"] == "data/insider_trading/prompts/**/*.yaml"
    assert metadata["prompt_paths"] == ["data/insider_trading/prompts/default.yaml"]
    assert len(metadata["prompt_sha256"]["data/insider_trading/prompts/default.yaml"]) == 64
    assert metadata["generation_settings"]["max_new_tokens"] == 16
    assert "flush_every" not in metadata["generation_settings"]


def test_generate_insider_trading_transcripts_rejects_incompatible_resume_metadata(tmp_path):
    prompt_dir = tmp_path / "data" / "insider_trading" / "prompts"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "default.yaml").write_text(
        """
messages:
  - role: user
    content: Should we trade?
""".strip()
    )
    output_path = tmp_path / "out.json"
    output_path.write_text(
        json.dumps(
            [
                {
                    "transcript": [],
                    "metadata": {
                        "run_id": "data/insider_trading/prompts/default.yaml::0",
                        "run_metadata": {"model_id": "other"},
                    },
                }
            ]
        )
    )

    with pytest.raises(ValueError, match="incompatible run metadata"):
        generate_insider_trading_transcripts(
            bundle=SimpleNamespace(model_id="Qwen/Qwen3-VL-8B-Thinking"),
            project_root=tmp_path,
            output_path=output_path,
            settings=GenerationSettings(),
        )
