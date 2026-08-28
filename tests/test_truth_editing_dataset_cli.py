from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_build_cli_compiles_a_local_jsonl(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "q1",
                "question": "Which element has atomic number one?",
                "correct_answer": "hydrogen",
                "wrong_answers": ["helium"],
            }
        )
        + "\n"
    )
    output = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_truth_editing_dataset.py",
            "--source",
            f"fixture={source}",
            "--revision",
            "fixture=fixture-r1",
            "--output-dir",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["audit"]["valid"] is True
    assert (output / "manifest.json").is_file()


def test_build_cli_requires_pinned_revision(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text("[]\n")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_truth_editing_dataset.py",
            "--source",
            f"fixture={source}",
            "--output-dir",
            str(tmp_path / "out"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--revision" in result.stderr
