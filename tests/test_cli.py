from __future__ import annotations

import pytest

pytest.importorskip("typer")

from typer.testing import CliRunner

from intelligent_liars.cli import app


def test_extract_activations_sycophancy_guard_runs_before_model_load(tmp_path, monkeypatch):
    def fail_model_load(*args, **kwargs):
        del args, kwargs
        raise AssertionError("model load should not be reached")

    monkeypatch.setattr("intelligent_liars.cli.load_model_and_processor", fail_model_load)
    monkeypatch.setattr("intelligent_liars.cli.load_nnsight_bundle", fail_model_load)

    result = CliRunner().invoke(
        app,
        [
            "extract-activations",
            "--task",
            "sycophancy__mmlu_stem_conf_part_2",
            "--project-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code != 0
    assert not isinstance(result.exception, AssertionError)
    assert result.exception is not None
    assert "Sycophancy source directory not found" in str(result.exception)
