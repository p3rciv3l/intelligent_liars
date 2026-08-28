from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "run_offline_judge_calibration.py"
)
SPEC = importlib.util.spec_from_file_location("run_offline_judge_calibration", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _fixture(tmp_path: Path) -> Path:
    path = tmp_path / "fixture.json"
    path.write_text('{"fixture":"stored-only"}\n', encoding="utf-8")
    return path


def test_cli_prints_canonical_report_to_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _fixture(tmp_path)
    seen: list[Path] = []

    def calibrate(path: Path) -> dict[str, object]:
        seen.append(path)
        return {"z": [2, 1], "a": {"unicode": "café"}}

    monkeypatch.setattr(MODULE, "calibrate_offline_judge_fixture", calibrate)

    assert MODULE.main([str(fixture)]) == 0

    assert seen == [fixture]
    assert capsys.readouterr().out == (
        '{"a":{"unicode":"café"},"z":[2,1]}\n'
    )


def test_cli_writes_new_output_without_also_printing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        MODULE,
        "calibrate_offline_judge_fixture",
        lambda _path: {"format": "report_v1", "count": 3},
    )

    assert MODULE.main([str(fixture), "--output", str(output)]) == 0

    assert output.read_bytes() == b'{"count":3,"format":"report_v1"}\n'
    assert capsys.readouterr().out == ""


def test_cli_refuses_to_clobber_an_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "report.json"
    output.write_text("user work\n", encoding="utf-8")
    monkeypatch.setattr(
        MODULE,
        "calibrate_offline_judge_fixture",
        lambda _path: {"format": "report_v1"},
    )

    with pytest.raises(FileExistsError):
        MODULE.main([str(fixture), "--output", str(output)])

    assert output.read_text(encoding="utf-8") == "user work\n"


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), -float("inf")])
def test_cli_fails_closed_before_emitting_a_non_finite_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    non_finite: float,
) -> None:
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(
        MODULE,
        "calibrate_offline_judge_fixture",
        lambda _path: {"invalid": non_finite},
    )

    with pytest.raises(ValueError, match="JSON compliant"):
        MODULE.main([str(fixture)])

    assert capsys.readouterr().out == ""


def test_cli_module_has_no_live_judge_or_configuration_surface() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "OpenRouter" not in source
    assert "judge_client" not in source
    assert "openrouter_client" not in source
    assert "os.environ" not in source
    assert "preflight" not in source
    assert "--model" not in source
    assert "--provider" not in source
    assert "--api" not in source

    parser = MODULE.build_parser()
    actions = {action.dest for action in parser._actions}
    assert actions == {"help", "fixture", "output"}
