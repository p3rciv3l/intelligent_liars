from __future__ import annotations

import os
import io
import json
import runpy
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/run_truth_editing_cuda_fleet_worker.py"


def test_paid_judge_failure_response_is_bound_to_request_and_worker_keeps_serving(
    monkeypatch,
) -> None:
    namespace = runpy.run_path(str(SCRIPT), run_name="truth_editing_worker_test")
    digest = "d" * 64
    requests = [
        {"request_sha256": "a" * 64},
        {"request_sha256": "b" * 64},
        {"command": "stop"},
    ]
    calls = 0

    def result(request, _run, _production):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise namespace["ProductionCompositionError"](
                "paid semantic judge failed closed; "
                f"failure_receipt_sha256={digest}"
            )
        return {
            "request_sha256": request["request_sha256"],
            "result": {
                "outcome_kind": "successful",
                "metrics": {"deception": 0.5},
                "detail": None,
            },
        }

    namespace["main"].__globals__["_result"] = result
    monkeypatch.setattr(
        namespace["ProductionRunConfig"], "open", lambda _path: object()
    )
    namespace["main"].__globals__["open_production_run"] = lambda _path: object()
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--config", "config.json"])
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO("".join(json.dumps(item) + "\n" for item in requests)),
    )
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)

    assert namespace["main"]() == 0
    lines = [json.loads(line) for line in output.getvalue().splitlines()]
    assert lines[0] == {
        "fatal": True,
        "request_sha256": "a" * 64,
        "failure_receipt_sha256": digest,
    }
    assert lines[1]["request_sha256"] == "b" * 64


def test_worker_imports_repository_package_without_install_or_pythonpath(
    tmp_path: Path,
) -> None:
    """A fresh clone can launch the worker directly from its script path."""

    repository_root = SCRIPT.parents[1]
    source_root = repository_root / "src"
    probe = f"""
import importlib.abc
import runpy
import sys

source_root = {str(source_root)!r}

class RejectInstalledProject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "intelligent_liars" and source_root not in sys.path:
            raise ModuleNotFoundError("installed project package intentionally unavailable")
        return None

sys.meta_path.insert(0, RejectInstalledProject())
runpy.run_path({str(SCRIPT)!r}, run_name="fresh_clone_worker")
"""
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
