from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/run_truth_editing_cuda_fleet_worker.py"


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
