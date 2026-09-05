from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
WRAPPER = REPOSITORY / "scripts" / "vastai_openai_ua.py"
LAUNCHERS = (
    REPOSITORY / "scripts" / "run_truth_editing_vast_prerequisites.py",
    REPOSITORY / "scripts" / "run_truth_editing_vast_production.py",
)
REQUIRED_USER_AGENT = "OpenAI File Downloader, XaiImageApiFetch/1.0"


def test_vast_wrapper_applies_required_user_agent_from_any_working_directory(
    tmp_path: Path,
) -> None:
    fake_modules = tmp_path / "fake-modules"
    fake_modules.mkdir()
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_vastai = fake_bin / "vastai"
    fake_vastai.write_text(f"#!{sys.executable}\n")
    fake_vastai.chmod(0o755)
    (fake_modules / "requests.py").write_text(
        """
class Session:
    def request(self, method, url, **kwargs):
        print(kwargs["headers"]["User-Agent"])
        return object()

class _Sessions:
    Session = Session

sessions = _Sessions()
""".lstrip()
    )
    (fake_modules / "vast.py").write_text(
        """
import requests

def main():
    requests.sessions.Session().request("GET", "https://example.invalid")
    return 0
""".lstrip()
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(fake_modules)
    environment["PATH"] = os.pathsep.join((str(fake_bin), environment["PATH"]))
    environment["PORTABILITY_TEST_SECRET"] = "must-not-be-printed"

    completed = subprocess.run(
        [sys.executable, str(WRAPPER)],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == REQUIRED_USER_AGENT
    assert "must-not-be-printed" not in completed.stdout + completed.stderr


def test_vast_wrapper_skips_broken_earlier_launcher(tmp_path: Path) -> None:
    fake_modules = tmp_path / "fake-modules"
    fake_modules.mkdir()
    (fake_modules / "requests.py").write_text(
        "class Session:\n"
        "    def request(self, method, url, **kwargs): return object()\n"
        "class _Sessions: Session = Session\n"
        "sessions = _Sessions()\n"
    )
    (fake_modules / "vast.py").write_text("def main(): return 0\n")
    bad_bin = tmp_path / "bad-bin"
    good_bin = tmp_path / "good-bin"
    bad_bin.mkdir()
    good_bin.mkdir()
    bad_python = tmp_path / "bad-python"
    bad_python.write_text("#!/bin/sh\nexit 1\n")
    bad_python.chmod(0o755)
    for directory, interpreter in ((bad_bin, bad_python), (good_bin, Path(sys.executable))):
        launcher = directory / "vastai"
        launcher.write_text(f"#!{interpreter}\n")
        launcher.chmod(0o755)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(fake_modules)
    environment["PATH"] = os.pathsep.join((str(bad_bin), str(good_bin), environment["PATH"]))

    completed = subprocess.run(
        [sys.executable, str(WRAPPER)],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_truth_editing_vast_launchers_default_to_committed_portable_wrapper(
    tmp_path: Path,
) -> None:
    for launcher in LAUNCHERS:
        completed = subprocess.run(
            [sys.executable, str(launcher), "--help"],
            cwd=tmp_path,
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 0, f"{launcher.name}: {completed.stderr}"
        compact_help = "".join(completed.stdout.split())
        assert str(WRAPPER) in compact_help, launcher.name
        assert ".local/vastai_openai_ua.py" not in compact_help, launcher.name
        assert "/Users/student/Library/Python/" not in compact_help, launcher.name
