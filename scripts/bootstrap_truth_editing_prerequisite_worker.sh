#!/usr/bin/env bash
set -euo pipefail

wheel=/tmp/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
wheel_url='https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl'
wheel_sha=d944fc7d2f962bce83fc4708c2fc0c21eaf8255962a0b350ae919362a51b7ef2
ninja_wheel=/tmp/ninja-1.11.1.1-py2.py3-none-manylinux1_x86_64.manylinux_2_5_x86_64.whl
ninja_url='https://files.pythonhosted.org/packages/6d/92/8d7aebd4430ab5ff65df2bfee6d5745f95c004284db2d8ca76dcbfd9de47/ninja-1.11.1.1-py2.py3-none-manylinux1_x86_64.manylinux_2_5_x86_64.whl'
ninja_sha=84502ec98f02a037a169c4b0d5d86075eaf6afc55e1879003d6cab51ced2ea4b

python -m pip install --require-hashes -r docker/truth-editing/requirements.lock
curl --fail --location --retry 3 \
  --user-agent 'OpenAI File Downloader, XaiImageApiFetch/1.0' \
  --output "${wheel}" "${wheel_url}"
printf '%s  %s\n' "${wheel_sha}" "${wheel}" | sha256sum --check --strict
python -m pip install --no-deps "${wheel}"
curl --fail --location --retry 3 \
  --user-agent 'OpenAI File Downloader, XaiImageApiFetch/1.0' \
  --output "${ninja_wheel}" "${ninja_url}"
printf '%s  %s\n' "${ninja_sha}" "${ninja_wheel}" | sha256sum --check --strict
python -m pip install --force-reinstall --no-deps "${ninja_wheel}"
if ! pip_check_output="$(python -m pip check 2>&1)"; then
  if [[ "${pip_check_output}" != 'ninja 1.11.1.1 is not supported on this platform' ]]; then
    printf '%s\n' "${pip_check_output}" >&2
    exit 1
  fi
  python - <<'PY'
import hashlib
import importlib.metadata
import platform
import subprocess
from pathlib import Path

import ninja

expected_binary_sha = "68f6c375c4234305bff9790aa232815b38924390cbb6ad4987ea0f94ad2bc410"
if platform.machine() not in {"x86_64", "AMD64"}:
    raise SystemExit("verified Ninja exception requires x86_64")
if importlib.metadata.version("ninja") != "1.11.1.1":
    raise SystemExit("verified Ninja distribution version changed")
binary = Path(ninja.BIN_DIR) / "ninja"
if binary.read_bytes()[:4] != b"\x7fELF":
    raise SystemExit("verified Ninja binary is not ELF")
if hashlib.sha256(binary.read_bytes()).hexdigest() != expected_binary_sha:
    raise SystemExit("verified Ninja binary hash changed")
version = subprocess.run(
    [binary, "--version"], check=True, capture_output=True, text=True, timeout=10
).stdout.strip()
if version != "1.11.1.git.kitware.jobserver-1":
    raise SystemExit("verified Ninja executable version changed")
PY
fi
python docker/truth-editing/validate_runtime.py --metadata-only
