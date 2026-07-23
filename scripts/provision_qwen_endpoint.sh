#!/usr/bin/env bash
set -euo pipefail

: "${QWEN_ENDPOINT_API_KEY:?QWEN_ENDPOINT_API_KEY must be set}"
: "${HF_HOME:?HF_HOME must be set}"

key_directory=/run/qwen-endpoint
key_path="${key_directory}/api-key"
install -d -m 0700 "${key_directory}"
umask 077
printf '%s' "${QWEN_ENDPOINT_API_KEY}" >"${key_path}"
chmod 0600 "${key_path}"

uv sync --frozen --no-dev --group endpoint --no-editable
export PATH="${PWD}/.venv/bin:${PATH}"
python -c 'import torch, torchvision; assert torch.__version__.split("+")[0] == "2.5.1"; assert torchvision.__version__.split("+")[0] == "0.20.1"'
qwen-endpoint --dry-run
export QWEN_ENDPOINT_API_KEY
exec qwen-endpoint --host 127.0.0.1 --port 8000
