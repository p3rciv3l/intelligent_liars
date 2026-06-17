#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv sync --frozen

if [[ ! -f .env ]]; then
  cp .env.example .env
fi

uv run intelligent-liars check-env --require-cuda
