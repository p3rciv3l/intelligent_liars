#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-ubuntu@45.250.254.57}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/ubuntu/intelligent_liars}"
SOURCE="${SOURCE:-artifacts/activations/activation_all_text_20260624/extracted_feats_all_text_qwen3-vl-8b-thinking.h5}"
DEST_DIR="$REMOTE:$REMOTE_ROOT/artifacts/activations/activation_all_text_20260624/"

exec rsync -a --partial --append --progress \
  -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o Compression=no" \
  "$SOURCE" "$DEST_DIR"
