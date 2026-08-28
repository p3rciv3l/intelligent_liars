# syntax=docker/dockerfile:1.7

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel@sha256:14611869895df612b7b07227d5925f30ec3cd6673bad58ce3d84ed107950e014

ARG IMAGE_REVISION=uncommitted

LABEL org.opencontainers.image.title="truth-editing-study" \
      org.opencontainers.image.description="Pinned CUDA runtime for persistent truth-editing Optuna studies" \
      org.opencontainers.image.revision="${IMAGE_REVISION}" \
      ai.intelligent-liars.runtime-manifest="/opt/truth-editing/runtime-manifest.json"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/workspace/cache/huggingface \
    HF_HUB_CACHE=/workspace/cache/huggingface/hub \
    HF_ASSETS_CACHE=/workspace/cache/huggingface/assets \
    TORCH_HOME=/workspace/cache/torch \
    XDG_CACHE_HOME=/workspace/cache/xdg \
    WANDB_CONSOLE=off \
    WANDB_DISABLE_CODE=true \
    WANDB_SILENT=true \
    TRUTH_EDITING_RUNTIME_MANIFEST=/opt/truth-editing/runtime-manifest.json

WORKDIR /opt/truth-editing

COPY requirements.in requirements.lock /opt/truth-editing/

# Torch and CUDA are supplied by the digest-pinned base. The lock deliberately
# excludes Torch, Triton, and NVIDIA wheels so pip cannot overlay that ABI.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --require-hashes -r /opt/truth-editing/requirements.lock

# The wheel is upstream-built for CPython 3.11, CUDA 12, Torch 2.5, and the
# base image's cxx11abiFALSE PyTorch build. A checksum rejects altered assets.
ADD --checksum=sha256:d944fc7d2f962bce83fc4708c2fc0c21eaf8255962a0b350ae919362a51b7ef2 \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl \
    /opt/truth-editing/wheels/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

RUN python -m pip install --no-deps /opt/truth-editing/wheels/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl \
    && python -m pip check \
    && mkdir -p /workspace/cache/huggingface/hub /workspace/cache/torch /workspace/cache/xdg /workspace/studies /workspace/studies/wandb

COPY Dockerfile runtime-manifest.json validate_runtime.py /opt/truth-editing/

# This checks every installed package pin and Optuna's journal API. CUDA kernels
# are checked on the execution host with validate_runtime.py (without flags).
RUN python /opt/truth-editing/validate_runtime.py --metadata-only

WORKDIR /workspace/intelligent_liars
