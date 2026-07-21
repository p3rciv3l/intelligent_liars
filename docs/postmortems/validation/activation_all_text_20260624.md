# Canonical All-Text Activation Validation

Date verified: 2026-06-25

This is the durable validation report for the canonical pre-probe all-text
activation artifact. It is intentionally small; raw logs and transcript dumps
are local forensic material, not the repo handoff surface.

## Canonical Artifact

```text
artifacts/activations/activation_all_text_20260624/extracted_feats_all_text_qwen3-vl-8b-thinking.h5
```

Do not rerun extraction to recreate this artifact unless the experiment design
changes. Future checkouts should recover the bytes through DVC and then validate
against this report.

## DVC Pointer

```text
artifacts/activations/activation_all_text_20260624/extracted_feats_all_text_qwen3-vl-8b-thinking.h5.dvc
```

Pointer contents:

```yaml
outs:
- md5: b6e82b698513d2372949f2752e17005a
  size: 62394047728
  hash: md5
  path: extracted_feats_all_text_qwen3-vl-8b-thinking.h5
```

Remote status command:

```bash
uvx --from "dvc[gdrive]" dvc status -c \
  artifacts/activations/activation_all_text_20260624/extracted_feats_all_text_qwen3-vl-8b-thinking.h5.dvc
```

Observed result:

```text
Cache and remote 'gdrive-artifacts' are in sync.
```

## Validation Proof

Validation command:

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_activation_hdf5.py \
  artifacts/activations/activation_all_text_20260624/extracted_feats_all_text_qwen3-vl-8b-thinking.h5 \
  --expected-task-count 20 \
  --expected-layer-count 36 \
  --expected-hidden-dim 4096 \
  --finite-check sample \
  --sha256
```

Observed summary:

```text
activation HDF5 validation: OK
sha256: c6e687f69256544121d0718cf8bba142ed5837221976121e1899552dc76f1a5a
tasks: 20
layers: 36
hidden_dim: 4096
finite_check: sample checked=720 non_finite=0
```

Task set:

```text
claims__definitional_gemini_600_full
claims__evidential_gemini_600_full
claims__fictional_gemini_600_full
claims__logical_gemini_600_full
ethics__commonsense
geometry_of_truth__best
geometry_of_truth__mixed
insider_trading__onpolicy
insider_trading__upscale
insider_trading_doubledown__onpolicy
internal_state__animals
internal_state__cities
internal_state__companies
internal_state__elements
internal_state__facts
internal_state__inventions
repe_honesty__IF_dishonest
roleplaying__plain
sandbagging_v2__wmdp_mmlu
sycophancy__mmlu_stem_same_conf_all
```

## Evidence Location

This file is the tracked evidence record. The validator implementation is
`scripts/validate_activation_hdf5.py`; the DVC pointer above is tracked next to
the ignored HDF5. Large binaries, upload logs, and transcript slices should stay
out of Git unless a later review explicitly asks for raw forensic material.
