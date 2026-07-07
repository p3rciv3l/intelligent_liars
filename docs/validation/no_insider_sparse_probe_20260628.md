# No-Insider Sparse Probe Validation

Date run: 2026-06-28

This validates the first no-insider sparse pooled-cache probe pass against the
canonical 58 GiB all-text activation HDF5. The run used cached mean-pooled
answer-token features, not raw activation reads during probe fitting.

## Source Artifact

```text
artifacts/activations/activation_all_text_20260624/extracted_feats_all_text_qwen3-vl-8b-thinking.h5
```

Source provenance:

- Size: `62,394,047,728` bytes
- Format: `qwen_answer_token_activations_v2`
- Model: `Qwen/Qwen3-VL-8B-Thinking`
- Hidden dim: `4096`
- Source mtime recorded in cache: `1782348993172045898`
- DVC pointer md5: `b6e82b698513d2372949f2752e17005a`
- Validation sha256: `c6e687f69256544121d0718cf8bba142ed5837221976121e1899552dc76f1a5a`
- Git commit at run time: `f8b6b52a9376f1afdbb7b069de47c207ea9aa6a1`

The activation HDF5 root metadata records the original extraction environment
as Python `3.12.13`, PyTorch `2.12.0+cu130`, Transformers `5.12.1`,
NNsight `0.7.0`, h5py `3.16.0`, NumPy `2.4.6`, and qwen-vl-utils `0.0.14`.

## Smoke Test

Smoke cache command:

```bash
PYTHONPATH=src uv run --no-sync intelligent-liars cache-pooled-features \
  --input artifacts/activations/activation_all_text_20260624/extracted_feats_all_text_qwen3-vl-8b-thinking.h5 \
  --output artifacts/probe_features/no_insider_layer3_two_task_smoke_pooled.h5 \
  --layers 3 \
  --task claims__definitional_gemini_600_full \
  --task claims__evidential_gemini_600_full \
  --overwrite
```

Smoke train command:

```bash
PYTHONPATH=src uv run --no-sync intelligent-liars train-probes-from-cache \
  --cache artifacts/probe_features/no_insider_layer3_two_task_smoke_pooled.h5 \
  --source artifacts/activations/activation_all_text_20260624/extracted_feats_all_text_qwen3-vl-8b-thinking.h5 \
  --output artifacts/probes/no_insider_layer3_two_task_smoke_probe_results.json \
  --layers 3 \
  --c 1.0 \
  --test-size 0.25 \
  --random-seed 0 \
  --general-domain-probe
```

Smoke result:

- Cache: `artifacts/probe_features/no_insider_layer3_two_task_smoke_pooled.h5`
- Probe JSON: `artifacts/probes/no_insider_layer3_two_task_smoke_probe_results.json`
- Cache size: `37,688,201` bytes
- Tasks: 2
- Layers: 1
- Hidden dim: 4096
- Probe output counts: 2 within-task, 2 cross-task, 2 final directions, 2 general-domain evaluations, 1 general-domain direction

This proves the new pooled-cache path can read and train against the real
canonical HDF5.

## Full Cache

Full cache command:

```bash
PYTHONPATH=src uv run --no-sync intelligent-liars cache-pooled-features \
  --input artifacts/activations/activation_all_text_20260624/extracted_feats_all_text_qwen3-vl-8b-thinking.h5 \
  --output artifacts/probe_features/no_insider_sparse_pooled.h5 \
  --layers 3,7,11,15,19,23,27,31,35 \
  --task claims__definitional_gemini_600_full \
  --task claims__evidential_gemini_600_full \
  --task claims__fictional_gemini_600_full \
  --task claims__logical_gemini_600_full \
  --task ethics__commonsense \
  --task geometry_of_truth__best \
  --task geometry_of_truth__mixed \
  --task internal_state__animals \
  --task internal_state__cities \
  --task internal_state__companies \
  --task internal_state__elements \
  --task internal_state__facts \
  --task internal_state__inventions \
  --task repe_honesty__IF_dishonest \
  --task roleplaying__plain \
  --task sandbagging_v2__wmdp_mmlu \
  --task sycophancy__mmlu_stem_same_conf_all
```

Full cache result:

- Cache: `artifacts/probe_features/no_insider_sparse_pooled.h5`
- Cache size: `7,734,086,958` bytes (`7.2G` by `ls -lh`)
- Format: `qwen_answer_token_pooled_features_v1`
- Pooling: `mean_answer_tokens_per_example`
- Dtype: `float32`
- Hidden dim: `4096`
- Layers: `3,7,11,15,19,23,27,31,35`
- Tasks: 17
- Feature datasets: 153

Excluded tasks:

```text
insider_trading__onpolicy
insider_trading__upscale
insider_trading_doubledown__onpolicy
```

## Baseline Probe

Train command:

```bash
PYTHONPATH=src uv run --no-sync intelligent-liars train-probes-from-cache \
  --cache artifacts/probe_features/no_insider_sparse_pooled.h5 \
  --source artifacts/activations/activation_all_text_20260624/extracted_feats_all_text_qwen3-vl-8b-thinking.h5 \
  --output artifacts/probes/no_insider_sparse_probe_results.json \
  --layers 3,7,11,15,19,23,27,31,35 \
  --c 1.0 \
  --test-size 0.25 \
  --random-seed 0 \
  --general-domain-probe
```

Output:

- Probe JSON: `artifacts/probes/no_insider_sparse_probe_results.json`
- Output size: `21M`
- Input kind: `pooled_feature_cache`
- Model: `sklearn.linear_model.LogisticRegression`
- Solver: `liblinear`
- Class weight: `balanced`
- `C`: `1.0`
- Test size: `0.25`
- Random seed: `0`
- General task-class cap: `1000`
- Output counts: 153 within-task, 2448 cross-task, 153 final directions, 153 general-domain evaluations, 9 general-domain directions

## Aggregate Layer Results

Mean over the 17 selected tasks:

| Layer | Within bal acc | Within AUC | Cross bal acc | Cross AUC | General bal acc | General AUC |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.621 | 0.661 | 0.505 | 0.518 | 0.565 | 0.599 |
| 7 | 0.698 | 0.750 | 0.520 | 0.544 | 0.625 | 0.666 |
| 11 | 0.789 | 0.858 | 0.534 | 0.574 | 0.689 | 0.742 |
| 15 | 0.896 | 0.953 | 0.580 | 0.652 | 0.818 | 0.874 |
| 19 | 0.933 | 0.975 | 0.633 | 0.736 | 0.876 | 0.935 |
| 23 | 0.928 | 0.974 | 0.603 | 0.718 | 0.863 | 0.921 |
| 27 | 0.916 | 0.964 | 0.591 | 0.688 | 0.816 | 0.883 |
| 31 | 0.901 | 0.952 | 0.592 | 0.682 | 0.778 | 0.843 |
| 35 | 0.893 | 0.948 | 0.583 | 0.678 | 0.796 | 0.871 |

Best aggregate layer by both balanced accuracy and AUC is layer 19 for
within-task, cross-task, and general-domain evaluation.

## Transfer Readout

General-domain probe average by held-out task across sparse layers:

| Task | Mean bal acc | Mean AUC | Best bal layer | Best bal acc |
|---|---:|---:|---:|---:|
| `geometry_of_truth__best` | 0.912 | 0.937 | 23 | 0.999 |
| `claims__definitional_gemini_600_full` | 0.844 | 0.913 | 15 | 0.943 |
| `sandbagging_v2__wmdp_mmlu` | 0.839 | 0.929 | 3 | 0.909 |
| `internal_state__companies` | 0.810 | 0.879 | 19 | 0.907 |
| `internal_state__elements` | 0.804 | 0.865 | 19 | 0.936 |
| `sycophancy__mmlu_stem_same_conf_all` | 0.804 | 0.848 | 35 | 0.938 |
| `claims__evidential_gemini_600_full` | 0.803 | 0.868 | 19 | 0.921 |
| `internal_state__cities` | 0.770 | 0.828 | 19 | 0.915 |
| `geometry_of_truth__mixed` | 0.768 | 0.836 | 19 | 0.848 |
| `ethics__commonsense` | 0.752 | 0.809 | 19 | 0.866 |
| `internal_state__animals` | 0.745 | 0.824 | 19 | 0.841 |
| `claims__fictional_gemini_600_full` | 0.740 | 0.810 | 19 | 0.857 |
| `roleplaying__plain` | 0.715 | 0.782 | 19 | 0.891 |
| `claims__logical_gemini_600_full` | 0.706 | 0.758 | 23 | 0.833 |
| `internal_state__inventions` | 0.705 | 0.760 | 19 | 0.866 |
| `internal_state__facts` | 0.698 | 0.747 | 19 | 0.864 |
| `repe_honesty__IF_dishonest` | 0.480 | 0.459 | 23 | 0.659 |

Cross-task transfer is much weaker than within-task performance. Averaged over
all train-task/test-task/layer pairs, the best cross-task layer is still layer
19, but only at balanced accuracy `0.633` and AUC `0.736`. The cross-task
target averages show `repe_honesty__IF_dishonest` is especially poor as a
target, with balanced accuracy `0.404` and AUC `0.302`; `roleplaying__plain`
is near chance as a cross-task target, with balanced accuracy `0.506` and AUC
`0.507`.

The general-domain probe looks more real than a purely dataset-specific
artifact because the pooled probe reaches strong held-out performance on many
task families and does not merely memorize one large source task. However, it
is not yet a clean universal truth direction: REPE honesty transfers poorly,
roleplaying is weaker outside the layer-19 peak, and cross-task one-source
probes are far below within-task probes. Treat layer 19 as the first candidate
for refinement, not as a final steering direction.

## Next Sweeps

Do not start all sweeps at once. Based on this baseline:

1. Regularization sweep on the cached features: `C=0.01,0.03,0.1,0.3,1,3,10`.
2. Layer refinement around the winning sparse band: dense layers around 15-23,
   with layer 19 as the center.
3. Training-source ablations:
   - Claims only.
   - Internal-state only.
   - Truth Spec-ish static tasks.
   - All no-insider tasks.
   - Exclude `repe_honesty__IF_dishonest` as a sensitivity check, because it
     is the clearest negative-transfer task.
