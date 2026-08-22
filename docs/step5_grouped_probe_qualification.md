# Step 5 grouped probe qualification

This qualification fits the previously missing Step 5 regularizer and evaluator
probe ensembles from the existing labeled Qwen3-VL activation cache. It is a
CPU-only legacy-data operation: it does not inspect the sealed Step 5 audit,
generate new model outputs, rent a GPU, or tune against Step 5 behavior rows.

## Leakage boundaries

The identity derivation uses only upstream metadata, never labels or activation
values:

- **source group:** normalized upstream task ID;
- **example group:** task ID plus upstream `example_source_index`, keeping paired
  honest/deceptive outputs together;
- **template group:** explicit `pair_index` when present, then a nested upstream
  `source_index`, then `example_source_index`.

The frozen config assigns six complete source groups to the evaluator side and
the remaining eleven to the regularizer side. Therefore no source group or
example used by the regularizer probe is used to fit an evaluator probe.

Layer calibration is confined to the regularizer side. Five decoder layers
(15, 18, 21, 24, and 27) are compared using three source-grouped folds and a
task-balanced cap of 256 examples per class per task. Token pooling has one
available legacy candidate, `mean_answer_tokens_per_example`; the code fails if
the cache does not exactly match that frozen candidate.

After calibration, one regularizer probe is fit on the regularizer sources. The
evaluator side is fit as six leave-one-source-out probes. Each evaluator probe's
reported metric comes only from its held-out source. The qualification compiler
then checks ensemble separation, layer/pooling/sign agreement, artifact hashes,
direction uniqueness, sign-flip controls, orthogonal controls, and complete
receipts.

Every fitted probe artifact includes the exact current Step 5 corpus-plan hash,
`5282aaf0696098970de476e6812534e4c75c2434d7ec369495f5a311d6c09f99`.
Changing that plan makes the fitter fail before it reads feature tensors.

## Actual result

The real run selected decoder layer 21:

| Layer | Regularizer-only grouped macro task ROC AUC |
|---:|---:|
| 15 | 0.7385 |
| 18 | 0.7875 |
| 21 | **0.8075** |
| 24 | 0.7693 |
| 27 | 0.7289 |

The six source-held-out evaluator results were:

| Held-out source | Balanced accuracy | ROC AUC |
|---|---:|---:|
| claims evidential | 0.9668 | 0.9969 |
| geometry of truth | 0.7285 | 1.0000 |
| internal-state facts | 0.8906 | 0.9886 |
| roleplaying | 0.6334 | 0.6627 |
| sandbagging | 0.8726 | 0.9776 |
| sycophancy | 0.7617 | 0.8619 |

The weaker roleplaying transfer is retained rather than hidden; the ensemble is
a diagnostic, not proof that the behavior intervention works. Step 5 still
requires held-out behavioral, preservation, safety, and multimodal gates.

The fitted report receipt is
`ab119fa5927cbe76149d621e97ca30463767fbbefff4b91e481aa70f178d6d9e`.
The final qualification receipt is
`1575fd3332442e6c8bdad34aaafc63c1767179249a3ccf0b9931f9c20e67a39a`.
Independent revalidation returned `valid: true` with zero issues.

## Reproduce or verify

Fit into a new output directory (the command refuses overwrite):

```bash
.venv/bin/python scripts/fit_step5_grouped_probe_ensemble.py \
  --pooled-cache artifacts/probe_features/no_insider_dense_15_27_pooled.h5 \
  --raw-activation-cache artifacts/activations/activation_all_text_20260624/extracted_feats_all_text_qwen3-vl-8b-thinking.h5 \
  --config configs/step5_grouped_probe_ensemble_v1.json \
  --step5-plan corpora/tinylora_deception_action_v1/step5_v1/manifest.json \
  --output-root artifacts/probes/step5_grouped_ensemble_v1
```

Verify the published qualification without refitting:

```bash
PYTHONPATH=src .venv/bin/python scripts/compile_step5_probe_qualification.py \
  --verify artifacts/probes/step5_grouped_ensemble_v1/probe_qualification.json \
  --artifact-root artifacts/probes/step5_grouped_ensemble_v1
```

The 10 GB pooled cache and 58 GB raw activation cache remain DVC-managed and
are not committed again. The committed qualification artifacts are JSON only.
