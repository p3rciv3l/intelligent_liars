# Step 5 grouped probe qualification

This qualification fits the previously missing Step 5 regularizer and evaluator
probe ensembles from the existing labeled Qwen3-VL activation cache. It is a
CPU-only legacy-data operation: it does not inspect the sealed Step 5 audit,
generate new model outputs, rent a GPU, or tune against Step 5 behavior rows.

## Leakage boundaries

The identity derivation uses only upstream metadata, never labels or activation
values:

- **source group:** normalized upstream dataset path, so task aliases cannot hide
  reuse of the same CSV or rollout source;
- **example group:** source group plus explicit `{csv,row}`, upstream pair ID,
  nested source index, or `example_source_index`, keeping paired outputs together;
- **template group:** explicit `pair_index` when present, then a nested upstream
  `source_index`, then `example_source_index`.

The frozen config assigns seven complete task views to the evaluator side and
the remaining ten to the regularizer side. The compiler then resolves their
actual upstream identities and fails if source, example, or template groups
overlap. This caught and removed a real aliasing problem: `geometry_of_truth`
`best` and `mixed` reuse the same upstream CSV rows, so both now remain on the
evaluator side.

Layer calibration is confined to the regularizer side. Five decoder layers
(15, 18, 21, 24, and 27) are compared using three source-grouped folds and a
task-balanced cap of 256 examples per class per task. Both
`mean_answer_tokens_per_example` and `last_answer_token_per_example` are tested;
the latter is read directly from the legacy token-level cache boundaries.

After calibration, one regularizer probe is fit on the regularizer sources. The
evaluator side is fit as five source-grouped cross-fit probes. Each evaluator
probe's reported metric comes only from its held-out source groups. The qualification compiler
then checks ensemble separation, layer/pooling/sign agreement, artifact hashes,
direction uniqueness, sign-flip controls, orthogonal controls, and complete
receipts.

Every fitted probe artifact includes the exact current Step 5 corpus-plan hash,
`5282aaf0696098970de476e6812534e4c75c2434d7ec369495f5a311d6c09f99`.
Changing that plan makes the fitter fail before it reads feature tensors.

## Actual result

The real run selected decoder layer 21 with last-answer-token pooling:

| Pooling | Layer | Regularizer-only grouped macro task ROC AUC |
|---|---:|---:|
| mean | 15 | 0.7253 |
| mean | 18 | 0.8011 |
| mean | 21 | 0.8290 |
| mean | 24 | 0.7896 |
| mean | 27 | 0.7743 |
| last | 15 | 0.7641 |
| last | 18 | 0.8087 |
| last | 21 | **0.8291** |
| last | 24 | 0.8188 |
| last | 27 | 0.8145 |

The five source-group-held-out evaluator results were:

| Cross-fit fold | Balanced accuracy | ROC AUC |
|---|---:|---:|
| 0 | 0.9292 | 0.9593 |
| 1 | 0.9196 | 0.9713 |
| 2 | 0.8688 | 0.9495 |
| 3 | 0.8724 | 0.9569 |
| 4 | 0.5512 | 0.5826 |

The weak fifth-fold transfer is retained rather than hidden; the ensemble is
a diagnostic, not proof that the behavior intervention works. Step 5 still
requires held-out behavioral, preservation, safety, and multimodal gates.

The fitted report receipt is
`d1c08b35e551dc817a7e239122ae1499061a6c978b4fa2bd7961717bb633bc3b`.
The final qualification receipt is
`f21781fdadab2eab6773d3e324d7500132e1f5f9e4bb38696c50837a07693b54`.
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
