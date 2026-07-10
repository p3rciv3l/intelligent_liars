# Probe Seed Robustness Summary

Status: `complete` — 20 valid runs, 0 invalid runs.

Population standard deviation is reported. Candidate ranking is lexicographic by general balanced accuracy, general AUC, cross-task balanced accuracy, then cross-task AUC.

Label convention: `HONEST=0, DECEPTIVE=1`

Direction sign convention: `sklearn_logistic_coef_positive_points_honest_to_deceptive`

## Selection interpretation

Formal pass/fail verdict: **not available**. No machine-readable pass/fail thresholds were recorded before these results were inspected, so this report does not retrofit them.

Descriptively, the provisional leader is `no_repe_layer21_c001`; this is evidence for the next staged ablation, not a final steering-direction selection.

Provenance limitation: the immutable v1 manifest identifies the pooled cache by portable path rather than content hash; the separately tracked DVC pointer remains the cache content source of truth.

## Candidate aggregates

| Candidate | Layer | C | General bal acc | General AUC | Cross bal acc | Cross AUC | Top-2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `no_repe_layer21_c001` | 21 | 0.0100 | 0.9267 ± 0.0035 (worst 0.9210) | 0.9734 ± 0.0021 (worst 0.9704) | 0.6899 ± 0.0036 (worst 0.6848) | 0.8475 ± 0.0021 (worst 0.8441) | 5/5 (1.0000) |
| `no_repe_layer21_c003` | 21 | 0.0300 | 0.9244 ± 0.0029 (worst 0.9200) | 0.9719 ± 0.0026 (worst 0.9680) | 0.6890 ± 0.0023 (worst 0.6849) | 0.8393 ± 0.0021 (worst 0.8357) | 4/5 (0.8000) |
| `no_repe_layer20_c003` | 20 | 0.0300 | 0.9213 ± 0.0060 (worst 0.9139) | 0.9708 ± 0.0016 (worst 0.9689) | 0.6921 ± 0.0024 (worst 0.6885) | 0.8381 ± 0.0019 (worst 0.8354) | 1/5 (0.2000) |
| `no_repe_layer19_c001` | 19 | 0.0100 | 0.9181 ± 0.0044 (worst 0.9119) | 0.9697 ± 0.0017 (worst 0.9676) | 0.6959 ± 0.0009 (worst 0.6945) | 0.8467 ± 0.0018 (worst 0.8434) | 0/5 (0.0000) |

## Per-seed ranks

| Candidate | Seed | Rank | Status | General bal acc | General AUC | Cross bal acc | Cross AUC |
|---|---:|---:|---|---:|---:|---:|---:|
| `no_repe_layer21_c001` | 0 | 1 | valid | 0.9302 | 0.9752 | 0.6903 | 0.8494 |
| `no_repe_layer21_c001` | 1 | 1 | valid | 0.9247 | 0.9713 | 0.6959 | 0.8497 |
| `no_repe_layer21_c001` | 2 | 1 | valid | 0.9210 | 0.9704 | 0.6879 | 0.8461 |
| `no_repe_layer21_c001` | 3 | 2 | valid | 0.9300 | 0.9753 | 0.6905 | 0.8441 |
| `no_repe_layer21_c001` | 4 | 1 | valid | 0.9278 | 0.9745 | 0.6848 | 0.8482 |
| `no_repe_layer21_c003` | 0 | 2 | valid | 0.9290 | 0.9739 | 0.6906 | 0.8408 |
| `no_repe_layer21_c003` | 1 | 2 | valid | 0.9231 | 0.9699 | 0.6909 | 0.8410 |
| `no_repe_layer21_c003` | 2 | 2 | valid | 0.9200 | 0.9680 | 0.6879 | 0.8383 |
| `no_repe_layer21_c003` | 3 | 3 | valid | 0.9248 | 0.9746 | 0.6904 | 0.8357 |
| `no_repe_layer21_c003` | 4 | 2 | valid | 0.9250 | 0.9734 | 0.6849 | 0.8408 |
| `no_repe_layer20_c003` | 0 | 3 | valid | 0.9259 | 0.9714 | 0.6927 | 0.8385 |
| `no_repe_layer20_c003` | 1 | 4 | valid | 0.9176 | 0.9694 | 0.6927 | 0.8397 |
| `no_repe_layer20_c003` | 2 | 3 | valid | 0.9139 | 0.9689 | 0.6885 | 0.8366 |
| `no_repe_layer20_c003` | 3 | 1 | valid | 0.9304 | 0.9736 | 0.6959 | 0.8354 |
| `no_repe_layer20_c003` | 4 | 3 | valid | 0.9189 | 0.9706 | 0.6907 | 0.8403 |
| `no_repe_layer19_c001` | 0 | 4 | valid | 0.9219 | 0.9720 | 0.6961 | 0.8483 |
| `no_repe_layer19_c001` | 1 | 3 | valid | 0.9214 | 0.9687 | 0.6974 | 0.8476 |
| `no_repe_layer19_c001` | 2 | 4 | valid | 0.9137 | 0.9689 | 0.6945 | 0.8460 |
| `no_repe_layer19_c001` | 3 | 4 | valid | 0.9217 | 0.9712 | 0.6956 | 0.8434 |
| `no_repe_layer19_c001` | 4 | 4 | valid | 0.9119 | 0.9676 | 0.6957 | 0.8480 |

## Per-task worst cases

### `no_repe_layer21_c001`

| Task | General bal acc | General AUC | Cross bal acc | Cross AUC |
|---|---:|---:|---:|---:|
| `claims__definitional_gemini_600_full` | 0.9697 (seed 4) | 0.9945 (seed 3) | 0.5017 (seed 1, train `roleplaying__plain`) | 0.6539 (seed 3, train `roleplaying__plain`) |
| `claims__evidential_gemini_600_full` | 0.9623 (seed 1) | 0.9844 (seed 1) | 0.5146 (seed 1, train `roleplaying__plain`) | 0.6968 (seed 3, train `roleplaying__plain`) |
| `claims__fictional_gemini_600_full` | 0.8767 (seed 2) | 0.9583 (seed 2) | 0.5000 (seed 1, train `roleplaying__plain`) | 0.6708 (seed 1, train `ethics__commonsense`) |
| `claims__logical_gemini_600_full` | 0.8889 (seed 3) | 0.9565 (seed 3) | 0.5000 (seed 0, train `roleplaying__plain`) | 0.5365 (seed 3, train `roleplaying__plain`) |
| `ethics__commonsense` | 0.8930 (seed 4) | 0.9629 (seed 0) | 0.4047 (seed 4, train `roleplaying__plain`) | 0.5160 (seed 4, train `roleplaying__plain`) |
| `geometry_of_truth__best` | 0.9977 (seed 3) | 0.9999 (seed 4) | 0.5000 (seed 0, train `claims__fictional_gemini_600_full`) | 0.2357 (seed 3, train `roleplaying__plain`) |
| `geometry_of_truth__mixed` | 0.8646 (seed 2) | 0.9402 (seed 0) | 0.5021 (seed 3, train `roleplaying__plain`) | 0.5027 (seed 3, train `roleplaying__plain`) |
| `internal_state__animals` | 0.8690 (seed 2) | 0.9538 (seed 2) | 0.5000 (seed 1, train `roleplaying__plain`) | 0.6058 (seed 3, train `roleplaying__plain`) |
| `internal_state__cities` | 0.9371 (seed 2) | 0.9817 (seed 0) | 0.5000 (seed 0, train `sandbagging_v2__wmdp_mmlu`) | 0.6053 (seed 3, train `sandbagging_v2__wmdp_mmlu`) |
| `internal_state__companies` | 0.9133 (seed 1) | 0.9668 (seed 0) | 0.5000 (seed 1, train `roleplaying__plain`) | 0.6014 (seed 3, train `roleplaying__plain`) |
| `internal_state__elements` | 0.9355 (seed 2) | 0.9703 (seed 2) | 0.5000 (seed 2, train `ethics__commonsense`) | 0.5859 (seed 3, train `roleplaying__plain`) |
| `internal_state__facts` | 0.9416 (seed 1) | 0.9847 (seed 3) | 0.5065 (seed 0, train `roleplaying__plain`) | 0.5485 (seed 3, train `roleplaying__plain`) |
| `internal_state__inventions` | 0.8986 (seed 2) | 0.9508 (seed 0) | 0.4989 (seed 0, train `sandbagging_v2__wmdp_mmlu`) | 0.6335 (seed 3, train `roleplaying__plain`) |
| `roleplaying__plain` | 0.7983 (seed 2) | 0.8796 (seed 1) | 0.4213 (seed 4, train `sycophancy__mmlu_stem_same_conf_all`) | 0.3582 (seed 2, train `sycophancy__mmlu_stem_same_conf_all`) |
| `sandbagging_v2__wmdp_mmlu` | 0.9365 (seed 0) | 0.9683 (seed 2) | 0.4589 (seed 1, train `geometry_of_truth__mixed`) | 0.3390 (seed 1, train `geometry_of_truth__mixed`) |
| `sycophancy__mmlu_stem_same_conf_all` | 0.8968 (seed 0) | 0.9569 (seed 1) | 0.4732 (seed 0, train `roleplaying__plain`) | 0.4199 (seed 3, train `roleplaying__plain`) |

### `no_repe_layer21_c003`

| Task | General bal acc | General AUC | Cross bal acc | Cross AUC |
|---|---:|---:|---:|---:|
| `claims__definitional_gemini_600_full` | 0.9630 (seed 1) | 0.9941 (seed 2) | 0.5017 (seed 3, train `sycophancy__mmlu_stem_same_conf_all`) | 0.6359 (seed 1, train `ethics__commonsense`) |
| `claims__evidential_gemini_600_full` | 0.9621 (seed 3) | 0.9835 (seed 1) | 0.5129 (seed 1, train `sandbagging_v2__wmdp_mmlu`) | 0.6982 (seed 3, train `roleplaying__plain`) |
| `claims__fictional_gemini_600_full` | 0.8733 (seed 2) | 0.9504 (seed 2) | 0.5000 (seed 1, train `roleplaying__plain`) | 0.6070 (seed 1, train `ethics__commonsense`) |
| `claims__logical_gemini_600_full` | 0.8785 (seed 3) | 0.9539 (seed 3) | 0.5000 (seed 1, train `roleplaying__plain`) | 0.5304 (seed 3, train `roleplaying__plain`) |
| `ethics__commonsense` | 0.8900 (seed 4) | 0.9612 (seed 0) | 0.4113 (seed 4, train `roleplaying__plain`) | 0.5197 (seed 4, train `roleplaying__plain`) |
| `geometry_of_truth__best` | 0.9977 (seed 0) | 0.9999 (seed 4) | 0.4577 (seed 4, train `roleplaying__plain`) | 0.2385 (seed 3, train `roleplaying__plain`) |
| `geometry_of_truth__mixed` | 0.8624 (seed 0) | 0.9386 (seed 0) | 0.4997 (seed 3, train `roleplaying__plain`) | 0.4950 (seed 3, train `roleplaying__plain`) |
| `internal_state__animals` | 0.8571 (seed 2) | 0.9499 (seed 1) | 0.5000 (seed 1, train `roleplaying__plain`) | 0.5919 (seed 3, train `roleplaying__plain`) |
| `internal_state__cities` | 0.9398 (seed 0) | 0.9831 (seed 0) | 0.5000 (seed 0, train `sandbagging_v2__wmdp_mmlu`) | 0.6044 (seed 3, train `sandbagging_v2__wmdp_mmlu`) |
| `internal_state__companies` | 0.9100 (seed 1) | 0.9689 (seed 0) | 0.5000 (seed 1, train `roleplaying__plain`) | 0.5971 (seed 3, train `roleplaying__plain`) |
| `internal_state__elements` | 0.9356 (seed 1) | 0.9717 (seed 2) | 0.5000 (seed 2, train `ethics__commonsense`) | 0.5803 (seed 3, train `roleplaying__plain`) |
| `internal_state__facts` | 0.9156 (seed 3) | 0.9808 (seed 3) | 0.5098 (seed 1, train `roleplaying__plain`) | 0.5445 (seed 3, train `roleplaying__plain`) |
| `internal_state__inventions` | 0.8755 (seed 2) | 0.9422 (seed 0) | 0.4989 (seed 0, train `sandbagging_v2__wmdp_mmlu`) | 0.6263 (seed 3, train `roleplaying__plain`) |
| `roleplaying__plain` | 0.8079 (seed 2) | 0.8836 (seed 1) | 0.4308 (seed 4, train `sycophancy__mmlu_stem_same_conf_all`) | 0.3546 (seed 2, train `sycophancy__mmlu_stem_same_conf_all`) |
| `sandbagging_v2__wmdp_mmlu` | 0.9008 (seed 4) | 0.9603 (seed 4) | 0.4658 (seed 1, train `geometry_of_truth__mixed`) | 0.3255 (seed 1, train `geometry_of_truth__mixed`) |
| `sycophancy__mmlu_stem_same_conf_all` | 0.9096 (seed 0) | 0.9619 (seed 1) | 0.4544 (seed 0, train `roleplaying__plain`) | 0.4092 (seed 3, train `roleplaying__plain`) |

### `no_repe_layer20_c003`

| Task | General bal acc | General AUC | Cross bal acc | Cross AUC |
|---|---:|---:|---:|---:|
| `claims__definitional_gemini_600_full` | 0.9630 (seed 3) | 0.9932 (seed 3) | 0.5017 (seed 2, train `sandbagging_v2__wmdp_mmlu`) | 0.6439 (seed 3, train `roleplaying__plain`) |
| `claims__evidential_gemini_600_full` | 0.9520 (seed 1) | 0.9824 (seed 1) | 0.5086 (seed 2, train `sandbagging_v2__wmdp_mmlu`) | 0.7034 (seed 3, train `roleplaying__plain`) |
| `claims__fictional_gemini_600_full` | 0.8600 (seed 2) | 0.9483 (seed 2) | 0.5017 (seed 1, train `roleplaying__plain`) | 0.6312 (seed 2, train `ethics__commonsense`) |
| `claims__logical_gemini_600_full` | 0.8785 (seed 3) | 0.9471 (seed 3) | 0.5009 (seed 1, train `roleplaying__plain`) | 0.5510 (seed 3, train `sandbagging_v2__wmdp_mmlu`) |
| `ethics__commonsense` | 0.8840 (seed 0) | 0.9541 (seed 0) | 0.4253 (seed 4, train `roleplaying__plain`) | 0.5202 (seed 4, train `roleplaying__plain`) |
| `geometry_of_truth__best` | 0.9965 (seed 4) | 1.0000 (seed 4) | 0.5000 (seed 0, train `claims__fictional_gemini_600_full`) | 0.3303 (seed 3, train `roleplaying__plain`) |
| `geometry_of_truth__mixed` | 0.8630 (seed 4) | 0.9404 (seed 2) | 0.5178 (seed 3, train `claims__fictional_gemini_600_full`) | 0.5905 (seed 3, train `roleplaying__plain`) |
| `internal_state__animals` | 0.8690 (seed 1) | 0.9466 (seed 2) | 0.4990 (seed 3, train `sandbagging_v2__wmdp_mmlu`) | 0.6247 (seed 3, train `roleplaying__plain`) |
| `internal_state__cities` | 0.9370 (seed 0) | 0.9838 (seed 2) | 0.5000 (seed 0, train `sandbagging_v2__wmdp_mmlu`) | 0.5868 (seed 3, train `sandbagging_v2__wmdp_mmlu`) |
| `internal_state__companies` | 0.9167 (seed 3) | 0.9658 (seed 0) | 0.5000 (seed 0, train `sandbagging_v2__wmdp_mmlu`) | 0.6341 (seed 3, train `roleplaying__plain`) |
| `internal_state__elements` | 0.9355 (seed 2) | 0.9643 (seed 2) | 0.5000 (seed 3, train `sandbagging_v2__wmdp_mmlu`) | 0.6522 (seed 0, train `roleplaying__plain`) |
| `internal_state__facts` | 0.9351 (seed 1) | 0.9801 (seed 3) | 0.5195 (seed 3, train `sandbagging_v2__wmdp_mmlu`) | 0.5809 (seed 3, train `roleplaying__plain`) |
| `internal_state__inventions` | 0.8474 (seed 2) | 0.9489 (seed 0) | 0.5000 (seed 0, train `sandbagging_v2__wmdp_mmlu`) | 0.6262 (seed 3, train `geometry_of_truth__mixed`) |
| `roleplaying__plain` | 0.7722 (seed 1) | 0.8684 (seed 1) | 0.4427 (seed 3, train `ethics__commonsense`) | 0.3992 (seed 3, train `geometry_of_truth__best`) |
| `sandbagging_v2__wmdp_mmlu` | 0.9008 (seed 4) | 0.9762 (seed 2) | 0.4406 (seed 2, train `geometry_of_truth__mixed`) | 0.2914 (seed 1, train `geometry_of_truth__mixed`) |
| `sycophancy__mmlu_stem_same_conf_all` | 0.9104 (seed 2) | 0.9581 (seed 1) | 0.5036 (seed 1, train `internal_state__companies`) | 0.5731 (seed 2, train `geometry_of_truth__mixed`) |

### `no_repe_layer19_c001`

| Task | General bal acc | General AUC | Cross bal acc | Cross AUC |
|---|---:|---:|---:|---:|
| `claims__definitional_gemini_600_full` | 0.9663 (seed 2) | 0.9927 (seed 3) | 0.5000 (seed 4, train `ethics__commonsense`) | 0.6371 (seed 3, train `roleplaying__plain`) |
| `claims__evidential_gemini_600_full` | 0.9657 (seed 1) | 0.9851 (seed 1) | 0.5017 (seed 1, train `sandbagging_v2__wmdp_mmlu`) | 0.7010 (seed 3, train `roleplaying__plain`) |
| `claims__fictional_gemini_600_full` | 0.8733 (seed 2) | 0.9589 (seed 0) | 0.5008 (seed 0, train `roleplaying__plain`) | 0.6618 (seed 3, train `roleplaying__plain`) |
| `claims__logical_gemini_600_full` | 0.8785 (seed 4) | 0.9481 (seed 3) | 0.5000 (seed 0, train `roleplaying__plain`) | 0.5583 (seed 3, train `sandbagging_v2__wmdp_mmlu`) |
| `ethics__commonsense` | 0.8870 (seed 0) | 0.9585 (seed 0) | 0.4460 (seed 4, train `roleplaying__plain`) | 0.4981 (seed 4, train `roleplaying__plain`) |
| `geometry_of_truth__best` | 0.9965 (seed 4) | 0.9999 (seed 4) | 0.4820 (seed 3, train `roleplaying__plain`) | 0.4517 (seed 3, train `roleplaying__plain`) |
| `geometry_of_truth__mixed` | 0.8633 (seed 2) | 0.9421 (seed 2) | 0.5147 (seed 0, train `claims__fictional_gemini_600_full`) | 0.6354 (seed 3, train `roleplaying__plain`) |
| `internal_state__animals` | 0.8730 (seed 2) | 0.9460 (seed 2) | 0.4950 (seed 3, train `sandbagging_v2__wmdp_mmlu`) | 0.6305 (seed 3, train `roleplaying__plain`) |
| `internal_state__cities` | 0.9398 (seed 2) | 0.9763 (seed 0) | 0.4979 (seed 4, train `ethics__commonsense`) | 0.5954 (seed 3, train `sandbagging_v2__wmdp_mmlu`) |
| `internal_state__companies` | 0.9167 (seed 0) | 0.9664 (seed 0) | 0.5000 (seed 0, train `ethics__commonsense`) | 0.5980 (seed 3, train `roleplaying__plain`) |
| `internal_state__elements` | 0.9355 (seed 2) | 0.9705 (seed 2) | 0.5000 (seed 3, train `sandbagging_v2__wmdp_mmlu`) | 0.6481 (seed 3, train `sandbagging_v2__wmdp_mmlu`) |
| `internal_state__facts` | 0.9286 (seed 1) | 0.9816 (seed 3) | 0.5016 (seed 0, train `roleplaying__plain`) | 0.5965 (seed 3, train `roleplaying__plain`) |
| `internal_state__inventions` | 0.8798 (seed 2) | 0.9541 (seed 0) | 0.5000 (seed 0, train `ethics__commonsense`) | 0.6475 (seed 3, train `roleplaying__plain`) |
| `roleplaying__plain` | 0.7887 (seed 4) | 0.8710 (seed 1) | 0.4821 (seed 0, train `sandbagging_v2__wmdp_mmlu`) | 0.4145 (seed 3, train `ethics__commonsense`) |
| `sandbagging_v2__wmdp_mmlu` | 0.8730 (seed 4) | 0.9802 (seed 4) | 0.4589 (seed 1, train `geometry_of_truth__mixed`) | 0.3888 (seed 1, train `geometry_of_truth__mixed`) |
| `sycophancy__mmlu_stem_same_conf_all` | 0.8504 (seed 2) | 0.9223 (seed 1) | 0.5090 (seed 0, train `internal_state__companies`) | 0.6236 (seed 4, train `roleplaying__plain`) |

## Direction cosine

Stored direction vectors are never modified. The aligned column applies the reported seed-0 alignment multipliers only for comparison.

### `no_repe_layer21_c001`

Sign convention: `sklearn_logistic_coef_positive_points_honest_to_deceptive`

| Seed A | Seed B | Align A | Align B | Raw cosine | Seed-0-aligned cosine |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 1 | 1 | 0.8579 | 0.8579 |
| 0 | 2 | 1 | 1 | 0.8621 | 0.8621 |
| 0 | 3 | 1 | 1 | 0.8545 | 0.8545 |
| 0 | 4 | 1 | 1 | 0.8715 | 0.8715 |
| 1 | 2 | 1 | 1 | 0.8705 | 0.8705 |
| 1 | 3 | 1 | 1 | 0.8668 | 0.8668 |
| 1 | 4 | 1 | 1 | 0.8748 | 0.8748 |
| 2 | 3 | 1 | 1 | 0.8680 | 0.8680 |
| 2 | 4 | 1 | 1 | 0.8737 | 0.8737 |
| 3 | 4 | 1 | 1 | 0.8586 | 0.8586 |

### `no_repe_layer21_c003`

Sign convention: `sklearn_logistic_coef_positive_points_honest_to_deceptive`

| Seed A | Seed B | Align A | Align B | Raw cosine | Seed-0-aligned cosine |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 1 | 1 | 0.8124 | 0.8124 |
| 0 | 2 | 1 | 1 | 0.8155 | 0.8155 |
| 0 | 3 | 1 | 1 | 0.8036 | 0.8036 |
| 0 | 4 | 1 | 1 | 0.8188 | 0.8188 |
| 1 | 2 | 1 | 1 | 0.8235 | 0.8235 |
| 1 | 3 | 1 | 1 | 0.8205 | 0.8205 |
| 1 | 4 | 1 | 1 | 0.8266 | 0.8266 |
| 2 | 3 | 1 | 1 | 0.8217 | 0.8217 |
| 2 | 4 | 1 | 1 | 0.8259 | 0.8259 |
| 3 | 4 | 1 | 1 | 0.8057 | 0.8057 |

### `no_repe_layer20_c003`

Sign convention: `sklearn_logistic_coef_positive_points_honest_to_deceptive`

| Seed A | Seed B | Align A | Align B | Raw cosine | Seed-0-aligned cosine |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 1 | 1 | 0.8323 | 0.8323 |
| 0 | 2 | 1 | 1 | 0.8263 | 0.8263 |
| 0 | 3 | 1 | 1 | 0.8345 | 0.8345 |
| 0 | 4 | 1 | 1 | 0.8107 | 0.8107 |
| 1 | 2 | 1 | 1 | 0.8269 | 0.8269 |
| 1 | 3 | 1 | 1 | 0.8306 | 0.8306 |
| 1 | 4 | 1 | 1 | 0.8114 | 0.8114 |
| 2 | 3 | 1 | 1 | 0.8331 | 0.8331 |
| 2 | 4 | 1 | 1 | 0.8207 | 0.8207 |
| 3 | 4 | 1 | 1 | 0.8296 | 0.8296 |

### `no_repe_layer19_c001`

Sign convention: `sklearn_logistic_coef_positive_points_honest_to_deceptive`

| Seed A | Seed B | Align A | Align B | Raw cosine | Seed-0-aligned cosine |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 1 | 1 | 0.8733 | 0.8733 |
| 0 | 2 | 1 | 1 | 0.8814 | 0.8814 |
| 0 | 3 | 1 | 1 | 0.8839 | 0.8839 |
| 0 | 4 | 1 | 1 | 0.8759 | 0.8759 |
| 1 | 2 | 1 | 1 | 0.8651 | 0.8651 |
| 1 | 3 | 1 | 1 | 0.8764 | 0.8764 |
| 1 | 4 | 1 | 1 | 0.8779 | 0.8779 |
| 2 | 3 | 1 | 1 | 0.8784 | 0.8784 |
| 2 | 4 | 1 | 1 | 0.8714 | 0.8714 |
| 3 | 4 | 1 | 1 | 0.8748 | 0.8748 |

## Missing or invalid results

None.
