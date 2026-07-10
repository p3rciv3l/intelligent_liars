# Apple-MPS Probe Sensitivity Comparison

Status: `valid_non_canonical_sensitivity_comparison`

This comparison checks whether a real Apple-MPS PyTorch fit recovers the same broad result as the canonical sklearn/liblinear probe. It is sensitivity evidence only: it does not replace the canonical result.

Both fits use layer 21, `C=0.01`, seed 0, the same example-level splits, and the exact 16 no-repeat tasks from the immutable robustness manifest.

| Fit | Mean balanced accuracy | Mean AUC | Direction norm |
|---|---:|---:|---:|
| Canonical sklearn/liblinear | 0.930190 | 0.975228 | 3.472851 |
| Apple MPS / PyTorch LBFGS | 0.929976 | 0.975232 | 3.472242 |

The MPS-minus-canonical differences are -0.000214 balanced accuracy and +0.000003 AUC. The maximum absolute per-task differences are 0.003333 balanced accuracy and 0.000133 AUC. The like-for-like final-direction cosine is 0.999982 under the shared honest-to-deceptive sign convention; both directions use the same capped all-example training population.

Interpretation: both the predictive result and the all-example direction are effectively unchanged across sklearn/liblinear and Apple-MPS/PyTorch LBFGS. This supports the layer-21 signal, while remaining sensitivity-only evidence rather than a replacement for the canonical fit.

The evaluation fit stopped after 105 optimizer steps (107 closure evaluations), taking 2.699 seconds at 6,087 training examples/second. The all-example direction fit stopped after 133 steps (134 closure evaluations), taking 3.573 seconds at 5,386 examples/second. Neither run met the strict `1e-7` gradient tolerance, so the artifact records optimizer termination without claiming numerical convergence. Its result explicitly records `canonical_evidence=false` and `replacement_for_sklearn_liblinear=false`.

Inputs:

- Canonical: `artifacts/probes/sweeps/dense_15_27_by_layer/no_repe_layer21_c001.json`
- MPS sensitivity: `artifacts/probes/robustness/gpu_sensitivity_v1/no_repe_layer21_c001_seed0_mps_v2.json`
- CPU robustness summary: `docs/validation/no_insider_seed_robustness_20260709.md`
