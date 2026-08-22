# Step 5 independent-probe qualification

Step 5 must not optimize against the same examples used to judge whether the
intervention worked. Before any GPU run, existing probe artifacts are therefore
assigned to two source-disjoint ensembles:

- **regularizer:** may contribute a differentiable training signal;
- **evaluator:** remains read-only and is used only for qualification metrics.

This step does not fit or select a probe. It freezes an already-made choice and
produces a content-addressed receipt. A run is not qualified if either ensemble
is empty, if a source group or example appears in both ensembles, if artifacts
are duplicated, or if layer, pooling, direction sign, vector dimension, artifact
bytes, controls, or receipts differ from the frozen contract.

## Registry contract

The registry is JSON with format
`intelligent_liars_step5_probe_registry_v1`. Its `qualification` object pins:

- `layer`;
- `token_pooling`;
- `direction_sign_convention`;
- `orthogonal_controls_per_probe` (at least one).

Each `probes` entry declares a unique `probe_id`, either `regularizer` or
`evaluator` as its `ensemble`, an artifact path, the JSON path of its direction
vector, nonempty `source_group_ids` and `example_ids`, and the same layer,
pooling, and sign values as the qualification object. The source-group ID must
identify the upstream split that was fixed before probe fitting; it must not be
an invented label for the artifact itself.

Example:

```json
{
  "format": "intelligent_liars_step5_probe_registry_v1",
  "qualification": {
    "layer": 19,
    "token_pooling": "mean_answer_tokens_per_example",
    "direction_sign_convention": "positive_logit_points_honest_to_deceptive",
    "orthogonal_controls_per_probe": 2
  },
  "probes": [
    {
      "probe_id": "regularizer_seed_0",
      "ensemble": "regularizer",
      "artifact_path": "probes/regularizer_seed_0.json",
      "artifact_direction_path": ["final_direction", "direction_vector"],
      "source_group_ids": ["apollo_train_fold_0"],
      "example_ids": ["apollo_0001", "apollo_0002"],
      "layer": 19,
      "token_pooling": "mean_answer_tokens_per_example",
      "direction_sign_convention": "positive_logit_points_honest_to_deceptive"
    },
    {
      "probe_id": "evaluator_seed_0",
      "ensemble": "evaluator",
      "artifact_path": "probes/evaluator_seed_0.json",
      "artifact_direction_path": ["final_direction", "direction_vector"],
      "source_group_ids": ["truthspec_heldout_fold_0"],
      "example_ids": ["truthspec_9001", "truthspec_9002"],
      "layer": 19,
      "token_pooling": "mean_answer_tokens_per_example",
      "direction_sign_convention": "positive_logit_points_honest_to_deceptive"
    }
  ]
}
```

Compile once, refusing overwrite:

```bash
python scripts/compile_step5_probe_qualification.py \
  --registry path/to/registry.json \
  --output path/to/qualification.json
```

Verify again immediately before a job consumes it:

```bash
python scripts/compile_step5_probe_qualification.py \
  --verify path/to/qualification.json
```

Use `--artifact-root` when relative artifact paths should resolve somewhere
other than the registry or manifest directory.

## Controls and receipts

Every real direction gets an exact sign-flip control and the requested number
of deterministic orthogonal controls. Orthogonal vectors use a fixed canonical
basis plus Gram–Schmidt construction seeded only by the probe ID; they are not
learned from behavior or evaluation labels. Their vectors and hashes are stored
in the manifest.

Each ensemble receives a split receipt over its sorted probe IDs, artifact and
direction hashes, source groups, and example IDs. The whole manifest receives a
qualification receipt. Verification recompiles the complete manifest from the
current artifact bytes, so an artifact edit, a moved split member, a changed
control, or a rewritten receipt fails closed.
