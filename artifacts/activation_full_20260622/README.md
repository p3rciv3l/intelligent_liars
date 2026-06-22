# Activation extraction full run - 2026-06-22

## Output

- HDF5: `artifacts/activation_full_20260622/extracted_feats_all_layers_qwen3-vl-8b-thinking.h5`
- Size: `11514713158` bytes
- Google Drive artifact folder: `https://drive.google.com/drive/folders/1C7KUkZHCfQXnpkt6debhCcA4YtV8eOHX?usp=sharing`
- Remote source: `/dev/shm/intelligent_liars_activation/results/extracted_feats_all_layers_qwen3-vl-8b-thinking.h5`
- Model: `Qwen/Qwen3-VL-8B-Thinking`
- Format: `qwen_answer_token_activations_v2`
- Storage dtype: `float16`

## Run Shape

- Hardware: 8x A100 80GB, one worker per GPU
- Workers: `8`
- Dynamic queue units: `61`
- Failed units: `0`
- Layers: `all` / 36 decoder layers
- Batch size: `1`
- Chunk target: `40000` estimated chars
- Max examples per chunk: `8`
- Capture logits: `false`
- Output storage: `/dev/shm` on remote, then copied locally

## Dataset Coverage

| Task | Examples | Answer-token rows |
| --- | ---: | ---: |
| `roleplaying__plain` | 329 | 11,576 |
| `sandbagging_v2__wmdp_mmlu` | 128 | 38,799 |
| Total | 457 | 50,375 |

The insider rollout files were not included in this full extraction because their raw labels are `unknown`, leaving zero usable examples for the current activation extractor.

## Timing

- Supervisor start: `2026-06-22T06:17:50+00:00`
- All extraction shards complete: `2026-06-22T06:20:36+00:00`
- Merge complete: `2026-06-22T06:35:18+00:00`
- Extraction wall time after startup: about 3 minutes
- Merge wall time: about 15 minutes

## Local Validation

Validated locally after clean transfer:

- Exact byte count matched remote: `11514713158`
- Tasks present: `roleplaying__plain`, `sandbagging_v2__wmdp_mmlu`
- Each task has 36 layer datasets
- First layer shapes:
  - `roleplaying__plain`: `(11576, 4096)`, `float16`
  - `sandbagging_v2__wmdp_mmlu`: `(38799, 4096)`, `float16`

## Notes

- The full HDF5 is intentionally not committed to GitHub. Keep it in Google Drive or local ignored storage.
- The first full attempt wrote outputs to the remote `/workspace` overlay and hit `ENOSPC`. The successful full run wrote shards and the merged output under `/dev/shm`.
- Transfer back to local was restarted cleanly into a `.tmp` file after an interrupted partial copy produced an oversized/corrupt local file.
- Worker logs contain repeated Transformers warnings about processor kwargs placement. These did not stop extraction, and the merged local HDF5 validated structurally.
