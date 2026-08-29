# Production throughput audit (2026-08-29)

Scope: static audit of the frozen Qwen production path and its local CUDA fleet. No
model loading, cloud work, or judge configuration changes were performed.

## Findings

| Area | Observed configuration | Assessment | Safe action |
| --- | --- | --- | --- |
| Model dtype | `ModelLoadConfig` loads `torch.bfloat16`; `_verify_loaded_runtime` rejects any other floating dtype. | Good for supported CUDA hardware; changing to FP16 would invalidate the pinned runtime identity and qualification. | Keep BF16. Re-qualify before considering another dtype. |
| Device placement | `device_map="cuda:0"`; every parameter is required on exactly `cuda:0`. The adaptive fleet launches eight CUDA-isolated subprocess workers, one GPU slot each. | Intentional fleet parallelism. `auto`/multi-GPU sharding would violate identity checks and complicate writer edits. | Keep the single-GPU map. Scale only through the already-bound worker fleet. |
| Attention | `flash_attention_2` is required in `ModelLoadConfig`, load kwargs, and runtime verification. | This is the fast path; silently falling back to eager attention would be a material throughput regression and is correctly rejected. | Keep FA2 and ensure the pinned wheel is present on worker images. |
| KV cache | Model config and both production `generate()` call sites set `use_cache=True`. | Correct for autoregressive generation. Disabling it would make decode substantially slower. | Keep enabled. |
| Study batch / runtime microbatch | Study contract is `batch_size=8`, but `TrialRuntime` defaults to `inference_microbatch_size=4`. Each 8-example trial therefore runs two chunks. | Main confirmed throughput opportunity: chunking doubles per-chunk processor/template overhead and launches two teacher-forcing forwards and two generation calls per trial (rather than one each). It may be a deliberate VRAM guard, so increasing it without a memory canary is unsafe. | For a future re-qualified runtime identity, benchmark `inference_microbatch_size=8` on the target GPU. If peak memory is unsafe, retain 4 and document it as the VRAM ceiling. Do not alter the frozen run in place. |
| Generation ceiling | Frozen production config sets `max_new_tokens=100`; the batch runtime forwards that value. The judge's `max_tokens=2048` is a separate API setting and does not control Qwen generation. | 100 is bounded and appropriate for the production answer path. Raising it linearly increases decode work; lowering it can truncate behavioral outputs and change evidence. | Keep 100 for the frozen study. Tune only in a separately hashed/re-qualified config. |
| Tokenizer / processor | Processor renders chat templates, uses `padding=True`, and tokenizer setup forces a pad token and `padding_side="left"`. Each chunk is rendered separately for generation, completed teacher-forcing, and blank-prefix scoring; multimodal inputs additionally call `process_vision_info`. | Padding and tokenizer identity are correctness-sensitive. Reusing rendered tensors could help CPU overhead, but changing template/tokenizer behavior risks evidence identity. The repeated render is measurable overhead, not a CUDA kernel killer. | Keep tokenizer/template contract. If optimizing, cache only within a trial and bind the cache/template identity; benchmark before changing production. |
| Multiprocessing | The adaptive controller uses eight persistent CUDA-isolated subprocess workers; the ordinary production composition is one model per worker. No Python `DataLoader`/process pool is used in the inference path. | Correct parallelism shape for one model per GPU. More processes per GPU would duplicate ~8B weights and contend for memory/kernels. | Keep one process per GPU. Do not add multiprocessing inside a worker. |

## Priority order

1. Measure a local, no-cloud canary with microbatch 4 versus 8 and record peak
   memory, trial wall time, and generated tokens/s. This is the only clear
   optimization found in the frozen path, and it requires a new identity if
   adopted.
2. Preserve BF16 + FA2 + cache + single-GPU placement as hard requirements.
3. Avoid changing tokenizer/template or answer token limits merely to improve
   timing; those changes alter scientific comparability.

This note is an audit artifact only; it does not authorize changing the frozen
judge, production config hash, or cloud lifecycle.
