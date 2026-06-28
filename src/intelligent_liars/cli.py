from __future__ import annotations

import typer
from dotenv import load_dotenv

from intelligent_liars.activations import (
    ActivationDataset,
    ActivationExtractionSettings,
    default_activation_output_path,
    extract_dataset_activations,
    extract_rollout_activations,
    merge_activation_hdf5_shards,
    parse_layer_spec,
)
from intelligent_liars.activation_backends import (
    ActivationBackend,
    TransformersHookBackend,
)
from intelligent_liars.cli_common import (
    app,
    console,
    _bind_judge_config_to_project_root,
    _find_project_root,
    _generation_settings,
    _infer_rollout_task_from_path,
    _judge_config_from_options,
    _project_path,
    _resolve_project_root,
    _rollout_task_needs_llm_judge,
    _selected_example_count,
    _selected_source_indices,
    _source_index_filter,
    _validate_insider_prompt_glob,
)
from intelligent_liars.environment import check_environment
from intelligent_liars.insider_trading import (
    LabelMode,
    generate_insider_trading_transcripts,
    validate_insider_trading_generation_resume,
)
from intelligent_liars.judging import (
    DEFAULT_INSIDER_GRADING_FLUSH_EVERY,
    DEFAULT_JUDGE_CONFIG,
    DEFAULT_ROLLOUT_GRADING_FLUSH_EVERY,
    DEFAULT_ROLLOUT_GRADING_MAX_WORKERS,
    JudgeConfig,
    grade_insider_trading_transcripts as grade_insider_trading_transcripts_with_judge,
    grade_rollout_file,
    infer_rollout_task,
    preflight_judge_config,
)
from intelligent_liars.models import (
    ModelBundle,
    load_model_and_processor,
    model_config_from_env,
    resolve_model_id,
)
from intelligent_liars.nnsight_backend import (
    NnsightActivationBackend,
    load_nnsight_bundle,
)
from intelligent_liars.probes import (
    build_pooled_feature_cache,
    preflight_probe_training,
    train_probe_directions,
    train_probe_directions_from_cache,
)
from intelligent_liars.rollouts import (
    DEFAULT_INSIDER_GENERATION_SETTINGS,
    DEFAULT_ROLLOUT_GENERATION_SETTINGS,
    DEFAULT_ROLLOUT_PROMPT_SET,
    DEFAULT_ROLLOUT_TASKS,
    GenerationSettings,
    MODEL_SLUG,
    generate_rollout_task,
    load_rollout_prompt_examples,
    merge_rollout_files,
    parse_task_name,
    pending_rollout_source_indices,
    qwen_model_slug,
    seed_everything,
)
from intelligent_liars.sycophancy import (
    STEM_TASKS,
    generate_sycophancy_dataset,
    pair_sycophancy_dataset,
)

# Importing command modules registers their Typer commands on the shared app.
from intelligent_liars.cli_activations import (
    extract_activations,
    merge_activation_shards,
)
from intelligent_liars.cli_env import check_env
from intelligent_liars.cli_insider import (
    generate_insider_trading,
    grade_insider_trading,
    run_insider_trading_sweep,
)
from intelligent_liars.cli_probes import (
    cache_pooled_features,
    preflight_probes,
    train_probes,
    train_probes_from_cache,
)
from intelligent_liars.cli_rollouts import (
    generate_rollouts,
    grade_rollouts,
    merge_rollouts,
    run_qwen_sweep,
)
from intelligent_liars.cli_sycophancy import generate_sycophancy

__all__ = [
    "ActivationBackend",
    "ActivationDataset",
    "ActivationExtractionSettings",
    "DEFAULT_INSIDER_GENERATION_SETTINGS",
    "DEFAULT_INSIDER_GRADING_FLUSH_EVERY",
    "DEFAULT_JUDGE_CONFIG",
    "DEFAULT_ROLLOUT_GENERATION_SETTINGS",
    "DEFAULT_ROLLOUT_GRADING_FLUSH_EVERY",
    "DEFAULT_ROLLOUT_GRADING_MAX_WORKERS",
    "DEFAULT_ROLLOUT_PROMPT_SET",
    "DEFAULT_ROLLOUT_TASKS",
    "GenerationSettings",
    "JudgeConfig",
    "LabelMode",
    "MODEL_SLUG",
    "ModelBundle",
    "NnsightActivationBackend",
    "STEM_TASKS",
    "TransformersHookBackend",
    "_bind_judge_config_to_project_root",
    "_find_project_root",
    "_generation_settings",
    "_infer_rollout_task_from_path",
    "_judge_config_from_options",
    "_project_path",
    "_resolve_project_root",
    "_rollout_task_needs_llm_judge",
    "_selected_example_count",
    "_selected_source_indices",
    "_source_index_filter",
    "_validate_insider_prompt_glob",
    "app",
    "build_pooled_feature_cache",
    "cache_pooled_features",
    "check_env",
    "check_environment",
    "console",
    "default_activation_output_path",
    "extract_dataset_activations",
    "extract_rollout_activations",
    "extract_activations",
    "generate_insider_trading",
    "generate_insider_trading_transcripts",
    "generate_rollout_task",
    "generate_rollouts",
    "generate_sycophancy",
    "generate_sycophancy_dataset",
    "grade_insider_trading",
    "grade_insider_trading_transcripts_with_judge",
    "grade_rollout_file",
    "grade_rollouts",
    "infer_rollout_task",
    "load_dotenv",
    "load_model_and_processor",
    "load_nnsight_bundle",
    "load_rollout_prompt_examples",
    "merge_activation_hdf5_shards",
    "merge_activation_shards",
    "merge_rollout_files",
    "merge_rollouts",
    "model_config_from_env",
    "pair_sycophancy_dataset",
    "parse_layer_spec",
    "parse_task_name",
    "pending_rollout_source_indices",
    "preflight_judge_config",
    "preflight_probe_training",
    "preflight_probes",
    "qwen_model_slug",
    "resolve_model_id",
    "run_insider_trading_sweep",
    "run_qwen_sweep",
    "seed_everything",
    "train_probe_directions",
    "train_probe_directions_from_cache",
    "train_probes",
    "train_probes_from_cache",
    "typer",
    "validate_insider_trading_generation_resume",
]
