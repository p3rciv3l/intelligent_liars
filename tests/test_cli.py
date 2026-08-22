from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("typer")

from typer.testing import CliRunner

from intelligent_liars.cli import (
    app,
    generate_insider_trading,
    generate_rollouts,
    grade_insider_trading,
    grade_rollouts,
    run_insider_trading_sweep,
    _resolve_project_root,
)
from intelligent_liars.activations import (
    ActivationDataset,
    ActivationExample,
    ActivationExtractionSummary,
    ActivationMergeSummary,
)
from intelligent_liars.activation_backends import ActivationSurface
from intelligent_liars.judging import (
    DEFAULT_INSIDER_GRADING_FLUSH_EVERY,
    DEFAULT_JUDGE_CONFIG,
    DEFAULT_ROLLOUT_GRADING_FLUSH_EVERY,
    DEFAULT_ROLLOUT_GRADING_MAX_WORKERS,
    GradingSummary,
    JudgePreflight,
)
from intelligent_liars.models import DEFAULT_MODEL_ID
from intelligent_liars.probes import PooledFeatureCacheSummary, ProbePreflightSummary, ProbeTrainingSummary
from intelligent_liars.rollouts import (
    DEFAULT_ROLLOUT_PROMPT_SET,
    DEFAULT_INSIDER_GENERATION_SETTINGS,
    DEFAULT_ROLLOUT_GENERATION_SETTINGS,
    GenerationSummary,
    PromptExample,
)


def _option_default(command, parameter_name: str):
    option = inspect.signature(command).parameters[parameter_name].default
    return option.default


def _write_probe_direction(path, *, layer: int = 21) -> None:
    path.write_text(
        json.dumps(
            {
                "direction_sign_convention": "sklearn_logistic_coef_positive_points_honest_to_deceptive",
                "general_domain": {
                    "directions": [
                        {
                            "task": "general_domain",
                            "layer": layer,
                            "direction_vector": [1.0, 2.0],
                            "intercept": -0.5,
                            "direction_sign_convention": "sklearn_logistic_coef_positive_points_honest_to_deceptive",
                        }
                    ]
                },
            }
        )
    )


def test_generation_cli_defaults_use_rollout_profiles():
    assert (
        _option_default(generate_rollouts, "batch_size")
        == DEFAULT_ROLLOUT_GENERATION_SETTINGS.batch_size
    )
    assert (
        _option_default(generate_rollouts, "max_new_tokens")
        == DEFAULT_ROLLOUT_GENERATION_SETTINGS.max_new_tokens
    )
    assert _option_default(generate_rollouts, "do_sample") == DEFAULT_ROLLOUT_GENERATION_SETTINGS.do_sample
    assert _option_default(generate_rollouts, "temperature") == DEFAULT_ROLLOUT_GENERATION_SETTINGS.temperature
    assert _option_default(generate_rollouts, "top_p") == DEFAULT_ROLLOUT_GENERATION_SETTINGS.top_p
    assert _option_default(generate_rollouts, "top_k") == DEFAULT_ROLLOUT_GENERATION_SETTINGS.top_k
    assert _option_default(generate_rollouts, "seed") == DEFAULT_ROLLOUT_GENERATION_SETTINGS.seed
    assert _option_default(generate_rollouts, "flush_every") == DEFAULT_ROLLOUT_GENERATION_SETTINGS.flush_every
    assert _option_default(generate_rollouts, "prompt_set") == DEFAULT_ROLLOUT_PROMPT_SET

    assert (
        _option_default(generate_insider_trading, "batch_size")
        == DEFAULT_INSIDER_GENERATION_SETTINGS.batch_size
    )
    assert (
        _option_default(generate_insider_trading, "max_new_tokens")
        == DEFAULT_INSIDER_GENERATION_SETTINGS.max_new_tokens
    )
    assert _option_default(generate_insider_trading, "do_sample") == DEFAULT_INSIDER_GENERATION_SETTINGS.do_sample
    assert (
        _option_default(generate_insider_trading, "temperature")
        == DEFAULT_INSIDER_GENERATION_SETTINGS.temperature
    )
    assert _option_default(generate_insider_trading, "top_p") == DEFAULT_INSIDER_GENERATION_SETTINGS.top_p
    assert _option_default(generate_insider_trading, "top_k") == DEFAULT_INSIDER_GENERATION_SETTINGS.top_k
    assert _option_default(generate_insider_trading, "seed") == DEFAULT_INSIDER_GENERATION_SETTINGS.seed
    assert (
        _option_default(generate_insider_trading, "flush_every")
        == DEFAULT_INSIDER_GENERATION_SETTINGS.flush_every
    )
    assert (
        _option_default(run_insider_trading_sweep, "batch_size")
        == DEFAULT_INSIDER_GENERATION_SETTINGS.batch_size
    )
    assert (
        _option_default(run_insider_trading_sweep, "max_new_tokens")
        == DEFAULT_INSIDER_GENERATION_SETTINGS.max_new_tokens
    )
    assert (
        _option_default(run_insider_trading_sweep, "generation_flush_every")
        == DEFAULT_INSIDER_GENERATION_SETTINGS.flush_every
    )


def test_judging_cli_defaults_use_judging_profiles():
    assert _option_default(grade_rollouts, "provider") == DEFAULT_JUDGE_CONFIG.provider
    assert _option_default(grade_rollouts, "judge_model") == DEFAULT_JUDGE_CONFIG.model
    assert _option_default(grade_rollouts, "max_workers") == DEFAULT_ROLLOUT_GRADING_MAX_WORKERS
    assert _option_default(grade_rollouts, "max_tokens") == DEFAULT_JUDGE_CONFIG.max_tokens
    assert _option_default(grade_rollouts, "max_retries") == DEFAULT_JUDGE_CONFIG.max_retries
    assert _option_default(grade_rollouts, "timeout") == DEFAULT_JUDGE_CONFIG.timeout
    assert _option_default(grade_rollouts, "structured_outputs") == DEFAULT_JUDGE_CONFIG.structured_outputs
    assert (
        _option_default(grade_rollouts, "require_structured_outputs")
        == DEFAULT_JUDGE_CONFIG.require_structured_outputs
    )
    assert _option_default(grade_rollouts, "flush_every") == DEFAULT_ROLLOUT_GRADING_FLUSH_EVERY

    assert _option_default(grade_insider_trading, "provider") == DEFAULT_JUDGE_CONFIG.provider
    assert _option_default(grade_insider_trading, "judge_model") == DEFAULT_JUDGE_CONFIG.model
    assert _option_default(grade_insider_trading, "max_tokens") == DEFAULT_JUDGE_CONFIG.max_tokens
    assert _option_default(grade_insider_trading, "max_retries") == DEFAULT_JUDGE_CONFIG.max_retries
    assert _option_default(grade_insider_trading, "timeout") == DEFAULT_JUDGE_CONFIG.timeout
    assert _option_default(grade_insider_trading, "structured_outputs") == DEFAULT_JUDGE_CONFIG.structured_outputs
    assert (
        _option_default(grade_insider_trading, "require_structured_outputs")
        == DEFAULT_JUDGE_CONFIG.require_structured_outputs
    )
    assert _option_default(grade_insider_trading, "flush_every") == DEFAULT_INSIDER_GRADING_FLUSH_EVERY
    assert _option_default(run_insider_trading_sweep, "provider") == DEFAULT_JUDGE_CONFIG.provider
    assert _option_default(run_insider_trading_sweep, "judge_model") == DEFAULT_JUDGE_CONFIG.model
    assert _option_default(run_insider_trading_sweep, "max_tokens") == DEFAULT_JUDGE_CONFIG.max_tokens
    assert _option_default(run_insider_trading_sweep, "max_retries") == DEFAULT_JUDGE_CONFIG.max_retries
    assert _option_default(run_insider_trading_sweep, "timeout") == DEFAULT_JUDGE_CONFIG.timeout
    assert (
        _option_default(run_insider_trading_sweep, "structured_outputs")
        == DEFAULT_JUDGE_CONFIG.structured_outputs
    )
    assert (
        _option_default(run_insider_trading_sweep, "require_structured_outputs")
        == DEFAULT_JUDGE_CONFIG.require_structured_outputs
    )
    assert (
        _option_default(run_insider_trading_sweep, "grading_flush_every")
        == DEFAULT_INSIDER_GRADING_FLUSH_EVERY
    )


def test_project_root_resolution_walks_up_from_package_dir(tmp_path, monkeypatch):
    package_dir = tmp_path / "src" / "intelligent_liars"
    package_dir.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='example'\n")
    monkeypatch.chdir(package_dir)

    assert _resolve_project_root(None) == tmp_path


def test_check_env_openrouter_missing_key_reports_concise_error(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("intelligent_liars.environment.load_dotenv", lambda: False)
    monkeypatch.setattr("intelligent_liars.clients.openrouter_client.load_dotenv", lambda: False)

    result = CliRunner().invoke(app, ["check-env", "--check-openrouter"])

    assert result.exit_code != 0
    assert "OpenRouter judge preflight failed" in result.output
    assert "Traceback" not in result.output


def test_extract_activations_sycophancy_guard_runs_before_model_load(tmp_path, monkeypatch):
    def fail_model_load(*args, **kwargs):
        del args, kwargs
        raise AssertionError("model load should not be reached")

    monkeypatch.setattr("intelligent_liars.cli.load_model_and_processor", fail_model_load)
    monkeypatch.setattr("intelligent_liars.cli.load_nnsight_bundle", fail_model_load)

    result = CliRunner().invoke(
        app,
        [
            "extract-activations",
            "--task",
            "sycophancy__mmlu_stem_conf_part_2",
            "--project-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code != 0
    assert not isinstance(result.exception, AssertionError)
    assert result.exception is not None
    assert "Sycophancy source directory not found" in str(result.exception)


def test_generate_sycophancy_pair_mode_does_not_load_model(tmp_path, monkeypatch):
    def fail_model_load(*args, **kwargs):
        del args, kwargs
        raise AssertionError("model load should not be reached")

    seen = {}

    def fake_pair(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(output_dir=kwargs["output_dir"], paired_examples=2)

    monkeypatch.setattr("intelligent_liars.cli.load_model_and_processor", fail_model_load)
    monkeypatch.setattr("intelligent_liars.cli.pair_sycophancy_dataset", fake_pair)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='example'\n")
    (tmp_path / "src" / "intelligent_liars").mkdir(parents=True)

    result = CliRunner().invoke(
        app,
        [
            "generate-sycophancy",
            "--project-root",
            str(tmp_path),
            "--mode",
            "pair",
            "--task",
            "abstract_algebra",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["output_dir"] == tmp_path / "data" / "sycophancy" / "qwen3-vl-8b-thinking"
    assert seen["tasks"] == ("abstract_algebra",)


def test_merge_activation_shards_cli_resolves_paths(tmp_path, monkeypatch):
    seen = {}

    def fake_merge(
        paths,
        *,
        output_path,
        overwrite,
        compression,
        merge_strategy,
        expected_queue_plan_id,
        require_queue_plan_id,
        force_stale_merge_lock,
    ):
        seen["paths"] = paths
        seen["output_path"] = output_path
        seen["overwrite"] = overwrite
        seen["compression"] = compression
        seen["merge_strategy"] = merge_strategy
        seen["expected_queue_plan_id"] = expected_queue_plan_id
        seen["require_queue_plan_id"] = require_queue_plan_id
        seen["force_stale_merge_lock"] = force_stale_merge_lock
        return ActivationMergeSummary(
            output_path=output_path,
            shard_paths=tuple(paths),
            tasks=("roleplaying__plain",),
            examples_by_task={"roleplaying__plain": 2},
            token_rows_by_task={"roleplaying__plain": 3},
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("intelligent_liars.cli.merge_activation_hdf5_shards", fake_merge)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='example'\n")
    (tmp_path / "src" / "intelligent_liars").mkdir(parents=True)

    result = CliRunner().invoke(
        app,
        [
            "merge-activation-shards",
            "tmp/shard-1.h5",
            "tmp/shard-2.h5",
            "--output",
            "results/merged.h5",
            "--overwrite",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["paths"] == [tmp_path / "tmp/shard-1.h5", tmp_path / "tmp/shard-2.h5"]
    assert seen["output_path"] == tmp_path / "results/merged.h5"
    assert seen["overwrite"] is True
    assert seen["compression"] == "lzf"
    assert seen["merge_strategy"] == "auto"
    assert seen["expected_queue_plan_id"] is None
    assert seen["require_queue_plan_id"] is False
    assert seen["force_stale_merge_lock"] is False
    assert "examples=2" in result.output
    assert "token_rows=3" in result.output


def test_cache_pooled_features_cli_resolves_paths(tmp_path, monkeypatch):
    seen = {}

    def fake_build(**kwargs):
        seen.update(kwargs)
        return PooledFeatureCacheSummary(
            output_path=kwargs["output_path"],
            input_path=kwargs["input_path"],
            tasks=tuple(kwargs["tasks"]),
            layers=(0, 1),
            hidden_dim=3,
            feature_datasets=4,
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("intelligent_liars.cli.build_pooled_feature_cache", fake_build)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='example'\n")
    (tmp_path / "src" / "intelligent_liars").mkdir(parents=True)

    result = CliRunner().invoke(
        app,
        [
            "cache-pooled-features",
            "--input",
            "tmp/raw.h5",
            "--output",
            "tmp/cache.h5",
            "--layers",
            "0,1",
            "--task",
            "roleplaying__plain",
            "--task",
            "sandbagging_v2__wmdp_mmlu",
            "--overwrite",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["input_path"] == tmp_path / "tmp/raw.h5"
    assert seen["output_path"] == tmp_path / "tmp/cache.h5"
    assert seen["layers"] == "0,1"
    assert seen["tasks"] == ["roleplaying__plain", "sandbagging_v2__wmdp_mmlu"]
    assert seen["compression"] == "lzf"
    assert seen["overwrite"] is True
    assert "Built pooled feature cache" in result.output


def test_train_probes_from_cache_cli_resolves_paths(tmp_path, monkeypatch):
    seen = {}

    def fake_train(**kwargs):
        seen.update(kwargs)
        return ProbeTrainingSummary(
            output_path=kwargs["output_path"],
            input_path=kwargs["cache_path"],
            tasks=tuple(kwargs["tasks"]),
            layers=(0,),
            within_task_results=1,
            cross_task_results=0,
            direction_results=1,
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("intelligent_liars.cli.train_probe_directions_from_cache", fake_train)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='example'\n")
    (tmp_path / "src" / "intelligent_liars").mkdir(parents=True)

    result = CliRunner().invoke(
        app,
        [
            "train-probes-from-cache",
            "--cache",
            "tmp/cache.h5",
            "--source",
            "tmp/raw.h5",
            "--output",
            "tmp/results.json",
            "--layers",
            "0",
            "--task",
            "roleplaying__plain",
            "--evaluation-task",
            "sandbagging_v2__wmdp_mmlu",
            "--test-size",
            "0.5",
            "--random-seed",
            "3",
            "--c",
            "0.25",
            "--overwrite",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["cache_path"] == tmp_path / "tmp/cache.h5"
    assert seen["source_path"] == tmp_path / "tmp/raw.h5"
    assert seen["output_path"] == tmp_path / "tmp/results.json"
    assert seen["layers"] == "0"
    assert seen["tasks"] == ["roleplaying__plain"]
    assert seen["evaluation_tasks"] == ["sandbagging_v2__wmdp_mmlu"]
    assert seen["test_size"] == 0.5
    assert seen["random_seed"] == 3
    assert seen["regularization_c"] == 0.25
    assert seen["overwrite"] is True
    assert "Trained probes from cache" in result.output


def test_run_insider_trading_sweep_preflights_before_qwen_load(tmp_path, monkeypatch):
    events: list[str] = []
    prompt_dir = tmp_path / "data" / "insider_trading" / "prompts" / "default"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "default.yaml").write_text("messages: []\n")

    def fake_preflight(config):
        assert config.model_config_path == tmp_path / "model_deployments.yaml"
        events.append("preflight")
        return JudgePreflight(
            provider="openrouter",
            alias="mock",
            resolved_model="mock",
            structured_outputs=True,
            require_structured_outputs=True,
            checked_api_key=False,
            checked_live_metadata=False,
        )

    def fake_load_model(config):
        del config
        events.append("load_model")
        return SimpleNamespace(model_id=DEFAULT_MODEL_ID)

    def fake_generate(**kwargs):
        assert events == ["preflight", "load_model"]
        assert kwargs["label_mode"] == "unknown"
        events.append("generate")
        return GenerationSummary(
            task="insider_trading",
            output_path=kwargs["output_path"],
            total_examples=1,
            generated_examples=1,
            skipped_existing=0,
        )

    def fake_grade(input_path, **kwargs):
        assert events == ["preflight", "load_model", "generate"]
        assert input_path == tmp_path / "out.json"
        events.append("grade")
        return GradingSummary(
            task="insider_trading",
            input_path=input_path,
            output_path=input_path,
            total_items=1,
            graded_items=1,
            skipped_items=0,
        )

    monkeypatch.setattr("intelligent_liars.cli.preflight_judge_config", fake_preflight)
    monkeypatch.setattr("intelligent_liars.cli.load_model_and_processor", fake_load_model)
    monkeypatch.setattr("intelligent_liars.cli.generate_insider_trading_transcripts", fake_generate)
    monkeypatch.setattr("intelligent_liars.cli.grade_insider_trading_transcripts_with_judge", fake_grade)

    result = CliRunner().invoke(
        app,
        [
            "run-insider-trading-sweep",
            "--project-root",
            str(tmp_path),
            "--output-path",
            "out.json",
            "--mock-response",
            '{"reasoning":"ok","judgement":"yes"}',
        ],
    )

    assert result.exit_code == 0, result.output
    assert events == ["preflight", "load_model", "generate", "grade"]


def test_run_qwen_sweep_preloads_prompts_and_grades_source_indices(tmp_path, monkeypatch):
    events: list[str] = []

    examples = [
        PromptExample(source_index=0, input_messages=[{"role": "user", "content": "zero"}]),
        PromptExample(source_index=1, input_messages=[{"role": "user", "content": "one"}]),
        PromptExample(source_index=2, input_messages=[{"role": "user", "content": "two"}]),
    ]

    def fake_load_examples(task, **kwargs):
        assert task == "roleplaying__plain"
        assert kwargs["project_root"] == tmp_path
        events.append("load_prompts")
        return examples

    def fake_preflight(config):
        assert config.model_config_path == tmp_path / "model_deployments.yaml"
        assert events == ["load_prompts"]
        events.append("preflight")
        return JudgePreflight(
            provider="openrouter",
            alias="mock",
            resolved_model="mock",
            structured_outputs=True,
            require_structured_outputs=True,
            checked_api_key=False,
            checked_live_metadata=False,
        )

    def fake_load_model(config):
        del config
        assert events == ["load_prompts", "preflight"]
        events.append("load_model")
        return SimpleNamespace(model_id=DEFAULT_MODEL_ID)

    def fake_generate(**kwargs):
        assert events == ["load_prompts", "preflight", "load_model"]
        assert kwargs["start"] == 1
        assert kwargs["limit"] == 1
        events.append("generate")
        return GenerationSummary(
            task="roleplaying__plain",
            output_path=kwargs["output_path"],
            total_examples=1,
            generated_examples=1,
            skipped_existing=0,
        )

    def fake_grade(path, **kwargs):
        assert events == ["load_prompts", "preflight", "load_model", "generate"]
        assert path == tmp_path / "rollouts" / "roleplaying__plain__qwen3-vl-8b-thinking.json"
        assert kwargs["start"] == 0
        assert kwargs["limit"] is None
        assert kwargs["source_indices"] == {1}
        events.append("grade")
        return GradingSummary(
            task="roleplaying__plain",
            input_path=path,
            output_path=path,
            total_items=1,
            graded_items=1,
            skipped_items=0,
        )

    monkeypatch.setattr("intelligent_liars.cli.load_rollout_prompt_examples", fake_load_examples)
    monkeypatch.setattr("intelligent_liars.cli.preflight_judge_config", fake_preflight)
    monkeypatch.setattr("intelligent_liars.cli.load_model_and_processor", fake_load_model)
    monkeypatch.setattr("intelligent_liars.cli.generate_rollout_task", fake_generate)
    monkeypatch.setattr("intelligent_liars.cli.grade_rollout_file", fake_grade)

    result = CliRunner().invoke(
        app,
        [
            "run-qwen-sweep",
            "--project-root",
            str(tmp_path),
            "--output-dir",
            "rollouts",
            "--task",
            "roleplaying__plain",
            "--start",
            "1",
            "--limit",
            "1",
            "--mock-response",
            '{"reasoning":"ok","judgement":7}',
        ],
    )

    assert result.exit_code == 0, result.output
    assert events == ["load_prompts", "preflight", "load_model", "generate", "grade"]


def test_run_qwen_sweep_missing_openrouter_key_aborts_before_qwen_load(tmp_path, monkeypatch):
    examples = [PromptExample(source_index=0, input_messages=[{"role": "user", "content": "zero"}])]

    monkeypatch.setattr(
        "intelligent_liars.cli.load_rollout_prompt_examples",
        lambda *args, **kwargs: examples,
    )

    def fail_preflight(config):
        del config
        raise ValueError("OpenRouter API key not provided")

    def fail_model_load(*args, **kwargs):
        del args, kwargs
        raise AssertionError("Qwen should not load when OpenRouter preflight fails")

    monkeypatch.setattr("intelligent_liars.cli.preflight_judge_config", fail_preflight)
    monkeypatch.setattr("intelligent_liars.cli.load_model_and_processor", fail_model_load)
    monkeypatch.setattr("intelligent_liars.cli.generate_rollout_task", fail_model_load)
    monkeypatch.setattr("intelligent_liars.cli.grade_rollout_file", fail_model_load)

    result = CliRunner().invoke(
        app,
        [
            "run-qwen-sweep",
            "--project-root",
            str(tmp_path),
            "--task",
            "roleplaying__plain",
        ],
    )

    assert result.exit_code != 0
    assert not isinstance(result.exception, AssertionError)
    assert "OpenRouter API key not provided" in str(result.exception)
    assert not (tmp_path / "data" / "rollouts").exists()


def test_run_qwen_sweep_resume_validation_runs_before_model_load(tmp_path, monkeypatch):
    events: list[str] = []
    examples = [PromptExample(source_index=0, input_messages=[{"role": "user", "content": "zero"}])]

    monkeypatch.setattr(
        "intelligent_liars.cli.load_rollout_prompt_examples",
        lambda *args, **kwargs: examples,
    )
    monkeypatch.setattr(
        "intelligent_liars.cli.preflight_judge_config",
        lambda config: JudgePreflight(
            provider=config.provider,
            alias="mock",
            resolved_model="mock",
            structured_outputs=config.structured_outputs,
            require_structured_outputs=config.require_structured_outputs,
            checked_api_key=False,
            checked_live_metadata=False,
        ),
    )

    def fail_pending_check(**kwargs):
        events.append("validate")
        raise ValueError(f"incompatible rollout metadata: {kwargs['output_path']}")

    def fail_model_load(*args, **kwargs):
        del args, kwargs
        raise AssertionError("Qwen should not load when resume validation fails")

    monkeypatch.setattr("intelligent_liars.cli.pending_rollout_source_indices", fail_pending_check)
    monkeypatch.setattr("intelligent_liars.cli.load_model_and_processor", fail_model_load)

    result = CliRunner().invoke(
        app,
        [
            "run-qwen-sweep",
            "--project-root",
            str(tmp_path),
            "--task",
            "roleplaying__plain",
            "--mock-response",
            '{"reasoning":"ok","judgement":7}',
        ],
    )

    assert result.exit_code != 0
    assert not isinstance(result.exception, AssertionError)
    assert events == ["validate"]
    assert "incompatible rollout metadata" in str(result.exception)


def test_run_qwen_sweep_passes_generation_options(tmp_path, monkeypatch):
    captured = {}
    examples = [PromptExample(source_index=0, input_messages=[{"role": "user", "content": "zero"}])]

    monkeypatch.setattr(
        "intelligent_liars.cli.load_rollout_prompt_examples",
        lambda *args, **kwargs: examples,
    )
    monkeypatch.setattr(
        "intelligent_liars.cli.preflight_judge_config",
        lambda config: JudgePreflight(
            provider=config.provider,
            alias="mock",
            resolved_model="mock",
            structured_outputs=config.structured_outputs,
            require_structured_outputs=config.require_structured_outputs,
            checked_api_key=False,
            checked_live_metadata=False,
        ),
    )
    monkeypatch.setattr("intelligent_liars.cli.load_model_and_processor", lambda config: SimpleNamespace(model_id=DEFAULT_MODEL_ID))
    monkeypatch.setattr(
        "intelligent_liars.cli.pending_rollout_source_indices",
        lambda **kwargs: {example.source_index for example in examples},
    )
    monkeypatch.setattr("intelligent_liars.cli.seed_everything", lambda seed: captured.setdefault("seed", seed))

    def fake_generate(**kwargs):
        captured["settings"] = kwargs["settings"]
        return GenerationSummary(
            task=kwargs["task"],
            output_path=kwargs["output_path"],
            total_examples=1,
            generated_examples=1,
            skipped_existing=0,
        )

    monkeypatch.setattr("intelligent_liars.cli.generate_rollout_task", fake_generate)
    monkeypatch.setattr(
        "intelligent_liars.cli.grade_rollout_file",
        lambda path, **kwargs: GradingSummary(
            task="roleplaying__plain",
            input_path=path,
            output_path=path,
            total_items=1,
            graded_items=1,
            skipped_items=0,
        ),
    )

    result = CliRunner().invoke(
        app,
        [
            "run-qwen-sweep",
            "--project-root",
            str(tmp_path),
            "--task",
            "roleplaying__plain",
            "--batch-size",
            "2",
            "--max-new-tokens",
            "123",
            "--do-sample",
            "--temperature",
            "0.2",
            "--top-p",
            "0.8",
            "--top-k",
            "40",
            "--seed",
            "99",
            "--flush-every",
            "3",
            "--mock-response",
            '{"reasoning":"ok","judgement":7}',
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["seed"] == 99
    settings = captured["settings"]
    assert settings.batch_size == 2
    assert settings.max_new_tokens == 123
    assert settings.do_sample is True
    assert settings.temperature == 0.2
    assert settings.top_p == 0.8
    assert settings.top_k == 40
    assert settings.seed == 99
    assert settings.flush_every == 3


def test_run_qwen_sweep_default_tasks_use_default_rollout_paths(tmp_path, monkeypatch):
    generated_paths = []
    graded_paths = []
    examples = [PromptExample(source_index=0, input_messages=[{"role": "user", "content": "zero"}])]

    monkeypatch.setattr(
        "intelligent_liars.cli.load_rollout_prompt_examples",
        lambda *args, **kwargs: examples,
    )
    monkeypatch.setattr(
        "intelligent_liars.cli.preflight_judge_config",
        lambda config: JudgePreflight(
            provider=config.provider,
            alias="mock",
            resolved_model="mock",
            structured_outputs=config.structured_outputs,
            require_structured_outputs=config.require_structured_outputs,
            checked_api_key=False,
            checked_live_metadata=False,
        ),
    )
    monkeypatch.setattr("intelligent_liars.cli.load_model_and_processor", lambda config: SimpleNamespace(model_id=DEFAULT_MODEL_ID))
    monkeypatch.setattr(
        "intelligent_liars.cli.pending_rollout_source_indices",
        lambda **kwargs: {example.source_index for example in examples},
    )

    def fake_generate(**kwargs):
        generated_paths.append(kwargs["output_path"])
        return GenerationSummary(
            task=kwargs["task"],
            output_path=kwargs["output_path"],
            total_examples=1,
            generated_examples=1,
            skipped_existing=0,
        )

    def fake_grade(path, **kwargs):
        graded_paths.append(path)
        return GradingSummary(
            task=path.name,
            input_path=path,
            output_path=path,
            total_items=1,
            graded_items=1,
            skipped_items=0,
        )

    monkeypatch.setattr("intelligent_liars.cli.generate_rollout_task", fake_generate)
    monkeypatch.setattr("intelligent_liars.cli.grade_rollout_file", fake_grade)

    result = CliRunner().invoke(
        app,
        [
            "run-qwen-sweep",
            "--project-root",
            str(tmp_path),
            "--mock-response",
            '{"reasoning":"ok","judgement":7}',
        ],
    )

    expected_paths = [
        tmp_path / "data" / "rollouts" / "roleplaying__plain__qwen3-vl-8b-thinking.json",
        tmp_path / "data" / "rollouts" / "sandbagging_v2__wmdp_mmlu__qwen3-vl-8b-thinking.json",
    ]
    assert result.exit_code == 0, result.output
    assert generated_paths == expected_paths
    assert graded_paths == expected_paths


@pytest.mark.parametrize(
    ("args", "required_flags"),
    [
        (["check-env"], ["--check-openrouter", "--require-cuda"]),
        (["generate-rollouts"], ["--prompt-set", "--output-dir"]),
        (["grade-rollouts"], ["structured", "require", "continue"]),
        (["merge-rollouts"], ["--output-path"]),
        (["run-qwen-sweep"], ["--prompt-set", "overwrite"]),
    ],
)
def test_remote_workflow_commands_show_expected_help(args, required_flags):
    result = CliRunner().invoke(app, [*args, "--help"], color=False)

    assert result.exit_code == 0, result.output
    for flag in required_flags:
        assert flag in result.output


def test_grade_rollouts_preflights_roleplaying_from_json_not_filename(tmp_path, monkeypatch):
    events: list[str] = []
    rollout_path = tmp_path / "renamed.json"
    rollout_path.write_text(
        '{"base_name":"roleplaying","variant":"plain","rollouts":[]}\n'
    )

    def fake_preflight(config):
        del config
        events.append("preflight")
        return JudgePreflight(
            provider="openrouter",
            alias="mock",
            resolved_model="mock",
            structured_outputs=True,
            require_structured_outputs=True,
            checked_api_key=False,
            checked_live_metadata=False,
        )

    def fake_grade(path, **kwargs):
        assert events == ["preflight"]
        assert path == rollout_path
        assert kwargs["task"] == "roleplaying__plain"
        events.append("grade")
        return GradingSummary(
            task="roleplaying__plain",
            input_path=path,
            output_path=path,
            total_items=0,
            graded_items=0,
            skipped_items=0,
        )

    monkeypatch.setattr("intelligent_liars.cli.preflight_judge_config", fake_preflight)
    monkeypatch.setattr("intelligent_liars.cli.grade_rollout_file", fake_grade)

    result = CliRunner().invoke(
        app,
        [
            "grade-rollouts",
            "--project-root",
            str(tmp_path),
            "--path",
            str(rollout_path),
            "--mock-response",
            '{"reasoning":"ok","judgement":7}',
        ],
    )

    assert result.exit_code == 0, result.output
    assert events == ["preflight", "grade"]


def test_grade_rollouts_preflights_insider_from_json_not_filename(tmp_path, monkeypatch):
    events: list[str] = []
    rollout_path = tmp_path / "renamed.json"
    rollout_path.write_text(
        '{"base_name":"insider_trading","variant":"onpolicy","rollouts":[]}\n'
    )

    def fake_preflight(config):
        del config
        events.append("preflight")
        return JudgePreflight(
            provider="openrouter",
            alias="mock",
            resolved_model="mock",
            structured_outputs=True,
            require_structured_outputs=True,
            checked_api_key=False,
            checked_live_metadata=False,
        )

    def fake_grade(path, **kwargs):
        assert events == ["preflight"]
        assert path == rollout_path
        assert kwargs["task"] == "insider_trading__onpolicy"
        events.append("grade")
        return GradingSummary(
            task="insider_trading__onpolicy",
            input_path=path,
            output_path=path,
            total_items=0,
            graded_items=0,
            skipped_items=0,
        )

    monkeypatch.setattr("intelligent_liars.cli.preflight_judge_config", fake_preflight)
    monkeypatch.setattr("intelligent_liars.cli.grade_rollout_file", fake_grade)

    result = CliRunner().invoke(
        app,
        [
            "grade-rollouts",
            "--project-root",
            str(tmp_path),
            "--path",
            str(rollout_path),
            "--mock-response",
            '{"reasoning":"ok","judgement":"explicit"}',
        ],
    )

    assert result.exit_code == 0, result.output
    assert events == ["preflight", "grade"]


def test_grade_rollouts_accepts_source_index_filter(tmp_path, monkeypatch):
    rollout_path = tmp_path / "sandbagging.json"
    rollout_path.write_text(
        '{"base_name":"sandbagging_v2","variant":"wmdp_mmlu","rollouts":[]}\n'
    )
    captured = {}

    def fake_grade(path, **kwargs):
        captured["path"] = path
        captured["kwargs"] = kwargs
        return GradingSummary(
            task="sandbagging_v2__wmdp_mmlu",
            input_path=path,
            output_path=path,
            total_items=0,
            graded_items=0,
            skipped_items=0,
        )

    monkeypatch.setattr("intelligent_liars.cli.grade_rollout_file", fake_grade)

    result = CliRunner().invoke(
        app,
        [
            "grade-rollouts",
            "--project-root",
            str(tmp_path),
            "--path",
            str(rollout_path),
            "--source-index",
            "3",
            "--source-start",
            "5",
            "--source-limit",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["path"] == rollout_path
    assert captured["kwargs"]["source_indices"] == {3, 5, 6}
    assert captured["kwargs"]["start"] == 0
    assert captured["kwargs"]["limit"] is None


def test_grade_rollouts_passes_continue_on_error(tmp_path, monkeypatch):
    rollout_path = tmp_path / "roleplaying.json"
    rollout_path.write_text(
        '{"base_name":"roleplaying","variant":"plain","rollouts":[]}\n'
    )
    captured = {}

    monkeypatch.setattr(
        "intelligent_liars.cli.preflight_judge_config",
        lambda config: JudgePreflight(
            provider=config.provider,
            alias="mock",
            resolved_model="mock",
            structured_outputs=config.structured_outputs,
            require_structured_outputs=config.require_structured_outputs,
            checked_api_key=False,
            checked_live_metadata=False,
        ),
    )

    def fake_grade(path, **kwargs):
        captured["path"] = path
        captured["kwargs"] = kwargs
        return GradingSummary(
            task="roleplaying__plain",
            input_path=path,
            output_path=path,
            total_items=0,
            graded_items=0,
            skipped_items=0,
        )

    monkeypatch.setattr("intelligent_liars.cli.grade_rollout_file", fake_grade)

    result = CliRunner().invoke(
        app,
        [
            "grade-rollouts",
            "--project-root",
            str(tmp_path),
            "--path",
            str(rollout_path),
            "--continue-on-error",
            "--mock-response",
            '{"reasoning":"ok","judgement":7}',
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["kwargs"]["continue_on_error"] is True


def test_extract_activations_nnsight_backend_passes_bundle_to_extractor(tmp_path, monkeypatch):
    events: list[str] = []
    dataset = ActivationDataset(
        task="claims__definitional_gemini_600_full",
        examples=[
            ActivationExample(
                task="claims__definitional_gemini_600_full",
                source_index=0,
                output_index=0,
                messages=[{"role": "assistant", "content": "yes", "detect": True}],
                label=0,
            )
        ],
    )

    monkeypatch.setattr(
        "intelligent_liars.cli.ActivationDataset.from_named_task",
        staticmethod(lambda *args, **kwargs: dataset),
    )
    monkeypatch.setattr(
        "intelligent_liars.cli.load_nnsight_bundle",
        lambda config: SimpleNamespace(
            model="model",
            processor="processor",
            tokenizer="tokenizer",
            model_id=DEFAULT_MODEL_ID,
            config=config,
        ),
    )

    class FakeNnsightBackend:
        name = "nnsight"
        model_id = DEFAULT_MODEL_ID
        processor = "processor"
        tokenizer = "tokenizer"

        def __init__(self, bundle):
            assert bundle.processor == "processor"
            events.append("backend")

        def decoder_layer_count(self):
            return 1

        def surface_for_decoder_layer(self, layer_idx):
            return ActivationSurface(name=f"layer_{layer_idx}", path=f"layers[{layer_idx}]", layer_idx=layer_idx)

    def fake_extract_dataset_activations(**kwargs):
        assert kwargs["bundle"].processor == "processor"
        assert kwargs["backend"].name == "nnsight"
        events.append("extract")
        return ActivationExtractionSummary(
            task=kwargs["dataset"].task,
            output_path=kwargs["output_path"],
            examples_seen=1,
            examples_extracted=1,
            skipped_labels=0,
            masked_tokens=1,
            layers=(0,),
            backend="nnsight",
        )

    monkeypatch.setattr("intelligent_liars.cli.NnsightActivationBackend", FakeNnsightBackend)
    monkeypatch.setattr("intelligent_liars.cli.extract_dataset_activations", fake_extract_dataset_activations)

    result = CliRunner().invoke(
        app,
        [
            "extract-activations",
            "--project-root",
            str(tmp_path),
            "--task",
            "claims__definitional_gemini_600_full",
            "--backend",
            "nnsight",
            "--layers",
            "0",
            "--output",
            "acts.h5",
        ],
    )

    assert result.exit_code == 0, result.output
    assert events == ["backend", "extract"]


def test_train_probes_cli_resolves_paths_and_options(tmp_path, monkeypatch):
    captured = {}
    (tmp_path / "pyproject.toml").write_text("[project]\nname='example'\n")
    (tmp_path / "src" / "intelligent_liars").mkdir(parents=True)

    def fake_train_probe_directions(**kwargs):
        captured.update(kwargs)
        return ProbeTrainingSummary(
            output_path=kwargs["output_path"],
            input_path=kwargs["input_path"],
            tasks=tuple(kwargs["tasks"]),
            layers=(0, 1),
            within_task_results=2,
            cross_task_results=0,
            direction_results=2,
        )

    monkeypatch.setattr("intelligent_liars.cli.train_probe_directions", fake_train_probe_directions)

    result = CliRunner().invoke(
        app,
        [
            "train-probes",
            "--project-root",
            str(tmp_path),
            "--input",
            "acts.h5",
            "--output",
            "results/probes.json",
            "--task",
            "roleplaying__plain",
            "--evaluation-task",
            "sandbagging_v2__wmdp_mmlu",
            "--layers",
            "0,1",
            "--test-size",
            "0.4",
            "--random-seed",
            "7",
            "--max-iter",
            "5",
            "--c",
            "0.25",
            "--general-domain-probe",
            "--general-task-class-cap",
            "17",
            "--overwrite",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["input_path"] == tmp_path / "acts.h5"
    assert captured["output_path"] == tmp_path / "results" / "probes.json"
    assert captured["tasks"] == ["roleplaying__plain"]
    assert captured["evaluation_tasks"] == ["sandbagging_v2__wmdp_mmlu"]
    assert captured["layers"] == "0,1"
    assert captured["test_size"] == 0.4
    assert captured["random_seed"] == 7
    assert captured["max_iter"] == 5
    assert captured["regularization_c"] == 0.25
    assert captured["train_general_domain_probe"] is True
    assert captured["general_task_class_cap"] == 17
    assert captured["overwrite"] is True
    assert "within_task=2" in result.output
    assert "directions=2" in result.output
    assert "general_domain=0" in result.output


def test_preflight_probes_cli_resolves_paths_and_options(tmp_path, monkeypatch):
    captured = {}
    (tmp_path / "pyproject.toml").write_text("[project]\nname='example'\n")
    (tmp_path / "src" / "intelligent_liars").mkdir(parents=True)

    def fake_preflight_probe_training(**kwargs):
        captured.update(kwargs)
        return ProbePreflightSummary(
            output_path=kwargs["output_path"],
            input_path=kwargs["input_path"],
            report={
                "recommendations": {
                    "trainable_tasks": ["roleplaying__plain"],
                    "eval_only_tasks": ["sycophancy__mmlu_stem_same_conf_all"],
                    "blocked_tasks": ["insider_trading__upscale"],
                }
            },
        )

    monkeypatch.setattr("intelligent_liars.cli.preflight_probe_training", fake_preflight_probe_training)

    result = CliRunner().invoke(
        app,
        [
            "preflight-probes",
            "--project-root",
            str(tmp_path),
            "--input",
            "acts.h5",
            "--output",
            "docs/validation/probe_preflight.json",
            "--task",
            "roleplaying__plain",
            "--test-size",
            "0.4",
            "--random-seed",
            "7",
            "--min-train-examples-per-class",
            "12",
            "--min-eval-examples-per-class",
            "3",
            "--min-test-examples-per-class",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["input_path"] == tmp_path / "acts.h5"
    assert captured["output_path"] == tmp_path / "docs" / "validation" / "probe_preflight.json"
    assert captured["tasks"] == ["roleplaying__plain"]
    assert captured["test_size"] == 0.4
    assert captured["random_seed"] == 7
    assert captured["min_train_examples_per_class"] == 12
    assert captured["min_eval_examples_per_class"] == 3
    assert captured["min_test_examples_per_class"] == 2
    assert "trainable=1" in result.output
    assert "eval_only=1" in result.output
    assert "blocked=1" in result.output


def test_build_intervention_cli_writes_portable_bundle(tmp_path):
    probe_path = tmp_path / "probe.json"
    output_path = tmp_path / "reflection.json"
    _write_probe_direction(probe_path)

    result = CliRunner().invoke(
        app,
        [
            "build-intervention",
            "--probe",
            str(probe_path),
            "--output",
            str(output_path),
            "--layer",
            "21",
            "--intervention-layer",
            "19",
            "--intervention-layer",
            "21",
            "--method",
            "one_sided_reflection",
            "--orthogonal-control-seed",
            "7",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output_path.read_text())
    assert payload["format"] == "qwen_truth_intervention_v2"
    assert payload["spec"]["layers"] == [19, 21]
    assert payload["spec"]["method"] == "one_sided_reflection"
    assert payload["spec"]["direction_mode"] == "seeded_orthogonal_control"
    assert payload["spec"]["control_seed"] == 7
    assert payload["direction"]["vector"] == [1.0, 2.0]


def test_materialize_writer_edit_rejects_runtime_only_method_before_model_load(
    monkeypatch, tmp_path
):
    probe_path = tmp_path / "probe.json"
    intervention_path = tmp_path / "reflection.json"
    _write_probe_direction(probe_path)
    build_result = CliRunner().invoke(
        app,
        [
            "build-intervention",
            "--probe",
            str(probe_path),
            "--output",
            str(intervention_path),
            "--layer",
            "21",
            "--method",
            "one_sided_reflection",
        ],
    )
    assert build_result.exit_code == 0, build_result.output

    def fail_load(*_args, **_kwargs):
        raise AssertionError("model should not load for a runtime-only intervention")

    monkeypatch.setattr(
        "intelligent_liars.cli_interventions.load_model_and_processor", fail_load
    )

    result = CliRunner().invoke(
        app,
        [
            "materialize-writer-edit",
            "--intervention",
            str(intervention_path),
            "--output",
            str(tmp_path / "model"),
        ],
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)
    assert "requires a runtime intervention" in str(result.exception)


def test_build_homogeneous_writer_edit_records_dropped_affine_intercept(tmp_path):
    probe_path = tmp_path / "probe.json"
    output_path = tmp_path / "writer-reflection.json"
    _write_probe_direction(probe_path)

    result = CliRunner().invoke(
        app,
        [
            "build-intervention",
            "--probe",
            str(probe_path),
            "--output",
            str(output_path),
            "--layer",
            "21",
            "--method",
            "full_reflection",
            "--homogeneous-writer-edit",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output_path.read_text())
    assert payload["direction"]["intercept"] == 0.0
    assert payload["direction"]["original_intercept"] == -0.5
    assert payload["spec"]["token_scope"] == "all"


def test_build_intervention_suite_writes_all_variants(tmp_path):
    probe_path = tmp_path / "probe.json"
    output_dir = tmp_path / "suite"
    _write_probe_direction(probe_path)

    result = CliRunner().invoke(
        app,
        [
            "build-intervention-suite",
            "--probe",
            str(probe_path),
            "--output",
            str(output_dir),
            "--layer",
            "21",
            "--deceptive-margin",
            "1.25",
            "--score-movement-budget",
            "3",
        ],
    )

    assert result.exit_code == 0, result.output
    paths = sorted(output_dir.glob("*.json"))
    assert len(paths) == 8
    payloads = {path.stem: json.loads(path.read_text()) for path in paths}
    assert set(payloads) == {
        "directed_scalar_add_deceptive",
        "directed_affine_project_deceptive",
        "directed_full_reflection",
        "directed_partial_reflection",
        "honest_boundary_ablation",
        "bounded_directed_inversion",
        "bounded_deceptive_margin",
        "seeded_orthogonal_full_reflection",
    }
    methods = {payload["spec"]["method"] for payload in payloads.values()}
    assert methods == {
        "scalar_addition",
        "affine_projection",
        "full_reflection",
        "partial_reflection",
        "one_sided_reflection",
        "bounded_remap",
        "bounded_margin_clamp",
    }
    assert all(
        payload["spec"]["score_directionality"] == "toward_deceptive"
        for name, payload in payloads.items()
        if name != "seeded_orthogonal_full_reflection"
    )
    assert (
        payloads["seeded_orthogonal_full_reflection"]["spec"]
        ["score_directionality"]
        == "symmetric"
    )
    assert (
        payloads["honest_boundary_ablation"]["spec"]["reflection_strength"]
        == 0.5
    )
    assert payloads["directed_scalar_add_deceptive"]["spec"]["score_delta"] == 3.0
    assert (
        payloads["directed_affine_project_deceptive"]["spec"][
            "projection_target"
        ]
        == 1.25
    )
    assert payloads["bounded_directed_inversion"]["spec"]["max_score_delta"] == 6.0
    assert payloads["bounded_deceptive_margin"]["spec"]["margin"] == 1.25
    assert payloads["bounded_deceptive_margin"]["spec"]["max_score_delta"] == 3.0

    portable_variant = output_dir / "directed_partial_reflection.json"
    portable_payload = json.loads(portable_variant.read_text())
    portable_payload["direction"]["source_path"] = "/another/machine/probe.json"
    portable_variant.write_text(json.dumps(portable_payload))

    prompts = tmp_path / "prompts.json"
    prompts.write_text("[]")
    preservation = tmp_path / "preservation.json"
    preservation.write_text("{}")
    plan_path = tmp_path / "plan.json"
    plan_args = []
    for path in paths:
        plan_args.extend(["--intervention", str(path)])
    plan_result = CliRunner().invoke(
        app,
        [
            "plan-intervention-model-fleet",
            *plan_args,
            "--prompts",
            str(prompts),
            "--preservation-teacher",
            str(preservation),
            "--output-root",
            str(tmp_path / "fleet"),
            "--plan",
            str(plan_path),
            "--base-revision",
            "dddddddddddddddddddddddddddddddddddddddd",
        ],
    )
    assert plan_result.exit_code == 0, plan_result.output
    plan = json.loads(plan_path.read_text())
    assert len(plan["variants"]) == 8
    assert plan["base_revision"] == "d" * 40
    assert plan["training"]["svd_rank"] == 2
    assert plan["training"]["projection_dim"] == 13
    assert plan["training"]["projection_seed"] == 42
    assert "weight_tying" not in plan["training"]
    assert "rank" not in plan["training"]
    assert "alpha" not in plan["training"]

    invalid_variant = output_dir / "directed_full_reflection.json"
    invalid_payload = json.loads(invalid_variant.read_text())
    invalid_payload["spec"]["score_directionality"] = "symmetric"
    invalid_variant.write_text(json.dumps(invalid_payload))
    invalid_plan_result = CliRunner().invoke(
        app,
        [
            "plan-intervention-model-fleet",
            *plan_args,
            "--prompts",
            str(prompts),
            "--preservation-teacher",
            str(preservation),
            "--output-root",
            str(tmp_path / "invalid-fleet"),
            "--plan",
            str(tmp_path / "invalid-plan.json"),
            "--base-revision",
            "dddddddddddddddddddddddddddddddddddddddd",
        ],
    )
    assert invalid_plan_result.exit_code != 0
    assert "canonical directional semantics" in str(invalid_plan_result.exception)

    invalid_payload["spec"]["score_directionality"] = "toward_deceptive"
    invalid_payload["spec"]["max_score_delta"] = 0.01
    invalid_variant.write_text(json.dumps(invalid_payload))
    capped_plan_result = CliRunner().invoke(
        app,
        [
            "plan-intervention-model-fleet",
            *plan_args,
            "--prompts",
            str(prompts),
            "--preservation-teacher",
            str(preservation),
            "--output-root",
            str(tmp_path / "capped-fleet"),
            "--plan",
            str(tmp_path / "capped-plan.json"),
            "--base-revision",
            "dddddddddddddddddddddddddddddddddddddddd",
        ],
    )
    assert capped_plan_result.exit_code != 0
    assert "canonical directional semantics" in str(capped_plan_result.exception)
