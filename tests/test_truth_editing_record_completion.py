from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from intelligent_liars.truth_editing_judge_contracts import (
    AbsoluteJudgeResult,
    AbsoluteResponseLabel,
    AbsoluteSemanticResult,
    JudgeCacheReceipt,
    TokenUsage,
    judge_cache_key_sha256,
)
from intelligent_liars.truth_editing_record_completion import (
    FileSemanticRecordCompletionStore,
    RecordCompletionError,
    RecordCompletionRequirement,
    RecordCompletionScope,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode()
    ).hexdigest()


def _evidence(
    requirement: RecordCompletionRequirement,
    *,
    adapter_code_sha256: str = _sha("adapter-v1"),
    price_usd: float = 0.00125,
) -> tuple[AbsoluteJudgeResult, JudgeCacheReceipt]:
    label = AbsoluteResponseLabel(
        response_id=requirement.record_id,
        asserted_answer="Rome",
        asserted_state="false",
        answer_type="direct",
        semantic_type_valid=True,
        plausible_falsehood="yes",
        coherent=True,
        relevant=True,
        internally_consistent=True,
        instruction_following=True,
        hedging_or_abstention=False,
        brief_evidence="A specific plausible city answer.",
        confidence="high",
    )
    result = AbsoluteJudgeResult(
        judge_config_sha256=_sha("judge"),
        rubric_sha256=_sha("rubric"),
        request_sha256=_sha(f"request:{requirement.record_id}"),
        operational_status="succeeded",
        operational_failure=None,
        result=AbsoluteSemanticResult(
            responses=(label,),
            false_reporting_persistent=False,
            retained_truth_behaviorally_supported="yes",
            failure_mode="ordinary_error",
            contradictions=(),
            unresolved_ambiguities=(),
        ),
        content_sha256="",
    )
    result = replace(
        result,
        content_sha256=_canonical_sha(
            {
                key: value
                for key, value in result.to_payload().items()
                if key != "content_sha256"
            }
        ),
    )
    cache_key = judge_cache_key_sha256(
        judge_kind="absolute",
        rubric_sha256=result.rubric_sha256,
        judge_config_sha256=result.judge_config_sha256,
        resolved_model="z-ai/glm-5.3-flash",
        provider_route="z-ai/fp8",
        request_parameters_sha256=_sha("request-parameters"),
        prompt_bundle_sha256=requirement.prompt_sha256,
        response_sha256s=(requirement.raw_generation_sha256,),
    )
    receipt = JudgeCacheReceipt(
        cache_key_sha256=cache_key,
        judge_kind="absolute",
        rubric_sha256=result.rubric_sha256,
        judge_config_sha256=result.judge_config_sha256,
        resolved_model="z-ai/glm-5.3-flash",
        provider_route="z-ai/fp8",
        request_parameters_sha256=_sha("request-parameters"),
        prompt_bundle_sha256=requirement.prompt_sha256,
        response_sha256s=(requirement.raw_generation_sha256,),
        raw_request_sha256=result.request_sha256,
        raw_response_sha256=_sha(f"response:{requirement.record_id}"),
        parsed_result_sha256=result.content_sha256,
        operational_status="succeeded",
        operational_failure=None,
        cache_status="miss",
        attempts=2,
        latency_ms=123.5,
        usage=TokenUsage(101, 19, 120),
        price_usd=price_usd,
        code_sha256=adapter_code_sha256,
        created_at="2026-08-30T12:34:56Z",
        content_sha256="",
    )
    receipt = replace(
        receipt,
        content_sha256=_canonical_sha(
            {
                key: value
                for key, value in receipt.to_payload().items()
                if key != "content_sha256"
            }
        ),
    )
    return result, receipt


def _scope(
    store: FileSemanticRecordCompletionStore,
    requirements: tuple[RecordCompletionRequirement, ...],
) -> RecordCompletionScope:
    return RecordCompletionScope.create(
        evaluator_config_sha256=_sha("evaluator"),
        dataset_manifest_sha256=_sha("dataset"),
        recipe_sha256=_sha("recipe"),
        edited_model_sha256=_sha("model"),
        output_bundle_sha256=_sha("outputs"),
        tier="discovery",
        judge_config_sha256=_sha("judge"),
        rubric_sha256=_sha("rubric"),
        judge_execution_identity_sha256=None,
        completion_store_identity_sha256=store.identity_sha256,
        requirements=requirements,
    )


def test_success_survives_restart_with_exact_request_cost_and_cache_identity(
    tmp_path,
) -> None:
    adapter_sha = _sha("adapter-v1")
    first = RecordCompletionRequirement("record-1", _sha("prompt-1"), _sha("raw-1"))
    second = RecordCompletionRequirement("record-2", _sha("prompt-2"), _sha("raw-2"))
    store = FileSemanticRecordCompletionStore(
        tmp_path / "record-completions",
        accepted_judge_adapter_code_sha256s=(adapter_sha,),
    )
    scope = _scope(store, (first, second))

    completion = store.resolve(scope, first, lambda: _evidence(first))

    assert store.missing_record_ids(scope) == ("record-2",)
    assert completion.cache_receipt.raw_request_sha256 == _sha("request:record-1")
    assert completion.cache_receipt.cache_key_sha256 == judge_cache_key_sha256(
        judge_kind="absolute",
        rubric_sha256=_sha("rubric"),
        judge_config_sha256=_sha("judge"),
        resolved_model="z-ai/glm-5.3-flash",
        provider_route="z-ai/fp8",
        request_parameters_sha256=_sha("request-parameters"),
        prompt_bundle_sha256=first.prompt_sha256,
        response_sha256s=(first.raw_generation_sha256,),
    )
    assert completion.cache_receipt.usage == TokenUsage(101, 19, 120)
    assert completion.cache_receipt.price_usd == pytest.approx(0.00125)
    assert completion.cache_receipt.cache_status == "miss"

    restarted = FileSemanticRecordCompletionStore(
        tmp_path / "record-completions",
        accepted_judge_adapter_code_sha256s=(adapter_sha,),
    )
    calls = 0

    def must_not_run():
        nonlocal calls
        calls += 1
        raise AssertionError("completed record was retried")

    replayed = restarted.resolve(scope, first, must_not_run)
    assert calls == 0
    assert replayed == completion
    assert restarted.completed(scope) == {"record-1": completion}


def test_failed_producer_leaves_only_that_record_missing_and_retryable(tmp_path) -> None:
    adapter_sha = _sha("adapter-v1")
    first = RecordCompletionRequirement("record-1", _sha("prompt-1"), _sha("raw-1"))
    second = RecordCompletionRequirement("record-2", _sha("prompt-2"), _sha("raw-2"))
    store = FileSemanticRecordCompletionStore(
        tmp_path / "record-completions",
        accepted_judge_adapter_code_sha256s=(adapter_sha,),
    )
    scope = _scope(store, (first, second))
    store.resolve(scope, first, lambda: _evidence(first))

    with pytest.raises(TimeoutError, match="judge interrupted"):
        store.resolve(
            scope,
            second,
            lambda: (_ for _ in ()).throw(TimeoutError("judge interrupted")),
        )

    assert store.missing_record_ids(scope) == ("record-2",)
    restarted = FileSemanticRecordCompletionStore(
        tmp_path / "record-completions",
        accepted_judge_adapter_code_sha256s=(adapter_sha,),
    )
    restarted.resolve(scope, second, lambda: _evidence(second))
    assert restarted.missing_record_ids(scope) == ()


def test_concurrent_resolve_invokes_the_paid_producer_once(tmp_path) -> None:
    adapter_sha = _sha("adapter-v1")
    requirement = RecordCompletionRequirement(
        "record-1", _sha("prompt-1"), _sha("raw-1")
    )
    first_store = FileSemanticRecordCompletionStore(
        tmp_path / "record-completions",
        accepted_judge_adapter_code_sha256s=(adapter_sha,),
    )
    second_store = FileSemanticRecordCompletionStore(
        tmp_path / "record-completions",
        accepted_judge_adapter_code_sha256s=(adapter_sha,),
    )
    scope = _scope(first_store, (requirement,))
    call_count = 0
    count_lock = threading.Lock()

    def paid_call():
        nonlocal call_count
        with count_lock:
            call_count += 1
        time.sleep(0.05)
        return _evidence(requirement)

    with ThreadPoolExecutor(max_workers=2) as pool:
        completions = tuple(
            pool.map(
                lambda store: store.resolve(scope, requirement, paid_call),
                (first_store, second_store),
            )
        )

    assert call_count == 1
    assert completions[0] == completions[1]


def test_incompatible_adapter_evidence_is_rejected_without_a_completion(tmp_path) -> None:
    requirement = RecordCompletionRequirement(
        "record-1", _sha("prompt-1"), _sha("raw-1")
    )
    store = FileSemanticRecordCompletionStore(
        tmp_path / "record-completions",
        accepted_judge_adapter_code_sha256s=(_sha("adapter-v1"),),
    )
    scope = _scope(store, (requirement,))

    with pytest.raises(RecordCompletionError, match="adapter code"):
        store.resolve(
            scope,
            requirement,
            lambda: _evidence(
                requirement, adapter_code_sha256=_sha("incompatible-adapter")
            ),
        )

    assert store.missing_record_ids(scope) == ("record-1",)


def test_store_accepts_monotonic_adapter_compatibility_growth(tmp_path) -> None:
    root = tmp_path / "record-completions"
    original = FileSemanticRecordCompletionStore(
        root,
        accepted_judge_adapter_code_sha256s=(_sha("adapter-v1"),),
    )

    upgraded = FileSemanticRecordCompletionStore(
        root,
        accepted_judge_adapter_code_sha256s=(
            _sha("adapter-v1"),
            _sha("adapter-v2"),
        ),
    )

    assert upgraded.identity_sha256 == original.identity_sha256
    with pytest.raises(RecordCompletionError, match="contract differs"):
        FileSemanticRecordCompletionStore(
            root,
            accepted_judge_adapter_code_sha256s=(_sha("adapter-v2"),),
        )
