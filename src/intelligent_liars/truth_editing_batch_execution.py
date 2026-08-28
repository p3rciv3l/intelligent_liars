"""Ordered, bounded evaluation scheduling for a synchronous study barrier.

The interface deliberately does not choose concurrency.  An evaluator may
implement ``evaluate_batch`` and safely batch requests inside one loaded model,
or omit it and receive deterministic scalar calls.  The study observes neither
path until every request in the barrier has a durable outcome.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar


ProposalT = TypeVar("ProposalT")
ResultT = TypeVar("ResultT")
ResultT_co = TypeVar("ResultT_co", covariant=True)


class BatchExecutionError(ValueError):
    """A batch adapter violated the ordered scheduling contract."""


@dataclass(frozen=True)
class BatchEvaluationRequest(Generic[ProposalT]):
    """One immutable trial request in journal order."""

    trial_id: str
    ordinal: int
    proposal: ProposalT
    record_ids: tuple[str, ...]
    objective_names: tuple[str, ...]


class BatchCapableEvaluator(Protocol[ProposalT, ResultT_co]):
    """Optional capability for evaluators that own safe internal batching."""

    @property
    def identity(self) -> Mapping[str, Any]: ...

    def evaluate_batch(
        self, requests: tuple[BatchEvaluationRequest[ProposalT], ...]
    ) -> Iterable[ResultT_co]: ...


def execute_ordered_batch(
    evaluator: object,
    requests: Sequence[BatchEvaluationRequest[ProposalT]],
    *,
    evaluate_one: Callable[[BatchEvaluationRequest[ProposalT]], ResultT],
    accept_result: Callable[[BatchEvaluationRequest[ProposalT], ResultT], None] | None = None,
) -> tuple[ResultT, ...]:
    """Evaluate one bounded barrier and return results in request order.

    ``evaluate_batch`` is a capability, not a concurrency promise.  A mutable
    one-model evaluator can implement it sequentially while reusing its model;
    isolated workers may implement bounded parallelism behind the same seam.
    """

    frozen = tuple(requests)
    if not frozen:
        return ()
    ordinals = tuple(item.ordinal for item in frozen)
    if ordinals != tuple(sorted(ordinals)) or len(set(ordinals)) != len(ordinals):
        raise BatchExecutionError("batch requests must have unique increasing ordinals")
    if len({item.trial_id for item in frozen}) != len(frozen):
        raise BatchExecutionError("batch requests must have unique trial IDs")

    batch_method = getattr(evaluator, "evaluate_batch", None)
    if callable(batch_method):
        raw_results = batch_method(frozen)
        if isinstance(raw_results, (str, bytes)) or not isinstance(raw_results, Iterable):
            raise BatchExecutionError("batch evaluator must return an ordered result iterable")
        iterator = iter(raw_results)
    else:
        iterator = (evaluate_one(item) for item in frozen)

    results: list[ResultT] = []
    for request in frozen:
        try:
            result = next(iterator)
        except StopIteration as error:
            raise BatchExecutionError(
                "batch evaluator must return one ordered result per request"
            ) from error
        if accept_result is not None:
            accept_result(request, result)
        results.append(result)
    try:
        next(iterator)
    except StopIteration:
        return tuple(results)
    raise BatchExecutionError("batch evaluator must return one ordered result per request")
