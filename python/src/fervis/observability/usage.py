"""Model-call usage observability over canonical lineage."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum

from fervis.observability.usage_types import CostSource, UsageKey
from fervis.lineage.enums import ModelUsageKind, ModelUsageUnit, RunStepKey
from fervis.lineage.run_chain import run_chain_ids
from fervis.observability.query import (
    ObservabilityModelCall,
    ObservabilityQueryPort,
    ObservabilityUsage,
)


class UsageScope(StrEnum):
    ANSWER = "answer"
    RUN = "run"
    QUESTION = "question"
    CONVERSATION = "conversation"


@dataclass(frozen=True)
class RuntimeUsageFilter:
    step_key: RunStepKey | None = None
    provider: str | None = None
    model_key: str | None = None
    usage_kind: ModelUsageKind | None = None


@dataclass(frozen=True)
class RuntimeUsageReport:
    scope: UsageScope
    scope_id: str
    calls: tuple[ObservabilityModelCall, ...]
    duration_ms_total: int = 0
    duration_ms_by_step: dict[RunStepKey, int] = field(default_factory=dict)
    usage_totals: dict[tuple[ModelUsageKind, ModelUsageUnit], int] = field(
        default_factory=dict
    )
    cost_micros_by_usage_kind: dict[ModelUsageKind, int] = field(default_factory=dict)
    cost_micros_by_currency: dict[str, int] = field(default_factory=dict)
    unpriced_usage_count: int = 0
    missing_cost_count: int = 0
    pricing_versions: tuple[str, ...] = ()


class ObservabilityRootNotFound(LookupError):
    """Raised when an observability root id does not exist."""


class RuntimeUsageService:
    def __init__(self, query: ObservabilityQueryPort) -> None:
        self._query = query

    def for_answer(
        self, answer_id: str, *, filters: RuntimeUsageFilter | None = None
    ) -> RuntimeUsageReport:
        run_id = self._query.run_id_for_answer(answer_id)
        if run_id is None:
            raise ObservabilityRootNotFound(f"answer {answer_id!r} does not exist")
        return self._report(
            scope=UsageScope.ANSWER,
            scope_id=answer_id,
            run_ids=run_chain_ids(
                run_id,
                get_run=self._query.run_by_id,
                missing=lambda item: ObservabilityRootNotFound(
                    f"run {item!r} does not exist"
                ),
            ),
            filters=filters or RuntimeUsageFilter(),
        )

    def for_run(
        self, run_id: str, *, filters: RuntimeUsageFilter | None = None
    ) -> RuntimeUsageReport:
        return self._report(
            scope=UsageScope.RUN,
            scope_id=run_id,
            run_ids=self._required_run_ids(
                self._query.run_ids_for_run(run_id),
                scope=UsageScope.RUN,
                scope_id=run_id,
            ),
            filters=filters or RuntimeUsageFilter(),
        )

    def for_question(
        self, question_id: str, *, filters: RuntimeUsageFilter | None = None
    ) -> RuntimeUsageReport:
        return self._report(
            scope=UsageScope.QUESTION,
            scope_id=question_id,
            run_ids=self._required_run_ids(
                self._query.run_ids_for_question(question_id),
                scope=UsageScope.QUESTION,
                scope_id=question_id,
            ),
            filters=filters or RuntimeUsageFilter(),
        )

    def for_conversation(
        self, conversation_id: str, *, filters: RuntimeUsageFilter | None = None
    ) -> RuntimeUsageReport:
        return self._report(
            scope=UsageScope.CONVERSATION,
            scope_id=conversation_id,
            run_ids=self._required_run_ids(
                self._query.run_ids_for_conversation(conversation_id),
                scope=UsageScope.CONVERSATION,
                scope_id=conversation_id,
            ),
            filters=filters or RuntimeUsageFilter(),
        )

    def _report(
        self,
        *,
        scope: UsageScope,
        scope_id: str,
        run_ids: tuple[str, ...],
        filters: RuntimeUsageFilter,
    ) -> RuntimeUsageReport:
        calls = self._query.model_calls_for_run_ids(run_ids, detail="cost")
        calls = _filter_calls(calls, filters)
        run_order = {run_id: index for index, run_id in enumerate(run_ids)}
        unexpected = sorted({call.run_id for call in calls} - set(run_order))
        if unexpected:
            raise ValueError(
                "observability query returned out-of-scope model calls: "
                + ", ".join(unexpected)
            )
        calls = tuple(sorted(calls, key=lambda call: _call_sort_key(call, run_order)))
        return RuntimeUsageReport(
            scope=scope,
            scope_id=scope_id,
            calls=calls,
            duration_ms_total=_duration_ms_total(calls),
            duration_ms_by_step=_duration_ms_by_step(calls),
            usage_totals=_usage_totals(calls, filters),
            cost_micros_by_usage_kind=_cost_micros_by_usage_kind(calls, filters),
            cost_micros_by_currency=_cost_micros_by_currency(calls, filters),
            unpriced_usage_count=_unpriced_usage_count(calls, filters),
            missing_cost_count=_missing_cost_count(calls, filters),
            pricing_versions=_pricing_versions(calls, filters),
        )

    @staticmethod
    def _required_run_ids(
        run_ids: tuple[str, ...], *, scope: UsageScope, scope_id: str
    ) -> tuple[str, ...]:
        if not run_ids:
            raise ObservabilityRootNotFound(
                f"{scope.value} {scope_id!r} does not exist"
            )
        return run_ids


def usage_payload_from_report(report: RuntimeUsageReport) -> dict[str, object]:
    input_tokens = _usage_quantity(report, ModelUsageKind.INPUT_TOKENS)
    output_tokens = _usage_quantity(report, ModelUsageKind.OUTPUT_TOKENS)
    thinking_tokens = _usage_quantity(report, ModelUsageKind.THINKING_TOKENS)
    total_usd_cost_micros = report.cost_micros_by_currency.get("USD", 0)
    return {
        UsageKey.INPUT_TOKENS: input_tokens,
        UsageKey.OUTPUT_TOKENS: output_tokens,
        UsageKey.THINKING_TOKENS: thinking_tokens,
        UsageKey.INPUT_COST_USD: _usage_cost_usd(report, ModelUsageKind.INPUT_TOKENS),
        UsageKey.OUTPUT_COST_USD: _usage_cost_usd(report, ModelUsageKind.OUTPUT_TOKENS),
        UsageKey.THINKING_COST_USD: _usage_cost_usd(
            report, ModelUsageKind.THINKING_TOKENS
        ),
        UsageKey.COST_USD: total_usd_cost_micros / 1_000_000,
        UsageKey.COST_SOURCE: CostSource.LINEAGE_MODEL_CALL_USAGE,
        UsageKey.PRICING_VERSION: ",".join(report.pricing_versions),
        "durationMs": report.duration_ms_total,
    }


def _usage_quantity(report: RuntimeUsageReport, usage_kind: ModelUsageKind) -> int:
    return sum(
        quantity
        for (kind, _unit), quantity in report.usage_totals.items()
        if kind is usage_kind
    )


def _usage_cost_usd(report: RuntimeUsageReport, usage_kind: ModelUsageKind) -> float:
    return report.cost_micros_by_usage_kind.get(usage_kind, 0) / 1_000_000


def _duration_ms_total(calls: tuple[ObservabilityModelCall, ...]) -> int:
    return sum(call.duration_ms or 0 for call in calls)


def _duration_ms_by_step(
    calls: tuple[ObservabilityModelCall, ...],
) -> dict[RunStepKey, int]:
    totals: defaultdict[RunStepKey, int] = defaultdict(int)
    for call in calls:
        totals[call.step_key] += call.duration_ms or 0
    return dict(totals)


def _usage_totals(
    calls: tuple[ObservabilityModelCall, ...],
    filters: RuntimeUsageFilter,
) -> dict[tuple[ModelUsageKind, ModelUsageUnit], int]:
    totals: defaultdict[tuple[ModelUsageKind, ModelUsageUnit], int] = defaultdict(int)
    for call in calls:
        for usage in _filtered_usage(call, filters):
            totals[(usage.usage_kind, usage.unit)] += usage.quantity
    return dict(totals)


def _cost_micros_by_currency(
    calls: tuple[ObservabilityModelCall, ...],
    filters: RuntimeUsageFilter,
) -> dict[str, int]:
    totals: defaultdict[str, int] = defaultdict(int)
    for call in calls:
        for usage in _filtered_usage(call, filters):
            if usage.cost_micros is None or not usage.currency:
                continue
            totals[usage.currency] += usage.cost_micros
    return dict(totals)


def _cost_micros_by_usage_kind(
    calls: tuple[ObservabilityModelCall, ...],
    filters: RuntimeUsageFilter,
) -> dict[ModelUsageKind, int]:
    totals: defaultdict[ModelUsageKind, int] = defaultdict(int)
    for call in calls:
        for usage in _filtered_usage(call, filters):
            if usage.cost_micros is None or usage.currency != "USD":
                continue
            totals[usage.usage_kind] += usage.cost_micros
    return dict(totals)


def _unpriced_usage_count(
    calls: tuple[ObservabilityModelCall, ...],
    filters: RuntimeUsageFilter,
) -> int:
    return sum(
        1
        for call in calls
        for usage in _filtered_usage(call, filters)
        if _cost_source(usage.price_basis_json) == CostSource.PROVIDER_USAGE_UNPRICED
    )


def _missing_cost_count(
    calls: tuple[ObservabilityModelCall, ...],
    filters: RuntimeUsageFilter,
) -> int:
    return sum(
        1
        for call in calls
        for usage in _filtered_usage(call, filters)
        if usage.cost_micros is None or not usage.currency
    )


def _pricing_versions(
    calls: tuple[ObservabilityModelCall, ...],
    filters: RuntimeUsageFilter,
) -> tuple[str, ...]:
    versions = {
        str(usage.price_basis_json.get(UsageKey.PRICING_VERSION) or "")
        for call in calls
        for usage in _filtered_usage(call, filters)
        if usage.price_basis_json
    }
    return tuple(sorted(version for version in versions if version))


def _call_sort_key(
    call: ObservabilityModelCall, run_order: dict[str, int]
) -> tuple[int, int, int]:
    return (run_order[call.run_id], call.step_sequence, call.call_index)


def _filter_calls(
    calls: tuple[ObservabilityModelCall, ...],
    filters: RuntimeUsageFilter,
) -> tuple[ObservabilityModelCall, ...]:
    output: list[ObservabilityModelCall] = []
    for call in calls:
        if filters.step_key is not None and call.step_key != filters.step_key:
            continue
        if filters.provider is not None and call.provider != filters.provider:
            continue
        if filters.model_key is not None and call.model_key != filters.model_key:
            continue
        output.append(call)
    return tuple(output)


def _filtered_usage(
    call: ObservabilityModelCall, filters: RuntimeUsageFilter
) -> tuple[ObservabilityUsage, ...]:
    if filters.usage_kind is None:
        return call.usage_rows
    return tuple(
        usage for usage in call.usage_rows if usage.usage_kind == filters.usage_kind
    )


def _cost_source(price_basis_json: dict[str, object] | None) -> str:
    if not price_basis_json:
        return ""
    return str(price_basis_json.get(UsageKey.COST_SOURCE) or "")
