"""Runtime run-limit tracking for model turns and cost."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from fervis.observability.usage_types import UsageKey
from fervis.lookup.errors import ErrorCode
from fervis.lookup.lineage.results import (
    RuntimeErrorTerminal,
    runtime_error_terminal_result,
)
from fervis.lookup.orchestration.request import LookupRuntimePorts
from fervis.lookup.orchestration.result import RunStatus
from fervis.observability.usage import (
    RuntimeUsageService,
    usage_payload_from_report,
)
from fervis.observability.query import ObservabilityQueryPort
from fervis.lookup.orchestration.result import LookupResult, PlannerRunResult


@dataclass
class RunLimitTracker:
    run_id: str
    tenant_id: str
    conversation_id: str
    max_budget_usd: Decimal
    observability_query: ObservabilityQueryPort

    def __post_init__(self) -> None:
        self.max_budget_usd = Decimal(str(self.max_budget_usd))

    def failure_before_next_model_turn(self) -> PlannerRunResult | None:
        report = RuntimeUsageService(self.observability_query).for_run(self.run_id)
        usage = usage_payload_from_report(report)
        unpriced_failure = self.failure_if_unpriced_usage(
            unpriced_count=report.unpriced_usage_count,
            usage=usage,
        )
        if unpriced_failure is not None:
            return unpriced_failure
        budget_failure = self.failure_if_budget_exceeded(usage=usage)
        if budget_failure is not None:
            return budget_failure
        return None

    def current_usage(self) -> dict[str, Any]:
        report = RuntimeUsageService(self.observability_query).for_run(self.run_id)
        return dict(usage_payload_from_report(report))

    def result_with_current_usage(self, result: PlannerRunResult) -> PlannerRunResult:
        result.usage = self.current_usage()
        if result.status != RunStatus.FAILED:
            report = RuntimeUsageService(self.observability_query).for_run(self.run_id)
            unpriced_failure = self.failure_if_unpriced_usage(
                unpriced_count=report.unpriced_usage_count,
                usage=result.usage,
            )
            if unpriced_failure is not None:
                return unpriced_failure
            budget_failure = self.failure_if_budget_exceeded(usage=result.usage)
            if budget_failure is not None:
                return budget_failure
        return result

    def failure_if_unpriced_usage(
        self,
        *,
        unpriced_count: int,
        usage: dict[str, Any],
    ) -> PlannerRunResult | None:
        if unpriced_count <= 0:
            return None
        return _limit_failure(
            error=ErrorCode.MAX_BUDGET_EXCEEDED,
            usage=usage,
            details={
                "maxBudgetUsd": str(self.max_budget_usd),
                "unpricedModelTurns": unpriced_count,
            },
        )

    def failure_if_budget_exceeded(
        self,
        *,
        usage: dict[str, Any],
    ) -> PlannerRunResult | None:
        cost = Decimal(str(usage.get(UsageKey.COST_USD) or 0))
        if self.max_budget_usd <= Decimal("0"):
            return _limit_failure(
                error=ErrorCode.MAX_BUDGET_EXCEEDED,
                usage=usage,
                details={
                    "maxBudgetUsd": str(self.max_budget_usd),
                    "costUsd": str(cost),
                },
            )
        if cost <= self.max_budget_usd:
            return None
        return _limit_failure(
            error=ErrorCode.MAX_BUDGET_EXCEEDED,
            usage=usage,
            details={
                "maxBudgetUsd": str(self.max_budget_usd),
                "costUsd": str(cost),
            },
        )


def _limit_failure(
    *,
    error: str,
    usage: dict[str, Any],
    details: dict[str, Any],
) -> PlannerRunResult:
    return PlannerRunResult(
        status=RunStatus.FAILED,
        error=error,
        result_data=details,
        usage=usage,
    )


def _merge_usage(*items: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for item in items:
        if item is None:
            continue
        for key, value in item.items():
            if isinstance(value, (int, float)) and isinstance(
                merged.get(key), (int, float)
            ):
                merged[key] += value
            elif key not in merged:
                merged[key] = value
    return merged


def _limit_before_next_model_turn(
    ports: LookupRuntimePorts,
    run_id: str,
) -> LookupResult | None:
    policy = ports.policy_port
    if policy is None:
        return None
    failure = policy.failure_before_next_model_turn()
    if failure is None:
        return None
    terminal = RuntimeErrorTerminal(
        run_id=run_id,
        status=failure.status,
        error_code=str(failure.error or ""),
        message=str(failure.error or ""),
        result_data=dict(failure.result_data or {}),
        usage=dict(failure.usage or {}),
    )
    sink = ports.lineage_step_sink
    return runtime_error_terminal_result(
        terminal,
        recorder=sink.recorder if sink is not None else None,
        lineage_required=getattr(ports, "lineage_required", False),
    )
