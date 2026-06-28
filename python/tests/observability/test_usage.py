from __future__ import annotations

import pytest

from fervis.observability.usage_types import CostSource, UsageKey
from fervis.lineage.enums import (
    ModelCallStatus,
    ModelUsageKind,
    ModelUsageUnit,
    RunStepKey,
)
from fervis.observability.usage import (
    ObservabilityRootNotFound,
    RuntimeUsageFilter,
    RuntimeUsageReport,
    RuntimeUsageService,
    UsageScope,
    usage_payload_from_report,
)
from fervis.observability.query import (
    ObservabilityModelCall,
    ObservabilityQueryPort,
    ObservabilityRun,
    ObservabilityUsage,
)


def test_runtime_usage_service_summarizes_answer_usage() -> None:
    service = RuntimeUsageService(
        _UsageReadPort(
            answer_run_ids={"answer_1": "run_1"},
            runs={"run_1": ObservabilityRun(run_id="run_1")},
            calls=(
                ObservabilityModelCall(
                    model_call_id="call_source",
                    run_id="run_1",
                    step_id="step_source_binding",
                    step_key=RunStepKey.SOURCE_BINDING,
                    step_sequence=2,
                    call_index=1,
                    provider="openai",
                    model_key="gpt-test",
                    status=ModelCallStatus.SUCCEEDED,
                    reasoning_effort="medium",
                    duration_ms=1200,
                    prompt_chars=100,
                    schema_chars=50,
                    tool_spec_chars=75,
                    usage_rows=(
                        ObservabilityUsage(
                            usage_kind=ModelUsageKind.INPUT_TOKENS,
                            quantity=20,
                            unit=ModelUsageUnit.TOKENS,
                            provider_usage_key="input_tokens",
                            cost_micros=1000,
                            currency="USD",
                            price_basis_json={UsageKey.PRICING_VERSION: "2026-06"},
                        ),
                        ObservabilityUsage(
                            usage_kind=ModelUsageKind.OUTPUT_TOKENS,
                            quantity=5,
                            unit=ModelUsageUnit.TOKENS,
                            provider_usage_key="output_tokens",
                            cost_micros=500,
                            currency="USD",
                            price_basis_json={UsageKey.PRICING_VERSION: "2026-06"},
                        ),
                    ),
                ),
                ObservabilityModelCall(
                    model_call_id="call_fact",
                    run_id="run_1",
                    step_id="step_fact_planning",
                    step_key=RunStepKey.FACT_PLANNING,
                    step_sequence=3,
                    call_index=1,
                    provider="anthropic",
                    model_key="claude-test",
                    status=ModelCallStatus.SUCCEEDED,
                    reasoning_effort="low",
                    duration_ms=800,
                    prompt_chars=80,
                    schema_chars=40,
                    tool_spec_chars=55,
                    usage_rows=(
                        ObservabilityUsage(
                            usage_kind=ModelUsageKind.INPUT_TOKENS,
                            quantity=10,
                            unit=ModelUsageUnit.TOKENS,
                            provider_usage_key="input_tokens",
                            cost_micros=250,
                            currency="USD",
                            price_basis_json={UsageKey.PRICING_VERSION: "2026-06"},
                        ),
                        ObservabilityUsage(
                            usage_kind=ModelUsageKind.OUTPUT_TOKENS,
                            quantity=2,
                            unit=ModelUsageUnit.TOKENS,
                            provider_usage_key="output_tokens",
                            cost_micros=80,
                            currency="USD",
                            price_basis_json={UsageKey.PRICING_VERSION: "2026-06"},
                        ),
                        ObservabilityUsage(
                            usage_kind=ModelUsageKind.REASONING_TOKENS,
                            quantity=3,
                            unit=ModelUsageUnit.TOKENS,
                            provider_usage_key="reasoning_tokens",
                            cost_micros=20,
                            currency="USD",
                            price_basis_json={UsageKey.PRICING_VERSION: "2026-06"},
                        ),
                    ),
                ),
            ),
        )
    )

    report = service.for_answer("answer_1")

    assert report.scope == UsageScope.ANSWER
    assert report.scope_id == "answer_1"
    assert (
        report.usage_totals[(ModelUsageKind.INPUT_TOKENS, ModelUsageUnit.TOKENS)] == 30
    )
    assert (
        report.usage_totals[(ModelUsageKind.OUTPUT_TOKENS, ModelUsageUnit.TOKENS)] == 7
    )
    assert (
        report.usage_totals[(ModelUsageKind.REASONING_TOKENS, ModelUsageUnit.TOKENS)]
        == 3
    )
    assert report.cost_micros_by_currency == {"USD": 1850}
    assert report.duration_ms_total == 2000
    assert report.duration_ms_by_step == {
        RunStepKey.SOURCE_BINDING: 1200,
        RunStepKey.FACT_PLANNING: 800,
    }
    assert report.pricing_versions == ("2026-06",)
    assert [
        (call.step_key, call.provider, call.model_key) for call in report.calls
    ] == [
        (RunStepKey.SOURCE_BINDING, "openai", "gpt-test"),
        (RunStepKey.FACT_PLANNING, "anthropic", "claude-test"),
    ]
    assert report.calls[0].reasoning_effort == "medium"
    assert report.calls[0].tool_spec_chars == 75


def test_runtime_usage_service_supports_step_scoped_usage() -> None:
    service = RuntimeUsageService(
        _UsageReadPort(
            run_ids={"run_1": ("run_1",)},
            calls=(
                ObservabilityModelCall(
                    model_call_id="call_source",
                    run_id="run_1",
                    step_id="step_source_binding",
                    step_key=RunStepKey.SOURCE_BINDING,
                    step_sequence=2,
                    call_index=1,
                    provider="openai",
                    model_key="gpt-test",
                    status=ModelCallStatus.SUCCEEDED,
                    usage_rows=(
                        ObservabilityUsage(
                            usage_kind=ModelUsageKind.INPUT_TOKENS,
                            quantity=20,
                            unit=ModelUsageUnit.TOKENS,
                            provider_usage_key="input_tokens",
                            cost_micros=1000,
                            currency="USD",
                        ),
                    ),
                ),
                ObservabilityModelCall(
                    model_call_id="call_fact",
                    run_id="run_1",
                    step_id="step_fact_planning",
                    step_key=RunStepKey.FACT_PLANNING,
                    step_sequence=3,
                    call_index=1,
                    provider="anthropic",
                    model_key="claude-test",
                    status=ModelCallStatus.SUCCEEDED,
                    usage_rows=(
                        ObservabilityUsage(
                            usage_kind=ModelUsageKind.OUTPUT_TOKENS,
                            quantity=2,
                            unit=ModelUsageUnit.TOKENS,
                            provider_usage_key="output_tokens",
                            cost_micros=80,
                            currency="USD",
                        ),
                    ),
                ),
            ),
        )
    )

    report = service.for_run(
        "run_1", filters=RuntimeUsageFilter(step_key=RunStepKey.FACT_PLANNING)
    )

    assert report.scope == UsageScope.RUN
    assert report.scope_id == "run_1"
    assert [call.model_call_id for call in report.calls] == ["call_fact"]
    assert report.usage_totals == {
        (ModelUsageKind.OUTPUT_TOKENS, ModelUsageUnit.TOKENS): 2
    }
    assert report.cost_micros_by_currency == {"USD": 80}


def test_runtime_usage_service_supports_question_and_conversation_scopes() -> None:
    call = ObservabilityModelCall(
        model_call_id="call_1",
        run_id="run_1",
        step_id="step_source_binding",
        step_key=RunStepKey.SOURCE_BINDING,
        step_sequence=1,
        call_index=1,
        provider="openai",
        model_key="gpt-test",
        status=ModelCallStatus.SUCCEEDED,
        usage_rows=(
            ObservabilityUsage(
                usage_kind=ModelUsageKind.INPUT_TOKENS,
                quantity=20,
                unit=ModelUsageUnit.TOKENS,
                provider_usage_key="input_tokens",
                cost_micros=1000,
                currency="USD",
            ),
        ),
    )
    service = RuntimeUsageService(
        _UsageReadPort(
            calls=(call,),
            question_run_ids={"question_1": ("run_1",)},
            conversation_run_ids={"conversation_1": ("run_1",)},
        )
    )

    question_report = service.for_question("question_1")
    conversation_report = service.for_conversation("conversation_1")

    assert question_report.scope == UsageScope.QUESTION
    assert question_report.usage_totals == {
        (ModelUsageKind.INPUT_TOKENS, ModelUsageUnit.TOKENS): 20
    }
    assert conversation_report.scope == UsageScope.CONVERSATION
    assert conversation_report.cost_micros_by_currency == {"USD": 1000}


def test_runtime_usage_service_supports_provider_model_and_usage_filters() -> None:
    service = RuntimeUsageService(
        _UsageReadPort(
            run_ids={"run_1": ("run_1",)},
            calls=(
                ObservabilityModelCall(
                    model_call_id="call_openai",
                    run_id="run_1",
                    step_id="step_source_binding",
                    step_key=RunStepKey.SOURCE_BINDING,
                    step_sequence=1,
                    call_index=1,
                    provider="openai",
                    model_key="gpt-test",
                    status=ModelCallStatus.SUCCEEDED,
                    usage_rows=(
                        ObservabilityUsage(
                            usage_kind=ModelUsageKind.INPUT_TOKENS,
                            quantity=20,
                            unit=ModelUsageUnit.TOKENS,
                            provider_usage_key="input_tokens",
                            cost_micros=1000,
                            currency="USD",
                        ),
                    ),
                ),
                ObservabilityModelCall(
                    model_call_id="call_anthropic",
                    run_id="run_1",
                    step_id="step_fact_planning",
                    step_key=RunStepKey.FACT_PLANNING,
                    step_sequence=2,
                    call_index=1,
                    provider="anthropic",
                    model_key="claude-test",
                    status=ModelCallStatus.SUCCEEDED,
                    usage_rows=(
                        ObservabilityUsage(
                            usage_kind=ModelUsageKind.OUTPUT_TOKENS,
                            quantity=2,
                            unit=ModelUsageUnit.TOKENS,
                            provider_usage_key="output_tokens",
                            cost_micros=80,
                            currency="USD",
                        ),
                    ),
                ),
            ),
        )
    )

    report = service.for_run(
        "run_1",
        filters=RuntimeUsageFilter(
            provider="anthropic",
            model_key="claude-test",
            usage_kind=ModelUsageKind.OUTPUT_TOKENS,
        ),
    )

    assert [call.model_call_id for call in report.calls] == ["call_anthropic"]
    assert report.usage_totals == {
        (ModelUsageKind.OUTPUT_TOKENS, ModelUsageUnit.TOKENS): 2
    }
    assert report.cost_micros_by_currency == {"USD": 80}


def test_runtime_usage_service_preserves_unpriced_usage_provenance() -> None:
    service = RuntimeUsageService(
        _UsageReadPort(
            run_ids={"run_1": ("run_1",)},
            calls=(
                ObservabilityModelCall(
                    model_call_id="call_1",
                    run_id="run_1",
                    step_id="step_source_binding",
                    step_key=RunStepKey.SOURCE_BINDING,
                    step_sequence=1,
                    call_index=1,
                    provider="openai",
                    model_key="gpt-test",
                    status=ModelCallStatus.SUCCEEDED,
                    usage_rows=(
                        ObservabilityUsage(
                            usage_kind=ModelUsageKind.INPUT_TOKENS,
                            quantity=20,
                            unit=ModelUsageUnit.TOKENS,
                            provider_usage_key="input_tokens",
                            price_basis_json={
                                UsageKey.COST_SOURCE: CostSource.PROVIDER_USAGE_UNPRICED
                            },
                        ),
                    ),
                ),
            ),
        )
    )

    report = service.for_run("run_1")

    assert report.unpriced_usage_count == 1
    assert report.missing_cost_count == 1
    assert report.cost_micros_by_currency == {}


def test_runtime_usage_payload_preserves_component_costs_and_zero_cost_rows() -> None:
    service = RuntimeUsageService(
        _UsageReadPort(
            run_ids={"run_1": ("run_1",)},
            calls=(
                ObservabilityModelCall(
                    model_call_id="call_1",
                    run_id="run_1",
                    step_id="step_source_binding",
                    step_key=RunStepKey.SOURCE_BINDING,
                    step_sequence=1,
                    call_index=1,
                    provider="openai",
                    model_key="gpt-test",
                    status=ModelCallStatus.SUCCEEDED,
                    duration_ms=175,
                    usage_rows=(
                        ObservabilityUsage(
                            usage_kind=ModelUsageKind.INPUT_TOKENS,
                            quantity=20,
                            unit=ModelUsageUnit.TOKENS,
                            provider_usage_key="inputTokens",
                            cost_micros=1000,
                            currency="USD",
                            price_basis_json={
                                UsageKey.COST_SOURCE: CostSource.CONFIGURED_PROVIDER_PRICING,
                                UsageKey.PRICING_VERSION: "test-pricing",
                            },
                        ),
                        ObservabilityUsage(
                            usage_kind=ModelUsageKind.OUTPUT_TOKENS,
                            quantity=5,
                            unit=ModelUsageUnit.TOKENS,
                            provider_usage_key="outputTokens",
                            cost_micros=500,
                            currency="USD",
                            price_basis_json={
                                UsageKey.COST_SOURCE: CostSource.CONFIGURED_PROVIDER_PRICING,
                                UsageKey.PRICING_VERSION: "test-pricing",
                            },
                        ),
                        ObservabilityUsage(
                            usage_kind=ModelUsageKind.THINKING_TOKENS,
                            quantity=3,
                            unit=ModelUsageUnit.TOKENS,
                            provider_usage_key="thinkingTokens",
                            cost_micros=0,
                            currency="USD",
                            price_basis_json={
                                UsageKey.COST_SOURCE: CostSource.CONFIGURED_PROVIDER_PRICING,
                                UsageKey.PRICING_VERSION: "test-pricing",
                            },
                        ),
                    ),
                ),
            ),
        )
    )

    report = service.for_run("run_1")
    payload = usage_payload_from_report(report)

    assert report.missing_cost_count == 0
    assert payload[UsageKey.INPUT_TOKENS] == 20
    assert payload[UsageKey.OUTPUT_TOKENS] == 5
    assert payload[UsageKey.THINKING_TOKENS] == 3
    assert payload[UsageKey.INPUT_COST_USD] == 0.001
    assert payload[UsageKey.OUTPUT_COST_USD] == 0.0005
    assert payload[UsageKey.THINKING_COST_USD] == 0.0
    assert payload[UsageKey.COST_USD] == 0.0015
    assert payload["durationMs"] == 175


def test_runtime_usage_payload_reports_cost_usd_from_usd_rows_only() -> None:
    report = RuntimeUsageReport(
        scope=UsageScope.RUN,
        scope_id="run_1",
        calls=(),
        cost_micros_by_currency={"USD": 1000, "EUR": 7000},
    )

    payload = usage_payload_from_report(report)

    assert payload[UsageKey.COST_USD] == 0.001


def test_runtime_usage_service_rejects_missing_roots() -> None:
    service = RuntimeUsageService(_UsageReadPort(calls=()))

    with pytest.raises(ObservabilityRootNotFound, match="missing_answer"):
        service.for_answer("missing_answer")


class _UsageReadPort(ObservabilityQueryPort):
    def __init__(
        self,
        *,
        calls: tuple[ObservabilityModelCall, ...],
        answer_run_ids: dict[str, str] | None = None,
        runs: dict[str, ObservabilityRun] | None = None,
        run_ids: dict[str, tuple[str, ...]] | None = None,
        question_run_ids: dict[str, tuple[str, ...]] | None = None,
        conversation_run_ids: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        self._calls = calls
        self._answer_run_ids = answer_run_ids or {}
        self._runs = runs or {}
        self._run_ids = run_ids or {}
        self._question_run_ids = question_run_ids or {}
        self._conversation_run_ids = conversation_run_ids or {}

    def run_id_for_answer(self, answer_id: str) -> str | None:
        return self._answer_run_ids.get(answer_id)

    def run_by_id(self, run_id: str) -> ObservabilityRun | None:
        return self._runs.get(run_id, ObservabilityRun(run_id=run_id))

    def run_ids_for_run(self, run_id: str) -> tuple[str, ...]:
        return self._run_ids.get(run_id, ())

    def run_ids_for_question(self, question_id: str) -> tuple[str, ...]:
        return self._question_run_ids.get(question_id, ())

    def run_ids_for_conversation(self, conversation_id: str) -> tuple[str, ...]:
        return self._conversation_run_ids.get(conversation_id, ())

    def model_calls_for_run_ids(
        self, run_ids: tuple[str, ...], *, detail: str = "inspection"
    ) -> tuple[ObservabilityModelCall, ...]:
        run_id_set = set(run_ids)
        return tuple(call for call in self._calls if call.run_id in run_id_set)

    def model_calls_for_run(
        self,
        run_id: str,
        step_key: RunStepKey | None = None,
        *,
        detail: str = "inspection",
    ) -> tuple[ObservabilityModelCall, ...]:
        return tuple(
            call
            for call in self._calls
            if call.run_id == run_id and (step_key is None or call.step_key == step_key)
        )
