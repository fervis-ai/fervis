"""Framework-neutral lookup runtime composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fervis.host_api.context import (
    HostApiContext,
    get_host_api_context,
)
from fervis.host_api.contracts.authority import ReadAuthority, ReadContextRef
from fervis.host_api.contracts.credentials import DelegatedReadCredential
from fervis.lookup.orchestration.result import AnswerSource
from fervis.lookup.orchestration.request import (
    LookupRequest,
    LookupProgressSink,
    LookupRuntimePorts,
)
from fervis.lookup.answer_program.persistence import LineageProgramInvocationBinding
from fervis.lookup.turn_prompts.context import HostPromptContext
from fervis.lineage.ports import LineageRecorderPort
from fervis.model_io.backbone.factory import (
    ProviderBackbone,
    build_provider_backbone,
)
from fervis.observability.query import ObservabilityQueryPort
from fervis.lookup.lineage.steps import LineageRuntimeStepSink
from fervis.lookup.orchestration.limits import RunLimitTracker
from fervis.lookup.orchestration.result import (
    PlannerRunResult,
    delivery_result_data,
)
from fervis.lookup.orchestration.host_runtime import (
    HostRelationDataAccess,
    host_relation_catalog,
)

if TYPE_CHECKING:
    from fervis.lookup.fact_planning.request import RuntimeValueContext
    from fervis.lookup.answer_program.persistence import PriorProgramInvocationReader
    from fervis.lookup.clarification.model import ClarificationOwnerResponse


class LookupService:
    def __init__(
        self,
        *,
        provider_backbone: ProviderBackbone | None = None,
        provider_name: str | None = None,
        host_api_context: HostApiContext | None = None,
        observability_query: ObservabilityQueryPort | None = None,
        lineage_recorder: LineageRecorderPort | None = None,
        prior_program_invocations: PriorProgramInvocationReader | None = None,
    ) -> None:
        self.host_api_context = host_api_context or get_host_api_context()
        self.provider_backbone = provider_backbone or build_provider_backbone(
            provider_name
        )
        if observability_query is None:
            raise RuntimeError("Observability query is required.")
        self.observability_query = observability_query
        self.lineage_recorder = lineage_recorder
        self.prior_program_invocations = prior_program_invocations
        self.model_router = self.provider_backbone.model_router

    def run_lookup(
        self,
        *,
        run_id: str,
        conversation_id: str,
        tenant_id: str,
        question: str,
        read_context_ref: ReadContextRef,
        provider: str,
        model_key: str,
        conversation_context: dict[str, Any],
        max_budget_usd: Any,
        max_thinking_tokens: int,
        delegated_credential: DelegatedReadCredential | None = None,
        user_context: dict[str, Any] | None = None,
        active_attempt: int | None = None,
        progress_sink: LookupProgressSink | None = None,
        clarification_responses: tuple[ClarificationOwnerResponse, ...] = (),
    ) -> PlannerRunResult:
        from fervis.lookup.orchestration.pipeline import run_lookup_question

        _require_model_prompt_host_context(self.host_api_context)
        limits = RunLimitTracker(
            run_id=run_id,
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            max_budget_usd=max_budget_usd,
            observability_query=self.observability_query,
        )
        failure = limits.failure_before_next_model_turn()
        if failure is not None:
            from fervis.lookup.lineage.results import (
                RuntimeErrorTerminal,
                runtime_error_terminal_result,
            )

            terminal = runtime_error_terminal_result(
                RuntimeErrorTerminal(
                    run_id=run_id,
                    status=failure.status,
                    error_code=str(failure.error or ""),
                    message=str(failure.error or ""),
                    result_data=dict(failure.result_data or {}),
                    usage=dict(failure.usage or {}),
                ),
                recorder=self.lineage_recorder,
                lineage_required=True,
            )
            return PlannerRunResult(
                status=terminal.status,
                error=terminal.error,
                result_data=dict(terminal.result_data or {}),
                usage=dict(terminal.usage or {}),
            )
        lineage_recorder = self.lineage_recorder
        if lineage_recorder is None:
            raise RuntimeError("Lineage recorder is required for lookup execution.")
        resolved_provider = self.provider_backbone.resolve_provider(
            requested_provider=provider,
            model_key=model_key,
        )
        authority = ReadAuthority(
            tenant_id=tenant_id,
            read_context_ref=read_context_ref,
            delegated_credential=delegated_credential,
        )
        result = run_lookup_question(
            LookupRequest(
                question=question,
                conversation_context=conversation_context,
                run_id=run_id,
                tenant_id=tenant_id,
                authority_ref=authority.evidence_ref,
                user_context={
                    **dict(user_context or {}),
                    "conversationId": conversation_id,
                },
                provider_preferences={
                    "provider": resolved_provider,
                    "modelKey": model_key,
                },
                max_thinking_tokens=max_thinking_tokens,
                runtime_values=_runtime_values(self.host_api_context),
                host=_host_prompt_context(self.host_api_context),
                active_attempt=active_attempt,
                clarification_responses=clarification_responses,
            ),
            LookupRuntimePorts(
                relation_catalog_port=_ConfiguredRelationCatalogProvider(
                    host_api_context=self.host_api_context,
                ),
                data_access_port=HostRelationDataAccess(
                    host_api_context=self.host_api_context,
                    authority=authority,
                ),
                planner_model_port=_SelectedModelPort(
                    model_port=self.model_router,
                    model_key=model_key,
                ),
                program_invocation_binding=LineageProgramInvocationBinding(
                    run_id=run_id,
                    recorder=lineage_recorder,
                ),
                prior_program_invocations=self.prior_program_invocations,
                lineage_step_sink=_lineage_step_sink(
                    run_id=run_id,
                    recorder=lineage_recorder,
                    active_attempt=active_attempt,
                ),
                lineage_required=True,
                policy_port=limits,
                progress_sink=progress_sink,
            ),
        )
        return limits.result_with_current_usage(
            PlannerRunResult(
                status=result.status,
                answer=result.answer,
                result_data=_result_data(result),
                usage=result.usage,
                error=result.error or None,
                answer_source=AnswerSource.RENDERED_FACT,
            )
        )


@dataclass(frozen=True)
class _SelectedModelPort:
    model_port: Any
    model_key: str

    def generate(self, **kwargs: Any) -> dict[str, Any]:
        return self.model_port.generate(model_key=self.model_key, **kwargs)


@dataclass(frozen=True)
class _ConfiguredRelationCatalogProvider:
    host_api_context: HostApiContext

    def build_relation_catalog(self):
        return host_relation_catalog(self.host_api_context)


def _runtime_values(
    host_api_context: HostApiContext,
) -> RuntimeValueContext:
    from fervis.lookup.fact_planning.request import RuntimeValueContext

    host = host_api_context.host_context
    return RuntimeValueContext(
        runtime_date=host.today().isoformat(),
        timezone=host.timezone,
    )


def _host_prompt_context(host_api_context: HostApiContext) -> HostPromptContext:
    host = host_api_context.host_context
    return HostPromptContext(
        organization_name=host.organization_name,
        about_api=host.about_api,
    )


def _require_model_prompt_host_context(
    host_api_context: HostApiContext,
) -> None:
    host = host_api_context.host_context
    missing_fields = tuple(
        field
        for field, value in (
            ("host.organization_name", host.organization_name),
            ("host.about_api", host.about_api),
        )
        if not value.strip()
    )
    if not missing_fields:
        return
    missing = " and ".join(missing_fields)
    raise RuntimeError(
        f"Fervis lookup cannot start: {missing} are required. "
        "Configure the organization, API domain, and supported factual-question "
        "scope in config/fervis.json."
    )


def _lineage_step_sink(
    *,
    run_id: str,
    recorder: LineageRecorderPort | None,
    active_attempt: int | None,
) -> Any:
    if recorder is None:
        return None
    return LineageRuntimeStepSink(
        run_id=run_id,
        recorder=recorder,
        attempt=active_attempt,
    )


def _result_data(result: Any) -> dict[str, Any] | None:
    rendered = result.rendered_fact
    if rendered is None:
        return delivery_result_data(result.result_data)
    from fervis.lookup.answer_rendering import rendered_fact_payload

    return delivery_result_data(rendered_fact_payload(rendered))
