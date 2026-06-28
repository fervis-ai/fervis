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
from fervis.host_api.contracts.read import ReadInvocation
from fervis.lookup.orchestration.result import AnswerSource
from fervis.lookup.relation_catalog.from_host_api import (
    relation_catalog_from_endpoint_contracts,
)
from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.orchestration.request import (
    LookupRequest,
    LookupProgressSink,
    LookupRuntimePorts,
)
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

if TYPE_CHECKING:
    from fervis.lookup.fact_planning.request import RuntimeValueContext


class LookupService:
    def __init__(
        self,
        *,
        provider_backbone: ProviderBackbone | None = None,
        provider_name: str | None = None,
        host_api_context: HostApiContext | None = None,
        observability_query: ObservabilityQueryPort | None = None,
        lineage_recorder: LineageRecorderPort | None = None,
    ) -> None:
        self.host_api_context = host_api_context or get_host_api_context()
        self.provider_backbone = provider_backbone or build_provider_backbone(
            provider_name
        )
        if observability_query is None:
            raise RuntimeError("Observability query is required.")
        self.observability_query = observability_query
        self.lineage_recorder = lineage_recorder
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
    ) -> PlannerRunResult:
        from fervis.lookup.orchestration.pipeline import run_lookup_question

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
        resolved_provider = self.provider_backbone.resolve_provider(
            requested_provider=provider,
            model_key=model_key,
        )
        result = run_lookup_question(
            LookupRequest(
                question=question,
                conversation_context=conversation_context,
                run_id=run_id,
                tenant_id=tenant_id,
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
            ),
            LookupRuntimePorts(
                relation_catalog_port=_ConfiguredRelationCatalogProvider(
                    host_api_context=self.host_api_context,
                ),
                data_access_port=_EndpointRelationDataAccess(
                    host_api_context=self.host_api_context,
                    authority=ReadAuthority(
                        tenant_id=tenant_id,
                        read_context_ref=read_context_ref,
                        delegated_credential=delegated_credential,
                    ),
                ),
                planner_model_port=_SelectedModelPort(
                    model_port=self.model_router,
                    model_key=model_key,
                ),
                lineage_step_sink=_lineage_step_sink(
                    run_id=run_id,
                    recorder=self.lineage_recorder,
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

    def build_relation_catalog(self) -> RelationCatalog:
        return relation_catalog_from_endpoint_contracts(
            self.host_api_context.describe_sources()
        )


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


@dataclass(frozen=True)
class _EndpointRelationDataAccess:
    host_api_context: HostApiContext
    authority: ReadAuthority

    def read(self, *, endpoint_name: str, args: dict[str, Any]) -> dict[str, Any]:
        contract = self.host_api_context.endpoint_contract(endpoint_name)
        if contract is None:
            raise ValueError(f"Unknown endpoint contract: {endpoint_name}")
        path_params: dict[str, Any] = {}
        query_params: dict[str, Any] = {}
        params = {
            f"{endpoint_name}.{param.source}.{param.name}": param
            for param in (*contract.path_params, *contract.query_params)
        }
        for param_ref, value in args.items():
            param = params.get(str(param_ref))
            if param is None:
                raise ValueError(f"Unknown endpoint parameter: {param_ref}")
            if param.source == "path":
                path_params[param.name] = value
            elif param.source == "query":
                query_params[param.name] = value
            else:
                raise ValueError(f"Unsupported endpoint parameter source: {param_ref}")
        return self.host_api_context.execute_read(
            authority=self.authority,
            invocation=ReadInvocation(
                endpoint_name=endpoint_name,
                path_params=path_params,
                query_params=query_params,
                page_policy={
                    "mode": "all_pages" if contract.paginated else "single_page"
                },
            ),
        ).to_public_dict()


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
