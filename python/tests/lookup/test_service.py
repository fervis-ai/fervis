import pytest

import fervis.host_api.context as host_api_context_module
from fervis.host_api.context import (
    HostApiContext,
)
from fervis.host_api.contracts.authority import ReadAuthority, ReadContextRef
from fervis.host_api.contracts.read import ReadInvocation
from fervis.lookup.orchestration.service import LookupService
from fervis.lookup.orchestration.service import _EndpointRelationDataAccess
from fervis.lookup.orchestration.service import (
    _ConfiguredRelationCatalogProvider,
)
from fervis.host_api.contracts import EndpointContract
from fervis.host_api.contracts import ParameterContract
from fervis.host_api.contracts import ResponseFieldContract
from fervis.host_api.contracts.ports import EndpointExecutionResult
from fervis.lookup.orchestration.result import LookupResult
from fervis.model_io.backbone.factory import build_test_provider_backbone


def test_host_api_context_fails_fast_when_not_configured(monkeypatch):
    monkeypatch.setattr(host_api_context_module, "_host_api_context", None)

    with pytest.raises(RuntimeError, match="host API context is not configured"):
        host_api_context_module.get_host_api_context()


def test_endpoint_relation_data_access_uses_all_pages_only_for_paginated_endpoints():
    invocations = []

    def execute_read(*, authority, invocation):
        invocations.append((authority, invocation))
        return EndpointExecutionResult(
            endpoint_name=invocation.endpoint_name,
            request_url="/v1/record/",
            query_params=dict(invocation.query_params),
            response_status=200,
            response_body={"data": {"id": "record-1"}},
        )

    context = _host_api_context(
        contracts=(
            EndpointContract(
                endpoint_name="get_record",
                url_name="record-detail",
                method="GET",
                path_template="/v1/record/",
                docstring="Record detail.",
                view_class="RecordDetail",
                paginated=False,
            ),
            EndpointContract(
                endpoint_name="list_records",
                url_name="record-list",
                method="GET",
                path_template="/v1/records/",
                docstring="Record list.",
                view_class="RecordList",
                paginated=True,
            ),
        ),
        execute_read=execute_read,
    )
    read_context_ref = ReadContextRef(scheme="delegated_capability", key="user_1")
    access = _EndpointRelationDataAccess(
        host_api_context=context,
        authority=ReadAuthority(
            tenant_id="tenant_1",
            read_context_ref=read_context_ref,
        ),
    )

    access.read(endpoint_name="get_record", args={})
    access.read(endpoint_name="list_records", args={})

    expected_authority = ReadAuthority(
        tenant_id="tenant_1",
        read_context_ref=read_context_ref,
    )
    assert invocations == [
        (
            expected_authority,
            ReadInvocation(
                endpoint_name="get_record",
                page_policy={"mode": "single_page"},
            ),
        ),
        (
            expected_authority,
            ReadInvocation(
                endpoint_name="list_records",
                page_policy={"mode": "all_pages"},
            ),
        ),
    ]


def test_endpoint_relation_data_access_routes_args_by_param_ref_source():
    calls = []

    def execute_read(*, authority, invocation):
        assert authority == ReadAuthority(
            tenant_id="tenant_1",
            read_context_ref=ReadContextRef(
                scheme="delegated_capability",
                key="user_1",
                tenant_key="tenant_1",
            ),
        )
        calls.append(
            {
                "path": dict(invocation.path_params),
                "query": dict(invocation.query_params),
            }
        )
        return EndpointExecutionResult(
            endpoint_name=invocation.endpoint_name,
            request_url="/v1/records/path-id/",
            query_params=dict(invocation.query_params),
            response_status=200,
            response_body={"data": {"id": "record-1"}},
        )

    contract = EndpointContract(
        endpoint_name="get_record",
        url_name="record-detail",
        method="GET",
        path_template="/v1/records/{id}/",
        docstring="Record detail.",
        view_class="RecordDetail",
        path_params=(
            ParameterContract(
                name="id",
                type="uuid",
                required=True,
                source="path",
            ),
        ),
        query_params=(
            ParameterContract(
                name="id",
                type="uuid",
                required=False,
                source="query",
            ),
        ),
    )
    context = _host_api_context(
        contracts=(contract,),
        execute_read=execute_read,
    )
    access = _EndpointRelationDataAccess(
        host_api_context=context,
        authority=ReadAuthority(
            tenant_id="tenant_1",
            read_context_ref=ReadContextRef(
                scheme="delegated_capability",
                key="user_1",
                tenant_key="tenant_1",
            ),
        ),
    )

    access.read(
        endpoint_name="get_record",
        args={
            "get_record.path.id": "path-id",
            "get_record.query.id": "query-id",
        },
    )

    assert calls == [{"path": {"id": "path-id"}, "query": {"id": "query-id"}}]


def test_relation_catalog_provider_uses_configured_sources_without_subject_filtering():
    contracts = (
        EndpointContract(
            endpoint_name="list_visible_records",
            url_name="visible-records",
            method="GET",
            path_template="/v1/visible/",
            docstring="Visible records.",
            view_class="VisibleRecords",
            resource_names=("records",),
            response_fields=(
                ResponseFieldContract(name="id", type="string", path="id"),
            ),
            public_access=True,
        ),
        EndpointContract(
            endpoint_name="list_private_records",
            url_name="private-records",
            method="GET",
            path_template="/v1/private/",
            docstring="Private records.",
            view_class="PrivateRecords",
            resource_names=("records",),
            public_access=False,
            admin_access=False,
            staff_access=False,
            agent_access=False,
            response_fields=(
                ResponseFieldContract(name="id", type="string", path="id"),
            ),
        ),
    )
    context = _host_api_context(contracts=contracts)

    catalog = _ConfiguredRelationCatalogProvider(
        host_api_context=context,
    ).build_relation_catalog()

    assert [read.id for read in catalog.reads] == [
        "list_visible_records",
        "list_private_records",
    ]


def test_fervis_runtime_resolves_provider_from_model_key(monkeypatch):
    captured = {}

    def run_lookup_question(request, ports):
        captured["provider_preferences"] = dict(request.provider_preferences)
        return LookupResult(status="COMPLETED", answer="ok")

    monkeypatch.setattr(
        "fervis.lookup.orchestration.pipeline.run_lookup_question",
        run_lookup_question,
    )

    context = _host_api_context()
    runtime = LookupService(
        host_api_context=context,
        provider_backbone=build_test_provider_backbone(adapters={}),
        observability_query=_EmptyObservabilityQuery(),
    )

    runtime.run_lookup(
        run_id="run_provider_resolution",
        conversation_id="conversation_provider_resolution",
        tenant_id="tenant_provider_resolution",
        question="How much sales today?",
        read_context_ref=ReadContextRef(scheme="delegated_capability", key="user_1"),
        provider="",
        model_key="GPT_5_4_MINI",
        conversation_context={},
        max_budget_usd="1",
        max_thinking_tokens=64,
    )

    assert captured["provider_preferences"]["provider"] == "openai"
    assert captured["provider_preferences"]["modelKey"] == "GPT_5_4_MINI"


def test_fervis_runtime_pre_lookup_limit_preserves_policy_error_without_lineage() -> (
    None
):
    context = _host_api_context()
    runtime = LookupService(
        host_api_context=context,
        provider_backbone=build_test_provider_backbone(adapters={}),
        observability_query=_EmptyObservabilityQuery(),
        lineage_recorder=None,
    )

    result = runtime.run_lookup(
        run_id="run_pre_lookup_limit",
        conversation_id="conversation_pre_lookup_limit",
        tenant_id="tenant_pre_lookup_limit",
        question="How much sales today?",
        read_context_ref=ReadContextRef(scheme="delegated_capability", key="user_1"),
        provider="",
        model_key="GPT_5_4_MINI",
        conversation_context={},
        max_budget_usd="0",
        max_thinking_tokens=64,
    )

    assert result.status == "FAILED"
    assert result.error == "max_budget_exceeded"


class _EmptyObservabilityQuery:
    def run_id_for_answer(self, answer_id):
        return None

    def run_by_id(self, run_id):
        return None

    def run_ids_for_run(self, run_id):
        return (run_id,)

    def run_ids_for_question(self, question_id):
        return ()

    def run_ids_for_conversation(self, conversation_id):
        return ()

    def model_calls_for_run_ids(self, run_ids, *, detail="inspection"):
        return ()

    def model_calls_for_run(self, run_id, step_key=None, *, detail="inspection"):
        return ()


def _host_api_context(
    *,
    contracts: tuple[EndpointContract, ...] = (),
    execute_read=None,
) -> HostApiContext:
    return HostApiContext(
        adapter=_FakeHostApiAdapter(
            contracts=contracts,
            execute_read=execute_read,
        )
    )


class _FakeHostApiAdapter:
    def __init__(self, *, contracts, execute_read=None):
        self.contracts = tuple(contracts)
        self._execute_read = execute_read or _default_execute_read

    def describe_sources(self):
        return self.contracts

    def capture_read_context(self, request):
        del request
        return ReadContextRef(scheme="delegated_capability", key="user_1")

    def execute_read(self, *, authority, invocation):
        return self._execute_read(
            authority=authority, invocation=invocation
        )


def _default_execute_read(*, authority, invocation):
    del authority, invocation
    return EndpointExecutionResult(
        endpoint_name="unused",
        request_url="",
        response_status=200,
        response_body={},
    )
