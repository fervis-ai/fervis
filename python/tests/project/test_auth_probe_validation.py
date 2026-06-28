from __future__ import annotations

from fervis.host_api.contracts import (
    CatalogEndpointContract,
    EndpointContract,
    FrameworkKind,
    ParameterContract,
    ReadAuthority,
    SourceNamespaceKind,
)
from fervis.host_api.contracts.ports import EndpointExecutionResult
from fervis.host_api.contracts.read import ReadInvocation
from fervis.project.auth_config.validation import _read_probe_check
from fervis.project.auth_config.validation import _probe_contract


def test_auth_probe_contract_skips_required_query_param_reads() -> None:
    selected = _probe_contract(
        (
            _endpoint(
                "needs_filter",
                query_params=(
                    ParameterContract(
                        name="user_id",
                        type="string",
                        required=True,
                    ),
                ),
            ),
            _endpoint("list_users"),
        ),
        source_name="default",
    )

    assert selected is not None
    assert selected.endpoint_name == "list_users"


def test_auth_probe_tries_executable_reads_until_one_succeeds() -> None:
    context = _FakeProbeContext(statuses={"get_teams": 401, "get_user_me": 200})

    check = _read_probe_check(
        context=context,
        contracts=(
            _endpoint("get_teams"),
            _endpoint("get_user_me"),
        ),
        source_name="default",
        authority=ReadAuthority(
            tenant_id="default",
            read_context_ref={"scheme": "anonymous"},
        ),
    )

    assert check.status == "passed"
    assert "default.get_user_me returned HTTP 200" in check.message


def _endpoint(
    endpoint_name: str,
    *,
    query_params: tuple[ParameterContract, ...] = (),
) -> EndpointContract:
    return EndpointContract(
        endpoint_name=endpoint_name,
        url_name=endpoint_name,
        method="GET",
        path_template="/api/users/",
        docstring="",
        view_class="",
        query_params=query_params,
        catalog_endpoint=CatalogEndpointContract(
            framework_kind=FrameworkKind.FLASK,
            source_namespace_kind=SourceNamespaceKind.PYTHON_MODULE,
            source_namespace_path=("default",),
            handler_ref=f"app:{endpoint_name}",
        ),
    )


class _FakeProbeContext:
    def __init__(self, *, statuses: dict[str, int]) -> None:
        self.statuses = statuses

    def execute_read(
        self,
        *,
        authority: ReadAuthority,
        invocation: ReadInvocation,
    ) -> EndpointExecutionResult:
        del authority
        status = self.statuses[invocation.endpoint_name]
        return EndpointExecutionResult(
            endpoint_name=invocation.endpoint_name,
            request_url=f"/{invocation.endpoint_name}",
            query_params={},
            response_status=status,
            response_body={},
        )
