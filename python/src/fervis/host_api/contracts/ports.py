"""Framework-neutral endpoint access ports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from fervis.host_api.contracts import EndpointContract
from fervis.host_api.contracts.authority import ReadAuthority, ReadContextRef
from fervis.host_api.contracts.credentials import DelegatedReadCredential
from fervis.host_api.contracts.read import ReadInvocation


@dataclass(frozen=True)
class EndpointExecutionResult:
    endpoint_name: str
    request_url: str
    query_params: dict[str, Any]
    response_status: int
    response_body: Any
    page_count: int = 1
    truncated: bool = False

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "endpointName": self.endpoint_name,
            "requestMethod": "GET",
            "requestUrl": self.request_url,
            "requestParams": dict(self.query_params),
            "responseStatus": self.response_status,
            "responseBody": self.response_body,
            "pageCount": self.page_count,
            "truncated": self.truncated,
        }


class EndpointExecutionError(ValueError):
    pass


class HostApiAdapter(Protocol):
    def describe_sources(self) -> tuple[EndpointContract, ...]: ...

    def capture_read_context(self, request: Any) -> ReadContextRef: ...

    def capture_delegated_credential(
        self,
        request: Any,
    ) -> DelegatedReadCredential | None: ...

    def execute_read(
        self,
        *,
        authority: ReadAuthority,
        invocation: ReadInvocation,
    ) -> EndpointExecutionResult: ...
