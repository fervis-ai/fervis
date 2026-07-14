"""Host API ports and context shared with Fervis runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone as datetime_timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fervis.host_api.contracts.authority import ReadAuthority
from fervis.host_api.contracts.read import ReadInvocation
from fervis.host_api.contracts.ports import (
    EndpointExecutionResult,
    HostApiAdapter,
)
from fervis.host_api.contracts import EndpointContract


@dataclass(frozen=True)
class HostContext:
    fervis_name: str = "Fervis"
    organization_name: str = ""
    about_api: str = ""
    timezone: str = "UTC"

    def today(self, *, now: datetime | None = None) -> date:
        current = now or datetime.now(datetime_timezone.utc)
        try:
            timezone = ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unsupported host timezone: {self.timezone}") from exc
        return current.astimezone(timezone).date()


@dataclass(frozen=True)
class HostApiContext:
    """The explicit host API context consumed by Fervis orchestration."""

    adapter: HostApiAdapter
    host_context: HostContext = field(default_factory=HostContext)

    def close(self) -> None:
        self.adapter.close()

    def describe_sources(self) -> tuple[EndpointContract, ...]:
        return self.adapter.describe_sources()

    def endpoint_contract(self, endpoint_name: str) -> EndpointContract | None:
        endpoint_name = str(endpoint_name or "")
        return next(
            (
                contract
                for contract in self.describe_sources()
                if contract.endpoint_name == endpoint_name
            ),
            None,
        )

    def execute_read(
        self,
        *,
        authority: ReadAuthority,
        invocation: ReadInvocation,
    ) -> EndpointExecutionResult:
        return self.adapter.execute_read(
            authority=authority,
            invocation=invocation,
        )


_host_api_context: HostApiContext | None = None


def configure_host_api_context(context: HostApiContext) -> None:
    global _host_api_context
    _host_api_context = context


def get_host_api_context() -> HostApiContext:
    if _host_api_context is None:
        raise RuntimeError("Fervis host API context is not configured.")
    return _host_api_context
