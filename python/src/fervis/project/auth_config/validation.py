"""Validate configured host reads for doctor."""

from __future__ import annotations

from dataclasses import dataclass

from fervis.interfaces.agent.actions import run_doctor_probe_action
from fervis.host_api.context import HostApiContext
from fervis.host_api.contracts import EndpointContract
from fervis.host_api.contracts.authority import ReadAuthority, ReadContextRef
from fervis.host_api.contracts.probe import is_probeable_get_contract
from fervis.host_api.contracts.read import ReadInvocation
from fervis.project.configuration import LoadedFervisConfig
from fervis.project.discovery import ProjectInspection
from fervis.project.host_api_context import host_api_context_from_config
from fervis.project.host_runtime import host_project_runtime


@dataclass(frozen=True)
class AuthProbeCheck:
    id: str
    status: str
    message: str
    fix: dict[str, object] | None = None

    @property
    def passed(self) -> bool:
        return self.status != "failed"


def read_context_probe_checks(
    *,
    project: ProjectInspection,
    loaded_config: LoadedFervisConfig,
    auth_schema: dict[str, object],
    probe_key: str | None,
) -> list[AuthProbeCheck]:
    read_context_ref = _probe_read_context_ref(
        auth_schema,
        probe_key=probe_key,
    )
    if read_context_ref is None:
        return [
            AuthProbeCheck(
                id="auth.probe_read_context",
                status="skipped",
                message=(
                    "Optional read-context probe was not run. Pass "
                    "--probe-read-context-key to validate read execution for one "
                    "concrete host context."
                ),
                fix=_configure_fix(),
            )
        ]
    checks = [
        AuthProbeCheck(
            id="auth.probe_read_context",
            status="passed",
            message=(
                f"Auth probe read context is configured as {read_context_ref.scheme}:"
                f"{read_context_ref.key}."
            ),
        )
    ]
    checks.extend(
        _read_probe_checks(
            project=project,
            loaded_config=loaded_config,
            read_context_ref=read_context_ref,
        )
    )
    return checks


def _read_probe_checks(
    *,
    project: ProjectInspection,
    loaded_config: LoadedFervisConfig,
    read_context_ref: ReadContextRef,
) -> list[AuthProbeCheck]:
    try:
        with host_project_runtime(project):
            context = host_api_context_from_config(
                project=project,
                loaded_config=loaded_config,
            )
            contracts = context.describe_sources()
    except Exception as exc:
        return [
            AuthProbeCheck(
                id="auth.probe_read",
                status="failed",
                message=f"Auth probe could not load configured source contracts: {exc}",
            )
        ]
    return [
        _read_probe_check(
            context=context,
            contracts=contracts,
            source_name=source_name,
            authority=ReadAuthority.from_read_context(read_context_ref),
        )
        for source_name in _source_names(loaded_config)
    ]


def _read_probe_check(
    *,
    context: HostApiContext,
    contracts: tuple[EndpointContract, ...],
    source_name: str,
    authority: ReadAuthority,
) -> AuthProbeCheck:
    check_id = f"auth.probe_read.{_check_id_segment(source_name)}"
    probe_contracts = _probe_contracts(contracts, source_name=source_name)
    if not probe_contracts:
        return AuthProbeCheck(
            id=check_id,
            status="skipped",
            message=(
                f"Configured source {source_name!r} has no GET endpoint that can be "
                "executed without path or required query params."
            ),
        )
    failures: list[str] = []
    for contract in probe_contracts:
        try:
            result = context.execute_read(
                authority=authority,
                invocation=ReadInvocation(endpoint_name=contract.endpoint_name),
            )
        except Exception as exc:
            failures.append(f"{contract.endpoint_name}: {exc}")
            continue
        if 200 <= result.response_status < 300:
            return AuthProbeCheck(
                id=check_id,
                status="passed",
                message=(
                    f"Auth probe read {source_name}.{contract.endpoint_name} returned "
                    f"HTTP {result.response_status} through the configured read context."
                ),
            )
        failures.append(f"{contract.endpoint_name}: HTTP {result.response_status}")
    return AuthProbeCheck(
        id=check_id,
        status="failed",
        message=(
            f"Auth probe read failed for every executable endpoint in {source_name}: "
            + "; ".join(failures)
        ),
    )


def _probe_contracts(
    contracts: tuple[EndpointContract, ...],
    *,
    source_name: str,
) -> tuple[EndpointContract, ...]:
    return tuple(
        contract
        for contract in contracts
        if _is_executable_probe_contract(contract, source_name=source_name)
    )


def _probe_contract(
    contracts: tuple[EndpointContract, ...],
    *,
    source_name: str,
) -> EndpointContract | None:
    return next(
        iter(_probe_contracts(contracts, source_name=source_name)),
        None,
    )


def _is_executable_probe_contract(
    contract: EndpointContract,
    *,
    source_name: str,
) -> bool:
    return is_probeable_get_contract(contract, source_name=source_name)


def _probe_read_context_ref(
    auth_schema: dict[str, object],
    *,
    probe_key: str | None,
) -> ReadContextRef | None:
    framework = str(auth_schema.get("framework") or "")
    key = str(probe_key or "")
    if not key.strip():
        return None
    if framework == "django":
        return ReadContextRef(scheme="django_principal", key=key)
    if framework == "fastapi":
        return ReadContextRef(scheme="fastapi_principal", key=key)
    if framework == "flask":
        return ReadContextRef(scheme="flask_principal", key=key)
    raise ValueError(f"Unsupported auth framework for read-context probe: {framework}")


def _source_names(loaded_config: LoadedFervisConfig) -> tuple[str, ...]:
    return tuple(source.name for source in loaded_config.config.sources)


def _check_id_segment(value: str) -> str:
    segment = "".join(
        char if char.isalnum() or char in {"_", "-"} else "_" for char in value
    )
    return segment or "unnamed"


def _configure_fix() -> dict[str, object]:
    return run_doctor_probe_action()
