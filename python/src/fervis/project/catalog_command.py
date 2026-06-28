"""Project catalog command service."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from fervis.interfaces.agent.actions import (
    add_schema_metadata_action,
    run_init_action,
)
from fervis.host_api.contracts import EndpointContract

from .configuration import (
    ConfigProblem,
    LoadedFervisConfig,
    load_fervis_project_config,
)
from .discovery import ProjectInspection
from .host_runtime import host_project_runtime
from .host_api_context import host_api_context_from_config
from .integration import DjangoAppSource, FastAPIAppSource, FlaskAppSource
from .catalog_errors import catalog_failure_action


CATALOG_RESULT_SCHEMA_VERSION = "v0.1"


@dataclass(frozen=True)
class CatalogCommandResult:
    payload: dict[str, object]
    exit_code: int = 0
    payload_schema: str = "fervis-catalog-result.v0.1"
    next_actions: list[dict[str, object]] = field(default_factory=list)


def catalog_view(project: ProjectInspection) -> CatalogCommandResult:
    loaded = load_fervis_project_config(project)
    if isinstance(loaded, ConfigProblem):
        return CatalogCommandResult(
            payload={
                "error": {
                    "code": loaded.code,
                    "message": loaded.message,
                    "retryable": False,
                }
            },
            exit_code=2,
            payload_schema="fervis-command-error.v0.1",
            next_actions=[run_init_action(project.framework)]
            if project.framework in {"django", "fastapi", "flask"}
            else [],
        )
    try:
        contracts = _configured_contracts(project, loaded)
    except Exception as exc:
        return CatalogCommandResult(
            payload=_blocked_catalog_payload(loaded, reason=str(exc)),
            exit_code=2,
            next_actions=[catalog_failure_action(exc, loaded=loaded)],
        )
    payload = _catalog_payload(loaded, contracts)
    return CatalogCommandResult(
        payload=payload,
        exit_code=2 if payload["status"] == "blocked" else 0,
        next_actions=_payload_next_actions(payload),
    )


def _configured_contracts(
    project: ProjectInspection,
    loaded: LoadedFervisConfig,
) -> tuple[EndpointContract, ...]:
    with host_project_runtime(project):
        return host_api_context_from_config(
            project=project,
            loaded_config=loaded,
        ).describe_sources()


def _catalog_payload(
    loaded: LoadedFervisConfig,
    contracts: tuple[EndpointContract, ...],
) -> dict[str, object]:
    sources = [
        _source_payload(source, contracts=contracts) for source in loaded.config.sources
    ]
    blocked_sources = [
        {
            "source": source["name"],
            "reason": "one_or_more_endpoints_not_read_eligible",
        }
        for source in sources
        if any(
            not bool(endpoint["eligible"])
            for endpoint in source.get("endpoints", ())
            if isinstance(endpoint, dict)
        )
    ]
    return {
        "schema_version": CATALOG_RESULT_SCHEMA_VERSION,
        "status": "blocked" if blocked_sources else "passed",
        "source_count": len(sources),
        "endpoint_count": sum(int(source["endpoint_count"]) for source in sources),
        "sources": sources,
        "blocked_sources": blocked_sources,
    }


def _blocked_catalog_payload(
    loaded: LoadedFervisConfig,
    *,
    reason: str,
) -> dict[str, object]:
    return {
        "schema_version": CATALOG_RESULT_SCHEMA_VERSION,
        "status": "blocked",
        "source_count": len(loaded.config.sources),
        "endpoint_count": 0,
        "sources": [
            _configured_source_payload(source) for source in loaded.config.sources
        ],
        "blocked_sources": [
            {
                "source": source.name,
                "reason": reason,
            }
            for source in loaded.config.sources
        ],
    }


def _payload_next_actions(payload: dict[str, object]) -> list[dict[str, object]]:
    actions: list[dict[str, object]] = []
    seen: set[str] = set()
    for source in payload.get("sources", ()):
        if not isinstance(source, dict):
            continue
        for endpoint in source.get("endpoints", ()):
            if not isinstance(endpoint, dict):
                continue
            for action in endpoint.get("next_actions", ()):
                if not isinstance(action, dict):
                    continue
                key = json.dumps(action, sort_keys=True, separators=(",", ":"))
                if key in seen:
                    continue
                actions.append(action)
                seen.add(key)
    return actions


def _source_payload(
    source: DjangoAppSource | FastAPIAppSource | FlaskAppSource,
    *,
    contracts: tuple[EndpointContract, ...],
) -> dict[str, object]:
    endpoints = [
        _endpoint_payload(contract)
        for contract in contracts
        if _contract_source_name(contract) == source.name
    ]
    return {
        **_configured_source_payload(source),
        "endpoint_count": len(endpoints),
        "endpoints": endpoints,
    }


def _configured_source_payload(
    source: DjangoAppSource | FastAPIAppSource | FlaskAppSource,
) -> dict[str, object]:
    configured: dict[str, object]
    kind: str
    if isinstance(source, DjangoAppSource):
        kind = "django_app"
        configured = {
            "app_modules": list(source.app_modules),
            "path_prefixes": list(source.path_prefixes),
        }
    elif isinstance(source, FastAPIAppSource):
        kind = "fastapi_app"
        configured = {
            "import_paths": list(source.import_paths),
            "path_prefixes": list(source.path_prefixes),
        }
    else:
        kind = "flask_app"
        configured = {
            "app": source.app,
            "app_args": list(source.app_args),
            "app_kwargs": dict(source.app_kwargs),
            "path_prefixes": list(source.path_prefixes),
            "blueprints": list(source.blueprints),
        }
    return {
        "name": source.name,
        "kind": kind,
        "configured": configured,
    }


def _endpoint_payload(contract: EndpointContract) -> dict[str, object]:
    readiness = _endpoint_readiness(contract)
    return {
        "name": contract.endpoint_name,
        "method": str(contract.method).upper(),
        "path": contract.path_template,
        "query_params": [_parameter_payload(param) for param in contract.query_params],
        "path_params": [_parameter_payload(param) for param in contract.path_params],
        "response_fields": [
            _response_field_payload(field) for field in contract.response_fields
        ],
        "quality": readiness["quality"],
        "eligible": readiness["eligible"],
        "blocked_reason": readiness["blocked_reason"],
        "next_actions": readiness["next_actions"],
        "capabilities": {
            "read": contract.supports_lookup_read(),
            "filter": bool(contract.query_params or contract.path_params),
            "aggregate_candidate": bool(contract.response_fields),
        },
    }


def _endpoint_readiness(contract: EndpointContract) -> dict[str, object]:
    if contract.supports_lookup_read():
        quality = (
            "schema_backed"
            if contract.response_schema_source != "missing"
            else "documented"
        )
        return {
            "quality": quality,
            "eligible": True,
            "blocked_reason": None,
            "next_actions": [],
        }
    reason = "response_schema_missing"
    return {
        "quality": "route_only",
        "eligible": False,
        "blocked_reason": reason,
        "next_actions": [add_schema_metadata_action(contract.endpoint_name)],
    }


def _parameter_payload(param: Any) -> dict[str, object]:
    return {
        "name": str(getattr(param, "name", "") or ""),
        "required": bool(getattr(param, "required", False)),
        "source": str(getattr(param, "source", "") or ""),
        "type": str(getattr(param, "type", "") or ""),
    }


def _response_field_payload(field: Any) -> dict[str, object]:
    return {
        "path": str(getattr(field, "path", "") or ""),
        "type": str(getattr(field, "type", "") or ""),
    }


def _contract_source_name(contract: EndpointContract) -> str:
    catalog_endpoint = contract.catalog_endpoint
    if catalog_endpoint is None or not catalog_endpoint.source_namespace_path:
        return ""
    return catalog_endpoint.source_namespace_path[0]
