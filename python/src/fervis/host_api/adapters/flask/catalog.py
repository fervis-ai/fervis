"""Flask endpoint catalog from runtime app routes."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from fervis.host_api.contracts import (
    CatalogEndpointContract,
    EndpointContract,
    FrameworkKind,
    ParameterContract,
    SourceNamespaceKind,
)

from fervis.host_api.adapters.jsonapi_schema import (
    enrich_contract_from_jsonapi_resource,
)
from fervis.host_api.adapters.openapi import endpoint_evidence_from_openapi
from fervis.host_api.adapters.runtime_output import suppress_host_output
from fervis.project.integration import FlaskAppSource
from fervis.project.source_paths import (
    normalize_source_path_prefixes,
    source_path_matches,
)

from .apispec import enrich_contract_from_flask_apispec
from .loading import import_flask_app
from .transport import FlaskInProcessReadTransport
from .flask_appbuilder import (
    enrich_contract_from_flask_appbuilder_metadata,
)

_FLASK_PARAM = re.compile(r"<(?:(?P<converter>[^:<>]+):)?(?P<name>[^<>]+)>")
_OPENAPI_SPEC_NAMES = (
    "openapi.json",
    "openapi.yaml",
    "openapi.yml",
    "swagger.json",
    "swagger.yaml",
    "swagger.yml",
    "apispec_1.json",
)
_DOCUMENTATION_PATH_SEGMENTS = frozenset(
    {"api-docs", "docs", "documentation", "swagger", "swagger-ui", "swaggerui"}
)
_DOCUMENTATION_ENDPOINT_TOKENS = ("apispec", "openapi", "swagger")


def get_flask_endpoint_contracts(
    *,
    sources: tuple[FlaskAppSource, ...],
    project_root: Path,
) -> tuple[EndpointContract, ...]:
    contracts: list[EndpointContract] = []
    for source in sources:
        prefixes = _normalized_path_prefixes(tuple(source.path_prefixes))
        app = import_flask_app(
            source.app,
            project_root=project_root,
            app_args=tuple(source.app_args),
            app_kwargs=dict(source.app_kwargs),
        )
        contracts.extend(
            _contracts_from_app(
                app,
                source=source,
                path_prefixes=prefixes,
            )
        )
    return tuple(sorted(contracts, key=lambda item: item.path_template))


def _contracts_from_app(
    app: Any,
    *,
    source: FlaskAppSource,
    path_prefixes: tuple[str, ...],
) -> tuple[EndpointContract, ...]:
    route_contracts: list[EndpointContract] = []
    blueprints = {item for item in source.blueprints if item}
    openapi_evidence = _openapi_evidence_by_route(
        app,
        path_prefixes=path_prefixes,
    )
    for rule in app.url_map.iter_rules():
        if _skip_framework_rule(rule):
            continue
        if "GET" not in rule.methods:
            continue
        path = _flask_rule_to_template(rule.rule)
        if not source_path_matches(path, path_prefixes):
            continue
        if _is_control_route(rule, path=path, path_prefixes=path_prefixes):
            continue
        blueprint_name = _blueprint_name(rule.endpoint)
        if blueprints and blueprint_name not in blueprints:
            continue
        route_contracts.append(
            _enriched_route_contract(
                app,
                rule=rule,
                source=source,
                path=path,
                blueprint_name=blueprint_name,
                openapi_evidence=openapi_evidence.get(("GET", path)),
            )
        )
    return tuple(route_contracts)


def _enriched_route_contract(
    app: Any,
    *,
    rule: Any,
    source: FlaskAppSource,
    path: str,
    blueprint_name: str,
    openapi_evidence: Any | None,
) -> EndpointContract:
    handler_ref = _handler_ref(app, rule.endpoint)
    contract = EndpointContract(
        endpoint_name=rule.endpoint,
        url_name=rule.endpoint,
        method="GET",
        path_template=path,
        docstring=_docstring(app, rule.endpoint),
        view_class=handler_ref,
        path_params=_path_params(rule),
        query_params=(),
        response_fields=(),
        response_schema={},
        query_schema_source="route",
        response_schema_source="missing",
        resource_names=_namespace_path(source, blueprint_name),
        catalog_endpoint=CatalogEndpointContract(
            framework_kind=FrameworkKind.FLASK.value,
            source_namespace_kind=SourceNamespaceKind.FLASK_BLUEPRINT.value,
            source_namespace_path=_namespace_path(source, blueprint_name),
            handler_ref=handler_ref,
            route_name=rule.endpoint,
        ),
    )
    return _enrich_contract(
        contract,
        view=app.view_functions.get(rule.endpoint),
        openapi_evidence=openapi_evidence,
    )


def _enrich_contract(
    contract: EndpointContract,
    *,
    view: object | None,
    openapi_evidence: Any | None,
) -> EndpointContract:
    enriched = _openapi_enriched_contract(contract, evidence=openapi_evidence)
    for enrich in (
        enrich_contract_from_flask_apispec,
        enrich_contract_from_jsonapi_resource,
        enrich_contract_from_flask_appbuilder_metadata,
    ):
        enriched = enrich(enriched, view=view)
    return enriched


def _openapi_enriched_contract(
    contract: EndpointContract,
    *,
    evidence: Any | None,
) -> EndpointContract:
    if evidence is None:
        return contract
    catalog_endpoint = contract.catalog_endpoint
    if catalog_endpoint is not None:
        catalog_endpoint = replace(
            catalog_endpoint,
            api_schema_operation_id=evidence.operation_id,
            domain_resource_names=evidence.resource_names,
        )
    return replace(
        contract,
        endpoint_name=evidence.operation_id,
        url_name=evidence.operation_id,
        docstring=evidence.summary or contract.docstring,
        path_params=evidence.path_params or contract.path_params,
        query_params=evidence.query_params,
        response_fields=evidence.response_fields,
        response_schema=evidence.response_schema,
        response_schema_source="openapi",
        response_cardinality=evidence.response_cardinality,
        pagination=evidence.pagination,
        query_schema_source="openapi",
        tags=evidence.tags,
        resource_names=evidence.resource_names or contract.resource_names,
        candidate_keys=evidence.candidate_keys,
        entity_references=evidence.entity_references,
        catalog_endpoint=catalog_endpoint,
    )


def _openapi_evidence_by_route(
    app: Any,
    *,
    path_prefixes: tuple[str, ...],
) -> dict[tuple[str, str], Any]:
    evidence: dict[tuple[str, str], Any] = {}
    for _, schema in _openapi_schemas_from_app(app):
        for item in endpoint_evidence_from_openapi(
            schema,
            path_prefixes=path_prefixes,
        ):
            evidence[(item.method.upper(), item.path_template)] = item
    return evidence


def _openapi_schemas_from_app(app: Any) -> tuple[tuple[str, dict[str, Any]], ...]:
    route_paths = {
        str(rule.rule)
        for rule in app.url_map.iter_rules()
        if "GET" in rule.methods and not _skip_framework_rule(rule)
    }
    schema_paths = sorted(path for path in route_paths if _is_openapi_spec_path(path))
    if not schema_paths:
        return ()
    schemas: list[tuple[str, dict[str, Any]]] = []
    transport = FlaskInProcessReadTransport(app)
    for path in schema_paths:
        with suppress_host_output():
            status, body = transport.get(path, {})
        if status != 200:
            continue
        if isinstance(body, dict) and isinstance(body.get("paths"), dict):
            schemas.append((path, body))
    return tuple(schemas)


def _is_openapi_spec_path(path: str) -> bool:
    normalized = path.strip("/")
    return any(normalized.endswith(name) for name in _OPENAPI_SPEC_NAMES)


def _skip_framework_rule(rule: Any) -> bool:
    if rule.endpoint == "static":
        return True
    return rule.endpoint.startswith("fervis.")


def _is_control_route(
    rule: Any,
    *,
    path: str,
    path_prefixes: tuple[str, ...],
) -> bool:
    if _is_openapi_spec_path(path):
        return True
    endpoint = str(getattr(rule, "endpoint", "") or "").lower()
    if any(token in endpoint for token in _DOCUMENTATION_ENDPOINT_TOKENS):
        if "swagger_ui" in endpoint or "swaggerui" in endpoint:
            return True
        segments = {
            segment.lower()
            for segment in path.strip("/").split("/")
            if "{" not in segment
        }
        return bool(segments & _DOCUMENTATION_PATH_SEGMENTS)
    if _endpoint_leaf(endpoint) in {"doc", "root"} and _is_source_root_path(
        path,
        path_prefixes=path_prefixes,
    ):
        return True
    return False


def _endpoint_leaf(endpoint: str) -> str:
    return endpoint.rsplit(".", 1)[-1]


def _is_source_root_path(path: str, *, path_prefixes: tuple[str, ...]) -> bool:
    normalized_path = path.rstrip("/") + "/"
    return any(normalized_path == prefix for prefix in path_prefixes)


def _path_params(rule: Any) -> tuple[ParameterContract, ...]:
    converters = getattr(rule, "_converters", {})
    return tuple(
        ParameterContract(
            name=name,
            type=_converter_type(converters.get(name)),
            required=True,
            source="path",
        )
        for name in sorted(rule.arguments)
    )


def _converter_type(converter: Any) -> str:
    name = converter.__class__.__name__.lower() if converter is not None else ""
    if name == "integerconverter":
        return "integer"
    if name == "floatconverter":
        return "decimal"
    if name == "uuidconverter":
        return "uuid"
    if name == "pathconverter":
        return "path"
    return "string"


def _flask_rule_to_template(rule: str) -> str:
    return _FLASK_PARAM.sub(lambda match: "{" + match.group("name") + "}", str(rule))


def _docstring(app: Any, endpoint: str) -> str:
    view = app.view_functions.get(endpoint)
    return str(getattr(view, "__doc__", "") or "").strip()


def _handler_ref(app: Any, endpoint: str) -> str:
    view = app.view_functions.get(endpoint)
    module = str(getattr(view, "__module__", "") or "")
    name = str(getattr(view, "__qualname__", "") or getattr(view, "__name__", "") or "")
    return f"{module}:{name}" if module and name else endpoint


def _blueprint_name(endpoint: str) -> str:
    return endpoint.split(".", 1)[0] if "." in endpoint else ""


def _namespace_path(source: FlaskAppSource, blueprint_name: str) -> tuple[str, ...]:
    if blueprint_name:
        return (source.name, blueprint_name)
    return (source.name,)


def _normalized_path_prefixes(path_prefixes: tuple[str, ...]) -> tuple[str, ...]:
    try:
        return normalize_source_path_prefixes(path_prefixes)
    except ValueError as exc:
        raise ValueError(f"FlaskAppSource.path_prefixes invalid: {exc}") from exc
