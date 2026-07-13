"""Cached public GET endpoint contract registry."""

from __future__ import annotations

import inspect
import re
from dataclasses import replace
from functools import lru_cache
from importlib.util import find_spec
from types import SimpleNamespace
from typing import Any

from django.urls import URLPattern, URLResolver, get_resolver
from rest_framework.filters import OrderingFilter
from rest_framework.permissions import AllowAny

from fervis.host_api.contracts import (
    CatalogEndpointContract,
    CandidateKeyContract,
    EndpointContract,
    FrameworkKind,
    ParameterContract,
    ResponseFieldContract,
    SourceNamespaceKind,
)
from fervis.host_api.contracts.capabilities import (
    EndpointCapabilities,
    capabilities_from_schema,
)
from fervis.project.source_paths import (
    normalize_source_path_prefixes,
    source_path_matches,
)
from fervis.project.source_scope import DjangoSourceScope

from .schema_introspection import (
    conditional_response_roots_from_serializer,
    inspect_response_serializer,
    optional_full_response_projection_param_names,
    path_param_candidate_key_authority,
    path_param_entity_target,
    query_params_from_serializer,
)
from .pagination import pagination_contract


def clear_endpoint_contract_cache() -> None:
    _cached_endpoint_contracts.cache_clear()


def _introspected_framework_kind() -> FrameworkKind:
    if find_spec("django") is None:
        raise RuntimeError("Django endpoint catalog adapter requires django")
    if find_spec("rest_framework") is None:
        raise RuntimeError("Django endpoint catalog adapter requires DRF")
    return FrameworkKind.DJANGO_DRF


def get_endpoint_contracts(
    *,
    sources: tuple[DjangoSourceScope, ...],
) -> tuple[EndpointContract, ...]:
    return _cached_endpoint_contracts(_normalized_sources(sources))


@lru_cache(maxsize=32)
def _cached_endpoint_contracts(
    sources: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...],
) -> tuple[EndpointContract, ...]:
    contracts: list[EndpointContract] = []
    for _source_name, app_modules, path_prefixes in sources:
        _walk_patterns(
            get_resolver().url_patterns,
            "",
            contracts,
            app_modules=app_modules,
            path_prefixes=path_prefixes,
        )
    return tuple(sorted(contracts, key=lambda item: item.path_template))


def get_endpoint_contract(
    endpoint_name: str,
    *,
    sources: tuple[DjangoSourceScope, ...],
) -> EndpointContract | None:
    return next(
        (
            contract
            for contract in get_endpoint_contracts(
                sources=sources,
            )
            if contract.endpoint_name == endpoint_name
        ),
        None,
    )


def _walk_patterns(
    patterns: list[Any],
    prefix: str,
    contracts: list[EndpointContract],
    *,
    app_modules: tuple[str, ...],
    path_prefixes: tuple[str, ...],
    converters: dict[str, object] | None = None,
    source_namespace_path: tuple[str, ...] = (),
) -> None:
    inherited_converters = dict(converters or {})
    for pattern in patterns:
        fragment = _pattern_fragment(pattern.pattern)
        if fragment is None:
            continue
        pattern_converters = {
            **inherited_converters,
            **dict(getattr(pattern.pattern, "converters", {}) or {}),
        }
        if isinstance(pattern, URLResolver):
            _walk_patterns(
                pattern.url_patterns,
                prefix + fragment,
                contracts,
                app_modules=app_modules,
                path_prefixes=path_prefixes,
                converters=pattern_converters,
                source_namespace_path=(
                    source_namespace_path + _resolver_namespace_path(pattern)
                ),
            )
            continue
        if not isinstance(pattern, URLPattern) or not pattern.name:
            continue
        path = _normalize_path(prefix + fragment)
        if not _path_matches_prefixes(path, path_prefixes):
            continue

        view_class = _view_class(pattern)
        if view_class is None:
            continue
        if not _module_is_allowed(view_class.__module__, app_modules=app_modules):
            continue
        if not _supports_get(pattern):
            continue

        contracts.append(
            _build_contract(
                path=path,
                url_name=pattern.name,
                view_class=view_class,
                converters=pattern_converters,
                source_namespace_path=source_namespace_path,
            )
        )


def _build_contract(
    *,
    path: str,
    url_name: str,
    view_class: type,
    converters: dict[str, object],
    source_namespace_path: tuple[str, ...] = (),
) -> EndpointContract:
    query_serializer_class = getattr(view_class, "query_serializer_class", None)
    response_serializer_class = _get_response_serializer_class(view_class)
    response_model = _view_queryset_model(view_class)
    query_params = query_params_from_serializer(query_serializer_class)
    resource_names = _resource_names_for_endpoint(
        path=path,
        url_name=url_name,
        view_class=view_class,
        query_serializer_class=query_serializer_class,
        response_serializer_class=response_serializer_class,
    )
    response_inspection = inspect_response_serializer(
        response_serializer_class,
        model_context=response_model,
    )
    response_model = response_model or response_inspection.relation_model
    response_fields = response_inspection.response_fields
    candidate_keys = response_inspection.candidate_keys
    optional_projection_params = optional_full_response_projection_param_names(
        response_serializer_class,
        query_params=query_params,
    )
    query_params = tuple(
        param for param in query_params if param.name not in optional_projection_params
    )
    query_params = _with_framework_param_semantics(
        query_params,
        response_fields=response_fields,
        view_class=view_class,
    )
    response_schema = response_inspection.response_schema
    conditional_roots = conditional_response_roots_from_serializer(
        response_serializer_class
    )
    unreachable_roots = _unreachable_conditional_response_roots(
        conditional_roots=conditional_roots,
        query_params=query_params,
    )
    if unreachable_roots:
        response_fields = _without_response_roots(
            response_fields,
            roots=unreachable_roots,
        )
        response_schema = _without_schema_roots(
            response_schema,
            roots=unreachable_roots,
        )
    response_fields = _with_conditional_requirements(
        response_fields,
        conditional_roots=conditional_roots,
        query_params=query_params,
    )
    path_param_names = tuple(_path_param_names(path))
    path_params = tuple(
        ParameterContract(
            name=name,
            type=_path_param_type(converters.get(name)),
            required=True,
            description=f"Path parameter {name}",
            source="path",
            entity_target=path_param_entity_target(
                response_model,
                param_name=name,
            ),
        )
        for name in path_param_names
    )
    path_param_authorities = tuple(
        authority
        for name in path_param_names
        for authority in (
            path_param_candidate_key_authority(response_model, param_name=name),
        )
        if authority is not None
    )
    candidate_key_authorities = tuple(
        dict.fromkeys(
            (*response_inspection.candidate_key_authorities, *path_param_authorities)
        )
    )

    permission_classes = tuple(getattr(view_class, "permission_classes", ()) or ())
    public_access = any(item is AllowAny for item in permission_classes)
    pagination = pagination_contract(view_class)

    endpoint_name = _endpoint_name(path, url_name)
    capabilities = _capabilities_for_endpoint(
        endpoint_name=endpoint_name,
        view_class=view_class,
        query_serializer_class=query_serializer_class,
        response_serializer_class=response_serializer_class,
        path_params=path_params,
        query_params=query_params,
        response_fields=response_fields,
        candidate_keys=candidate_keys,
    )

    return EndpointContract(
        endpoint_name=endpoint_name,
        url_name=url_name,
        method="GET",
        path_template=_path_template(path),
        docstring=_docstring(view_class),
        view_class=f"{view_class.__module__}.{view_class.__name__}",
        path_params=path_params,
        query_params=query_params,
        response_fields=response_fields,
        response_schema=response_schema,
        capabilities=capabilities,
        capability_sources=_capability_sources(capabilities),
        agent_access=bool(getattr(view_class, "agent_access", False)),
        staff_access=bool(getattr(view_class, "staff_access", False)),
        admin_access=True,
        public_access=public_access,
        pagination=pagination,
        query_schema_source=(
            f"{query_serializer_class.__module__}.{query_serializer_class.__name__}"
            if query_serializer_class is not None
            else "missing"
        ),
        response_schema_source=(
            f"{response_serializer_class.__module__}.{response_serializer_class.__name__}"
            if response_serializer_class is not None
            else "missing"
        ),
        tags=_tags_for(path=path, view_class=view_class),
        resource_names=resource_names,
        candidate_keys=candidate_keys,
        candidate_key_authorities=candidate_key_authorities,
        entity_references=response_inspection.entity_references,
        catalog_endpoint=_catalog_endpoint_contract(
            url_name=url_name,
            view_class=view_class,
            source_namespace_path=source_namespace_path,
            resource_names=resource_names,
        ),
    )


def _get_response_serializer_class(view_class: type) -> type | None:
    if "get_serializer_class" not in getattr(view_class, "__dict__", {}):
        return getattr(view_class, "serializer_class", None)

    view = view_class()
    view.request = SimpleNamespace(method="GET")
    view.action = "list"
    view.kwargs = {}
    return view.get_serializer_class()


def _catalog_endpoint_contract(
    *,
    url_name: str,
    view_class: type,
    source_namespace_path: tuple[str, ...],
    resource_names: tuple[str, ...],
) -> CatalogEndpointContract:
    namespace_kind, namespace_path = _source_namespace(
        view_class=view_class,
        source_namespace_path=source_namespace_path,
    )
    handler_ref = f"{view_class.__module__}.{view_class.__name__}"
    return CatalogEndpointContract(
        framework_kind=_introspected_framework_kind().value,
        source_namespace_kind=namespace_kind.value,
        source_namespace_path=namespace_path,
        handler_ref=handler_ref,
        route_name=url_name,
        domain_resource_names=tuple(resource_names),
    )


def _source_namespace(
    *,
    view_class: type,
    source_namespace_path: tuple[str, ...],
) -> tuple[SourceNamespaceKind, tuple[str, ...]]:
    if source_namespace_path:
        return SourceNamespaceKind.DJANGO_APP, source_namespace_path
    module_app = _django_app_from_module(view_class.__module__)
    if module_app:
        return SourceNamespaceKind.DJANGO_APP, (module_app,)
    return SourceNamespaceKind.PYTHON_MODULE, tuple(
        part for part in view_class.__module__.split(".") if part
    )


def _resolver_namespace_path(pattern: URLResolver) -> tuple[str, ...]:
    values = (getattr(pattern, "app_name", ""), getattr(pattern, "namespace", ""))
    output: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in output:
            output.append(text)
    return tuple(output)


def _django_app_from_module(module: str) -> str:
    parts = tuple(part for part in module.split(".") if part)
    if len(parts) >= 2 and parts[0] == "apps":
        return parts[1]
    return ""


def _with_framework_param_semantics(
    query_params: tuple[ParameterContract, ...],
    *,
    response_fields: tuple[ResponseFieldContract, ...],
    view_class: type,
) -> tuple[ParameterContract, ...]:
    response_shape_param_names = _response_shape_param_names_from_framework(
        query_params,
        response_fields=response_fields,
        view_class=view_class,
    )
    if not response_shape_param_names:
        return query_params
    return tuple(
        (
            replace(param, semantics="response_shape")
            if param.name in response_shape_param_names and not param.semantics
            else param
        )
        for param in query_params
    )


def _response_shape_param_names_from_framework(
    query_params: tuple[ParameterContract, ...],
    *,
    response_fields: tuple[ResponseFieldContract, ...],
    view_class: type,
) -> frozenset[str]:
    names = set(_ordering_filter_param_names(view_class))
    orderable_field_keys = _orderable_field_keys(response_fields)
    names.update(
        param.name
        for param in query_params
        if _choice_values_are_ordering_terms(param, orderable_field_keys)
    )
    return frozenset(names)


def _ordering_filter_param_names(view_class: type) -> tuple[str, ...]:
    names: list[str] = []
    for backend_class in tuple(getattr(view_class, "filter_backends", ()) or ()):
        if not isinstance(backend_class, type):
            continue
        if not _is_ordering_filter_backend(backend_class):
            continue
        ordering_param = str(getattr(backend_class, "ordering_param", "") or "")
        if not ordering_param:
            try:
                ordering_param = str(
                    getattr(backend_class(), "ordering_param", "") or ""
                )
            except Exception:
                ordering_param = ""
        if ordering_param:
            names.append(ordering_param)
    return tuple(dict.fromkeys(names))


def _is_ordering_filter_backend(backend_class: type) -> bool:
    try:
        return issubclass(backend_class, OrderingFilter)
    except TypeError:
        return False


def _orderable_field_keys(
    response_fields: tuple[ResponseFieldContract, ...],
) -> frozenset[str]:
    keys: set[str] = set()
    for field in response_fields:
        name = str(field.name or "").strip()
        path = str(field.path or "").strip()
        if name:
            keys.add(name)
        if path:
            keys.add(path)
            keys.add(path.rsplit(".", 1)[-1])
    return frozenset(keys)


def _choice_values_are_ordering_terms(
    param: ParameterContract,
    orderable_field_keys: frozenset[str],
) -> bool:
    if param.type != "choice" or not param.choices or not orderable_field_keys:
        return False
    normalized = tuple(str(choice).strip() for choice in param.choices)
    if not normalized or not any(choice.startswith("-") for choice in normalized):
        return False
    return all(
        bool(choice) and choice.lstrip("-") in orderable_field_keys
        for choice in normalized
    )


def _normalized_sources(
    sources: tuple[DjangoSourceScope, ...],
) -> tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...]:
    normalized = tuple(
        (
            str(source.name).strip(),
            _normalized_app_modules(source.app_modules),
            _normalized_path_prefixes(source.path_prefixes),
        )
        for source in sources
    )
    if not normalized:
        raise ValueError("Django Fervis sources must not be empty.")
    return normalized


def _normalized_app_modules(app_modules: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(
        dict.fromkeys(
            str(item).strip().rstrip(".") for item in app_modules if str(item).strip()
        )
    )
    if not normalized:
        raise ValueError("DjangoAppSource.app_modules must list eligible API apps.")
    return normalized


def _normalized_path_prefixes(path_prefixes: tuple[str, ...]) -> tuple[str, ...]:
    try:
        return normalize_source_path_prefixes(path_prefixes)
    except ValueError as exc:
        raise ValueError(f"DjangoAppSource.path_prefixes invalid: {exc}") from exc


def _path_matches_prefixes(path: str, path_prefixes: tuple[str, ...]) -> bool:
    return source_path_matches(path, path_prefixes)


def _module_is_allowed(module: str, *, app_modules: tuple[str, ...]) -> bool:
    if module.startswith("fervis."):
        return False
    return any(module == item or module.startswith(f"{item}.") for item in app_modules)


def _capabilities_for_endpoint(
    *,
    endpoint_name: str,
    view_class: type,
    query_serializer_class: type | None,
    response_serializer_class: type | None,
    path_params: tuple[ParameterContract, ...],
    query_params: tuple[ParameterContract, ...],
    response_fields: tuple[ResponseFieldContract, ...],
    candidate_keys: tuple[CandidateKeyContract, ...],
) -> EndpointCapabilities:
    return capabilities_from_schema(
        path_params=path_params,
        query_params=query_params,
        response_fields=response_fields,
        candidate_keys=candidate_keys,
    )


def _unreachable_conditional_response_roots(
    *,
    conditional_roots: tuple[str, ...],
    query_params: tuple[ParameterContract, ...],
) -> set[str]:
    query_param_names = {param.name for param in query_params}
    return {
        root for root in conditional_roots if f"include_{root}" not in query_param_names
    }


def _without_response_roots(
    response_fields: tuple[Any, ...],
    *,
    roots: set[str],
) -> tuple[Any, ...]:
    return tuple(
        field
        for field in response_fields
        if str(getattr(field, "path", "") or "").split(".", 1)[0] not in roots
    )


def _without_schema_roots(
    response_schema: dict[str, Any],
    *,
    roots: set[str],
) -> dict[str, Any]:
    return {
        key: value
        for key, value in dict(response_schema or {}).items()
        if str(key) not in roots
    }


def _with_conditional_requirements(
    response_fields: tuple[ResponseFieldContract, ...],
    *,
    conditional_roots: tuple[str, ...],
    query_params: tuple[ParameterContract, ...],
) -> tuple[ResponseFieldContract, ...]:
    query_param_names = {param.name for param in query_params}
    conditional_query_params = {
        root: f"include_{root}"
        for root in conditional_roots
        if f"include_{root}" in query_param_names
    }
    if not conditional_query_params:
        return response_fields
    updated: list[ResponseFieldContract] = []
    for field in response_fields:
        root = str(field.path).split(".", 1)[0]
        query_param = conditional_query_params.get(root)
        if not query_param:
            updated.append(field)
            continue
        updated.append(
            ResponseFieldContract(
                name=field.name,
                type=field.type,
                path=field.path,
                description=field.description,
                choices=field.choices,
                requires={"queryParam": query_param, "value": True},
            )
        )
    return tuple(updated)


def _capability_sources(capabilities: EndpointCapabilities) -> tuple[str, ...]:
    return tuple(sorted({item.source.value for item in capabilities.items}))


def _resource_names_for_endpoint(
    *,
    path: str,
    url_name: str,
    view_class: type,
    query_serializer_class: type | None,
    response_serializer_class: type | None,
) -> tuple[str, ...]:
    names: set[str] = set()
    names.update(
        _resource_names_from_framework_metadata(
            path=path,
            url_name=url_name,
            view_class=view_class,
            query_serializer_class=query_serializer_class,
            response_serializer_class=response_serializer_class,
        )
    )
    names.update(_declared_resource_names(view_class))
    names.update(_declared_resource_names(query_serializer_class))
    names.update(_declared_resource_names(response_serializer_class))
    for model in (
        _view_queryset_model(view_class),
        _serializer_model(query_serializer_class),
        _serializer_model(response_serializer_class),
    ):
        if model is None:
            continue
        names.update(_resource_names_for_model(model))
    return tuple(sorted(names))


_ENDPOINT_ACTION_WORDS = frozenset(
    {
        "create",
        "delete",
        "detail",
        "get",
        "list",
        "patch",
        "post",
        "put",
        "retrieve",
        "update",
        "view",
    }
)
_SERIALIZER_ROLE_WORDS = frozenset(
    {
        "crud",
        "input",
        "output",
        "query",
        "read",
        "request",
        "response",
        "serializer",
        "write",
    }
)


def _resource_names_from_framework_metadata(
    *,
    path: str,
    url_name: str,
    view_class: type,
    query_serializer_class: type | None,
    response_serializer_class: type | None,
) -> set[str]:
    names: set[str] = set()
    for words in (
        _url_name_resource_words(url_name),
        _path_resource_words(path),
        _class_resource_words(view_class.__name__),
        _class_resource_words(getattr(query_serializer_class, "__name__", "")),
        _class_resource_words(getattr(response_serializer_class, "__name__", "")),
    ):
        value = " ".join(words).strip()
        if value:
            names.add(value)
    return names


def _url_name_resource_words(url_name: str) -> tuple[str, ...]:
    return _strip_action_suffix(_split_identifier_words(url_name))


def _path_resource_words(path: str) -> tuple[str, ...]:
    segments = [
        segment
        for segment in str(path or "").strip("/").split("/")
        if segment and "<" not in segment and "{" not in segment
    ]
    if not segments:
        return ()
    return _split_identifier_words(segments[-1])


def _class_resource_words(class_name: str) -> tuple[str, ...]:
    words = _split_identifier_words(class_name)
    while words and words[-1] in _SERIALIZER_ROLE_WORDS | _ENDPOINT_ACTION_WORDS:
        words = words[:-1]
    return words


def _strip_action_suffix(words: tuple[str, ...]) -> tuple[str, ...]:
    while len(words) > 1 and words[-1] in _ENDPOINT_ACTION_WORDS:
        words = words[:-1]
    return words


def _split_identifier_words(value: str) -> tuple[str, ...]:
    raw = _camel_words(str(value or "").replace("-", "_")).split()
    return tuple(word for word in raw if word)


def _declared_resource_names(owner: type | None) -> set[str]:
    return {
        str(item).strip()
        for item in getattr(owner, "fervis_resource_names", ()) or ()
        if str(item).strip()
    }


def _view_queryset_model(view_class: type) -> type | None:
    queryset = getattr(view_class, "queryset", None)
    model = getattr(queryset, "model", None)
    if model is not None:
        return model
    get_queryset = getattr(view_class, "get_queryset", None)
    if get_queryset is None:
        return None
    view = view_class()
    view.request = SimpleNamespace(query_params={})
    view.kwargs = {}
    try:
        queryset = view.get_queryset()
    except Exception:
        return None
    return getattr(queryset, "model", None)


def _serializer_model(serializer_class: type | None) -> type | None:
    meta = getattr(serializer_class, "Meta", None)
    return getattr(meta, "model", None)


def _resource_names_for_model(model: type) -> set[str]:
    meta = getattr(model, "_meta", None)
    if meta is None:
        return set()
    value = (
        _camel_words(getattr(meta, "object_name", "") or getattr(model, "__name__", ""))
        or str(getattr(meta, "model_name", "") or "").replace("_", " ").strip().lower()
    )
    return {value} if value else set()


def _camel_words(value: str) -> str:
    parts = re.findall(r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|\d+", str(value or ""))
    return " ".join(part.lower() for part in parts if part)


def _view_class(pattern: URLPattern) -> type | None:
    callback = pattern.callback
    return getattr(callback, "cls", None) or getattr(callback, "view_class", None)


def _supports_get(pattern: URLPattern) -> bool:
    actions = getattr(pattern.callback, "actions", None)
    if isinstance(actions, dict):
        return "get" in {str(method).lower() for method in actions}
    view_class = _view_class(pattern)
    return view_class is not None and callable(getattr(view_class, "get", None))


def _normalize_path(path: str) -> str:
    path = re.sub(r"<\w+:(\w+)>", r"<\1>", path)
    return path if path.endswith("/") else f"{path}/"


def _pattern_fragment(pattern: object) -> str | None:
    route = getattr(pattern, "_route", None)
    if route is not None:
        converters = dict(getattr(pattern, "converters", {}) or {})
        if "format" in converters:
            return None
        return str(pattern)
    regex = str(getattr(getattr(pattern, "regex", None), "pattern", "") or "")
    if not regex or "(?P<format>" in regex:
        return None
    return _regex_fragment_to_route(regex)


def _regex_fragment_to_route(regex: str) -> str:
    route = regex
    route = route.removeprefix("^")
    route = re.sub(r"\\Z$", "", route)
    route = route.removesuffix("$")
    route = re.sub(r"\(\?P<(\w+)>[^)]+\)", r"<\1>", route)
    route = route.replace(r"\-", "-")
    route = route.replace(r"\/", "/")
    return route


def _path_template(path: str) -> str:
    template = re.sub(r"<(\w+)>", lambda match: "{" + match.group(1) + "}", path)
    return "/" + template


def _path_param_names(path: str) -> list[str]:
    return re.findall(r"<(\w+)>", path)


def _path_param_type(converter: object | None) -> str:
    converter_name = converter.__class__.__name__ if converter is not None else ""
    if converter_name == "UUIDConverter":
        return "uuid"
    if converter_name == "IntConverter":
        return "integer"
    return "string"


def _endpoint_name(path: str, url_name: str) -> str:
    name = url_name.replace("-", "_")
    if "<" in path:
        return name if name.startswith("get_") else f"get_{name}"
    return name if name.startswith(("list_", "get_")) else f"list_{name}"


def _docstring(view_class: type) -> str:
    doc = inspect.getdoc(view_class) or ""
    return " ".join(doc.split())


def _tags_for(*, path: str, view_class: type) -> tuple[str, ...]:
    raw = f"{path} {view_class.__name__} {view_class.__module__}".replace("_", " ")
    return tuple(
        sorted({part.lower() for part in re.split(r"[^a-zA-Z0-9]+", raw) if part})
    )
