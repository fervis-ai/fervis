"""Projection from endpoint contracts to Lookup relation catalog."""

from __future__ import annotations

from dataclasses import replace

from fervis.host_api.contracts.response_envelope import TOTAL_COUNT_FIELD
from fervis.host_api.contracts.endpoint import (
    EndpointContract,
    make_catalog_endpoint_key,
)
from fervis.lookup.relation_catalog import (
    CatalogEndpointMetadata,
    CatalogField,
    CatalogParam,
    CompletenessPolicy,
    EndpointRead,
    FieldRequirement,
    IdentityMetadata,
    PaginationMetadata,
    PaginationMode,
    ParamSource,
    RelationCatalog,
    ResponseEnvelopeMetadata,
    RowCardinality,
    RowPath,
    validate_relation_catalog,
)


def relation_catalog_from_endpoint_contracts(
    contracts: tuple[EndpointContract, ...],
) -> RelationCatalog:
    _validate_endpoint_contracts(contracts)
    catalog = RelationCatalog(
        reads=tuple(
            _endpoint_read(item) for item in contracts if item.supports_lookup_read()
        )
    )
    return validate_relation_catalog(_with_propagated_param_identities(catalog))


def _validate_endpoint_contracts(contracts: tuple[EndpointContract, ...]) -> None:
    for contract in contracts:
        for param in (*contract.path_params, *contract.query_params):
            _catalog_param(contract.endpoint_name, param)


def _endpoint_read(contract: EndpointContract) -> EndpointRead:
    row_paths = _row_paths(contract)
    fields = tuple(
        _catalog_field(contract, item, row_paths) for item in contract.response_fields
    )
    return EndpointRead(
        id=contract.endpoint_name,
        endpoint_name=contract.endpoint_name,
        method=contract.method,
        path=contract.path_template,
        resource_names=tuple(str(item) for item in contract.resource_names),
        params=tuple(
            _catalog_param(contract.endpoint_name, item)
            for item in (*contract.path_params, *contract.query_params)
        ),
        row_paths=row_paths,
        fields=fields,
        response_envelope=_response_envelope(contract),
        pagination=_pagination(contract),
        access=_access(contract),
        catalog_endpoint=_catalog_endpoint_metadata(contract),
        source_metadata=_source_metadata(contract),
    )


def _source_metadata(contract: EndpointContract) -> dict[str, object]:
    return {"description": contract.docstring}


def _catalog_endpoint_metadata(
    contract: EndpointContract,
) -> CatalogEndpointMetadata | None:
    if contract.catalog_endpoint is None:
        return None
    return CatalogEndpointMetadata(
        catalog_endpoint_key=make_catalog_endpoint_key(contract),
        endpoint_name=contract.endpoint_name,
        framework_kind=contract.catalog_endpoint.framework_kind,
        source_namespace_kind=contract.catalog_endpoint.source_namespace_kind,
        source_namespace_path=contract.catalog_endpoint.source_namespace_path,
        route_method=contract.method,
        route_path_template=contract.path_template,
        route_name=contract.catalog_endpoint.route_name,
        api_schema_operation_id=contract.catalog_endpoint.api_schema_operation_id,
        handler_ref=contract.catalog_endpoint.handler_ref,
        domain_resource_names=contract.catalog_endpoint.domain_resource_names,
    )


def _catalog_param(endpoint_name: str, param: object) -> CatalogParam:
    raw_source = str(getattr(param, "source", "") or "")
    if not raw_source:
        raise ValueError(f"{endpoint_name}.{param.name} param source is required")
    source = ParamSource(raw_source)
    return CatalogParam(
        ref=f"{endpoint_name}.{source.value}.{param.name}",
        name=str(param.name),
        source=source,
        type=str(param.type),
        description=str(getattr(param, "description", "") or ""),
        required=bool(param.required),
        choices=tuple(str(item) for item in getattr(param, "choices", ()) or ()),
        choice_labels={
            str(key): str(value)
            for key, value in (getattr(param, "choice_labels", {}) or {}).items()
        },
        default=getattr(param, "default", None),
        identity=_param_identity_metadata(param),
        semantics=str(getattr(param, "semantics", "") or ""),
    )


def _param_identity_metadata(param: object) -> IdentityMetadata | None:
    identity = getattr(param, "identity", {}) or {}
    entity_ref = str(identity.get("entityRef") or "")
    id_field = str(identity.get("idField") or "")
    if not entity_ref or not id_field:
        return None
    return IdentityMetadata(
        entity_ref=entity_ref,
        identity_field=id_field,
        primary_key=True,
        stable=True,
    )


def _with_propagated_param_identities(catalog: RelationCatalog) -> RelationCatalog:
    identities_by_field = _primary_identity_by_field(catalog)
    if not identities_by_field:
        return catalog
    reads = []
    for read in catalog.reads:
        params = tuple(
            (
                param
                if param.identity is not None
                else replace(param, identity=identities_by_field.get(param.name))
            )
            for param in read.params
        )
        reads.append(replace(read, params=params))
    return RelationCatalog(reads=tuple(reads), facts=catalog.facts)


def _primary_identity_by_field(catalog: RelationCatalog) -> dict[str, IdentityMetadata]:
    identities: dict[str, IdentityMetadata] = {}
    for read in catalog.reads:
        for field in read.fields:
            identity = field.identity
            if identity is None or not identity.primary_key or not identity.stable:
                continue
            if not field.path:
                continue
            if identity.identity_field:
                identities.setdefault(identity.identity_field, identity)
                continue
            identities.setdefault(field.path.split(".")[-1], identity)
    return identities


def _row_paths(contract: EndpointContract) -> tuple[RowPath, ...]:
    paths = {
        "root": RowPath(
            id="root",
            path="",
            cardinality=(
                RowCardinality.MANY
                if contract.response_cardinality == "many"
                else RowCardinality.ONE
            ),
        )
    }
    if contract.paginated:
        paths["data"] = RowPath(
            id="data",
            path="data",
            cardinality=RowCardinality.MANY,
        )
    for field in contract.response_fields:
        path = _catalog_path(contract, str(field.path or ""))
        if not path:
            continue
        if field.type == "array":
            paths[path] = RowPath(
                id=_row_path_id(path),
                path=path,
                cardinality=RowCardinality.MANY,
                parent_path=_parent_row_path(path),
            )
            continue
        if field.type == "object" and not _field_row_path(path, paths):
            paths[path] = RowPath(
                id=_row_path_id(path),
                path=path,
                cardinality=RowCardinality.ONE,
                parent_path=_parent_row_path(path),
            )
            continue
        parent = _field_row_path(path, paths)
        if parent and parent not in paths:
            paths[parent] = RowPath(
                id=_row_path_id(parent),
                path=parent,
                cardinality=RowCardinality.MANY,
                parent_path=_parent_row_path(parent),
            )
    return tuple(paths[key] for key in sorted(paths))


def _catalog_field(
    contract: EndpointContract,
    field: object,
    row_paths: tuple[RowPath, ...],
) -> CatalogField:
    raw_path = str(field.path or field.name or "")
    path = _catalog_path(contract, raw_path)
    row_path = _field_row_path(path, {item.path: item for item in row_paths})
    return CatalogField(
        ref=f"field.{path}",
        path=path,
        row_path_id=_row_path_id(row_path),
        type=str(field.type),
        nullable=False,
        choices=tuple(str(item) for item in getattr(field, "choices", ()) or ()),
        identity=_identity_metadata(contract, field),
        requirements=_field_requirements(contract, field),
        metadata={
            "name": str(field.name or ""),
            "description": str(getattr(field, "description", "") or ""),
        },
    )


def _field_row_path(path: str, row_paths: dict[str, RowPath]) -> str:
    segments = path.split(".")
    for index in range(len(segments), 0, -1):
        candidate = ".".join(segments[:index])
        if candidate in row_paths and row_paths[candidate].path:
            return candidate
    return ""


def _identity_metadata(
    contract: EndpointContract,
    field: object,
) -> IdentityMetadata | None:
    raw_path = str(field.path or "")
    path = _catalog_path(contract, raw_path)
    identity = getattr(field, "identity", {}) or {}
    entity_ref = str(identity.get("entityRef") or "")
    if identity and entity_ref:
        return IdentityMetadata(
            entity_ref=entity_ref,
            identity_field=str(identity.get("idField") or path),
            primary_key=bool(identity.get("primaryKey")),
            stable=True,
            display_fields=tuple(
                str(item) for item in identity.get("displayFields", ())
            ),
        )
    if path in contract.primary_key_fields or raw_path in contract.primary_key_fields:
        return IdentityMetadata(
            entity_ref=path,
            identity_field=path.split(".")[-1],
            primary_key=True,
            stable=True,
        )
    return None


def _field_requirements(
    contract: EndpointContract,
    field: object,
) -> tuple[FieldRequirement, ...]:
    requires = getattr(field, "requires", {}) or {}
    query_param = str(requires.get("queryParam") or "")
    if not query_param:
        return ()
    return (
        FieldRequirement(
            param_ref=f"{contract.endpoint_name}.{ParamSource.QUERY.value}.{query_param}",
            value=requires.get("value"),
        ),
    )


def _response_envelope(contract: EndpointContract) -> ResponseEnvelopeMetadata:
    return ResponseEnvelopeMetadata(
        results_path="data" if contract.paginated or _has_data_array(contract) else "",
        count_path=TOTAL_COUNT_FIELD if contract.paginated else "",
    )


def _pagination(contract: EndpointContract) -> PaginationMetadata:
    if not contract.paginated:
        return PaginationMetadata(mode=PaginationMode.NONE)
    return PaginationMetadata(
        mode=PaginationMode.PAGE_NUMBER,
        completeness_policy=CompletenessPolicy.ALL_PAGES,
    )


def _access(contract: EndpointContract) -> tuple[str, ...]:
    values: list[str] = []
    if contract.admin_access:
        values.append("admin")
    if contract.staff_access:
        values.append("staff")
    if contract.agent_access:
        values.append("agent")
    if contract.public_access:
        values.append("public")
    return tuple(values)


def _has_data_array(contract: EndpointContract) -> bool:
    return any(
        field.path == "data" and field.type == "array"
        for field in contract.response_fields
    )


def _catalog_path(contract: EndpointContract, path: str) -> str:
    if not contract.paginated or not path:
        return path
    if path == "data" or path.startswith("data."):
        return path
    return f"data.{path}"


def _row_path_id(path: str) -> str:
    return path.replace(".", "_") if path else "root"


def _parent_row_path(path: str) -> str:
    parts = path.split(".")
    if len(parts) <= 1:
        return ""
    return ".".join(parts[:-1])
