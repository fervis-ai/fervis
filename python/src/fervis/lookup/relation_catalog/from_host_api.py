"""Projection from endpoint contracts to Lookup relation catalog."""

from __future__ import annotations

from fervis.host_api.contracts.response_envelope import TOTAL_COUNT_FIELD
from fervis.host_api.contracts.endpoint import (
    CandidateKeyAuthorityContract,
    CandidateKeyContract,
    EndpointContract,
    EntityReferenceContract,
    ParameterContract,
    ResponseFieldContract,
    make_catalog_endpoint_key,
)
from fervis.lookup.relation_catalog import (
    CandidateKeyAuthority,
    CandidateKeyAuthorityComponent,
    CandidateKey,
    CandidateKeyComponent,
    CatalogEndpointMetadata,
    CatalogField,
    CatalogParam,
    CatalogValidationError,
    CompletenessPolicy,
    EndpointRead,
    EntityKeyComponentTarget,
    EntityReference,
    EntityReferenceComponent,
    FieldRequirement,
    PaginationMetadata,
    PaginationMode,
    ParamSource,
    RelationCatalog,
    ResponseEnvelopeMetadata,
    RowCardinality,
    RowPath,
    parse_relation_catalog,
)


def relation_catalog_from_endpoint_contracts(
    contracts: tuple[EndpointContract, ...],
) -> RelationCatalog:
    _validate_endpoint_contracts(contracts)
    catalog = RelationCatalog(
        reads=tuple(
            _endpoint_read(item) for item in contracts if item.supports_lookup_read()
        ),
        candidate_key_authorities=_candidate_key_authorities(contracts),
    )
    return parse_relation_catalog(catalog)


def _validate_endpoint_contracts(contracts: tuple[EndpointContract, ...]) -> None:
    for contract in contracts:
        for param in (*contract.path_params, *contract.query_params):
            _catalog_param(contract.endpoint_name, param)


def _endpoint_read(contract: EndpointContract) -> EndpointRead:
    row_paths = _row_paths(contract)
    fields = tuple(
        field
        for item in contract.response_fields
        for field in (_catalog_field(contract, item, row_paths),)
        if field is not None
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
        candidate_keys=_candidate_keys(contract, fields=fields),
        entity_references=_entity_references(contract, fields=fields),
        response_envelope=_response_envelope(contract),
        pagination=_pagination(contract),
        access=_access(contract),
        catalog_endpoint=_catalog_endpoint_metadata(contract),
        source_metadata=_source_metadata(contract),
    )


def _candidate_keys(
    contract: EndpointContract,
    *,
    fields: tuple[CatalogField, ...],
) -> tuple[CandidateKey, ...]:
    fields_by_path = {field.path: field.ref for field in fields}
    keys = tuple(
        _candidate_key(
            key,
            contract=contract,
            fields_by_path=fields_by_path,
        )
        for key in contract.candidate_keys
    )
    return keys


def _candidate_key_authorities(
    contracts: tuple[EndpointContract, ...],
) -> tuple[CandidateKeyAuthority, ...]:
    authorities = tuple(
        _candidate_key_authority(authority)
        for contract in contracts
        for authority in contract.candidate_key_authorities
    )
    return tuple(dict.fromkeys(authorities))


def _candidate_key_authority(
    authority: CandidateKeyAuthorityContract,
) -> CandidateKeyAuthority:
    components = tuple(
        CandidateKeyAuthorityComponent(
            id=component.component_id,
            type=component.type,
        )
        for component in authority.components
    )
    return CandidateKeyAuthority(
        id=authority.key_id,
        entity_kind=authority.entity_kind,
        components=components,
        primary=authority.primary,
        stable=authority.stable,
    )


def _candidate_key(
    key: CandidateKeyContract,
    *,
    contract: EndpointContract,
    fields_by_path: dict[str, str],
) -> CandidateKey:
    components = tuple(
        CandidateKeyComponent(
            id=component.component_id,
            field_ref=fields_by_path[
                _required_catalog_path(contract, component.field_path)
            ],
        )
        for component in key.components
    )
    context_field_refs = tuple(
        fields_by_path[_required_catalog_path(contract, path)]
        for path in key.context_field_paths
    )
    return CandidateKey(
        id=key.key_id,
        entity_kind=key.entity_kind,
        components=components,
        primary=key.primary,
        stable=key.stable,
        context_field_refs=context_field_refs,
    )


def _entity_references(
    contract: EndpointContract,
    *,
    fields: tuple[CatalogField, ...],
) -> tuple[EntityReference, ...]:
    fields_by_path = {field.path: field.ref for field in fields}
    references = tuple(
        _entity_reference(
            reference,
            contract=contract,
            fields_by_path=fields_by_path,
        )
        for reference in contract.entity_references
    )
    return references


def _entity_reference(
    reference: EntityReferenceContract,
    *,
    contract: EndpointContract,
    fields_by_path: dict[str, str],
) -> EntityReference:
    components = tuple(
        EntityReferenceComponent(
            target_component_id=component.target_component_id,
            local_field_ref=fields_by_path[
                _required_catalog_path(contract, component.local_field_path)
            ],
        )
        for component in reference.components
    )
    context_field_refs = tuple(
        fields_by_path[_required_catalog_path(contract, path)]
        for path in reference.context_field_paths
    )
    return EntityReference(
        id=reference.reference_id,
        target_entity_kind=reference.target_entity_kind,
        target_key_id=reference.target_key_id,
        components=components,
        context_field_refs=context_field_refs,
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


def _catalog_param(endpoint_name: str, param: ParameterContract) -> CatalogParam:
    raw_source = param.source
    if not raw_source:
        raise CatalogValidationError(
            f"{endpoint_name}.{param.name} param source is required"
        )
    source = ParamSource(raw_source)
    return CatalogParam(
        ref=f"{endpoint_name}.{source.value}.{param.name}",
        name=str(param.name),
        source=source,
        type=str(param.type),
        description=param.description,
        required=bool(param.required),
        choices=param.choices,
        choice_labels={
            str(key): str(value) for key, value in param.choice_labels.items()
        },
        default=param.default,
        entity_target=_param_entity_target(param),
        semantics=param.semantics,
    )


def _param_entity_target(
    param: ParameterContract,
) -> EntityKeyComponentTarget | None:
    target = param.entity_target
    if target is None:
        return None
    return EntityKeyComponentTarget(
        entity_kind=str(target.entity_kind),
        key_id=str(target.key_id),
        component_id=str(target.component_id),
    )


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
    if contract.pagination is not None:
        paths["data"] = RowPath(
            id="data",
            path="data",
            cardinality=RowCardinality.MANY,
        )
    for field in contract.response_fields:
        path = _catalog_path(contract, str(field.path or ""))
        if not path:
            continue
        if field.type == "array" and _field_has_descendants(contract, path):
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


def _field_has_descendants(contract: EndpointContract, field_path: str) -> bool:
    descendant_prefix = f"{field_path}."
    return any(
        _catalog_path(contract, str(field.path or "")).startswith(descendant_prefix)
        for field in contract.response_fields
    )


def _catalog_field(
    contract: EndpointContract,
    field: ResponseFieldContract,
    row_paths: tuple[RowPath, ...],
) -> CatalogField | None:
    raw_path = str(field.path or field.name or "")
    path = _catalog_path(contract, raw_path)
    if not path:
        return None
    row_path = _field_row_path(path, {item.path: item for item in row_paths})
    return CatalogField(
        ref=f"field.{path}",
        path=path,
        row_path_id=_row_path_id(row_path),
        type=str(field.type),
        nullable=False,
        choices=tuple(str(item) for item in getattr(field, "choices", ()) or ()),
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


def _field_requirements(
    contract: EndpointContract,
    field: ResponseFieldContract,
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
        results_path=(
            "data"
            if contract.pagination is not None or _has_data_array(contract)
            else ""
        ),
        count_path=(
            f"pagination.{TOTAL_COUNT_FIELD}"
            if contract.pagination is not None
            else ""
        ),
    )


def _pagination(contract: EndpointContract) -> PaginationMetadata:
    if contract.pagination is None:
        return PaginationMetadata(mode=PaginationMode.NONE)
    mode = (
        PaginationMode.LIMIT_OFFSET
        if contract.pagination.kind.value == "offset"
        else PaginationMode.PAGE_NUMBER
    )
    return PaginationMetadata(
        mode=mode,
        default_page_size=contract.pagination.page_size,
        max_page_size=contract.pagination.max_page_size,
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
    if contract.pagination is None or not path:
        return path
    results_path = contract.pagination.results_path
    if path == results_path:
        return "data"
    results_prefix = f"{results_path}."
    if path.startswith(results_prefix):
        return f"data.{path.removeprefix(results_prefix)}"
    if contract.pagination.total_path and path == contract.pagination.total_path:
        return f"pagination.{TOTAL_COUNT_FIELD}"
    if not _declares_results_envelope(contract):
        return f"data.{path}"
    return ""


def _declares_results_envelope(contract: EndpointContract) -> bool:
    results_path = contract.pagination.results_path if contract.pagination else ""
    return any(
        str(field.path or field.name) == results_path and field.type == "array"
        for field in contract.response_fields
    )


def _required_catalog_path(contract: EndpointContract, path: str) -> str:
    canonical_path = _catalog_path(contract, path)
    if not canonical_path:
        raise ValueError(
            "relation identity field is unavailable after response normalization"
        )
    return canonical_path


def _row_path_id(path: str) -> str:
    return path.replace(".", "_") if path else "root"


def _parent_row_path(path: str) -> str:
    parts = path.split(".")
    if len(parts) <= 1:
        return ""
    return ".".join(parts[:-1])
