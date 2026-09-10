"""Build row-source projections from API catalog and memory relations."""

from __future__ import annotations

from fervis.host_api.contracts import ParameterSemantics

from itertools import product
from dataclasses import replace
from collections.abc import Mapping
from typing import TYPE_CHECKING

from fervis.lookup.relation_catalog.model import (
    CandidateKey,
    CatalogFactAvailability,
    CatalogField,
    CatalogParam,
    EndpointRead,
    EntityReference,
    RelationCatalog,
    RowCardinality,
    RowPath,
)
from fervis.lookup.relation_catalog.parameter_values import CatalogParameterValue
from fervis.lookup.answer_program.relations import (
    FieldBindingRole,
    Relation,
)

from .field_paths import (
    _allowed_roles,
    _row_context_path_ids,
    _field_fact_refs,
    _field_ids,
    _field_label,
    _field_row_path,
    _field_row_path_id,
    _opaque_id,
    _param_ids,
    _read_catalog_facts,
    _relative_field_path,
    _symbol,
)
from .model import (
    CALENDAR_DATE_FIELD_ID,
    CALENDAR_END_PARAM_ID,
    CALENDAR_END_PARAM_REF,
    CALENDAR_ROW_SOURCE_ID,
    CALENDAR_START_PARAM_ID,
    CALENDAR_START_PARAM_REF,
    _MISSING,
    RowSource,
    RowSourceCandidateKey,
    RowSourceEntityReference,
    RowSourceEntityReferenceComponent,
    RowSourceKeyComponent,
    RowSourceBlockedFact,
    RowSourceCatalog,
    RowSourceField,
    RowSourceKind,
    RowSourceParam,
    RowSourceValueType,
    row_source_value_type,
)

if TYPE_CHECKING:
    from fervis.lookup.plan_execution.relations import RelationRows


def build_row_source_catalog(
    catalog: RelationCatalog,
    *,
    memory_relations: tuple["RelationRows", ...] = (),
) -> RowSourceCatalog:
    return RowSourceCatalog(
        sources=(
            _generated_calendar_row_source(),
            *build_api_row_source_catalog(catalog).sources,
            *tuple(_memory_row_source(relation) for relation in memory_relations),
        )
    )


def build_api_row_source_catalog(catalog: RelationCatalog) -> RowSourceCatalog:
    """Project only the API reads in one catalog-selection batch."""

    return RowSourceCatalog(
        sources=tuple(
            _with_declared_entity_kind(source)
            for read in catalog.reads
            for source in _api_row_sources(read, catalog=catalog)
        )
    )


def _with_declared_entity_kind(source: RowSource) -> RowSource:
    kinds = {key.entity_kind for key in source.candidate_keys}
    if len(kinds) != 1:
        return source
    (kind,) = kinds
    field_id = "metadata:entity_kind"
    if any(field.id == field_id for field in source.fields):
        raise ValueError("source field collides with declared entity metadata")
    return replace(
        source,
        fields=(
            *source.fields,
            RowSourceField(
                id=field_id,
                field_ref=f"source_metadata:{source.id}:entity_kind",
                label="Declared entity class of this row",
                type=RowSourceValueType.STRING,
                allowed_roles=(FieldBindingRole.OUTPUT, FieldBindingRole.PREDICATE),
                choices=(kind,),
                declared_entity_kind=kind,
                description="Fixed row-class value supplied by the source's declared candidate-key entity kind.",
            ),
        ),
    )


def row_source_ids_for_read_ids(
    read_ids: tuple[str, ...],
    *,
    row_sources: RowSourceCatalog,
) -> tuple[str, ...]:
    selected = set(read_ids)
    return tuple(
        source.id for source in row_sources.sources if source.read_id in selected
    )


def row_sources_for_read_id(
    read_id: str,
    *,
    row_sources: RowSourceCatalog,
) -> tuple[RowSource, ...]:
    return tuple(
        source
        for source in row_sources.sources
        if source.kind == RowSourceKind.API_READ and source.read_id == read_id
    )


def api_row_source_id(read_id: str, row_path_id: str) -> str:
    return _opaque_id("rs", read_id, row_path_id or "root")


def memory_row_source_id(memory_relation_id: str) -> str:
    return _opaque_id("mem", memory_relation_id)


def _api_row_source_for_relation(
    relation: Relation,
    *,
    row_sources: RowSourceCatalog,
) -> RowSource:
    if not relation.source.read_id:
        raise KeyError("api read source requires read_id")
    selected_field_ids = {field.field_id for field in relation.fields}
    candidates = row_sources_for_read_id(
        relation.source.read_id,
        row_sources=row_sources,
    )
    if not candidates:
        raise KeyError(relation.source.read_id)
    covering = tuple(
        source
        for source in candidates
        if selected_field_ids <= {field.id for field in source.fields}
    )
    selectable = covering or candidates
    return sorted(
        selectable,
        key=lambda source: (
            _row_path_depth(source.row_path),
            len(source.fields),
            source.id,
        ),
    )[0]


def _row_path_depth(row_path: str) -> int:
    return len(tuple(part for part in row_path.split(".") if part))


def _api_row_sources(
    read: EndpointRead,
    *,
    catalog: RelationCatalog,
) -> tuple[RowSource, ...]:
    if (read.source_metadata or {}).get("representation_status") == "read_failed":
        return ()
    row_paths = read.row_paths or ()
    if not row_paths:
        if not read.fields:
            return ()
        return _api_row_sources_for_path(
            read,
            row_path_id="root",
            row_path="",
            row_cardinality=RowCardinality.ONE,
            label=read.endpoint_name,
            catalog=catalog,
            row_paths=(),
        )
    return tuple(
        source
        for index, row_path in enumerate(row_paths, start=1)
        for source in _api_row_sources_for_path(
            read,
            row_path_id=row_path.id,
            row_path=row_path.path,
            parent_row_path=row_path.parent_path,
            parent_row_cardinality=_parent_row_cardinality(
                row_path,
                row_paths=row_paths,
            ),
            row_cardinality=row_path.cardinality,
            label=f"{read.endpoint_name} row set {index}",
            catalog=catalog,
            row_paths=row_paths,
        )
    )


def _api_row_sources_for_path(
    read: EndpointRead,
    *,
    row_path_id: str,
    row_path: str,
    parent_row_path: str = "",
    parent_row_cardinality: RowCardinality | None = None,
    row_cardinality: RowCardinality,
    label: str,
    catalog: RelationCatalog,
    row_paths: tuple[RowPath, ...],
) -> tuple[RowSource, ...]:
    selected_fields = _selected_api_fields(
        read.fields,
        row_path_id=row_path_id if row_paths else "",
        row_paths=row_paths,
    )
    field_defaults = _field_requirement_defaults(selected_fields)
    response_shape_param_refs = _response_shape_param_refs(read.fields)
    return tuple(
        RowSource(
            id=api_row_source_id(
                read.id,
                _row_source_id_seed(row_path_id, source_defaults),
            ),
            kind=RowSourceKind.API_READ,
            label=_source_label(label, params=read.params, defaults=source_defaults),
            read_id=read.id,
            endpoint_name=read.endpoint_name,
            resource_names=read.resource_names,
            description=_source_description(
                read,
                params=read.params,
                defaults=source_defaults,
            ),
            row_path_id=row_path_id,
            row_path=row_path,
            parent_row_path=parent_row_path,
            parent_row_cardinality=parent_row_cardinality,
            row_cardinality=row_cardinality,
            fields=_api_fields(
                read.fields,
                read_id=read.id,
                row_path_id=row_path_id if row_paths else "",
                row_path=row_path,
                catalog=catalog,
                row_paths=row_paths,
                identity_field_refs=_identity_field_refs(read),
            ),
            candidate_keys=_row_source_candidate_keys(
                read,
                row_path_id=row_path_id if row_paths else "",
                row_paths=row_paths,
            ),
            entity_references=_row_source_entity_references(
                read,
                row_path_id=row_path_id if row_paths else "",
                row_paths=row_paths,
            ),
            params=_api_params(
                read.params,
                defaults=source_defaults,
                response_shape_param_refs=response_shape_param_refs,
            ),
            blocked_facts=_api_blocked_facts(
                read,
                row_path_id=row_path_id if row_paths else "",
                row_path=row_path,
                catalog=catalog,
                row_paths=row_paths,
            ),
        )
        for source_defaults in _source_default_variants(
            read.params,
            field_defaults=field_defaults,
        )
    )


def _parent_row_cardinality(
    row_path: RowPath,
    *,
    row_paths: tuple[RowPath, ...],
) -> RowCardinality | None:
    if not row_path.path:
        return None
    for candidate in row_paths:
        if candidate.path == row_path.parent_path:
            return candidate.cardinality
    if not row_path.parent_path:
        return None
    raise ValueError(f"row path {row_path.id} references unknown parent row path")


def _row_source_candidate_keys(
    read: EndpointRead,
    *,
    row_path_id: str,
    row_paths: tuple[RowPath, ...],
) -> tuple[RowSourceCandidateKey, ...]:
    fields_by_ref = {field.ref: field for field in read.fields}
    field_ids = _row_source_field_ids(
        read,
        row_path_id=row_path_id,
        row_paths=row_paths,
    )
    candidate_keys: list[RowSourceCandidateKey] = []
    for key in read.candidate_keys:
        if not _candidate_key_belongs_to_row_path(
            key,
            row_path_id=row_path_id,
            fields_by_ref=fields_by_ref,
            row_paths=row_paths,
        ):
            continue
        components = tuple(
            RowSourceKeyComponent(
                id=component.id,
                field_id=field_ids[component.field_ref],
            )
            for component in key.components
        )
        context_field_ids = tuple(
            field_ids[field_ref] for field_ref in key.context_field_refs
        )
        candidate_key = RowSourceCandidateKey(
            id=key.id,
            entity_kind=key.entity_kind,
            components=components,
            primary=key.primary,
            stable=key.stable,
            context_field_ids=context_field_ids,
        )
        candidate_keys.append(candidate_key)
    return tuple(candidate_keys)


def _candidate_key_belongs_to_row_path(
    key: CandidateKey,
    *,
    row_path_id: str,
    fields_by_ref: dict[str, CatalogField],
    row_paths: tuple[RowPath, ...],
) -> bool:
    component_refs = tuple(component.field_ref for component in key.components)
    required_refs = (*component_refs, *key.context_field_refs)
    if any(field_ref not in fields_by_ref for field_ref in required_refs):
        return False
    return all(
        _field_row_path_id(
            fields_by_ref[field_ref],
            row_paths=row_paths,
        )
        == row_path_id
        for field_ref in component_refs
    )


def _row_path_for_id(row_path_id: str, *, row_paths: tuple[RowPath, ...]) -> str:
    for row_path in row_paths:
        if row_path.id == row_path_id:
            return row_path.path
    return ""


def _row_source_entity_references(
    read: EndpointRead,
    *,
    row_path_id: str,
    row_paths: tuple[RowPath, ...],
) -> tuple[RowSourceEntityReference, ...]:
    fields_by_ref = {field.ref: field for field in read.fields}
    field_ids = _row_source_field_ids(
        read,
        row_path_id=row_path_id,
        row_paths=row_paths,
    )
    references: list[RowSourceEntityReference] = []
    for reference in read.entity_references:
        required_refs = (
            *tuple(component.local_field_ref for component in reference.components),
            *reference.context_field_refs,
        )
        if not all(ref in field_ids for ref in required_refs):
            continue
        if not _entity_reference_belongs_to_row_path(
            reference,
            row_path_id=row_path_id,
            fields_by_ref=fields_by_ref,
            row_paths=row_paths,
        ):
            continue
        components = tuple(
            RowSourceEntityReferenceComponent(
                target_component_id=component.target_component_id,
                local_field_id=field_ids[component.local_field_ref],
            )
            for component in reference.components
        )
        context_field_ids = tuple(
            field_ids[field_ref] for field_ref in reference.context_field_refs
        )
        row_source_reference = RowSourceEntityReference(
            id=reference.id,
            target_entity_kind=reference.target_entity_kind,
            target_key_id=reference.target_key_id,
            components=components,
            context_field_ids=context_field_ids,
        )
        references.append(row_source_reference)
    return tuple(references)


def _row_source_field_ids(
    read: EndpointRead,
    *,
    row_path_id: str,
    row_paths: tuple[RowPath, ...],
) -> dict[str, str]:
    fields = _selected_api_fields_for_source(
        read.fields,
        row_path_id=row_path_id,
        row_paths=row_paths,
    )
    row_path = _row_path_for_id(row_path_id, row_paths=row_paths)
    return _field_ids(fields, row_path=row_path, row_paths=row_paths)


def _entity_reference_belongs_to_row_path(
    reference: EntityReference,
    *,
    row_path_id: str,
    fields_by_ref: dict[str, CatalogField],
    row_paths: tuple[RowPath, ...],
) -> bool:
    component_refs = tuple(
        component.local_field_ref for component in reference.components
    )
    required_refs = (*component_refs, *reference.context_field_refs)
    if any(field_ref not in fields_by_ref for field_ref in required_refs):
        return False
    visible_paths = set(_row_context_path_ids(row_path_id, row_paths=row_paths))
    return all(
        _field_row_path_id(fields_by_ref[field_ref], row_paths=row_paths)
        in visible_paths
        for field_ref in component_refs
    )


def _response_shape_param_refs(fields: tuple[CatalogField, ...]) -> frozenset[str]:
    return frozenset(
        requirement.param_ref
        for field in fields
        for requirement in field.requirements
        if requirement.param_ref
    )


def _api_fields(
    fields: tuple[CatalogField, ...],
    *,
    read_id: str,
    row_path_id: str,
    row_path: str = "",
    catalog: RelationCatalog,
    row_paths: tuple[RowPath, ...],
    identity_field_refs: frozenset[str],
) -> tuple[RowSourceField, ...]:
    selected = (
        _selected_api_fields_for_source(
            fields,
            row_path_id=row_path_id,
            row_paths=row_paths,
        )
        if row_paths
        else fields
    )
    ids = _field_ids(selected, row_path=row_path, row_paths=row_paths)
    return tuple(
        RowSourceField(
            id=ids[field.ref],
            field_ref=field.ref,
            label=_field_label(
                field,
                row_path=_field_row_path(field, row_paths=row_paths),
                field_id=ids[field.ref],
            ),
            type=row_source_value_type(field.type),
            choices=field.choices,
            allowed_roles=_allowed_roles(
                field,
                identity_field_refs=identity_field_refs,
            ),
            fact_refs=_field_fact_refs(field, catalog=catalog, read_id=read_id),
            answer_output_ids=_field_answer_output_ids(field),
            path=field.path,
            response_path=_relative_field_path(
                field.path,
                _field_row_path(field, row_paths=row_paths),
            ),
            description=_field_description(field),
            nullable=field.nullable,
            declared_value_domain=(field.metadata or {}).get("representation_authority")
            != "observed_response",
        )
        for field in selected
        if not row_path or field.path != row_path
    )


def _identity_field_refs(read: EndpointRead) -> frozenset[str]:
    key_field_refs = (
        component.field_ref
        for key in read.candidate_keys
        for component in key.components
    )
    reference_field_refs = (
        component.local_field_ref
        for reference in read.entity_references
        for component in reference.components
    )
    return frozenset((*key_field_refs, *reference_field_refs))


def _field_description(field: CatalogField) -> str:
    if not field.metadata:
        return ""
    return " ".join(str(value or "") for value in field.metadata.values())


def _field_answer_output_ids(field: CatalogField) -> tuple[str, ...]:
    metadata = field.metadata if isinstance(field.metadata, dict) else {}
    raw = metadata.get("fervis_answer_output_ids")
    if isinstance(raw, str):
        return (raw,) if raw.strip() else ()
    if isinstance(raw, tuple | list):
        return tuple(str(item) for item in raw if str(item).strip())
    return ()


def _selected_api_fields(
    fields: tuple[CatalogField, ...],
    *,
    row_path_id: str,
    row_paths: tuple[RowPath, ...],
) -> tuple[CatalogField, ...]:
    return tuple(
        field
        for field in fields
        if _field_row_path_id(field, row_paths=row_paths) == row_path_id
    )


def _selected_api_fields_for_source(
    fields: tuple[CatalogField, ...],
    *,
    row_path_id: str,
    row_paths: tuple[RowPath, ...],
) -> tuple[CatalogField, ...]:
    ancestor_ids = _row_context_path_ids(row_path_id, row_paths=row_paths)[:-1]
    local_fields = tuple(
        field
        for field in fields
        if _field_row_path_id(field, row_paths=row_paths) == row_path_id
    )
    if not ancestor_ids:
        return local_fields

    def container(field: CatalogField) -> str:
        owner_path = _field_row_path(field, row_paths=row_paths)
        return _relative_field_path(field.path, owner_path).split(".", 1)[0]

    local_containers = {container(field) for field in local_fields}
    ancestor_fields = tuple(
        field
        for field in fields
        if _field_row_path_id(field, row_paths=row_paths) in ancestor_ids
        and container(field) not in local_containers
    )
    return (*ancestor_fields, *local_fields)


def _api_params(
    params: tuple[CatalogParam, ...],
    *,
    defaults: dict[str, CatalogParameterValue],
    response_shape_param_refs: frozenset[str],
) -> tuple[RowSourceParam, ...]:
    ids = _param_ids(params)
    return tuple(
        RowSourceParam(
            id=ids[param.ref],
            param_ref=param.ref,
            name=param.name,
            type=row_source_value_type(param.type),
            source=param.source,
            required=param.required,
            choices=param.choices,
            choice_labels=param.choice_labels,
            description=param.description,
            default=defaults.get(param.ref, param.default),
            default_is_known=param.ref in defaults or param.default_is_known,
            default_source="source_variant" if param.ref in defaults else "",
            population=param.population,
            entity_target=param.entity_target,
            semantics=_param_semantics(
                param,
                response_shape_param_refs=response_shape_param_refs,
            ),
        )
        for param in params
    )


def _param_semantics(
    param: CatalogParam,
    *,
    response_shape_param_refs: frozenset[str],
) -> ParameterSemantics:
    if param.ref in response_shape_param_refs:
        return ParameterSemantics.RESPONSE_SHAPE
    if param.semantics:
        try:
            return ParameterSemantics(param.semantics)
        except ValueError as exc:
            raise ValueError(f"{param.ref} has unsupported param semantics") from exc
    return ParameterSemantics.OPAQUE_QUERY_PARAM


def _source_default_variants(
    params: tuple[CatalogParam, ...],
    *,
    field_defaults: dict[str, CatalogParameterValue],
) -> tuple[dict[str, CatalogParameterValue], ...]:
    source_shape_params = tuple(
        param
        for param in params
        if _source_variant_param(param) and param.ref not in field_defaults
    )
    if not source_shape_params:
        return (dict(field_defaults),)
    return tuple(
        {
            **field_defaults,
            **{
                param.ref: value
                for param, value in zip(source_shape_params, values, strict=True)
            },
        }
        for values in product(*(param.choices for param in source_shape_params))
    )


def _source_variant_param(param: CatalogParam) -> bool:
    return (
        param.required
        and param.default is None
        and bool(param.choices)
        and param.semantics == ParameterSemantics.RESPONSE_SHAPE.value
    )


def _field_requirement_defaults(
    fields: tuple[CatalogField, ...],
) -> dict[str, CatalogParameterValue]:
    defaults: dict[str, CatalogParameterValue] = {}
    for field in fields:
        for requirement in field.requirements:
            existing = defaults.get(requirement.param_ref, _MISSING)
            if existing is not _MISSING and existing != requirement.value:
                raise ValueError(
                    f"conflicting field requirements for {requirement.param_ref}"
                )
            defaults[requirement.param_ref] = requirement.value
    return defaults


def _row_source_id_seed(
    row_path_id: str,
    defaults: dict[str, CatalogParameterValue],
) -> str:
    base = row_path_id or "root"
    if not defaults:
        return base
    suffix = "__".join(
        f"{_symbol(param_ref.rsplit('.', 1)[-1])}_{_symbol(str(value))}"
        for param_ref, value in sorted(defaults.items())
    )
    return f"{base}__{suffix}"


def _source_label(
    label: str,
    *,
    params: tuple[CatalogParam, ...],
    defaults: dict[str, CatalogParameterValue],
) -> str:
    default_label = _defaults_label(params, defaults=defaults)
    if not default_label:
        return label
    return f"{label} ({default_label})"


def _source_description(
    read: EndpointRead,
    *,
    params: tuple[CatalogParam, ...],
    defaults: Mapping[str, CatalogParameterValue],
) -> str:
    description = read.description
    default_label = _defaults_label(params, defaults=defaults)
    if not default_label:
        return description
    suffix = f"Source defaults: {default_label}."
    if not description:
        return suffix
    return f"{description} {suffix}"


def _defaults_label(
    params: tuple[CatalogParam, ...],
    *,
    defaults: Mapping[str, CatalogParameterValue],
) -> str:
    if not defaults:
        return ""
    labels = []
    for param in params:
        if param.ref in defaults:
            labels.append(f"{param.name}={defaults[param.ref]}")
    return ", ".join(labels)


def _api_blocked_facts(
    read: EndpointRead,
    *,
    row_path_id: str,
    row_path: str,
    catalog: RelationCatalog,
    row_paths: tuple[RowPath, ...],
) -> tuple[RowSourceBlockedFact, ...]:
    selected_fields = tuple(
        field
        for field in read.fields
        if _field_row_path_id(field, row_paths=row_paths) == row_path_id
    )
    field_ids = _field_ids(selected_fields, row_path=row_path, row_paths=row_paths)
    output: list[RowSourceBlockedFact] = []
    for fact in _read_catalog_facts(read, catalog=catalog):
        if fact.availability == CatalogFactAvailability.AVAILABLE:
            continue
        field_id = ""
        if fact.field_ref:
            if fact.field_ref not in field_ids:
                continue
            field_id = field_ids[fact.field_ref]
        output.append(
            RowSourceBlockedFact(
                fact_ref=fact.ref,
                availability=fact.availability,
                field_id=field_id,
                proof_refs=tuple(fact.proof_refs),
            )
        )
    return tuple(output)


def _memory_row_source(relation: "RelationRows") -> RowSource:
    field_ids = _memory_field_ids(relation)
    return RowSource(
        id=memory_row_source_id(relation.id),
        kind=RowSourceKind.MEMORY_READ,
        label=relation.id,
        memory_ref=relation.id,
        row_cardinality=RowCardinality.MANY,
        fields=tuple(
            RowSourceField(
                id=field_id,
                field_ref=field_id,
                label=field_id,
                type=row_source_value_type(
                    (relation.field_types or {}).get(field_id, "unknown")
                ),
                allowed_roles=_memory_roles(field_id, relation=relation),
            )
            for field_id in field_ids
        ),
    )


def _generated_calendar_row_source() -> RowSource:
    return RowSource(
        id=CALENDAR_ROW_SOURCE_ID,
        kind=RowSourceKind.GENERATED_CALENDAR,
        label="calendar days",
        description="Calendar day rows generated from an interval start and interval end.",
        row_cardinality=RowCardinality.MANY,
        fields=(
            RowSourceField(
                id=CALENDAR_DATE_FIELD_ID,
                field_ref=CALENDAR_DATE_FIELD_ID,
                label="runtime date",
                type=RowSourceValueType.DATE,
                allowed_roles=(
                    FieldBindingRole.IDENTITY,
                    FieldBindingRole.OUTPUT,
                    FieldBindingRole.PREDICATE,
                ),
            ),
        ),
        params=(
            RowSourceParam(
                id=CALENDAR_START_PARAM_ID,
                param_ref=CALENDAR_START_PARAM_REF,
                name="interval start",
                type=RowSourceValueType.DATE,
                source="generated",
                required=True,
            ),
            RowSourceParam(
                id=CALENDAR_END_PARAM_ID,
                param_ref=CALENDAR_END_PARAM_REF,
                name="interval end",
                type=RowSourceValueType.DATE,
                source="generated",
                required=True,
            ),
        ),
    )


def _memory_field_ids(relation: "RelationRows") -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for field_id in (
        *relation.grain_keys,
        *(relation.field_types or {}),
        *(key for row in relation.rows for key in row),
    ):
        field = str(field_id)
        if field and field not in seen:
            seen.add(field)
            output.append(field)
    return tuple(output)


def _memory_roles(
    field_id: str,
    *,
    relation: "RelationRows",
) -> tuple[FieldBindingRole, ...]:
    roles = [FieldBindingRole.OUTPUT, FieldBindingRole.PREDICATE]
    if field_id in set(relation.grain_keys):
        roles.insert(0, FieldBindingRole.IDENTITY)
    return tuple(roles)
