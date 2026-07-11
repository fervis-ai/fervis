"""Source, relation, catalog, and expression checks for program verification."""

from ._shared import (
    AnswerProgram,
    AuthorizedExecutionSources,
    CatalogField,
    CatalogSelectionResult,
    FieldBindingRole,
    Relation,
    RelationCatalog,
    RelationSource,
    RowSource,
    RowSourceCatalog,
    RowSourceKind,
    SourceKind,
    VerificationError,
    instantiate_program_expressions,
    row_source_for_relation,
)
from fervis.lookup.answer_program.expression_instantiation import (
    InstantiatedProgramInputs,
)


def _verify_program_expression_targets(
    answer: AnswerProgram,
    *,
    bindings,
    catalog: RelationCatalog | None,
    row_sources: RowSourceCatalog,
) -> None:
    instantiate_program_expressions(
        bindings=bindings,
        catalog=catalog or RelationCatalog(),
        relations=answer.relations,
        parameters=answer.parameters,
        row_sources=row_sources,
    )


def _verify_required_source_params(
    answer: AnswerProgram,
    *,
    row_sources: RowSourceCatalog,
) -> None:
    provided: set[tuple[str, str]] = set()
    for relation in answer.relations:
        if relation.source.kind not in {
            SourceKind.API_READ,
            SourceKind.GENERATED_CALENDAR,
            SourceKind.MEMORY_READ,
        }:
            continue
        row_source = _row_source_for_relation(relation, row_sources=row_sources)
        for binding in relation.source.param_bindings:
            provided.add((relation.id, binding.param_id))
    for relation in answer.relations:
        source = relation.source
        if source.kind not in {
            SourceKind.API_READ,
            SourceKind.GENERATED_CALENDAR,
            SourceKind.MEMORY_READ,
        }:
            continue
        row_source = _row_source_for_relation(relation, row_sources=row_sources)
        if row_source.kind not in {
            RowSourceKind.API_READ,
            RowSourceKind.GENERATED_CALENDAR,
        }:
            continue
        for param in row_source.params:
            if not param.required or param.default is not None:
                continue
            if (relation.id, param.id) not in provided:
                raise VerificationError(
                    f"relation {relation.id} requires source param {param.id}"
                )


def _verify_sources(
    answer: AnswerProgram,
    *,
    row_sources: RowSourceCatalog,
    allowed_read_ids: frozenset[str] | None = None,
) -> None:
    for relation in answer.relations:
        _verify_source(
            relation.source,
            row_sources=row_sources,
            allowed_read_ids=allowed_read_ids,
        )


def _verify_source(
    source: RelationSource,
    *,
    row_sources: RowSourceCatalog,
    allowed_read_ids: frozenset[str] | None = None,
) -> None:
    kind = source.kind
    if kind == SourceKind.API_READ:
        if not source.read_id:
            raise VerificationError("api_read source requires read_id")
        if allowed_read_ids is not None and source.read_id not in allowed_read_ids:
            raise VerificationError("relation uses source outside selected catalog")
        if not any(
            item.kind == RowSourceKind.API_READ and item.read_id == source.read_id
            for item in row_sources.sources
        ):
            raise VerificationError("relation references unknown API read")
        return
    if kind == SourceKind.GENERATED_CALENDAR:
        if source.calendar_id != "calendar_days":
            raise VerificationError("generated_calendar source requires calendar_id")
        return
    if kind == SourceKind.MEMORY_READ:
        if source.param_bindings:
            raise VerificationError("param bindings require api_read source")
        if not source.memory_relation_id:
            raise VerificationError("memory_read source requires memory_relation_id")
        return
    raise VerificationError(f"unsupported relation source kind: {kind.value}")


def _row_source_for_relation(
    relation: Relation,
    *,
    row_sources: RowSourceCatalog,
) -> RowSource:
    try:
        return row_source_for_relation(relation, row_sources=row_sources)
    except KeyError as exc:
        raise VerificationError(
            f"relation {relation.id} references unknown source"
        ) from exc


def _allowed_read_ids(
    *,
    catalog_selection: CatalogSelectionResult | None,
    authorized_sources: AuthorizedExecutionSources | None = None,
) -> frozenset[str] | None:
    if authorized_sources is not None:
        authorized = authorized_sources
    elif catalog_selection is not None:
        authorized = AuthorizedExecutionSources.from_catalog_selection(
            catalog_selection
        )
    else:
        return None
    return authorized.allowed_read_ids


def _verify_relations(relations: tuple[Relation, ...]) -> None:
    relation_ids: set[str] = set()
    for relation in relations:
        if not relation.id:
            raise VerificationError("relation requires id")
        if relation.id in relation_ids:
            raise VerificationError(f"duplicate relation {relation.id}")
        relation_ids.add(relation.id)
        _verify_unique_relation_field_ids(relation)
        field_ids = {item.field_id for item in relation.fields}
        fields_by_id = {item.field_id: item for item in relation.fields}
        for grain_key in relation.grain_keys:
            if grain_key not in field_ids:
                raise VerificationError(
                    f"relation {relation.id} references unknown grain key"
                )
            if FieldBindingRole.IDENTITY not in fields_by_id[grain_key].roles:
                raise VerificationError(
                    f"relation {relation.id} grain key requires identity binding"
                )


def _verify_unique_relation_field_ids(relation: Relation) -> None:
    seen: set[str] = set()
    for field in relation.fields:
        if field.field_id in seen:
            raise VerificationError(
                f"relation {relation.id} has duplicate field {field.field_id}"
            )
        if not field.roles:
            raise VerificationError(f"relation {relation.id} field requires role")
        seen.add(field.field_id)


def _verify_api_relation_catalog_refs(
    relations: tuple[Relation, ...],
    catalog: RelationCatalog,
    *,
    row_sources: RowSourceCatalog,
    instantiated_inputs: InstantiatedProgramInputs,
) -> None:
    endpoint_arg_values = _endpoint_arg_values(
        relations=relations,
        row_sources=row_sources,
        instantiated_inputs=instantiated_inputs,
    )
    for relation in relations:
        if relation.source.kind != SourceKind.API_READ:
            continue
        row_source = _row_source_for_relation(relation, row_sources=row_sources)
        if row_source.kind != RowSourceKind.API_READ:
            continue
        for field in relation.fields:
            try:
                row_source_field = row_source.field(field.field_id)
            except KeyError as exc:
                raise VerificationError(
                    f"relation {relation.id} references unknown source field"
                ) from exc
            for role in field.roles:
                if role not in row_source_field.allowed_roles:
                    raise VerificationError(
                        f"relation {relation.id} field role is not allowed"
                    )
            _verify_field_requirements(
                relation=relation,
                field=_catalog_field(
                    catalog, row_source.read_id, row_source_field.field_ref
                ),
                row_source=row_source,
                endpoint_arg_values=endpoint_arg_values,
            )


def _endpoint_arg_values(
    *,
    relations: tuple[Relation, ...],
    row_sources: RowSourceCatalog,
    instantiated_inputs: InstantiatedProgramInputs,
) -> dict[tuple[str, str], object]:
    return {
        (
            item.relation_id,
            _endpoint_param_id(
                relations,
                row_sources=row_sources,
                relation_id=item.relation_id,
                param_ref=item.param_ref,
            ),
        ): item.value
        for item in instantiated_inputs.endpoint_args
    }


def _endpoint_param_id(
    relations: tuple[Relation, ...],
    *,
    row_sources: RowSourceCatalog,
    relation_id: str,
    param_ref: str,
) -> str:
    relation = next((item for item in relations if item.id == relation_id), None)
    if relation is None:
        raise VerificationError("endpoint argument references unknown relation")
    row_source = _row_source_for_relation(relation, row_sources=row_sources)
    return _param_id_for_ref(row_source, param_ref)


def _verify_field_requirements(
    *,
    relation: Relation,
    field: CatalogField,
    row_source: RowSource,
    endpoint_arg_values: dict[tuple[str, str], object],
) -> None:
    for requirement in field.requirements:
        param_id = _param_id_for_ref(row_source, requirement.param_ref)
        actual = endpoint_arg_values.get((relation.id, param_id))
        if not _requirement_value_matches(actual, requirement.value):
            raise VerificationError(
                f"relation {relation.id} field requires endpoint argument {param_id}"
            )


def _param_id_for_ref(row_source: RowSource, param_ref: str) -> str:
    for param in row_source.params:
        if param.param_ref == param_ref:
            return param.id
    raise VerificationError(f"row source {row_source.id} lacks required param")


def _catalog_field(
    catalog: RelationCatalog,
    read_id: str,
    field_ref: str,
) -> CatalogField:
    read = catalog.read(read_id)
    for field in read.fields:
        if field.ref == field_ref:
            return field
    raise VerificationError(f"row source field {field_ref} is unavailable")


def _requirement_value_matches(actual: object, expected: object) -> bool:
    if actual is None:
        return False
    return _requirement_value(actual) == _requirement_value(expected)


def _requirement_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip().lower()
