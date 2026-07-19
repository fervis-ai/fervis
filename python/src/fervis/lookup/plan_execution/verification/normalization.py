"""Backend-owned fact-plan normalization before verification."""

from ._shared import (
    AggregateSpec,
    AnswerProgram,
    FactPlan,
    FieldBindingRole,
    Operation,
    ProjectField,
    ProjectSpec,
    OrderSpec,
    Relation,
    RelationCatalog,
    RelationField,
    RelationRows,
    RowSourceCatalog,
    build_row_source_catalog,
    replace,
    row_source_for_relation,
)


def _normalize_fact_plan_for_verification(
    plan: FactPlan,
    *,
    catalog: RelationCatalog | None,
    memory_relations: tuple[RelationRows, ...],
) -> FactPlan:
    outcome = plan.outcome
    if isinstance(outcome, AnswerProgram) and catalog is not None:
        row_sources = build_row_source_catalog(
            catalog,
            memory_relations=memory_relations,
        )
        normalized_outcome = _with_backend_identity_fields(outcome, row_sources)
        if normalized_outcome is outcome:
            return plan
        return FactPlan(outcome=normalized_outcome)
    return plan


def _with_backend_identity_fields(
    answer: AnswerProgram,
    row_sources: RowSourceCatalog,
) -> AnswerProgram:
    relations = tuple(
        _relation_with_backend_identity_fields(relation, row_sources=row_sources)
        for relation in answer.relations
    )
    grain_by_relation = {relation.id: relation.grain_keys for relation in relations}
    operations: list[Operation] = []
    for operation in answer.operations:
        operation = _operation_with_projected_identity_fields(
            operation,
            grain_by_relation=grain_by_relation,
        )
        operations.append(operation)
        if operation.output_relation:
            grain_by_relation[operation.output_relation] = _operation_output_grain(
                operation,
                grain_by_relation=grain_by_relation,
            )
    if relations == answer.relations and tuple(operations) == answer.operations:
        return answer
    return replace(
        answer,
        relations=relations,
        operations=tuple(operations),
    )


def _relation_with_backend_identity_fields(
    relation: Relation,
    *,
    row_sources: RowSourceCatalog,
) -> Relation:
    try:
        row_source = row_source_for_relation(relation, row_sources=row_sources)
    except KeyError:
        return relation
    if relation.grain_keys:
        return relation
    identity_field_ids = tuple(
        field.id
        for field in row_source.fields
        if FieldBindingRole.IDENTITY in field.allowed_roles
    )
    if len(identity_field_ids) != 1:
        return relation
    fields_by_id = {field.field_id: field for field in relation.fields}
    fields = (
        *relation.fields,
        *(
            RelationField(
                field_id=field_id,
                roles=(FieldBindingRole.IDENTITY,),
            )
            for field_id in identity_field_ids
            if field_id not in fields_by_id
        ),
    )
    return replace(relation, fields=fields)


def _operation_with_projected_identity_fields(
    operation: Operation,
    *,
    grain_by_relation: dict[str, tuple[str, ...]],
) -> Operation:
    spec = operation.spec
    if not isinstance(spec, ProjectSpec):
        return operation
    input_grain = grain_by_relation.get(spec.input_relation, ())
    if not input_grain:
        return operation
    projected_sources = {field.source for field in spec.fields}
    missing = tuple(field for field in input_grain if field not in projected_sources)
    if not missing:
        return operation
    return replace(
        operation,
        spec=replace(
            spec,
            fields=(
                *(ProjectField(source=field) for field in missing),
                *spec.fields,
            ),
        ),
    )


def _operation_output_grain(
    operation: Operation,
    *,
    grain_by_relation: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    spec = operation.spec
    if isinstance(spec, ProjectSpec):
        input_grain = grain_by_relation.get(spec.input_relation, ())
        projections = {
            field.source: field.output or field.source for field in spec.fields
        }
        if all(field in projections for field in input_grain):
            return tuple(projections[field] for field in input_grain)
        return ()
    if isinstance(spec, AggregateSpec):
        return spec.group_by
    if isinstance(spec, OrderSpec):
        return grain_by_relation.get(spec.input_relation, ())
    return ()
