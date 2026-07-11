"""Instantiate closed answer-program expressions into executable inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, assert_never, cast

from fervis.lookup.relation_catalog.model import RelationCatalog
from fervis.lookup.plan_execution.errors import VerificationError
from fervis.lookup.answer_program.relations import (
    EndpointParamBinding,
    PopulationChoiceControllerKind,
    Relation,
    RelationSourcePopulationChoice,
    RelationSourceReviewScopeDecision,
    RelationSourceRowFilter,
    SourceKind,
)
from fervis.lookup.fact_plan.row_sources import (
    RowSource,
    RowSourceCatalog,
    RowSourceKind,
    RowSourceParam,
    RowSourceValueType,
    build_row_source_catalog,
    row_source_for_relation,
    row_source_param_evidence_ref,
)
from fervis.lookup.canonical_data import canonical_runtime_json
from fervis.lookup.answer_program.values import (
    EnvironmentRef,
    FactValue,
    LiteralType,
    ParameterRef,
    TimeComponent,
    ValueComponent,
    ValueFilterOperator,
)
from fervis.lookup.fact_planning.value_components import value_component
from fervis.lookup.answer_program.contracts import (
    AnswerProgramContractError,
    BindingSet,
    ParameterDeclaration,
)
from fervis.lookup.answer_program.inputs import (
    ResolvedValueExpression,
    resolve_value_expression,
)


@dataclass(frozen=True)
class ResolvedEndpointArg:
    relation_id: str
    read_id: str
    param_ref: str
    value: Any
    proof_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedRowFilter:
    relation_id: str
    field_id: str
    operator: ValueFilterOperator
    value: Any
    proof_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedPopulationChoice:
    relation_id: str
    controller_kind: PopulationChoiceControllerKind
    controller_id: str
    field_id: str
    requested_fact_ids: tuple[str, ...]
    semantic_control_ref: str
    included_values: tuple[str, ...]
    excluded_values: tuple[str, ...]
    proof_refs: tuple[str, ...] = ()
    review_scope_decisions: tuple[RelationSourceReviewScopeDecision, ...] = ()


@dataclass(frozen=True)
class InstantiatedProgramInputs:
    endpoint_args: tuple[ResolvedEndpointArg, ...] = ()
    row_filters: tuple[ResolvedRowFilter, ...] = ()
    population_choices: tuple[ResolvedPopulationChoice, ...] = ()


def instantiate_program_expressions(
    *,
    bindings: BindingSet,
    catalog: RelationCatalog,
    relations: tuple[Relation, ...] = (),
    parameters: tuple[ParameterDeclaration, ...] = (),
    row_sources: RowSourceCatalog | None = None,
) -> InstantiatedProgramInputs:
    row_source_catalog = row_sources or build_row_source_catalog(catalog)
    relation_sources = _relation_row_sources(
        relations,
        row_source_catalog,
    )

    endpoint_args: list[ResolvedEndpointArg] = []
    row_filters: list[ResolvedRowFilter] = []
    population_choices: list[ResolvedPopulationChoice] = []
    endpoint_arg_targets: set[tuple[str, str]] = set()

    _append_relation_source_row_filters(
        row_filters,
        bindings=bindings,
        parameters=parameters,
        relations=relations,
    )
    _append_relation_source_endpoint_args(
        endpoint_args,
        endpoint_arg_targets=endpoint_arg_targets,
        relations=relations,
        relation_sources=relation_sources,
        bindings=bindings,
        parameters=parameters,
    )
    _append_relation_source_population_choices(
        population_choices,
        row_filters=row_filters,
        relations=relations,
        bindings=bindings,
        parameters=parameters,
    )
    return InstantiatedProgramInputs(
        endpoint_args=tuple(endpoint_args),
        row_filters=_canonical_row_filters(tuple(row_filters)),
        population_choices=tuple(population_choices),
    )


def _canonical_row_filters(
    row_filters: tuple[ResolvedRowFilter, ...],
) -> tuple[ResolvedRowFilter, ...]:
    output: list[ResolvedRowFilter] = []
    by_field: dict[tuple[str, str], ResolvedRowFilter] = {}
    for row_filter in row_filters:
        key = (row_filter.relation_id, row_filter.field_id)
        existing = by_field.get(key)
        if existing is None:
            by_field[key] = row_filter
            output.append(row_filter)
            continue
        if not _same_row_filter_constraint(existing, row_filter):
            raise VerificationError(
                "relation "
                f"{row_filter.relation_id} has conflicting row filters for "
                f"field {row_filter.field_id}"
            )
        merged = ResolvedRowFilter(
            relation_id=existing.relation_id,
            field_id=existing.field_id,
            operator=existing.operator,
            value=existing.value,
            proof_refs=_dedupe_refs((*existing.proof_refs, *row_filter.proof_refs)),
        )
        by_field[key] = merged
        output[output.index(existing)] = merged
    return tuple(output)


def _same_row_filter_constraint(
    left: ResolvedRowFilter,
    right: ResolvedRowFilter,
) -> bool:
    return left.operator == right.operator and left.value == right.value


def _append_relation_source_row_filters(
    row_filters: list[ResolvedRowFilter],
    *,
    bindings: BindingSet,
    parameters: tuple[ParameterDeclaration, ...],
    relations: tuple[Relation, ...],
) -> None:
    for relation in relations:
        for source_row_filter in relation.source.row_filters:
            _require_bound_relation_field(relation, source_row_filter.field_id)
            resolved = _resolve_omittable_expression(
                source_row_filter.value_expr,
                bindings=bindings,
                parameters=parameters,
            )
            if resolved is None:
                continue
            _require_filter_value(resolved.fact_value)
            row_filters.append(
                ResolvedRowFilter(
                    relation_id=relation.id,
                    field_id=source_row_filter.field_id,
                    operator=_source_row_filter_operator(source_row_filter),
                    value=_source_row_filter_value(
                        source_row_filter,
                        resolved.value,
                    ),
                    proof_refs=_dedupe_refs(
                        (*source_row_filter.proof_refs, *resolved.proof_refs)
                    ),
                )
            )
        for source_filter in relation.source.applied_filters:
            resolved = _resolve_omittable_expression(
                source_filter.value_expr,
                bindings=bindings,
                parameters=parameters,
            )
            if resolved is None:
                continue
            _require_filter_value(resolved.fact_value)
            for field_id in source_filter.predicate_field_ids:
                _require_bound_relation_field(relation, field_id)
                row_filters.append(
                    ResolvedRowFilter(
                        relation_id=relation.id,
                        field_id=field_id,
                        operator=ValueFilterOperator.EQUALS,
                        value=resolved.value,
                        proof_refs=resolved.proof_refs,
                    )
                )


def _source_row_filter_operator(
    source_filter: RelationSourceRowFilter,
) -> ValueFilterOperator:
    try:
        return ValueFilterOperator(source_filter.operator)
    except ValueError as exc:
        raise VerificationError(
            f"unsupported relation source row filter operator: {source_filter.operator}"
        ) from exc


def _source_row_filter_value(
    source_filter: RelationSourceRowFilter,
    value: object,
) -> object:
    operator = _source_row_filter_operator(source_filter)
    values = value if isinstance(value, tuple) else (value,)
    if operator == ValueFilterOperator.EQUALS and len(values) == 1:
        return values[0]
    return tuple(values)


def _append_relation_source_population_choices(
    population_choices: list[ResolvedPopulationChoice],
    *,
    row_filters: list[ResolvedRowFilter],
    relations: tuple[Relation, ...],
    bindings: BindingSet,
    parameters: tuple[ParameterDeclaration, ...],
) -> None:
    for relation in relations:
        for choice in relation.source.population_choices:
            compiled = _compiled_population_choice(
                relation_id=relation.id,
                choice=choice,
                bindings=bindings,
                parameters=parameters,
            )
            if compiled is not None:
                population_choices.append(compiled)
                if (
                    compiled.controller_kind
                    is PopulationChoiceControllerKind.ROW_PREDICATE
                    and compiled.excluded_values
                ):
                    _require_bound_relation_field(relation, compiled.field_id)
                    row_filters.append(
                        ResolvedRowFilter(
                            relation_id=relation.id,
                            field_id=compiled.field_id,
                            operator=ValueFilterOperator.IN,
                            value=compiled.included_values,
                            proof_refs=compiled.proof_refs,
                        )
                    )


def _compiled_population_choice(
    *,
    relation_id: str,
    choice: RelationSourcePopulationChoice,
    bindings: BindingSet,
    parameters: tuple[ParameterDeclaration, ...],
) -> ResolvedPopulationChoice | None:
    resolved = _resolve_omittable_expression(
        choice.selection_expr,
        bindings=bindings,
        parameters=parameters,
    )
    if resolved is None:
        return None
    included_values = cast(tuple[str, ...], resolved.value)
    excluded_values = tuple(
        value for value in choice.allowed_values if value not in set(included_values)
    )
    semantic_control_ref = _population_choice_semantic_control_ref(
        choice,
        parameters=parameters,
    )
    return ResolvedPopulationChoice(
        relation_id=relation_id,
        controller_kind=choice.controller_kind,
        controller_id=choice.controller_id,
        field_id=choice.field_id,
        requested_fact_ids=choice.requested_fact_ids,
        semantic_control_ref=semantic_control_ref,
        included_values=included_values,
        excluded_values=excluded_values,
        proof_refs=_dedupe_refs((*choice.proof_refs, *resolved.proof_refs)),
        review_scope_decisions=choice.review_scope_decisions,
    )


def _population_choice_semantic_control_ref(
    choice: RelationSourcePopulationChoice,
    *,
    parameters: tuple[ParameterDeclaration, ...],
) -> str:
    expression = choice.selection_expr
    declaration = next(
        (
            parameter
            for parameter in parameters
            if parameter.id == expression.parameter_id
        ),
        None,
    )
    if declaration is None:
        raise VerificationError(
            f"population choice references unknown parameter {expression.parameter_id}"
        )
    if not declaration.semantic_control_ref:
        raise VerificationError(
            "population choice parameter requires semantic-control identity"
        )
    return declaration.semantic_control_ref


def _append_relation_source_endpoint_args(
    endpoint_args: list[ResolvedEndpointArg],
    *,
    endpoint_arg_targets: set[tuple[str, str]],
    relations: tuple[Relation, ...],
    relation_sources: dict[str, RowSource],
    bindings: BindingSet,
    parameters: tuple[ParameterDeclaration, ...],
) -> None:
    for relation in relations:
        if relation.source.kind not in {
            SourceKind.API_READ,
            SourceKind.GENERATED_CALENDAR,
        }:
            if relation.source.param_bindings:
                raise VerificationError(
                    f"relation {relation.id} param bindings require api_read source"
                )
            continue
        row_source = _row_source_for_relation(relation.id, relation_sources)
        if row_source.kind not in {
            RowSourceKind.API_READ,
            RowSourceKind.GENERATED_CALENDAR,
        }:
            continue
        for binding in relation.source.param_bindings:
            try:
                param = row_source.param(binding.param_id)
            except KeyError as exc:
                raise VerificationError(
                    f"relation {relation.id} references unknown source param"
                ) from exc
            resolved = _resolve_endpoint_binding(
                binding,
                row_source=row_source,
                param=param,
                bindings=bindings,
                parameters=parameters,
            )
            if resolved is None:
                if param.required and param.default is None:
                    raise VerificationError(
                        f"relation {relation.id} requires source param {param.id}"
                    )
                continue
            values = (
                resolved.value
                if isinstance(resolved.value, tuple)
                else (resolved.value,)
            )
            if param.choices and any(value not in param.choices for value in values):
                raise VerificationError(
                    f"relation {relation.id} param binding has unknown choice"
                )
            _append_endpoint_arg(
                endpoint_args,
                endpoint_arg_targets=endpoint_arg_targets,
                arg=ResolvedEndpointArg(
                    relation_id=relation.id,
                    read_id=row_source.read_id,
                    param_ref=param.param_ref,
                    value=resolved.value,
                    proof_refs=(
                        *binding.proof_refs,
                        *resolved.proof_refs,
                        row_source_param_evidence_ref(
                            row_source_id=row_source.id,
                            param_id=param.id,
                        ),
                    ),
                ),
            )


def _resolve_endpoint_binding(
    binding: EndpointParamBinding,
    *,
    row_source: RowSource,
    param: RowSourceParam,
    bindings: BindingSet,
    parameters: tuple[ParameterDeclaration, ...],
):
    expression = binding.value_expr
    if isinstance(expression, EnvironmentRef):
        expected_source_ref = f"{row_source.id}:{param.id}"
        if (
            expression.key != "catalog_param_default"
            or expression.source_ref != expected_source_ref
            or param.default is None
        ):
            raise VerificationError("endpoint environment value is unavailable")
        return ResolvedValueExpression(
            value=param.default,
            fact_value=_fact_value_for_endpoint_default(param),
        )
    return _resolve_omittable_expression(
        expression,
        bindings=bindings,
        parameters=parameters,
    )


def _resolve_expression(expression, *, bindings: BindingSet):
    try:
        return resolve_value_expression(expression, bindings=bindings)
    except AnswerProgramContractError as exc:
        raise VerificationError(f"{exc.code}: {exc}") from exc


def _resolve_omittable_expression(
    expression,
    *,
    bindings: BindingSet,
    parameters: tuple[ParameterDeclaration, ...],
):
    if not isinstance(expression, ParameterRef):
        return _resolve_expression(expression, bindings=bindings)
    if bindings.get(expression.parameter_id) is not None:
        return _resolve_expression(expression, bindings=bindings)
    declaration = next(
        (
            parameter
            for parameter in parameters
            if parameter.id == expression.parameter_id
        ),
        None,
    )
    if declaration is not None and not declaration.required:
        return None
    return _resolve_expression(expression, bindings=bindings)


def _fact_value_for_endpoint_default(param: RowSourceParam) -> FactValue:
    value = param.default
    value_id = f"catalog-default.{param.id}"
    match param.type:
        case RowSourceValueType.BOOLEAN:
            return FactValue.literal(
                id=value_id,
                literal_type=LiteralType.BOOLEAN,
                value=str(value).lower(),
            )
        case (
            RowSourceValueType.INTEGER
            | RowSourceValueType.NUMBER
            | RowSourceValueType.DOUBLE
            | RowSourceValueType.FLOAT
        ):
            return FactValue.literal(
                id=value_id,
                literal_type=LiteralType.NUMBER,
                value=str(value),
            )
        case RowSourceValueType.ARRAY | RowSourceValueType.LIST:
            return FactValue.literal(
                id=value_id,
                literal_type=LiteralType.STRING,
                value=canonical_runtime_json(value),
            )
        case RowSourceValueType.JSON | RowSourceValueType.OBJECT:
            return FactValue.literal(
                id=value_id,
                literal_type=LiteralType.STRING,
                value=canonical_runtime_json(value),
            )
        case (
            RowSourceValueType.CHOICE
            | RowSourceValueType.DATE
            | RowSourceValueType.DATETIME
            | RowSourceValueType.DECIMAL
            | RowSourceValueType.DURATION
            | RowSourceValueType.PATH
            | RowSourceValueType.PK
            | RowSourceValueType.STRING
            | RowSourceValueType.TIME
            | RowSourceValueType.UUID
        ):
            return FactValue.literal(
                id=value_id,
                literal_type=LiteralType.STRING,
                value=str(value),
            )
        case RowSourceValueType.ANY | RowSourceValueType.UNKNOWN:
            return FactValue.literal(
                id=value_id,
                literal_type=LiteralType.STRING,
                value=canonical_runtime_json(value),
            )
        case _:
            assert_never(param.type)


def _append_endpoint_arg(
    endpoint_args: list[ResolvedEndpointArg],
    *,
    endpoint_arg_targets: set[tuple[str, str]],
    arg: ResolvedEndpointArg,
) -> None:
    target = (arg.relation_id, arg.param_ref)
    if target in endpoint_arg_targets:
        existing = next(
            item
            for item in endpoint_args
            if item.relation_id == arg.relation_id and item.param_ref == arg.param_ref
        )
        if existing.value == arg.value:
            merged_refs = _dedupe_refs((*existing.proof_refs, *arg.proof_refs))
            if merged_refs != existing.proof_refs:
                endpoint_args[endpoint_args.index(existing)] = ResolvedEndpointArg(
                    relation_id=existing.relation_id,
                    read_id=existing.read_id,
                    param_ref=existing.param_ref,
                    value=existing.value,
                    proof_refs=merged_refs,
                )
            return
        raise VerificationError(
            f"duplicate endpoint argument {arg.param_ref} on {arg.relation_id}"
        )
    endpoint_arg_targets.add(target)
    endpoint_args.append(arg)


def _relation_row_sources(
    relations: tuple[Relation, ...],
    row_sources: RowSourceCatalog,
) -> dict[str, RowSource]:
    relation_sources: dict[str, RowSource] = {}
    for relation in relations:
        if relation.source.kind not in {
            SourceKind.API_READ,
            SourceKind.GENERATED_CALENDAR,
            SourceKind.MEMORY_READ,
        }:
            continue
        try:
            relation_sources[relation.id] = row_source_for_relation(
                relation,
                row_sources=row_sources,
            )
        except KeyError as exc:
            raise VerificationError(
                f"relation {relation.id} references unknown source"
            ) from exc
    return relation_sources


def _row_source_for_relation(
    relation_id: str,
    relation_sources: dict[str, RowSource],
) -> RowSource:
    row_source = relation_sources.get(relation_id)
    if row_source is None:
        raise VerificationError(f"unknown plan relation {relation_id}")
    return row_source


def _filter_value(
    value: FactValue,
    component: ValueComponent | TimeComponent,
) -> Any:
    _require_filter_value(value)
    return value_component(value, component)


def _require_filter_value(value: FactValue) -> None:
    if value.payload.row_filter_error:
        raise VerificationError(f"{value.payload.row_filter_error}: {value.id}")


def _require_relation_field(
    relations: tuple[Relation, ...],
    relation_id: str,
    field_id: str,
) -> None:
    relation = _relation(relations, relation_id)
    _require_bound_relation_field(relation, field_id)


def _require_bound_relation_field(relation: Relation, field_id: str) -> None:
    if any(field.field_id == field_id for field in relation.fields):
        return
    raise VerificationError(f"unknown field {field_id} on relation {relation.id}")


def _relation(relations: tuple[Relation, ...], relation_id: str) -> Relation:
    for item in relations:
        if item.id == relation_id:
            return item
    raise VerificationError(f"unknown relation {relation_id}")


def _dedupe_refs(refs: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(ref for ref in refs if ref))
