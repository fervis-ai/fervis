"""Deterministic compilation for typed value uses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fervis.lookup.relation_catalog.model import RelationCatalog
from fervis.lookup.plan_execution.errors import VerificationError
from fervis.lookup.fact_planning.grounded_params import (
    unique_grounded_param_values,
)
from fervis.lookup.fact_plan.relations import (
    PopulationChoiceControllerKind,
    Relation,
    RelationSourceAppliedFilter,
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
    build_row_source_catalog,
    row_source_for_relation,
    row_source_param_evidence_ref,
)
from fervis.lookup.fact_plan.values import (
    ScalarInputUse,
    FactValue,
    IdentityValuePayload,
    IdentitySetValuePayload,
    LiteralType,
    LiteralValuePayload,
    RankLimitUse,
    RowFilterUse,
    TimeComponent,
    ValueComponent,
    ValueFilterOperator,
    ValueKind,
    ValueUse,
    known_input_id_for_value,
)
from fervis.lookup.fact_planning.value_components import value_component


@dataclass(frozen=True)
class CompiledEndpointArg:
    relation_id: str
    read_id: str
    param_ref: str
    value: Any
    proof_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompiledRowFilter:
    relation_id: str
    field_id: str
    operator: ValueFilterOperator
    value: Any
    proof_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompiledScalarInput:
    operation_id: str
    input_id: str
    value: Any
    source_refs: tuple[str, ...] = ()
    proof_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompiledRankLimit:
    operation_id: str
    value: Any
    proof_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompiledPopulationChoice:
    relation_id: str
    controller_kind: PopulationChoiceControllerKind
    controller_id: str
    field_id: str
    included_values: tuple[str, ...]
    excluded_values: tuple[str, ...]
    proof_refs: tuple[str, ...] = ()
    review_scope_decisions: tuple[RelationSourceReviewScopeDecision, ...] = ()


@dataclass(frozen=True)
class CompiledValueUses:
    endpoint_args: tuple[CompiledEndpointArg, ...] = ()
    row_filters: tuple[CompiledRowFilter, ...] = ()
    scalar_inputs: tuple[CompiledScalarInput, ...] = ()
    rank_limits: tuple[CompiledRankLimit, ...] = ()
    population_choices: tuple[CompiledPopulationChoice, ...] = ()


def compile_value_uses(
    *,
    values: tuple[FactValue, ...],
    value_uses: tuple[ValueUse, ...],
    catalog: RelationCatalog,
    relations: tuple[Relation, ...] = (),
    row_sources: RowSourceCatalog | None = None,
    grounded_input_uses: tuple[Any, ...] = (),
) -> CompiledValueUses:
    values_by_id = {item.id: item for item in values}
    row_source_catalog = row_sources or build_row_source_catalog(catalog)
    relation_sources = _relation_row_sources(
        relations,
        row_source_catalog,
    )

    endpoint_args: list[CompiledEndpointArg] = []
    row_filters: list[CompiledRowFilter] = []
    scalar_inputs: list[CompiledScalarInput] = []
    rank_limits: list[CompiledRankLimit] = []
    population_choices: list[CompiledPopulationChoice] = []
    endpoint_arg_targets: set[tuple[str, str]] = set()

    for use in value_uses:
        value = values_by_id.get(use.value_id)
        if value is None:
            raise VerificationError(f"value use {use.id} references unknown value")
        target = use.target
        if isinstance(target, RowFilterUse):
            _require_relation_field(relations, target.relation_id, target.field_id)
            row_filters.append(
                CompiledRowFilter(
                    relation_id=target.relation_id,
                    field_id=target.field_id,
                    operator=target.operator,
                    value=_filter_value(value, target.value_component),
                    proof_refs=tuple(value.proof_refs),
                )
            )
        elif isinstance(target, ScalarInputUse):
            scalar_inputs.append(
                CompiledScalarInput(
                    operation_id=target.operation_id,
                    input_id=target.input_id,
                    value=value_component(value, ValueComponent.VALUE),
                    source_refs=tuple(value.source_refs),
                    proof_refs=tuple(value.proof_refs),
                )
            )
        elif isinstance(target, RankLimitUse):
            rank_limits.append(
                CompiledRankLimit(
                    operation_id=target.operation_id,
                    value=value_component(value, ValueComponent.VALUE),
                    proof_refs=tuple(value.proof_refs),
                )
            )
        else:
            raise VerificationError(f"value use {use.id} has unsupported target")

    _append_relation_source_row_filters(
        row_filters,
        values_by_id=values_by_id,
        relations=relations,
    )
    _append_grounded_endpoint_args(
        endpoint_args,
        endpoint_arg_targets=endpoint_arg_targets,
        values_by_id=values_by_id,
        relation_sources=relation_sources,
        grounded_input_uses=grounded_input_uses,
    )
    _append_relation_source_endpoint_args(
        endpoint_args,
        endpoint_arg_targets=endpoint_arg_targets,
        relations=relations,
        relation_sources=relation_sources,
    )
    _append_relation_source_population_choices(
        population_choices,
        relations=relations,
    )
    _append_default_endpoint_args(
        endpoint_args,
        endpoint_arg_targets=endpoint_arg_targets,
        relation_sources=relation_sources,
    )

    return CompiledValueUses(
        endpoint_args=tuple(endpoint_args),
        row_filters=tuple(row_filters),
        scalar_inputs=tuple(scalar_inputs),
        rank_limits=tuple(rank_limits),
        population_choices=tuple(population_choices),
    )


def _append_relation_source_row_filters(
    row_filters: list[CompiledRowFilter],
    *,
    values_by_id: dict[str, FactValue],
    relations: tuple[Relation, ...],
) -> None:
    for relation in relations:
        for source_row_filter in relation.source.row_filters:
            _require_bound_relation_field(relation, source_row_filter.field_id)
            row_filters.append(
                CompiledRowFilter(
                    relation_id=relation.id,
                    field_id=source_row_filter.field_id,
                    operator=_source_row_filter_operator(source_row_filter),
                    value=tuple(source_row_filter.values),
                    proof_refs=tuple(source_row_filter.proof_refs),
                )
            )
        for source_filter in relation.source.applied_filters:
            value = _source_filter_value(
                source_filter,
                values_by_id=values_by_id,
                relation_id=relation.id,
            )
            for field_id in source_filter.predicate_field_ids:
                _require_bound_relation_field(relation, field_id)
                row_filters.append(
                    CompiledRowFilter(
                        relation_id=relation.id,
                        field_id=field_id,
                        operator=ValueFilterOperator.EQUALS,
                        value=_filter_value(value, ValueComponent.VALUE),
                        proof_refs=tuple(value.proof_refs),
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


def _append_relation_source_population_choices(
    population_choices: list[CompiledPopulationChoice],
    *,
    relations: tuple[Relation, ...],
) -> None:
    for relation in relations:
        for choice in relation.source.population_choices:
            population_choices.append(
                _compiled_population_choice(
                    relation_id=relation.id,
                    choice=choice,
                )
            )


def _compiled_population_choice(
    *, relation_id: str, choice: RelationSourcePopulationChoice
) -> CompiledPopulationChoice:
    return CompiledPopulationChoice(
        relation_id=relation_id,
        controller_kind=choice.controller_kind,
        controller_id=choice.controller_id,
        field_id=choice.field_id,
        included_values=choice.included_values,
        excluded_values=choice.excluded_values,
        proof_refs=choice.proof_refs,
        review_scope_decisions=choice.review_scope_decisions,
    )


def _source_filter_value(
    source_filter: RelationSourceAppliedFilter,
    *,
    values_by_id: dict[str, FactValue],
    relation_id: str,
) -> FactValue:
    candidates = tuple(
        value
        for value in values_by_id.values()
        if known_input_id_for_value(value) == source_filter.known_input_id
        and _value_matches_source_filter(value, source_filter=source_filter)
    )
    if len(candidates) != 1:
        raise VerificationError(
            f"relation {relation_id} applied filter cannot resolve value"
        )
    return candidates[0]


def _value_matches_source_filter(
    value: FactValue,
    *,
    source_filter: RelationSourceAppliedFilter,
) -> bool:
    if source_filter.value_kind and value.kind.value != source_filter.value_kind:
        return False
    if not source_filter.identity_type:
        return True
    if isinstance(value.payload, IdentityValuePayload):
        return value.payload.identity_type == source_filter.identity_type
    if isinstance(value.payload, IdentitySetValuePayload):
        return value.payload.identity_type == source_filter.identity_type
    return False


def _append_grounded_endpoint_args(
    endpoint_args: list[CompiledEndpointArg],
    *,
    endpoint_arg_targets: set[tuple[str, str]],
    values_by_id: dict[str, FactValue],
    relation_sources: dict[str, RowSource],
    grounded_input_uses: tuple[Any, ...],
) -> None:
    grounded_params = unique_grounded_param_values(
        values=tuple(values_by_id.values()),
        grounded_input_uses=grounded_input_uses,
    )
    for relation_id, row_source in relation_sources.items():
        if row_source.kind not in {
            RowSourceKind.API_READ,
            RowSourceKind.GENERATED_CALENDAR,
        }:
            continue
        for param in row_source.params:
            grounded_param = grounded_params.get((row_source.id, param.id))
            if grounded_param is None:
                continue
            _append_endpoint_arg(
                endpoint_args,
                endpoint_arg_targets=endpoint_arg_targets,
                arg=CompiledEndpointArg(
                    relation_id=relation_id,
                    read_id=row_source.read_id,
                    param_ref=param.param_ref,
                    value=grounded_param.value,
                    proof_refs=grounded_param.proof_refs,
                ),
            )


def _append_relation_source_endpoint_args(
    endpoint_args: list[CompiledEndpointArg],
    *,
    endpoint_arg_targets: set[tuple[str, str]],
    relations: tuple[Relation, ...],
    relation_sources: dict[str, RowSource],
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
            if param.choices and binding.value not in param.choices:
                raise VerificationError(
                    f"relation {relation.id} param binding has unknown choice"
                )
            _append_endpoint_arg(
                endpoint_args,
                endpoint_arg_targets=endpoint_arg_targets,
                arg=CompiledEndpointArg(
                    relation_id=relation.id,
                    read_id=row_source.read_id,
                    param_ref=param.param_ref,
                    value=_endpoint_arg_value(param=param, value=binding.value),
                    proof_refs=(
                        *binding.proof_refs,
                        row_source_param_evidence_ref(
                            row_source_id=row_source.id,
                            param_id=param.id,
                        ),
                    ),
                ),
            )


def _append_default_endpoint_args(
    endpoint_args: list[CompiledEndpointArg],
    *,
    endpoint_arg_targets: set[tuple[str, str]],
    relation_sources: dict[str, RowSource],
) -> None:
    for relation_id, row_source in relation_sources.items():
        if row_source.kind not in {
            RowSourceKind.API_READ,
            RowSourceKind.GENERATED_CALENDAR,
        }:
            continue
        for param in row_source.params:
            if param.default is None:
                continue
            if (relation_id, param.param_ref) in endpoint_arg_targets:
                continue
            _append_endpoint_arg(
                endpoint_args,
                endpoint_arg_targets=endpoint_arg_targets,
                arg=CompiledEndpointArg(
                    relation_id=relation_id,
                    read_id=row_source.read_id,
                    param_ref=param.param_ref,
                    value=_endpoint_arg_value(param=param, value=param.default),
                    proof_refs=(
                        row_source_param_evidence_ref(
                            row_source_id=row_source.id,
                            param_id=param.id,
                        ),
                    ),
                ),
            )


def _endpoint_arg_value(*, param: RowSourceParam, value: Any) -> Any:
    if isinstance(value, tuple):
        return tuple(_endpoint_arg_value(param=param, value=item) for item in value)
    if param.type == "boolean" and value in {"true", "false"}:
        return value == "true"
    return value


def _append_endpoint_arg(
    endpoint_args: list[CompiledEndpointArg],
    *,
    endpoint_arg_targets: set[tuple[str, str]],
    arg: CompiledEndpointArg,
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
                endpoint_args[endpoint_args.index(existing)] = CompiledEndpointArg(
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
    if value.kind == ValueKind.IDENTITY_SET and isinstance(
        value.payload, IdentitySetValuePayload
    ):
        raise VerificationError(f"identity set {value.id} cannot be row filter")
    if value.kind == ValueKind.LITERAL and isinstance(
        value.payload, LiteralValuePayload
    ):
        if value.payload.literal_type == LiteralType.STRING:
            raise VerificationError(f"literal value {value.id} cannot be row filter")
        raise VerificationError(f"literal value {value.id} requires scalar sink")
    return value_component(value, component)


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
