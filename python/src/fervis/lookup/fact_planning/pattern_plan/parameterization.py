"""Close compiler-front source values into canonical answer-program relations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any
from typing_extensions import assert_never

from fervis.lookup.answer_program.compiler_inputs import CompilerInputContext
from fervis.lookup.source_binding.compiler_ir import (
    DraftEndpointParamBinding,
    DraftRelationSource,
    DraftRelationSourceAppliedFilter,
    DraftRelationSourcePopulationChoice,
    RelationInputOrigin,
)
from fervis.lookup.answer_program.contracts import (
    BindingProvenance,
    BindingProvenanceKind,
    BindingSet,
    ParameterBinding,
    ParameterDeclaration,
    ParameterRole,
    ProgramInputs,
    canonical_fact_value,
    parameter_value_type,
)
from fervis.lookup.answer_program.inputs import (
    CompiledProgramInputs,
    compile_program_inputs,
)
from fervis.lookup.answer_program.relations import (
    EndpointParamBinding,
    PopulationChoiceControllerKind,
    PopulationCoverageClaim,
    Relation,
    RelationField,
    RelationSource,
    RelationSourcePopulationChoice,
    SourceKind,
)
from fervis.lookup.answer_program.operations import (
    FilterSpec,
    Operation,
    Predicate,
    PredicateOperator,
)
from fervis.lookup.fact_plan.row_sources import (
    CALENDAR_END_PARAM_ID,
    CALENDAR_START_PARAM_ID,
)
from fervis.lookup.answer_program.values import (
    ConstantRef,
    FactValue,
    LiteralType,
    ParameterRef,
    TimeComponent,
    ValueKind,
)
from fervis.lookup.answer_program.expressions import FieldRef


@dataclass(frozen=True)
class ParameterizedRelation:
    relation: Relation
    output_relation_id: str
    operations: tuple[Operation, ...] = ()


def parameterize_relation(
    *,
    relation_id: str,
    source: DraftRelationSource,
    fields: tuple[RelationField, ...],
    input_context: CompilerInputContext,
    parameters: dict[str, ParameterDeclaration],
    bindings: dict[str, ParameterBinding],
) -> ParameterizedRelation:
    source_param_bindings = source.param_bindings
    if source.kind == SourceKind.GENERATED_CALENDAR and not source_param_bindings:
        source_param_bindings = _calendar_param_bindings(input_context)
    population_choices = tuple(
        _parameterize_population_choice(
            choice,
            source=source,
            parameters=parameters,
            bindings=bindings,
        )
        for choice in source.population_choices
    )
    returned_predicates = (
        *(
            predicate
            for item in source.applied_filters
            for predicate in _parameterize_applied_filter(
                item,
                input_context=input_context,
            )
        ),
        *(
            Predicate(
                left=FieldRef(choice.field_id),
                operator=PredicateOperator.IN,
                right=parameterized.selection_expr,
            )
            for choice, parameterized in zip(
                source.population_choices,
                population_choices,
                strict=True,
            )
            if choice.controller_kind is PopulationChoiceControllerKind.ROW_PREDICATE
            and choice.excluded_values
        ),
    )
    returned_filter_proof_refs = tuple(
        dict.fromkeys(
            (
                *(ref for item in source.applied_filters for ref in item.proof_refs),
                *(
                    ref
                    for item in source.population_choices
                    if item.controller_kind
                    is PopulationChoiceControllerKind.ROW_PREDICATE
                    and item.excluded_values
                    for ref in item.proof_refs
                ),
            )
        )
    )
    returned_filter_claims, source_claims = _partition_population_claims(
        source.population_coverage_claims,
        returned_filter_proof_refs=returned_filter_proof_refs,
    )
    _require_returned_predicate_fields(returned_predicates, fields=fields)
    source_relation_id = (
        f"{relation_id}__source" if returned_predicates else relation_id
    )
    relation = Relation(
        id=source_relation_id,
        source=RelationSource(
            kind=source.kind,
            read_id=source.read_id,
            row_source_id=source.row_source_id,
            calendar_id=source.calendar_id,
            memory_relation_id=source.memory_relation_id,
            param_bindings=tuple(
                _parameterize_endpoint_binding(
                    binding,
                    relation_id=relation_id,
                    source=source,
                    input_context=input_context,
                    parameters=parameters,
                    bindings=bindings,
                )
                for binding in source_param_bindings
            ),
            population_choices=population_choices,
            population_binding=source.population_binding,
            population_coverage_claims=source_claims,
            proof_refs=source.proof_refs,
        ),
        fields=fields,
    )
    operations: list[Operation] = []
    input_relation_id = source_relation_id
    for index, predicate in enumerate(returned_predicates, start=1):
        output_relation_id = (
            relation_id
            if index == len(returned_predicates)
            else f"{relation_id}__filter_{index}"
        )
        operations.append(
            Operation(
                id=f"{relation_id}__filter_operation_{index}",
                spec=FilterSpec(
                    input_relation=input_relation_id,
                    predicate=predicate,
                    proof_refs=returned_filter_proof_refs,
                    population_coverage_claims=(
                        returned_filter_claims
                        if index == len(returned_predicates)
                        else ()
                    ),
                ),
                output_relation=output_relation_id,
            )
        )
        input_relation_id = output_relation_id
    return ParameterizedRelation(
        relation=relation,
        output_relation_id=relation_id,
        operations=tuple(operations),
    )


def _partition_population_claims(
    claims: tuple[PopulationCoverageClaim, ...],
    *,
    returned_filter_proof_refs: tuple[str, ...],
) -> tuple[tuple[PopulationCoverageClaim, ...], tuple[PopulationCoverageClaim, ...]]:
    filter_refs = set(returned_filter_proof_refs)
    returned = tuple(
        claim
        for claim in claims
        if set(claim.proof_refs).intersection(filter_refs)
    )
    return returned, tuple(claim for claim in claims if claim not in returned)


def _require_returned_predicate_fields(
    predicates: tuple[Predicate, ...],
    *,
    fields: tuple[RelationField, ...],
) -> None:
    field_ids = {field.field_id for field in fields}
    for predicate in predicates:
        if (
            not isinstance(predicate.left, FieldRef)
            or predicate.left.field_id not in field_ids
        ):
            raise ValueError("returned-row filter requires a bound relation field")


def compiled_program_inputs(
    *,
    parameters: dict[str, ParameterDeclaration],
    bindings: dict[str, ParameterBinding],
) -> CompiledProgramInputs:
    return compile_program_inputs(
        ProgramInputs(
            parameters=tuple(parameters.values()),
            bindings=BindingSet.from_bindings(tuple(bindings.values())),
        )
    )


def _calendar_param_bindings(
    input_context: CompilerInputContext,
) -> tuple[DraftEndpointParamBinding, ...]:
    time_value_ids = tuple(
        value_id
        for value_id in input_context.expressions_by_value_id
        if _input_fact_value(value_id, input_context=input_context).kind
        == ValueKind.TIME
    )
    if len(time_value_ids) != 1:
        raise ValueError("calendar source requires exactly one grounded time input")
    value_id = time_value_ids[0]
    return (
        DraftEndpointParamBinding(
            param_id=CALENDAR_START_PARAM_ID,
            value_expr=input_context.expression_for_value(
                value_id,
                component=TimeComponent.START.value,
            ),
        ),
        DraftEndpointParamBinding(
            param_id=CALENDAR_END_PARAM_ID,
            value_expr=input_context.expression_for_value(
                value_id,
                component=TimeComponent.END.value,
            ),
        ),
    )


def _input_fact_value(
    value_id: str,
    *,
    input_context: CompilerInputContext,
) -> FactValue:
    expression = input_context.expressions_by_value_id[value_id]
    if isinstance(expression, ConstantRef):
        return expression.value
    if isinstance(expression, ParameterRef):
        binding = input_context.program_inputs.bindings.get(expression.parameter_id)
        if binding is None:
            raise ValueError(f"unbound compiler input {expression.parameter_id}")
        return binding.value
    raise ValueError(f"compiler input {value_id} has no materialized fact value")


def _parameterize_endpoint_binding(
    binding: DraftEndpointParamBinding,
    *,
    relation_id: str,
    source: DraftRelationSource,
    input_context: CompilerInputContext,
    parameters: dict[str, ParameterDeclaration],
    bindings: dict[str, ParameterBinding],
) -> EndpointParamBinding:
    if binding.value_expr is not None:
        expression = binding.value_expr
    elif binding.origin_kind == RelationInputOrigin.QUESTION_INPUT:
        if not binding.value_id:
            raise ValueError("question-input endpoint binding requires value id")
        expression = input_context.expression_for_value(
            binding.value_id,
            component=binding.value_component,
            item_index=binding.value_item_index,
        )
    elif binding.origin_kind == RelationInputOrigin.SEMANTIC_CONTROL:
        parameter_id = binding.parameter_id or (
            f"semantic.{source.read_id}.{binding.param_id}"
        )
        expression = replace(
            _add_parameter(
                parameter_id=parameter_id,
                role=ParameterRole.SEMANTIC_CONTROL,
                value=_fact_value(
                    value_id=f"binding.{parameter_id}",
                    value=binding.value,
                    proof_refs=binding.proof_refs,
                ),
                proof_refs=binding.proof_refs,
                parameters=parameters,
                bindings=bindings,
            ),
            item_index=binding.value_item_index,
        )
    elif binding.origin_kind == RelationInputOrigin.CONTEXT_CONSTANT:
        expression = ConstantRef(
            constant_id=f"source.{relation_id}.{binding.param_id}",
            version_ref="source-binding@1",
            value=_fact_value(
                value_id=f"constant.{relation_id}.{binding.param_id}",
                value=binding.value,
                proof_refs=binding.proof_refs,
            ),
        )
    else:
        assert_never(binding.origin_kind)
    return EndpointParamBinding(
        param_id=binding.param_id,
        value_expr=expression,
        proof_refs=binding.proof_refs,
    )


def _parameterize_applied_filter(
    source_filter: DraftRelationSourceAppliedFilter,
    *,
    input_context: CompilerInputContext,
) -> tuple[Predicate, ...]:
    if source_filter.value_expr is not None:
        expression = source_filter.value_expr
    else:
        value_id = source_filter.value_id or _value_id_for_known_input(
            source_filter.known_input_id,
            input_context=input_context,
        )
        if not value_id:
            raise ValueError("applied filter requires explicit input value")
        expression = input_context.expression_for_value(
            value_id,
            component=source_filter.value_component,
        )
    return tuple(
        Predicate(
            left=FieldRef(field_id),
            operator=PredicateOperator(source_filter.operator),
            right=expression,
        )
        for field_id in source_filter.predicate_field_ids
    )


def _parameterize_population_choice(
    choice: DraftRelationSourcePopulationChoice,
    *,
    source: DraftRelationSource,
    parameters: dict[str, ParameterDeclaration],
    bindings: dict[str, ParameterBinding],
) -> RelationSourcePopulationChoice:
    if choice.selection_expr is not None:
        return RelationSourcePopulationChoice(
            controller_kind=choice.controller_kind,
            controller_id=choice.controller_id,
            field_id=choice.field_id,
            requested_fact_ids=choice.requested_fact_ids,
            selection_expr=choice.selection_expr,
            allowed_values=choice.allowed_values,
            proof_refs=choice.proof_refs,
            review_scope_decisions=choice.review_scope_decisions,
        )
    parameter_id = choice.parameter_id or (
        f"semantic.{source.read_id}.{choice.controller_kind.value}."
        f"{choice.controller_id}"
    )
    allowed_values = tuple(
        dict.fromkeys((*choice.included_values, *choice.excluded_values))
    )
    expression = _add_parameter(
        parameter_id=parameter_id,
        role=ParameterRole.SEMANTIC_CONTROL,
        value=FactValue.string_set(
            id=f"binding.{parameter_id}",
            values=choice.included_values,
            proof_refs=choice.proof_refs,
        ),
        proof_refs=choice.proof_refs,
        allowed_values=allowed_values,
        semantic_control_ref=(f"{choice.controller_kind.value}:{choice.controller_id}"),
        parameters=parameters,
        bindings=bindings,
    )
    return RelationSourcePopulationChoice(
        controller_kind=choice.controller_kind,
        controller_id=choice.controller_id,
        field_id=choice.field_id,
        requested_fact_ids=choice.requested_fact_ids,
        selection_expr=expression,
        allowed_values=allowed_values,
        proof_refs=choice.proof_refs,
        review_scope_decisions=choice.review_scope_decisions,
    )


def _add_parameter(
    *,
    parameter_id: str,
    role: ParameterRole,
    value: FactValue,
    proof_refs: tuple[str, ...],
    parameters: dict[str, ParameterDeclaration],
    bindings: dict[str, ParameterBinding],
    allowed_values: tuple[str, ...] = (),
    semantic_control_ref: str = "",
) -> ParameterRef:
    declaration = ParameterDeclaration(
        id=parameter_id,
        role=role,
        value_type=parameter_value_type(value),
        allowed_values=allowed_values,
        semantic_control_ref=semantic_control_ref,
    )
    parameter_binding = ParameterBinding(
        parameter_id=parameter_id,
        value=value,
        provenance=BindingProvenance(
            kind=(
                BindingProvenanceKind.SEMANTIC_CHOICE
                if role == ParameterRole.SEMANTIC_CONTROL
                else BindingProvenanceKind.PLAN_CHOICE
            ),
            refs=proof_refs,
        ),
    )
    existing_declaration = parameters.get(parameter_id)
    existing_binding = bindings.get(parameter_id)
    if existing_declaration is not None:
        if (
            existing_declaration.role != declaration.role
            or existing_declaration.value_type != declaration.value_type
            or existing_declaration.required != declaration.required
        ):
            raise ValueError(f"conflicting parameter declaration {parameter_id}")
        if (
            existing_declaration.allowed_values
            and declaration.allowed_values
            and existing_declaration.allowed_values != declaration.allowed_values
        ):
            raise ValueError(f"conflicting parameter allowed values {parameter_id}")
        if (
            existing_declaration.semantic_control_ref
            and declaration.semantic_control_ref
            and existing_declaration.semantic_control_ref
            != declaration.semantic_control_ref
        ):
            raise ValueError(f"conflicting parameter semantic control {parameter_id}")
    if existing_binding is not None and canonical_fact_value(
        existing_binding.value
    ) != canonical_fact_value(value):
        raise ValueError(f"conflicting parameter binding {parameter_id}")
    parameters.setdefault(parameter_id, declaration)
    bindings.setdefault(parameter_id, parameter_binding)
    return ParameterRef(parameter_id=parameter_id)


def _value_id_for_known_input(
    known_input_id: str,
    *,
    input_context: CompilerInputContext,
) -> str:
    if not known_input_id:
        return ""
    return next(
        (
            value_id
            for value_id, expression in input_context.expressions_by_value_id.items()
            if isinstance(expression, ParameterRef)
            and expression.parameter_id == f"question.{known_input_id}"
        ),
        "",
    )


def _fact_value(
    *,
    value_id: str,
    value: Any,
    proof_refs: tuple[str, ...] = (),
) -> FactValue:
    if isinstance(value, bool):
        return FactValue.literal(
            id=value_id,
            literal_type=LiteralType.BOOLEAN,
            value=str(value).lower(),
            proof_refs=proof_refs,
        )
    if isinstance(value, (int, float)):
        return FactValue.literal(
            id=value_id,
            literal_type=LiteralType.NUMBER,
            value=str(value),
            proof_refs=proof_refs,
        )
    if isinstance(value, tuple):
        return FactValue.string_set(
            id=value_id,
            values=tuple(str(item) for item in value),
            proof_refs=proof_refs,
        )
    if value is None:
        raise ValueError(f"{value_id} has no value")
    return FactValue.literal(
        id=value_id,
        literal_type=LiteralType.STRING,
        value=str(value),
        proof_refs=proof_refs,
    )
