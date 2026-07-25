"""Deterministically lower verified semantic facts into one AnswerProgram."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TypeVar

from fervis.lookup.answer_program.compiler_inputs import CompilerInputContext
from fervis.lookup.answer_program.expressions import (
    BinaryExpression,
    Expression,
    FieldRef,
    FunctionExpression,
    ExpressionFunction,
    UnaryExpression,
)
from fervis.lookup.answer_program.model import (
    AnswerProgram,
    FactFulfillment,
    RelationGuaranteeDeclaration,
)
from fervis.lookup.answer_program.compatibility import build_program_compatibility
from fervis.lookup.answer_program.operations import (
    AggregateSpec,
    AggregationFunction,
    AggregationSpec,
    KeepAll,
    JoinKey,
    JoinSpec,
    Operation,
    OrderSpec,
    ProjectSpec,
    ProjectToKeySpec,
    NamedExpression,
    SortDirection,
    SortKey,
    Take,
    UnionSpec,
    ComputeSpec,
)
from fervis.lookup.answer_program.relations import (
    EndpointParamBinding,
    FieldBindingRole,
    Relation,
    RelationField,
    RelationSource,
    SourceKind,
)
from fervis.lookup.answer_program.result_projection import (
    EntityKeyProjection,
    EntityKeyProjectionComponent,
    RelationResultOutput,
    ResultProjection,
    ScalarResultOutput,
)
from fervis.lookup.answer_program.values import (
    ANCHOR_TIMEZONE_REF,
    ConstantRef,
    EnvironmentRef,
    FactValue,
    LiteralType,
    ParameterRef,
    NodeOutputRef,
    ValueProjectionKind,
    BindingSet,
)
from fervis.lookup.expression_operators import (
    ExpressionBinaryOperator,
    ExpressionUnaryOperator,
)
from fervis.lookup.qualification import QualificationGuarantee, qualification_dnf
from fervis.lookup.available_sources import SourceChoiceSurfaceKind
from fervis.lookup.relation_catalog.row_sources import (
    RowSource,
    RowSourceCatalog,
    RowSourceKind,
)
from fervis.lookup.question_contract import (
    RequestedFact,
    RequestedFactSemanticIndex,
    Singleton,
)
from fervis.lookup.question_contract import (
    Aggregate,
    Arithmetic,
    BooleanComposition,
    BooleanCompositionOperator,
    Comparison,
    ExpressionNode,
    FactTerm,
    FactLocalKind,
    FactLocalRef,
    FirstRankWithTies,
    NullCheck,
    Quantifier,
    Quantify,
    TemporalBucket,
)
from fervis.lookup.semantic_types import IdentifierType
from fervis.lookup.source_binding.param_binding_sets import (
    ParamBindingSetAlternatives,
    RelationInputOrigin,
    alternate_param_binding_sets,
    combine_param_binding_sets,
    equivalent_param_binding_sets,
    intersect_param_binding_sets,
    merge_equivalent_param_binding_sets,
    parameter_binding_sets,
)
from fervis.lookup.source_binding.param_values import fact_value_parameter_projection
from fervis.lookup.source_binding import (
    AssociationRealizationKind,
    FactRealization,
    SetRealization,
    SourceBindingPlan,
    SourceMechanicKind,
    SubjectSurfaceReview,
    VerifiedSourceStrategy,
)

from .inputs import semantic_compiler_inputs
from .model import FactCompilationResult


T = TypeVar("T")


@dataclass
class _ProgramBuilder:
    verified: VerifiedSourceStrategy
    inputs: CompilerInputContext
    relations: list
    operations: list[Operation]
    fulfillments: list[FactFulfillment]
    relation_outputs: list[RelationResultOutput]
    scalar_outputs: list[ScalarResultOutput]
    fields_by_relation: dict[str, tuple[RelationField, ...]]
    grain_by_relation: dict[str, tuple[str, ...]]
    scalar_aggregate_outputs: dict[FactLocalRef, NodeOutputRef]
    relation_guarantees: list[RelationGuaranteeDeclaration]
    projected_fields: dict[FactLocalRef, str]


def compile_verified_source_strategy(
    verified: VerifiedSourceStrategy,
) -> FactCompilationResult:
    """Lower one fully verified requested fact without another model decision."""

    inputs = semantic_compiler_inputs(verified)
    builder = _ProgramBuilder(
        verified,
        inputs,
        [],
        [],
        [],
        [],
        [],
        {},
        {},
        {},
        [],
        {},
    )
    _compile_requested_fact(builder)
    program = AnswerProgram(
        inputs=tuple(verified.request.index.input_by_ref.values()),
        input_denotations=tuple(
            verified.request.index.input_denotation_by_ref.values()
        ),
        fact_template=(verified.request.index.requested_fact,),
        fulfillment=tuple(builder.fulfillments),
        relation_guarantees=tuple(builder.relation_guarantees),
        parameters=inputs.program_inputs.parameters,
        relations=tuple(builder.relations),
        operations=tuple(builder.operations),
        result_projection=ResultProjection(
            relation_outputs=tuple(builder.relation_outputs),
            scalar_outputs=tuple(builder.scalar_outputs),
        ),
    )
    program = replace(
        program,
        compatibility=build_program_compatibility(
            program,
            row_sources=RowSourceCatalog(verified.request.source_catalog.sources),
        ),
    )
    return FactCompilationResult(
        answer_program=program,
        initial_bindings=inputs.program_inputs.bindings,
    )


def compile_verified_source_strategies(
    verified_strategies: tuple[VerifiedSourceStrategy, ...],
) -> FactCompilationResult:
    """Compile atomic requested facts into one immutable executable program."""

    if not verified_strategies:
        raise ValueError("fact compilation requires a verified source strategy")
    compiled = tuple(
        compile_verified_source_strategy(verified) for verified in verified_strategies
    )
    programs = tuple(item.answer_program for item in compiled)
    parameters = _unique_equal(
        tuple(parameter for program in programs for parameter in program.parameters),
        key=lambda item: item.id,
        label="program parameter",
    )
    bindings = _unique_equal(
        tuple(
            binding for item in compiled for binding in item.initial_bindings.bindings
        ),
        key=lambda item: item.parameter_id,
        label="initial parameter binding",
    )
    inputs = _unique_equal(
        tuple(input_term for program in programs for input_term in program.inputs),
        key=lambda item: item.id,
        label="semantic input",
    )
    input_denotations = _unique_equal(
        tuple(
            denotation
            for program in programs
            for denotation in program.input_denotations
        ),
        key=lambda item: item.input_ref,
        label="input denotation",
    )
    fact_template = tuple(
        fact for program in programs for fact in program.fact_template
    )
    if len({fact.id for fact in fact_template}) != len(fact_template):
        raise ValueError("fact compilation repeats a requested fact")
    row_sources = RowSourceCatalog(
        _unique_equal(
            tuple(
                source
                for verified in verified_strategies
                for source in verified.request.source_catalog.sources
            ),
            key=lambda item: item.id,
            label="row source",
        )
    )
    labeled_relation_outputs, labeled_scalar_outputs = _multi_fact_outputs(programs)
    program = AnswerProgram(
        inputs=inputs,
        input_denotations=input_denotations,
        fact_template=fact_template,
        fulfillment=tuple(item for program in programs for item in program.fulfillment),
        relation_guarantees=tuple(
            item for program in programs for item in program.relation_guarantees
        ),
        parameters=parameters,
        relations=tuple(
            relation for program in programs for relation in program.relations
        ),
        operations=tuple(
            operation for program in programs for operation in program.operations
        ),
        result_projection=ResultProjection(
            relation_outputs=labeled_relation_outputs,
            scalar_outputs=labeled_scalar_outputs,
        ),
    )
    program = replace(
        program,
        compatibility=build_program_compatibility(program, row_sources=row_sources),
    )
    return FactCompilationResult(
        answer_program=program,
        initial_bindings=BindingSet.from_bindings(bindings),
    )


def _multi_fact_outputs(
    programs: tuple[AnswerProgram, ...],
) -> tuple[tuple[RelationResultOutput, ...], tuple[ScalarResultOutput, ...]]:
    if len(programs) == 1:
        projection = programs[0].result_projection
        return projection.relation_outputs, projection.scalar_outputs
    relation_outputs: list[RelationResultOutput] = []
    scalar_outputs: list[ScalarResultOutput] = []
    for program in programs:
        [fact] = program.fact_template
        label_by_result_id = {
            fulfillment.result_output_id: _multi_fact_output_label(
                fact,
                answer_output_id=fulfillment.answer_output_id,
            )
            for fulfillment in program.fulfillment
        }
        relation_outputs.extend(
            replace(output, label=label_by_result_id[output.id])
            for output in program.result_projection.relation_outputs
        )
        scalar_outputs.extend(
            replace(output, label=label_by_result_id[output.id])
            for output in program.result_projection.scalar_outputs
        )
    return tuple(relation_outputs), tuple(scalar_outputs)


def _multi_fact_output_label(
    fact: RequestedFact,
    *,
    answer_output_id: str,
) -> str:
    requested_output = next(
        output for output in fact.outputs if output.id == answer_output_id
    )
    if len(fact.outputs) == 1:
        return fact.origin.meaning
    return f"{fact.origin.meaning}: {requested_output.origin.meaning}"


def _unique_equal(
    values: tuple[T, ...],
    *,
    key: Callable[[T], str],
    label: str,
) -> tuple[T, ...]:
    output: dict[str, T] = {}
    for value in values:
        item_id = key(value)
        prior = output.get(item_id)
        if prior is not None and prior != value:
            raise ValueError(f"{label} {item_id} has conflicting declarations")
        output[item_id] = value
    return tuple(output.values())


def _compile_requested_fact(builder: _ProgramBuilder) -> None:
    index = builder.verified.request.index
    plan = builder.verified.binding_plan
    branch_outputs = tuple(
        _compile_branch(builder, branch.branch_id, plan=plan, index=index)
        for branch in builder.verified.request.strategy.branches
    )
    input_relation = (
        branch_outputs[0]
        if len(branch_outputs) == 1
        else _union_relations(
            builder,
            relation_ids=branch_outputs,
            output_relation=f"{index.requested_fact_id}.qualified",
        )
    )
    _compile_result(builder, input_relation=input_relation, index=index)


def _compile_branch(
    builder: _ProgramBuilder,
    branch_id: str,
    *,
    plan: SourceBindingPlan,
    index: RequestedFactSemanticIndex,
) -> str:
    branch = next(
        item
        for item in builder.verified.request.strategy.branches
        if item.branch_id == branch_id
    )
    source_relations = {
        source_ref: _compile_source_relation(
            builder,
            plan=plan,
            branch_id=branch_id,
            source_ref=source_ref,
            source_position=position,
        )
        for position, source_ref in enumerate(branch.source_refs, start=1)
    }
    current = _join_branch_sources(
        builder,
        plan=plan,
        branch_id=branch_id,
        source_relations=source_relations,
    )
    returned_atoms = _returned_atom_refs(plan, branch_id=branch_id)
    clause_refs = set(branch.qualification_clause_refs)
    clauses = tuple(
        clause
        for clause in index.qualification.clauses
        if clause.clause_ref in clause_refs
    )
    if len(clauses) != len(clause_refs):
        raise ValueError("strategy branch references an unknown qualification clause")
    row_conditions = tuple(
        _compile_clause(
            builder,
            clause.atom_refs,
            returned_atoms=returned_atoms,
            branch_id=branch_id,
        )
        for clause in clauses
    )
    row_condition = (
        _combine_boolean_expressions(
            tuple(item for item in row_conditions if item is not None),
            operator=ExpressionBinaryOperator.OR,
        )
        if row_conditions and all(item is not None for item in row_conditions)
        else None
    )
    if row_condition is not None:
        from fervis.lookup.answer_program.operations import FilterSpec

        output_relation = f"{current}.qualified"
        builder.operations.append(
            Operation(
                id=f"{output_relation}.filter",
                spec=FilterSpec(
                    input_relation=current,
                    condition=row_condition,
                    proof_refs=builder.verified.evidence_refs,
                ),
                output_relation=output_relation,
            )
        )
        builder.fields_by_relation[output_relation] = builder.fields_by_relation[
            current
        ]
        builder.grain_by_relation[output_relation] = builder.grain_by_relation[current]
        current = output_relation
    selected_formula = qualification_dnf(
        index.requested_fact_id,
        tuple(frozenset(clause.atom_refs) for clause in clauses),
    )
    selected_atoms = {
        atom for clause in selected_formula.clauses for atom in clause.atom_refs
    }
    proved_atoms = {
        proof.atom_ref: proof
        for proof in builder.verified.qualification_guarantee.atom_proofs
    }
    builder.relation_guarantees.append(
        RelationGuaranteeDeclaration(
            relation_id=current,
            qualification=QualificationGuarantee(
                requested_fact_id=index.requested_fact_id,
                formula=selected_formula,
                atom_proofs=tuple(
                    proved_atoms[atom] for atom in sorted(selected_atoms)
                ),
            ),
            subject=builder.verified.subject_guarantee,
        )
    )
    return current


def _compile_source_relation(
    builder: _ProgramBuilder,
    *,
    plan: SourceBindingPlan,
    branch_id: str,
    source_ref: str,
    source_position: int,
) -> str:
    source = builder.verified.request.source_catalog.source(source_ref)
    fields = _relation_fields(builder, plan, branch_id=branch_id, source_ref=source_ref)
    relation_base = (
        f"{builder.verified.request.index.requested_fact_id}."
        f"{branch_id.rsplit(':', 1)[-1]}.source_{source_position}"
    )
    binding_sets = _source_parameter_binding_sets(
        builder,
        branch_id=branch_id,
        source_ref=source_ref,
    )
    relation_ids: list[str] = []
    for alternative_index, draft_bindings in enumerate(binding_sets, start=1):
        relation_id = (
            relation_base
            if len(binding_sets) == 1
            else f"{relation_base}.alternative_{alternative_index}"
        )
        relation = Relation(
            id=relation_id,
            source=RelationSource(
                kind=_source_kind(source.kind),
                read_id=source.read_id,
                row_source_id=source.id,
                memory_relation_id=source.memory_ref,
                param_bindings=tuple(
                    EndpointParamBinding(
                        param_id=binding.param_id,
                        value_expr=builder.inputs.expression_for_value(
                            binding.value_id,
                            component=binding.value_component,
                            item_index=binding.value_item_index,
                        ),
                        proof_refs=binding.proof_refs,
                    )
                    for binding in draft_bindings
                ),
                proof_refs=(
                    builder.verified.request.source_catalog.contract_snapshot.ref,
                    source_ref,
                    *_source_mechanic_proof_refs(
                        builder,
                        branch_id=branch_id,
                        source_ref=source_ref,
                    ),
                ),
            ),
            fields=fields,
        )
        builder.relations.append(relation)
        builder.fields_by_relation[relation.id] = fields
        builder.grain_by_relation[relation.id] = _source_grain_field_ids(source)
        relation_ids.append(relation.id)
    current = (
        relation_ids[0]
        if len(relation_ids) == 1
        else _union_relations(
            builder,
            relation_ids=tuple(relation_ids),
            output_relation=f"{relation_base}.alternatives",
        )
    )
    return _compile_returned_subject_filters(
        builder,
        branch_id=branch_id,
        source_ref=source_ref,
        input_relation=current,
    )


def _join_branch_sources(
    builder: _ProgramBuilder,
    *,
    plan: SourceBindingPlan,
    branch_id: str,
    source_relations: dict[str, str],
) -> str:
    [first, *remaining] = tuple(source_relations)
    current_source_refs = {first}
    current = source_relations[first]
    for position, source_ref in enumerate(remaining, start=1):
        realization = next(
            (
                item
                for values in plan.association_bindings.values()
                for item in values
                if item.branch_id == branch_id
                and source_ref in item.source_refs
                and current_source_refs.intersection(item.source_refs)
                and item.relation_evidence_ref is not None
            ),
            None,
        )
        if realization is None:
            raise ValueError("multi-source branch lacks declared relation evidence")
        evidence = next(
            item
            for item in builder.verified.request.source_catalog.relation_evidence
            if item.evidence_ref == realization.relation_evidence_ref
        )
        if evidence.left_source_ref in current_source_refs:
            left_fields, right_fields = (
                evidence.left_field_refs,
                evidence.right_field_refs,
            )
        else:
            left_fields, right_fields = (
                evidence.right_field_refs,
                evidence.left_field_refs,
            )
        output_relation = (
            f"{builder.verified.request.index.requested_fact_id}."
            f"{branch_id.rsplit(':', 1)[-1]}.join_{position}"
        )
        builder.operations.append(
            Operation(
                id=f"{output_relation}.operation",
                spec=JoinSpec(
                    left=current,
                    right=source_relations[source_ref],
                    join_keys=tuple(
                        JoinKey(left=left, right=right)
                        for left, right in zip(left_fields, right_fields, strict=True)
                    ),
                ),
                output_relation=output_relation,
            )
        )
        builder.fields_by_relation[output_relation] = tuple(
            dict.fromkeys(
                (
                    *builder.fields_by_relation[current],
                    *builder.fields_by_relation[source_relations[source_ref]],
                )
            )
        )
        builder.grain_by_relation[output_relation] = tuple(
            dict.fromkeys(
                (
                    *builder.grain_by_relation[current],
                    *builder.grain_by_relation[source_relations[source_ref]],
                )
            )
        )
        current = output_relation
        current_source_refs.add(source_ref)
    return current


def _union_relations(
    builder: _ProgramBuilder,
    *,
    relation_ids: tuple[str, ...],
    output_relation: str,
) -> str:
    first_fields = builder.fields_by_relation[relation_ids[0]]
    field_ids = tuple(item.field_id for item in first_fields)
    if any(
        tuple(item.field_id for item in builder.fields_by_relation[relation_id])
        != field_ids
        for relation_id in relation_ids[1:]
    ):
        raise ValueError("union branches require the same declared fields")
    identity_fields = builder.grain_by_relation[relation_ids[0]]
    if not identity_fields:
        raise ValueError("union requires stable identity fields")
    if any(
        builder.grain_by_relation[relation_id] != identity_fields
        for relation_id in relation_ids[1:]
    ):
        raise ValueError("union branches require the same stable identity")
    builder.operations.append(
        Operation(
            id=f"{output_relation}.operation",
            spec=UnionSpec(
                inputs=relation_ids,
                output_fields=field_ids,
                identity_fields=identity_fields,
            ),
            output_relation=output_relation,
        )
    )
    builder.fields_by_relation[output_relation] = first_fields
    builder.grain_by_relation[output_relation] = identity_fields
    return output_relation


def _compile_returned_subject_filters(
    builder: _ProgramBuilder,
    *,
    branch_id: str,
    source_ref: str,
    input_relation: str,
) -> str:
    reviews = tuple(
        review
        for realization in builder.verified.binding_plan.subject_binding.branch_realizations
        if realization.branch_id == branch_id
        for review in realization.surface_reviews
        for surface in (
            builder.verified.request.source_catalog.choice_surface(review.surface_ref),
        )
        if surface.source_ref == source_ref
        and surface.kind is SourceChoiceSurfaceKind.RETURNED_FIELD
        and review.mechanics
    )
    if not reviews:
        return input_relation
    conditions = tuple(
        _returned_choice_condition(builder, review=review) for review in reviews
    )
    condition = _combine_boolean_expressions(
        conditions,
        operator=ExpressionBinaryOperator.AND,
    )
    from fervis.lookup.answer_program.operations import FilterSpec

    output_relation = f"{input_relation}.subject"
    builder.operations.append(
        Operation(
            id=f"{output_relation}.filter",
            spec=FilterSpec(
                input_relation=input_relation,
                condition=condition,
                proof_refs=builder.verified.evidence_refs,
            ),
            output_relation=output_relation,
        )
    )
    builder.fields_by_relation[output_relation] = builder.fields_by_relation[
        input_relation
    ]
    builder.grain_by_relation[output_relation] = builder.grain_by_relation[
        input_relation
    ]
    return output_relation


def _returned_choice_condition(
    builder: _ProgramBuilder,
    *,
    review: SubjectSurfaceReview,
) -> Expression:
    surface = builder.verified.request.source_catalog.choice_surface(review.surface_ref)
    source = builder.verified.request.source_catalog.source(surface.source_ref)
    field_id = next(
        field.id for field in source.fields if field.field_ref == surface.surface_ref
    )
    comparisons = tuple(
        BinaryExpression(
            operator=ExpressionBinaryOperator.EQUALS,
            left=FieldRef(field_id),
            right=ConstantRef(
                constant_id=value.value_ref,
                version_ref=builder.verified.request.source_catalog.contract_snapshot.ref,
                value=FactValue.literal(
                    id=value.value_ref,
                    literal_type=LiteralType.STRING,
                    value=str(value.value),
                    label=value.label,
                    proof_refs=(
                        builder.verified.request.source_catalog.contract_snapshot.ref,
                        surface.surface_ref,
                        value.value_ref,
                    ),
                    source_refs=(surface.source_ref,),
                ),
            ),
        )
        for choice_ref in review.included_choice_refs
        for value in (builder.verified.request.source_catalog.choice_value(choice_ref),)
    )
    return _combine_boolean_expressions(
        comparisons,
        operator=ExpressionBinaryOperator.OR,
    )


def _combine_boolean_expressions(
    expressions: tuple[Expression, ...],
    *,
    operator: ExpressionBinaryOperator,
) -> Expression:
    if not expressions:
        raise ValueError("boolean expression requires at least one operand")
    current = expressions[0]
    for expression in expressions[1:]:
        current = BinaryExpression(operator, current, expression)
    return current


def _relation_fields(
    builder: _ProgramBuilder,
    plan: SourceBindingPlan,
    *,
    branch_id: str,
    source_ref: str,
) -> tuple[RelationField, ...]:
    source = builder.verified.request.source_catalog.source(source_ref)
    roles: dict[str, set[FieldBindingRole]] = {
        field_ref: {FieldBindingRole.IDENTITY}
        for field_ref in _source_grain_field_refs(source)
    }
    for set_realizations in plan.set_bindings.values():
        for set_realization in set_realizations:
            if (
                set_realization.branch_id == branch_id
                and set_realization.source_ref == source_ref
            ):
                for field_ref in set_realization.identity_field_refs:
                    roles.setdefault(field_ref, set()).add(FieldBindingRole.IDENTITY)
    for fact_realizations in plan.fact_bindings.values():
        for fact_realization in fact_realizations:
            if (
                fact_realization.branch_id == branch_id
                and fact_realization.source_ref == source_ref
            ):
                for field_ref in fact_realization.field_refs:
                    roles.setdefault(field_ref, set()).add(FieldBindingRole.OUTPUT)
    relation_evidence = {
        item.evidence_ref: item
        for item in builder.verified.request.source_catalog.relation_evidence
    }
    for association_realizations in plan.association_bindings.values():
        for realization in association_realizations:
            if (
                realization.branch_id != branch_id
                or source_ref not in realization.source_refs
                or realization.relation_evidence_ref is None
            ):
                continue
            evidence = relation_evidence[realization.relation_evidence_ref]
            field_ids = (
                evidence.left_field_refs
                if evidence.left_source_ref == source_ref
                else evidence.right_field_refs
            )
            for field_id in field_ids:
                roles.setdefault(source.field(field_id).field_ref, set()).add(
                    FieldBindingRole.PREDICATE
                )
    for subject_realization in plan.subject_binding.branch_realizations:
        if subject_realization.branch_id != branch_id:
            continue
        for review in subject_realization.surface_reviews:
            surface = builder.verified.request.source_catalog.choice_surface(
                review.surface_ref
            )
            if (
                surface.source_ref == source_ref
                and surface.kind is SourceChoiceSurfaceKind.RETURNED_FIELD
                and review.mechanics
            ):
                roles.setdefault(surface.surface_ref, set()).add(
                    FieldBindingRole.PREDICATE
                )
    return tuple(
        RelationField(
            field_id=next(
                field.id for field in source.fields if field.field_ref == field_ref
            ),
            roles=tuple(sorted(field_roles, key=lambda item: item.value)),
        )
        for field_ref, field_roles in sorted(roles.items())
    )


def _source_grain_field_refs(source: RowSource) -> tuple[str, ...]:
    stable_keys = tuple(key for key in source.candidate_keys if key.stable)
    key = next((key for key in stable_keys if key.primary), None)
    if key is None and stable_keys:
        key = stable_keys[0]
    if key is None:
        return ()
    return tuple(source.field(component.field_id).field_ref for component in key.components)


def _source_grain_field_ids(source: RowSource) -> tuple[str, ...]:
    grain_refs = set(_source_grain_field_refs(source))
    return tuple(field.id for field in source.fields if field.field_ref in grain_refs)


def _source_kind(kind: RowSourceKind) -> SourceKind:
    return {
        RowSourceKind.API_READ: SourceKind.API_READ,
        RowSourceKind.GENERATED_CALENDAR: SourceKind.GENERATED_CALENDAR,
        RowSourceKind.MEMORY_READ: SourceKind.MEMORY_READ,
    }[kind]


def _source_parameter_binding_sets(
    builder: _ProgramBuilder,
    *,
    branch_id: str,
    source_ref: str,
) -> ParamBindingSetAlternatives:
    source = builder.verified.request.source_catalog.source(source_ref)
    groups_by_target: dict[str, list[tuple[str, ParamBindingSetAlternatives]]] = {}
    for application in builder.verified.binding_plan.invocation_applications:
        if application.branch_id != branch_id or application.source_ref != source_ref:
            continue
        for target in application.target_applications:
            param = next(
                item for item in source.params if item.param_ref == target.target_ref
            )
            value = _bound_fact_value(builder, target.value_ref)
            projected = fact_value_parameter_projection(
                value,
                projection=target.projection,
                component_id=target.component_ref,
                type_name=param.type.value,
                choices=tuple(str(item) for item in param.choices),
            )
            groups_by_target.setdefault(target.target_ref, []).append(
                (
                    application.application_ref,
                    parameter_binding_sets(
                        param_id=param.id,
                        value=projected,
                        parameter_type=param.type.value,
                        origin_kind=(
                            RelationInputOrigin.QUESTION_INPUT
                            if any(
                                item.canonical_value_id == target.value_ref
                                for item in builder.verified.request.canonical_values
                            )
                            else RelationInputOrigin.PLAN_CONTROL
                        ),
                        value_id=target.value_ref,
                        value_component=_projection_component(target),
                        proof_refs=(application.application_ref, target.target_ref),
                    ),
                )
            )
    independent_groups: list[ParamBindingSetAlternatives] = []
    mechanics = _source_mechanics(
        builder,
        branch_id=branch_id,
        source_ref=source_ref,
    )
    for target_ref, entries in groups_by_target.items():
        distinct_entries: list[tuple[set[str], ParamBindingSetAlternatives]] = []
        for application_ref, group in entries:
            matching_index = next(
                (
                    index
                    for index, item in enumerate(distinct_entries)
                    if equivalent_param_binding_sets(item[1], group)
                ),
                None,
            )
            if matching_index is None:
                distinct_entries.append(({application_ref}, group))
                continue
            application_refs_for_group, existing_group = distinct_entries[
                matching_index
            ]
            application_refs_for_group.add(application_ref)
            distinct_entries[matching_index] = (
                application_refs_for_group,
                merge_equivalent_param_binding_sets(existing_group, group),
            )
        owned_application_refs: set[str] = set()
        constraints: list[ParamBindingSetAlternatives] = []
        for mechanic in mechanics:
            owned = tuple(
                group
                for application_refs, group in distinct_entries
                if application_refs & set(mechanic.application_refs)
            )
            if not owned:
                continue
            owned_application_refs.update(mechanic.application_refs)
            constraints.append(alternate_param_binding_sets(owned))
        constraints.extend(
            group
            for application_refs, group in distinct_entries
            if not application_refs & owned_application_refs
        )
        intersection = intersect_param_binding_sets(constraints)
        if not intersection:
            raise ValueError(f"invocation constraints conflict on target {target_ref}")
        independent_groups.append(intersection)
    return combine_param_binding_sets(independent_groups)


def _bound_fact_value(builder: _ProgramBuilder, value_ref: str) -> FactValue:
    matches = tuple(
        binding.value
        for binding in builder.inputs.program_inputs.bindings.bindings
        if binding.value.id == value_ref
    )
    if len(matches) != 1:
        raise ValueError(f"invocation value {value_ref} lacks one typed binding")
    return matches[0]


def _source_mechanic_proof_refs(
    builder: _ProgramBuilder,
    *,
    branch_id: str,
    source_ref: str,
) -> tuple[str, ...]:
    mechanics = _source_mechanics(
        builder,
        branch_id=branch_id,
        source_ref=source_ref,
    )
    plan = builder.verified.binding_plan
    reviewed_surfaces = tuple(
        review.surface_ref
        for realization in plan.subject_binding.branch_realizations
        if realization.branch_id == branch_id
        for review in realization.surface_reviews
        if builder.verified.request.source_catalog.choice_surface(
            review.surface_ref
        ).source_ref
        == source_ref
    )
    return tuple(
        dict.fromkeys(
            (
                *reviewed_surfaces,
                *(
                    ref
                    for mechanic in mechanics
                    for ref in (
                        mechanic.source_ref,
                        *mechanic.application_refs,
                        *mechanic.contract_evidence_refs,
                    )
                ),
            )
        )
    )


def _source_mechanics(
    builder: _ProgramBuilder,
    *,
    branch_id: str,
    source_ref: str,
):
    plan = builder.verified.binding_plan
    return tuple(
        mechanic
        for realizations in plan.boolean_bindings.values()
        for realization in realizations
        if realization.branch_id == branch_id
        for mechanic in realization.mechanics
        if mechanic.source_ref == source_ref
    ) + tuple(
        mechanic
        for realization in plan.subject_binding.branch_realizations
        if realization.branch_id == branch_id
        for review in realization.surface_reviews
        for mechanic in review.mechanics
        if mechanic.source_ref == source_ref
    )


def _projection_component(target) -> str:
    if target.projection is ValueProjectionKind.WHOLE_VALUE:
        return "value"
    if target.projection is ValueProjectionKind.TEMPORAL_START:
        return "start"
    if target.projection is ValueProjectionKind.TEMPORAL_END:
        return "end"
    if target.projection is ValueProjectionKind.IDENTITY_COMPONENT:
        if target.component_ref is None:
            raise ValueError("identity application lacks a key component")
        return f"key_component:{target.component_ref}"
    raise TypeError("unsupported invocation value projection")


def _returned_atom_refs(plan: SourceBindingPlan, *, branch_id: str) -> frozenset[str]:
    return frozenset(
        requirement_ref
        for requirement_ref, realizations in plan.boolean_bindings.items()
        for realization in realizations
        if realization.branch_id == branch_id
        and any(
            mechanic.kind is SourceMechanicKind.RETURNED_ROW_PREDICATE
            for mechanic in realization.mechanics
        )
    )


def _compile_clause(
    builder: _ProgramBuilder,
    atom_refs,
    *,
    returned_atoms: frozenset[str],
    branch_id: str,
) -> Expression | None:
    expressions: list[Expression] = []
    requirements = {
        item.requirement_ref: item
        for item in builder.verified.request.index.boolean_requirements
    }
    for atom in sorted(atom_refs):
        requirement = next(
            (item for item in requirements.values() if item.atom_ref == atom),
            None,
        )
        if requirement is None:
            raise ValueError("qualification atom lacks a scoped requirement")
        if requirement.requirement_ref not in returned_atoms:
            continue
        expression = _compile_semantic_ref(
            builder,
            FactLocalRef.from_token(atom.value_ref),
            branch_id=branch_id,
        )
        if atom.polarity.value == "negative":
            expression = UnaryExpression(ExpressionUnaryOperator.NOT, expression)
        expressions.append(expression)
    if not expressions:
        return None
    current = expressions[0]
    for expression in expressions[1:]:
        current = BinaryExpression(ExpressionBinaryOperator.AND, current, expression)
    return current


def _compile_result(
    builder: _ProgramBuilder,
    *,
    input_relation: str,
    index: RequestedFactSemanticIndex,
) -> None:
    requested_fact = index.requested_fact
    required_expression_refs = {
        dependency
        for requirement in index.output_requirements
        for dependency in requirement.dependencies
        if isinstance(dependency, FactLocalRef)
    } | set(index.ordering_refs)
    aggregates = tuple(
        (ref, node)
        for ref, node in index.expression_by_ref.items()
        if isinstance(node, Aggregate) and ref in required_expression_refs
    )
    current = input_relation
    current = _project_semantic_values(
        builder,
        input_relation=current,
        refs=index.grouping_refs,
        aggregate_fields={},
        label="group_values",
    )
    group_fields = tuple(
        _result_field(builder, ref, aggregate_fields={}) for ref in index.grouping_refs
    )
    aggregate_fields: dict[FactLocalRef, str] = {}
    if aggregates:
        specs: list[AggregationSpec] = []
        for position, (ref, aggregate) in enumerate(aggregates, start=1):
            output_field = f"aggregate_{position}"
            aggregate_fields[ref] = output_field
            argument = _semantic_ref(index, aggregate.argument_ref)
            specs.append(
                AggregationSpec(
                    function={
                        "sum": AggregationFunction.SUM,
                        "count": AggregationFunction.COUNT,
                        "minimum": AggregationFunction.MIN,
                        "maximum": AggregationFunction.MAX,
                        "average": AggregationFunction.AVG,
                    }[aggregate.function.value],
                    output_field=output_field,
                    input_field=(
                        ""
                        if argument.kind is FactLocalKind.SET
                        else _compiled_field_all_branches(builder, argument)
                    ),
                    filter=(
                        None
                        if aggregate.filter_ref is None
                        else _compile_semantic_ref(
                            builder,
                            _semantic_ref(index, aggregate.filter_ref),
                            branch_id=_representative_branch(builder),
                        )
                    ),
                    distinct_argument=aggregate.distinct_argument,
                )
            )
        output_relation = f"{index.requested_fact_id}.aggregate"
        aggregate_operation_id = f"{index.requested_fact_id}.aggregate_operation"
        builder.operations.append(
            Operation(
                id=aggregate_operation_id,
                spec=AggregateSpec(
                    input_relation=current,
                    group_by=group_fields,
                    aggregations=tuple(specs),
                ),
                output_relation=output_relation,
            )
        )
        builder.fields_by_relation[output_relation] = tuple(
            RelationField(
                field_id=field_id,
                roles=(FieldBindingRole.OUTPUT,),
            )
            for field_id in (*group_fields, *aggregate_fields.values())
        )
        if not group_fields:
            builder.scalar_aggregate_outputs.update(
                {
                    ref: NodeOutputRef(aggregate_operation_id, output_field)
                    for ref, output_field in aggregate_fields.items()
                }
            )
        current = output_relation
    result_value_refs = tuple(
        dict.fromkeys(
            (
                *(item.value_ref for item in index.output_requirements),
                *index.ordering_refs,
            )
        )
    )
    current = _project_semantic_values(
        builder,
        input_relation=current,
        refs=tuple(
            ref
            for ref in result_value_refs
            if isinstance(ref, FactLocalRef)
            and ref.kind is FactLocalKind.EXPRESSION
            and ref not in aggregate_fields
            and not (
                isinstance(index.result_grain, Singleton)
                and _is_scalar_computation(index, ref)
            )
        ),
        aggregate_fields=aggregate_fields,
        label="result_values",
    )
    if requested_fact.distinct_by:
        output_relation = f"{index.requested_fact_id}.distinct"
        key_fields = tuple(
            _result_field(
                builder,
                _semantic_ref(index, local_id),
                aggregate_fields=aggregate_fields,
            )
            for local_id in requested_fact.distinct_by
        )
        builder.operations.append(
            Operation(
                id=f"{index.requested_fact_id}.distinct_operation",
                spec=ProjectToKeySpec(
                    input_relation=current,
                    key_fields=key_fields,
                ),
                output_relation=output_relation,
            )
        )
        builder.fields_by_relation[output_relation] = tuple(
            field
            for field in builder.fields_by_relation[current]
            if field.field_id in set(key_fields)
        )
        current = output_relation
    if requested_fact.ordering:
        output_relation = f"{index.requested_fact_id}.ordered"
        builder.operations.append(
            Operation(
                id=f"{index.requested_fact_id}.order_operation",
                spec=OrderSpec(
                    input_relation=current,
                    order_by=tuple(
                        SortKey(
                            field=_result_field(
                                builder,
                                _semantic_ref(index, item.expression_ref),
                                aggregate_fields=aggregate_fields,
                            ),
                            direction={
                                "ascending": SortDirection.ASC,
                                "descending": SortDirection.DESC,
                            }[item.direction.value],
                        )
                        for item in requested_fact.ordering
                    ),
                    selection=_selection(builder, requested_fact.selection),
                ),
                output_relation=output_relation,
            )
        )
        builder.fields_by_relation[output_relation] = builder.fields_by_relation[
            current
        ]
        current = output_relation
    for output_position, requirement in enumerate(index.output_requirements, start=1):
        result_id = f"{index.requested_fact_id}.output_{output_position}"
        builder.fulfillments.append(
            FactFulfillment(
                requested_fact_id=index.requested_fact_id,
                answer_output_id=requirement.output_ref.local_id,
                result_output_id=result_id,
            )
        )
        if (
            isinstance(index.result_grain, Singleton)
            and requirement.value_ref not in aggregate_fields
            and _is_scalar_computation(index, requirement.value_ref)
        ):
            scalar_id = f"{result_id}.scalar"
            builder.operations.append(
                Operation(
                    id=f"{result_id}.compute",
                    spec=ComputeSpec(
                        expression=_compile_scalar_result_expression(
                            builder,
                            requirement.value_ref,
                        ),
                        output_scalar=scalar_id,
                    ),
                )
            )
            builder.scalar_outputs.append(
                ScalarResultOutput(
                    id=result_id,
                    scalar_id=scalar_id,
                    role="answer_value",
                )
            )
        else:
            entity_key = _result_entity_key(builder, requirement.value_ref)
            builder.relation_outputs.append(
                RelationResultOutput(
                    id=result_id,
                    relation_id=current,
                    field_id=(
                        ""
                        if entity_key is not None
                        else _result_field(
                            builder,
                            requirement.value_ref,
                            aggregate_fields=aggregate_fields,
                        )
                    ),
                    entity_key=entity_key,
                    role="answer_value",
                )
            )


def _is_scalar_computation(index: RequestedFactSemanticIndex, ref) -> bool:
    if not isinstance(ref, FactLocalRef) or ref.kind is not FactLocalKind.EXPRESSION:
        return False
    return not isinstance(index.expression_by_ref[ref], Aggregate)


def _compile_scalar_result_expression(
    builder: _ProgramBuilder,
    ref,
) -> Expression:
    if isinstance(ref, str):
        return builder.inputs.expression_for_question_input(ref)
    aggregate = builder.scalar_aggregate_outputs.get(ref)
    if aggregate is not None:
        return aggregate
    node = builder.verified.request.index.expression_by_ref.get(ref)
    if node is None:
        raise ValueError("scalar result references a non-scalar semantic value")

    def value(local_id: str) -> Expression:
        if local_id in builder.inputs.expressions_by_input_ref:
            return builder.inputs.expression_for_question_input(local_id)
        return _compile_scalar_result_expression(
            builder,
            _semantic_ref(builder.verified.request.index, local_id),
        )

    if isinstance(node, Arithmetic):
        arguments = tuple(value(item) for item in node.argument_refs)
        return _arithmetic_expression(node.operator, arguments)
    if isinstance(node, Comparison):
        return _compile_comparison(
            builder,
            node,
            value=value,
        )
    if isinstance(node, BooleanComposition):
        arguments = tuple(value(item) for item in node.argument_refs)
        if node.operator is BooleanCompositionOperator.NOT:
            return UnaryExpression(ExpressionUnaryOperator.NOT, arguments[0])
        operator = ExpressionBinaryOperator(node.operator.value)
        current = arguments[0]
        for argument in arguments[1:]:
            current = BinaryExpression(operator, current, argument)
        return current
    if isinstance(node, NullCheck):
        return UnaryExpression(node.operator, value(node.argument_ref))
    raise ValueError(
        f"semantic node {type(node).__name__} is not a scalar result computation"
    )


def _compile_semantic_ref(
    builder: _ProgramBuilder,
    ref: FactLocalRef,
    *,
    branch_id: str,
) -> Expression:
    if ref.kind is FactLocalKind.FACT:
        return FieldRef(_compiled_field(builder, ref, branch_id=branch_id))
    node = builder.verified.request.index.expression_by_ref.get(ref)
    if node is None:
        raise ValueError("semantic expression reference is unavailable")
    return _compile_node(
        builder,
        node,
        branch_id=branch_id,
    )


def _compile_node(
    builder: _ProgramBuilder,
    node: ExpressionNode,
    *,
    branch_id: str,
) -> Expression:
    index = builder.verified.request.index

    def value(ref: str) -> Expression:
        if ref in builder.inputs.expressions_by_input_ref:
            return builder.inputs.expression_for_question_input(ref)
        return _compile_semantic_ref(
            builder,
            _semantic_ref(index, ref),
            branch_id=branch_id,
        )

    if isinstance(node, Comparison):
        return _compile_comparison(
            builder,
            node,
            value=value,
        )
    if isinstance(node, Arithmetic):
        arguments = tuple(value(item) for item in node.argument_refs)
        return _arithmetic_expression(node.operator, arguments)
    if isinstance(node, BooleanComposition):
        arguments = tuple(value(item) for item in node.argument_refs)
        if node.operator is BooleanCompositionOperator.NOT:
            return UnaryExpression(ExpressionUnaryOperator.NOT, arguments[0])
        operator = ExpressionBinaryOperator(node.operator.value)
        current = arguments[0]
        for argument in arguments[1:]:
            current = BinaryExpression(operator, current, argument)
        return current
    if isinstance(node, NullCheck):
        return UnaryExpression(node.operator, value(node.argument_ref))
    if isinstance(node, TemporalBucket):
        return FunctionExpression(
            ExpressionFunction.TEMPORAL_BUCKET,
            (
                value(node.value_ref),
                ConstantRef(
                    constant_id=f"temporal_grain.{node.id}",
                    version_ref="semantic-question-contract@1",
                    value=FactValue.literal(
                        id=f"temporal_grain.{node.id}",
                        literal_type=LiteralType.STRING,
                        value=node.grain.value,
                    ),
                ),
                EnvironmentRef(key=ANCHOR_TIMEZONE_REF),
            ),
        )
    if isinstance(node, Quantify):
        return _compile_co_resident_exists(
            builder,
            node,
            branch_id=branch_id,
        )
    raise ValueError(
        f"semantic node {type(node).__name__} requires relational lowering"
    )


def _compile_co_resident_exists(
    builder: _ProgramBuilder,
    node: Quantify,
    *,
    branch_id: str,
) -> Expression:
    """Reduce a proved subject-grain co-resident EXISTS to its row condition."""

    if node.quantifier is not Quantifier.EXISTS or not node.association_refs:
        raise ValueError("quantifier requires relational lowering")
    plan = builder.verified.binding_plan
    association_sources: set[str] | None = None
    for association_ref in node.association_refs:
        token = FactLocalRef(
            builder.verified.request.index.requested_fact_id,
            FactLocalKind.ASSOCIATION,
            association_ref,
        ).token
        realizations = tuple(
            item
            for item in plan.association_bindings.get(token, ())
            if item.branch_id == branch_id
        )
        if len(realizations) != 1 or (
            realizations[0].kind is not AssociationRealizationKind.CO_RESIDENT
        ):
            raise ValueError("quantifier requires relational lowering")
        sources = set(realizations[0].source_refs)
        association_sources = (
            sources
            if association_sources is None
            else association_sources.intersection(sources)
        )
    subject_ref = builder.verified.request.index.subject_obligation.subject_set_ref.token
    subject_is_identified_at_association_grain = any(
        realization.branch_id == branch_id
        and realization.source_ref in (association_sources or set())
        and realization.identity_ref is not None
        and bool(realization.identity_field_refs)
        for realization in plan.set_bindings.get(subject_ref, ())
    )
    if not subject_is_identified_at_association_grain:
        raise ValueError("co-resident quantifier lacks complete subject grain")
    return _compile_semantic_ref(
        builder,
        _semantic_ref(builder.verified.request.index, node.condition_ref),
        branch_id=branch_id,
    )


def _arithmetic_expression(
    operator: ExpressionUnaryOperator | ExpressionBinaryOperator,
    arguments: tuple[Expression, ...],
) -> Expression:
    if isinstance(operator, ExpressionUnaryOperator):
        if len(arguments) != 1:
            raise ValueError("unary arithmetic requires one argument")
        return UnaryExpression(operator, arguments[0])
    if len(arguments) != 2:
        raise ValueError("binary arithmetic requires two arguments")
    return BinaryExpression(operator, arguments[0], arguments[1])


def _compile_comparison(
    builder: _ProgramBuilder,
    node: Comparison,
    *,
    value: Callable[[str], Expression],
) -> Expression:
    if node.operator is not ExpressionBinaryOperator.WITHIN:
        return BinaryExpression(
            node.operator,
            value(node.left_ref),
            value(node.right_ref),
        )
    lower_bound = builder.inputs.expression_for_question_input(
        node.right_ref,
        component="start",
    )
    upper_bound = builder.inputs.expression_for_question_input(
        node.right_ref,
        component="end",
    )
    return BinaryExpression(
        ExpressionBinaryOperator.AND,
        BinaryExpression(
            ExpressionBinaryOperator.GTE,
            value(node.left_ref),
            lower_bound,
        ),
        BinaryExpression(
            ExpressionBinaryOperator.LTE,
            value(node.left_ref),
            upper_bound,
        ),
    )


def _compiled_field(
    builder: _ProgramBuilder,
    ref: FactLocalRef,
    *,
    branch_id: str,
) -> str:
    realizations = builder.verified.binding_plan.fact_bindings.get(ref.token, ())
    matching = tuple(item for item in realizations if item.branch_id == branch_id)
    if len(matching) != 1 or len(matching[0].field_refs) != 1:
        raise ValueError("semantic scalar requires one realized source field")
    [realization] = matching
    source = builder.verified.request.source_catalog.source(realization.source_ref)
    return next(
        field.id
        for field in source.fields
        if field.field_ref == realization.field_refs[0]
    )


def _result_field(
    builder: _ProgramBuilder,
    ref,
    *,
    aggregate_fields: dict[FactLocalRef, str],
) -> str:
    if isinstance(ref, str):
        raise ValueError("input values are not direct result fields")
    if ref in aggregate_fields:
        return aggregate_fields[ref]
    if ref in builder.projected_fields:
        return builder.projected_fields[ref]
    return _compiled_field_all_branches(builder, ref)


def _result_entity_key(
    builder: _ProgramBuilder,
    ref,
) -> EntityKeyProjection | None:
    if not isinstance(ref, FactLocalRef):
        return None
    realizations: tuple[SetRealization | FactRealization, ...]
    if ref.kind is FactLocalKind.SET:
        realizations = builder.verified.binding_plan.set_bindings.get(ref.token, ())
    elif ref.kind is FactLocalKind.FACT:
        term = builder.verified.request.index.term_by_ref[ref]
        if not isinstance(term, FactTerm) or not isinstance(
            term.value_type, IdentifierType
        ):
            return None
        realizations = builder.verified.binding_plan.fact_bindings.get(ref.token, ())
    else:
        return None
    identity_refs = {item.identity_ref for item in realizations}
    if None in identity_refs or len(identity_refs) != 1:
        raise ValueError("identifier output requires one declared identity authority")
    identity_ref = next(iter(identity_refs))
    assert identity_ref is not None
    evidence = builder.verified.request.source_catalog.identity(identity_ref)
    projections = {
        _identity_projection_components(
            builder,
            source_ref=item.source_ref,
            identity_ref=item.identity_ref,
        )
        for item in realizations
    }
    if len(projections) != 1:
        raise ValueError("strategy branches expose incompatible identity components")
    return EntityKeyProjection(
        entity_kind=evidence.entity_kind,
        key_id=evidence.key_id,
        components=next(iter(projections)),
    )


def _identity_projection_components(
    builder: _ProgramBuilder,
    *,
    source_ref: str,
    identity_ref: str | None,
) -> tuple[EntityKeyProjectionComponent, ...]:
    if identity_ref is None:
        raise ValueError("identifier realization lacks identity authority")
    source = builder.verified.request.source_catalog.source(source_ref)
    evidence = builder.verified.request.source_catalog.identity(identity_ref)
    candidate_key = next(
        (
            key
            for key in source.candidate_keys
            if key.entity_kind == evidence.entity_kind and key.id == evidence.key_id
        ),
        None,
    )
    if candidate_key is not None:
        return tuple(
            EntityKeyProjectionComponent(component.id, component.field_id)
            for component in candidate_key.components
        )
    entity_reference = next(
        (
            reference
            for reference in source.entity_references
            if reference.target_entity_kind == evidence.entity_kind
            and reference.target_key_id == evidence.key_id
        ),
        None,
    )
    if entity_reference is None:
        raise ValueError("identifier authority is absent from its declared source")
    return tuple(
        EntityKeyProjectionComponent(
            component.target_component_id,
            component.local_field_id,
        )
        for component in entity_reference.components
    )


def _project_semantic_values(
    builder: _ProgramBuilder,
    *,
    input_relation: str,
    refs: tuple[FactLocalRef, ...],
    aggregate_fields: dict[FactLocalRef, str],
    label: str,
) -> str:
    pending = tuple(
        ref
        for ref in dict.fromkeys(refs)
        if ref.kind is FactLocalKind.EXPRESSION
        and ref not in aggregate_fields
        and ref not in builder.projected_fields
    )
    if not pending:
        return input_relation
    existing_fields = builder.fields_by_relation[input_relation]
    projected = {
        ref: f"semantic_value_{len(builder.projected_fields) + position}"
        for position, ref in enumerate(pending, start=1)
    }
    outputs = tuple(
        NamedExpression(field.field_id, FieldRef(field.field_id))
        for field in existing_fields
    ) + tuple(
        NamedExpression(
            output_field=field_id,
            expression=_compile_relation_result_expression(
                builder,
                ref,
                aggregate_fields=aggregate_fields,
            ),
        )
        for ref, field_id in projected.items()
    )
    fact_id = builder.verified.request.index.requested_fact_id
    output_relation = f"{fact_id}.{label}"
    builder.operations.append(
        Operation(
            id=f"{output_relation}.operation",
            spec=ProjectSpec(input_relation=input_relation, outputs=outputs),
            output_relation=output_relation,
        )
    )
    builder.fields_by_relation[output_relation] = (
        *existing_fields,
        *(
            RelationField(field_id, (FieldBindingRole.OUTPUT,))
            for field_id in projected.values()
        ),
    )
    builder.projected_fields.update(projected)
    return output_relation


def _compile_relation_result_expression(
    builder: _ProgramBuilder,
    ref: FactLocalRef,
    *,
    aggregate_fields: dict[FactLocalRef, str],
) -> Expression:
    index = builder.verified.request.index
    if ref in aggregate_fields:
        return FieldRef(aggregate_fields[ref])
    if ref.kind is FactLocalKind.FACT:
        return FieldRef(_compiled_field_all_branches(builder, ref))
    node = index.expression_by_ref[ref]

    def value(local_id: str) -> Expression:
        if local_id in builder.inputs.expressions_by_input_ref:
            return builder.inputs.expression_for_question_input(local_id)
        return _compile_relation_result_expression(
            builder,
            _semantic_ref(index, local_id),
            aggregate_fields=aggregate_fields,
        )

    if isinstance(node, Arithmetic):
        arguments = tuple(value(item) for item in node.argument_refs)
        return _arithmetic_expression(node.operator, arguments)
    if isinstance(node, Comparison):
        return _compile_comparison(
            builder,
            node,
            value=value,
        )
    if isinstance(node, BooleanComposition):
        arguments = tuple(value(item) for item in node.argument_refs)
        if node.operator is BooleanCompositionOperator.NOT:
            return UnaryExpression(ExpressionUnaryOperator.NOT, arguments[0])
        operator = ExpressionBinaryOperator(node.operator.value)
        current = arguments[0]
        for argument in arguments[1:]:
            current = BinaryExpression(operator, current, argument)
        return current
    if isinstance(node, NullCheck):
        return UnaryExpression(node.operator, value(node.argument_ref))
    if isinstance(node, TemporalBucket):
        return FunctionExpression(
            ExpressionFunction.TEMPORAL_BUCKET,
            (
                value(node.value_ref),
                ConstantRef(
                    constant_id=f"temporal_grain.{node.id}",
                    version_ref="semantic-question-contract@1",
                    value=FactValue.literal(
                        id=f"temporal_grain.{node.id}",
                        literal_type=LiteralType.STRING,
                        value=node.grain.value,
                    ),
                ),
                EnvironmentRef(key=ANCHOR_TIMEZONE_REF),
            ),
        )
    raise ValueError(
        f"semantic node {type(node).__name__} requires relational lowering"
    )


def _selection(builder: _ProgramBuilder, selection):
    if isinstance(selection, FirstRankWithTies):
        return Take(
            limit=ConstantRef(
                constant_id="selection.first_rank",
                version_ref="semantic-question-contract@1",
                value=FactValue.literal(
                    id="selection.first_rank",
                    literal_type=LiteralType.NUMBER,
                    value="1",
                ),
            )
        )
    if hasattr(selection, "limit_input_ref"):
        expression = builder.inputs.expression_for_question_input(
            selection.limit_input_ref
        )
        if not isinstance(expression, ParameterRef):
            raise ValueError("selection limit requires a parameter")
        return Take(limit=expression)
    return KeepAll()


def _semantic_ref(index: RequestedFactSemanticIndex, local_id: str) -> FactLocalRef:
    try:
        return index.fact_local_ref_by_local_id[local_id]
    except KeyError as exc:
        raise ValueError(f"unknown semantic reference {local_id}") from exc


def _compiled_field_all_branches(
    builder: _ProgramBuilder,
    ref: FactLocalRef,
) -> str:
    field_ids = {
        _compiled_field(builder, ref, branch_id=branch.branch_id)
        for branch in builder.verified.request.strategy.branches
    }
    if len(field_ids) != 1:
        raise ValueError("strategy branches expose incompatible semantic fields")
    return next(iter(field_ids))


def _representative_branch(builder: _ProgramBuilder) -> str:
    return builder.verified.request.strategy.branches[0].branch_id


__all__ = [
    "compile_verified_source_strategies",
    "compile_verified_source_strategy",
]
