"""Deterministically lower verified semantic facts into one AnswerProgram."""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass, replace, field
from typing import TypeVar

from fervis.lookup.answer_program.compiler_inputs import CompilerInputContext
from fervis.lookup.answer_program.graph import prune_unused_program_nodes
from .occurrences import RelationalValue
from fervis.lookup.question_contract.domains import (
    value_set_dependencies,
    relational_free_sets,
)
from fervis.lookup.source_binding.occurrences import OccurrenceScope, occurrence_scope
from fervis.lookup.question_contract.analysis import (
    RowDomain,
    AssociationExpandedDomain,
)
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
    JoinMode,
    Operation,
    OrderSpec,
    ProjectSpec,
    ProjectToKeySpec,
    NamedExpression,
    SortDirection,
    SortKey,
    Take,
    AtPosition,
    UnionSpec,
    ComputeSpec,
    CrossJoinSpec,
    FilterSpec,
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
    TimeValuePayload,
    LiteralType,
    ParameterRef,
    NodeOutputRef,
    BindingSet,
)
from fervis.lookup.expression_operators import (
    ExpressionBinaryOperator,
    ExpressionUnaryOperator,
)
from fervis.lookup.qualification import (
    QualificationGuarantee,
    qualification_dnf,
    BooleanAtomRef,
    BooleanPolarity,
    BooleanRequirementUseSite,
)
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
    SubjectRows,
)
from fervis.lookup.question_contract import (
    Aggregate,
    AssociationTerm,
    Arithmetic,
    BooleanComposition,
    BooleanCompositionOperator,
    Comparison,
    Coverage,
    ExpressionNode,
    FactTerm,
    FactLocalKind,
    FactLocalRef,
    FirstRankWithTies,
    PositionWithTies,
    NullCheck,
    Quantifier,
    Quantify,
    RelatedRow,
    TemporalBucket,
)
from fervis.lookup.semantic_types import IdentifierType
from fervis.lookup.source_binding import (
    FactRealization,
    SetRealization,
    SourceBindingPlan,
    SourceMechanicKind,
    SubjectSurfaceReview,
    VerifiedSourceStrategy,
)

from .inputs import semantic_compiler_inputs
from fervis.lookup.available_sources import source_choice_literal
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
    occurrence_scopes: dict[str, OccurrenceScope] = field(default_factory=dict)
    occurrence_relations: dict[tuple[str, str], str] = field(default_factory=dict)
    relational_values: dict[tuple[str, FactLocalRef], RelationalValue] = field(
        default_factory=dict
    )
    attached_values: dict[tuple[str, str], str] = field(default_factory=dict)
    qualified_branches: dict[str, str] = field(default_factory=dict)
    normalized_relations: dict[str, str] = field(default_factory=dict)
    global_relational_values: dict[FactLocalRef, RelationalValue] = field(
        default_factory=dict
    )


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
    builder.occurrence_scopes = {
        branch.branch_id: occurrence_scope(
            verified.request, verified.binding_plan, branch.branch_id
        )
        for branch in verified.request.strategy.branches
    }
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
    program = prune_unused_program_nodes(program)
    from fervis.lookup.source_reads.access_compilation import expand_read_access
    program = expand_read_access(program,verified.request.source_catalog.read_access)
    program = replace(
        program,
        compatibility=build_program_compatibility(
            program,
            row_sources=RowSourceCatalog(verified.request.source_catalog.execution_sources),
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
                for source in verified.request.source_catalog.execution_sources
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


def _needs_qualified_row_union(index: RequestedFactSemanticIndex) -> bool:
    # Global quantifiers consume each qualifying branch through idempotent
    # Boolean folds. They do not require a row-identity union first.
    return not isinstance(index.result_grain, Singleton) or any(
        isinstance(node, Aggregate)
        and not isinstance(index.evaluation_domain_by_ref[ref], RowDomain)
        for ref, node in index.expression_by_ref.items()
    )


def _compile_requested_fact(builder: _ProgramBuilder) -> None:
    index = builder.verified.request.index
    plan = builder.verified.binding_plan
    branch_outputs = tuple(
        _compile_branch(builder, branch.branch_id, plan=plan, index=index)
        for branch in builder.verified.request.strategy.branches
    )
    input_relation = (
        branch_outputs[0]
        if len(branch_outputs) == 1 or not _needs_qualified_row_union(index)
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
    scope = builder.occurrence_scopes[branch_id]
    for position, occurrence in enumerate(scope.occurrences, start=1):
        builder.occurrence_relations[branch_id, occurrence.id] = (
            _compile_source_relation(
                builder,
                plan=plan,
                branch_id=branch_id,
                source_ref=occurrence.source_ref,
                source_position=position,
                occurrence_ref=occurrence.id,
            )
        )
    root_refs = (
        tuple(requirement.value_ref for requirement in index.output_requirements)
        + index.grouping_refs
        + index.ordering_refs
    )
    if index.requested_fact.qualification_ref is not None:
        root_refs += (_semantic_ref(index, index.requested_fact.qualification_ref),)
    root_sets = {index.subject_obligation.subject_set_ref.token}
    for ref in root_refs:
        root_sets.update(value_set_dependencies(builder.verified.request.index, ref))
    root_occurrences = _connected_occurrences(builder, branch_id, root_sets)
    current = _compile_relational_scope(
        builder, branch_id=branch_id, occurrence_refs=root_occurrences
    )
    for ref in root_refs:
        current = _attach_relational_values(builder, current, ref, branch_id=branch_id)
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
    builder.qualified_branches[branch_id] = current
    if len(
        builder.verified.request.strategy.branches
    ) > 1 and _needs_qualified_row_union(index):
        current = _normalize_branch_fields(
            builder, branch_id=branch_id, input_relation=current
        )
    return current


def _normalize_branch_fields(
    builder: _ProgramBuilder,
    *,
    branch_id: str,
    input_relation: str,
) -> str:
    """Project equivalent semantic obligations to the representative branch."""
    if input_relation in builder.normalized_relations:
        return builder.normalized_relations[input_relation]
    plan = builder.verified.binding_plan
    representative = _representative_branch(builder)
    projections: dict[str, str] = {}
    fields = {
        field.field_id: field for field in builder.fields_by_relation[input_relation]
    }
    observed = {ref.token for ref in builder.verified.request.index.observed_fact_refs}
    binding_groups = list(plan.set_bindings.items()) + [
        (ref, values) for ref, values in plan.fact_bindings.items() if ref in observed
    ]
    for logical_ref, values in binding_groups:
        current = next(
            (value for value in values if value.branch_id == branch_id), None
        )
        target = next(
            (value for value in values if value.branch_id == representative), None
        )
        if current is None or target is None:
            raise ValueError(
                "union branches must realize the same semantic obligations"
            )
        for input_field, output_field in _equivalent_realization_fields(
            builder, current, target, logical_ref=logical_ref
        ):
            if input_field not in fields:
                continue
            if output_field in projections and projections[output_field] != input_field:
                raise ValueError(
                    "union branch has conflicting realizations for one semantic field"
                )
            projections[output_field] = input_field
    for (value_branch, ref), value in builder.relational_values.items():
        if value_branch != branch_id or value.value_field not in fields:
            continue
        target_value = builder.relational_values.get((representative, ref))
        if target_value is None:
            raise ValueError("union branches disagree on exposed relational values")
        projections[target_value.value_field] = value.value_field
    grain = builder.grain_by_relation[input_relation]
    if not grain or not set(grain) <= set(projections.values()):
        raise ValueError(
            "union normalization requires the complete realized row identity"
        )
    outputs = tuple(sorted(projections))
    normalized_grain = tuple(
        sorted(target for target, source in projections.items() if source in grain)
    )
    if outputs == tuple(fields) and all(
        target == source for target, source in projections.items()
    ):
        return input_relation
    relation_id = f"{input_relation}.semantic"
    builder.operations.append(
        Operation(
            id=f"{relation_id}.operation",
            output_relation=relation_id,
            spec=ProjectSpec(
                input_relation=input_relation,
                outputs=tuple(
                    NamedExpression(target, FieldRef(projections[target]))
                    for target in outputs
                ),
            ),
        )
    )
    builder.fields_by_relation[relation_id] = tuple(
        replace(fields[projections[target]], field_id=target) for target in outputs
    )
    builder.grain_by_relation[relation_id] = normalized_grain
    builder.normalized_relations[input_relation] = relation_id
    return relation_id


def _equivalent_realization_fields(
    builder: _ProgramBuilder, current, target, *, logical_ref
) -> tuple[tuple[str, str], ...]:
    if current.identity_ref is not None or target.identity_ref is not None:
        catalog = builder.verified.request.source_catalog
        if current.identity_ref is None or target.identity_ref is None:
            raise ValueError("union branches disagree on identity authority")
        left, right = (
            catalog.identity(current.identity_ref),
            catalog.identity(target.identity_ref),
        )
        if (left.entity_kind, left.key_id) != (right.entity_kind, right.key_id):
            raise ValueError("union branches disagree on identity authority")
        source_components = {
            item.component_id: item.field_id
            for item in _identity_projection_components(
                builder,
                source_ref=current.source_ref,
                identity_ref=current.identity_ref,
                logical_ref=logical_ref,
                branch_id=current.branch_id,
            )
        }
        target_components = {
            item.component_id: item.field_id
            for item in _identity_projection_components(
                builder,
                source_ref=target.source_ref,
                identity_ref=target.identity_ref,
                logical_ref=logical_ref,
                branch_id=target.branch_id,
            )
        }
        if source_components.keys() != target_components.keys():
            raise ValueError("union branches disagree on identity components")
        return tuple(
            (source_components[key], target_components[key])
            for key in sorted(source_components)
        )
    current_refs = (
        current.identity_field_refs
        if isinstance(current, SetRealization)
        else current.field_refs
    )
    target_refs = (
        target.identity_field_refs
        if isinstance(target, SetRealization)
        else target.field_refs
    )
    if len(current_refs) != len(target_refs):
        raise ValueError("union branches disagree on semantic field arity")

    def physical(realization, field_ref):
        source = builder.verified.request.source_catalog.source(realization.source_ref)
        field_id = next(
            field.id for field in source.fields if field.field_ref == field_ref
        )
        return _execution_field_id(
            builder,
            source.id,
            field_id,
            logical_ref=logical_ref,
            branch_id=realization.branch_id,
        )

    return tuple(
        (physical(current, left), physical(target, right))
        for left, right in zip(current_refs, target_refs, strict=True)
    )


def _compile_source_relation(
    builder: _ProgramBuilder,
    *,
    plan: SourceBindingPlan,
    branch_id: str,
    source_ref: str,
    source_position: int,
    occurrence_ref: str,
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
        occurrence_ref=occurrence_ref,
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
    scoped_relation = f"{relation_base}.scoped"
    scoped_fields = tuple(
        replace(
            item,
            field_id=_execution_field_id(
                builder, source_ref, item.field_id, occurrence_ref=occurrence_ref
            ),
        )
        for item in fields
    )
    row_key = _occurrence_row_key(occurrence_ref)
    builder.operations.append(
        Operation(
            id=f"{scoped_relation}.operation",
            output_relation=scoped_relation,
            spec=ProjectSpec(
                current,
                (
                    *tuple(
                        NamedExpression(scoped.field_id, FieldRef(physical.field_id))
                        for physical, scoped in zip(fields, scoped_fields, strict=True)
                    ),
                    NamedExpression(
                        row_key, FunctionExpression(ExpressionFunction.ROW_NUMBER, ())
                    ),
                ),
            ),
        )
    )
    builder.fields_by_relation[scoped_relation] = (
        *scoped_fields,
        RelationField(row_key, (FieldBindingRole.IDENTITY, FieldBindingRole.PREDICATE)),
    )
    builder.grain_by_relation[scoped_relation] = tuple(
        _execution_field_id(builder, source_ref, key, occurrence_ref=occurrence_ref)
        for key in builder.grain_by_relation[current]
    ) or (row_key,)
    current = scoped_relation
    from fervis.lookup.source_binding.membership import map_membership
    occurrence = next(item for item in builder.occurrence_scopes[branch_id].occurrences if item.id == occurrence_ref)
    for set_ref in occurrence.set_refs:
        realization = next(item for item in plan.set_bindings[set_ref] if item.branch_id == branch_id)
        if realization.membership is None:
            continue
        if len(occurrence.set_refs) != 1:
            raise ValueError("restricted set membership requires an independent occurrence")
        def membership_binary(node, left, right):
            if node.operator is not ExpressionBinaryOperator.WITHIN:
                return BinaryExpression(node.operator, left, right)
            from fervis.lookup.answer_program.values import ParameterRef
            if not isinstance(node.right, ParameterRef):
                raise ValueError("within requires a certified temporal value")
            scope_value = next(item.typed_value.payload for item in builder.verified.request.canonical_values
                               if item.canonical_value_id == node.right.parameter_id)
            if not isinstance(scope_value, TimeValuePayload):
                raise ValueError("within requires a certified temporal scope")
            return _compile_temporal_scope(point=left, temporal_value=scope_value,
                lower_bound=builder.inputs.expression_for_value(node.right.parameter_id, component="start"),
                upper_bound=builder.inputs.expression_for_value(node.right.parameter_id, component="end"),
                constant_id=f"membership_day.{set_ref}.{node.right.parameter_id}")
        condition = map_membership(realization.membership,
            binary=membership_binary,
            field=lambda ref: FieldRef(_execution_field_id(builder, source_ref, ref.field_id, occurrence_ref=occurrence_ref)),
            value=lambda ref: builder.inputs.expression_for_value(ref.parameter_id))
        output = f"{scoped_relation}.members"
        builder.operations.append(Operation(id=f"{output}.filter", output_relation=output,
            spec=FilterSpec(current, condition, proof_refs=realization.contract_evidence_refs)))
        builder.fields_by_relation[output] = builder.fields_by_relation[current]
        builder.grain_by_relation[output] = builder.grain_by_relation[current]
        current = output
    return current


def _source_join_mode(
    builder: _ProgramBuilder, *, branch_id: str, source_ref: str, occurrence_ref: str
) -> JoinMode:
    # A predicate pushed into a related read requires a matching related row.
    # Otherwise joining observations must preserve candidate rows with no match.
    population_refs = {
        requirement.requirement_ref
        for requirement in builder.verified.request.index.boolean_requirements
        if requirement.use_site.value == "population"
    }
    required_match = any(
        mechanic.source_ref == source_ref
        and mechanic.kind is SourceMechanicKind.INVOCATION_PREDICATE
        and _owner_applies_to_occurrence(
            builder,
            ref,
            branch_id=branch_id,
            source_ref=source_ref,
            occurrence_ref=occurrence_ref,
        )
        for ref, realizations in builder.verified.binding_plan.boolean_bindings.items()
        if ref in population_refs
        for realization in realizations
        if realization.branch_id == branch_id
        for mechanic in realization.mechanics
    )
    return JoinMode.INNER if required_match else JoinMode.LEFT


def _compile_relational_scope(
    builder: _ProgramBuilder,
    *,
    branch_id: str,
    occurrence_refs: tuple[str, ...],
    association_refs: tuple[str, ...] | None = None,
    label: str = "join",
    initial_relation: str | None = None,
    initial_occurrences: tuple[str, ...] = (),
) -> str:
    scope = builder.occurrence_scopes[branch_id]
    by_id = {item.id: item for item in scope.occurrences}
    first = occurrence_refs[0]
    joined = set(initial_occurrences) if initial_relation is not None else {first}
    pending = [item for item in occurrence_refs if item not in joined]
    current = initial_relation or builder.occurrence_relations[branch_id, first]
    allowed = None if association_refs is None else set(association_refs)
    links = tuple(
        link
        for link in scope.links
        if allowed is None or link.association_ref in allowed
    )
    position = 0
    while pending:
        next_ref = next(
            (
                ref
                for ref in pending
                if any(
                    (link.left_occurrence == ref and link.right_occurrence in joined)
                    or (link.right_occurrence == ref and link.left_occurrence in joined)
                    for link in links
                )
            ),
            None,
        )
        if next_ref is None:
            raise ValueError(
                "relational scope lacks a declared connection between occurrences"
            )
        connecting = tuple(
            link
            for link in links
            if (link.left_occurrence == next_ref and link.right_occurrence in joined)
            or (link.right_occurrence == next_ref and link.left_occurrence in joined)
        )
        keys: list[JoinKey] = []
        for link in connecting:
            if link.left_occurrence in joined:
                left, right, left_fields, right_fields = (
                    link.left_occurrence,
                    link.right_occurrence,
                    link.left_fields,
                    link.right_fields,
                )
            else:
                left, right, left_fields, right_fields = (
                    link.right_occurrence,
                    link.left_occurrence,
                    link.right_fields,
                    link.left_fields,
                )
            keys.extend(
                JoinKey(
                    _execution_field_id(
                        builder, by_id[left].source_ref, lfield, occurrence_ref=left
                    ),
                    _execution_field_id(
                        builder, by_id[right].source_ref, rfield, occurrence_ref=right
                    ),
                )
                for lfield, rfield in zip(left_fields, right_fields, strict=True)
            )
        position += 1
        output_relation = f"{builder.verified.request.index.requested_fact_id}.{branch_id.rsplit(':', 1)[-1]}.{label}_{position}"
        right_relation = builder.occurrence_relations[branch_id, next_ref]
        builder.operations.append(
            Operation(
                id=f"{output_relation}.operation",
                output_relation=output_relation,
                spec=JoinSpec(
                    current,
                    right_relation,
                    tuple(dict.fromkeys(keys)),
                    mode=_source_join_mode(
                        builder,
                        branch_id=branch_id,
                        source_ref=by_id[next_ref].source_ref,
                        occurrence_ref=next_ref,
                    ),
                ),
            )
        )
        builder.fields_by_relation[output_relation] = tuple(
            dict.fromkeys(
                (
                    *builder.fields_by_relation[current],
                    *builder.fields_by_relation[right_relation],
                )
            )
        )
        builder.grain_by_relation[output_relation] = tuple(
            dict.fromkeys(
                (
                    *builder.grain_by_relation[current],
                    *builder.grain_by_relation[right_relation],
                )
            )
        )
        current = output_relation
        joined.add(next_ref)
        pending.remove(next_ref)
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


def _returned_choice_condition(
    builder: _ProgramBuilder,
    *,
    review: SubjectSurfaceReview,
    occurrence_ref=None,
    logical_ref=None,
    branch_id=None,
) -> Expression:
    surface = builder.verified.request.source_catalog.choice_surface(review.surface_ref)
    source = builder.verified.request.source_catalog.source(surface.source_ref)
    field_id = next(
        field.id for field in source.fields if field.field_ref == surface.target_ref
    )
    comparisons = tuple(
        BinaryExpression(
            operator=ExpressionBinaryOperator.EQUALS,
            left=FieldRef(
                _execution_field_id(
                    builder,
                    source.id,
                    field_id,
                    occurrence_ref=occurrence_ref,
                    logical_ref=logical_ref,
                    branch_id=branch_id,
                )
            ),
            right=ConstantRef(
                constant_id=value.value_ref,
                version_ref=builder.verified.request.source_catalog.contract_snapshot.ref,
                value=source_choice_literal(
                    value,
                    snapshot_ref=builder.verified.request.source_catalog.contract_snapshot.ref,
                ),
            ),
        )
        for choice_ref in review.included_choice_refs
        for value in (builder.verified.request.source_catalog.choice_value(choice_ref),)
    )
    if not comparisons:
        constant_id = f"membership.empty:{review.surface_ref}"
        return ConstantRef(
            constant_id,
            builder.verified.request.source_catalog.contract_snapshot.ref,
            FactValue.literal(
                id=constant_id, literal_type=LiteralType.BOOLEAN, value="false"
            ),
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
        for field_ref in source.stable_grain_field_refs
    }
    for param in source.params:
        population = builder.verified.request.parameter_population(source_ref, param.param_ref)
        if population is not None and population.field_path:
            for field in source.fields:
                if population.field_path in (field.path, field.response_path, field.field_ref):
                    roles.setdefault(field.field_ref, set()).add(FieldBindingRole.PREDICATE)
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
    from fervis.lookup.answer_program.expressions import expression_references
    for set_members in plan.set_bindings.values():
        for membership_realization in set_members:
            if membership_realization.branch_id == branch_id and membership_realization.source_ref == source_ref and membership_realization.membership is not None:
                for ref in expression_references(membership_realization.membership).fields:
                    roles.setdefault(source.field(ref.field_id).field_ref, set()).add(FieldBindingRole.PREDICATE)
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
                roles.setdefault(surface.target_ref, set()).add(
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


def _source_grain_field_ids(source: RowSource) -> tuple[str, ...]:
    grain_refs = set(source.stable_grain_field_refs)
    return tuple(field.id for field in source.fields if field.field_ref in grain_refs)


def _source_kind(kind: RowSourceKind) -> SourceKind:
    return {
        RowSourceKind.API_READ: SourceKind.API_READ,
        RowSourceKind.GENERATED_CALENDAR: SourceKind.GENERATED_CALENDAR,
        RowSourceKind.MEMORY_READ: SourceKind.MEMORY_READ,
    }[kind]


def _owner_applies_to_occurrence(
    builder: _ProgramBuilder,
    owner_ref: str | None,
    *,
    branch_id: str,
    source_ref: str,
    occurrence_ref: str,
) -> bool:
    return builder.occurrence_scopes[branch_id].owner_applies(
        builder.verified.request,
        owner_ref,
        source_ref=source_ref,
        occurrence_ref=occurrence_ref,
    )


def _source_parameter_binding_sets(builder, *, branch_id, source_ref, occurrence_ref):
    from fervis.lookup.source_binding.invocation_bindings import invocation_binding_sets

    scope = builder.occurrence_scopes[branch_id]
    occurrence = next(item for item in scope.occurrences if item.id == occurrence_ref)
    return invocation_binding_sets(
        request=builder.verified.request,
        plan=builder.verified.binding_plan,
        scope=scope,
        occurrence=occurrence,
        branch_id=branch_id,
    )


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


def _returned_atom_refs(plan: SourceBindingPlan, *, branch_id: str) -> frozenset[str]:
    return frozenset(
        requirement_ref
        for requirement_ref, realizations in plan.boolean_bindings.items()
        for realization in realizations
        if realization.branch_id == branch_id
        and any(
            mechanic.kind
            in {
                SourceMechanicKind.RETURNED_ROW_PREDICATE,
                SourceMechanicKind.RETURNED_CHOICE_PREDICATE,
            }
            for mechanic in realization.mechanics
        )
    )


def _compile_clause(
    builder: _ProgramBuilder,
    atom_refs,
    *,
    returned_atoms: frozenset[str],
    branch_id: str,
    owner_expression_ref: str | None = None,
    use_site: BooleanRequirementUseSite = BooleanRequirementUseSite.POPULATION,
) -> Expression | None:
    expressions: list[Expression] = []
    requirements = {
        item.requirement_ref: item
        for item in builder.verified.request.index.boolean_requirements
    }
    for atom in sorted(atom_refs):
        requirement = next(
            (
                item
                for item in requirements.values()
                if item.atom_ref == atom
                and item.owner_expression_ref == owner_expression_ref
                and item.use_site is use_site
            ),
            None,
        )
        if requirement is None:
            raise ValueError("qualification atom lacks a scoped requirement")
        if requirement.requirement_ref not in returned_atoms:
            continue
        choice_conditions = tuple(
            _returned_choice_condition(
                builder,
                review=replace(
                    surface,
                    included_choice_refs=tuple(
                        choice.choice_ref
                        for choice in surface.choice_reviews
                        if requirement.requirement_ref
                        in choice.selection_requirement_refs
                    ),
                ),
                branch_id=branch_id,
                logical_ref=FactLocalRef.from_token(
                    builder.verified.request.requirement_fact_refs(
                        requirement.requirement_ref
                    )[0]
                ),
            )
            for branch in builder.verified.binding_plan.subject_binding.branch_realizations
            if branch.branch_id == branch_id
            for surface in branch.surface_reviews
            if any(
                mechanic.kind is SourceMechanicKind.RETURNED_CHOICE_PREDICATE
                for mechanic in surface.mechanics
            )
            and any(
                requirement.requirement_ref in choice.selection_requirement_refs
                for choice in surface.choice_reviews
            )
        )
        if choice_conditions:
            expressions.append(
                _combine_boolean_expressions(
                    choice_conditions, operator=ExpressionBinaryOperator.AND
                )
            )
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


def _compile_scoped_condition(
    builder: _ProgramBuilder,
    ref: FactLocalRef,
    *,
    branch_id: str,
    owner_expression_ref: str,
    use_site: BooleanRequirementUseSite,
    negated: bool = False,
) -> Expression:
    index = builder.verified.request.index
    node = index.expression_by_ref.get(ref)
    if isinstance(node, BooleanComposition):
        if node.operator is BooleanCompositionOperator.NOT:
            return _compile_scoped_condition(
                builder,
                _semantic_ref(index, node.argument_refs[0]),
                branch_id=branch_id,
                owner_expression_ref=owner_expression_ref,
                use_site=use_site,
                negated=not negated,
            )
        operator = ExpressionBinaryOperator(node.operator.value)
        if negated:
            operator = (
                ExpressionBinaryOperator.OR
                if operator is ExpressionBinaryOperator.AND
                else ExpressionBinaryOperator.AND
            )
        return _combine_boolean_expressions(
            tuple(
                _compile_scoped_condition(
                    builder,
                    _semantic_ref(index, child),
                    branch_id=branch_id,
                    owner_expression_ref=owner_expression_ref,
                    use_site=use_site,
                    negated=negated,
                )
                for child in node.argument_refs
            ),
            operator=operator,
        )
    condition = _compile_clause(
        builder,
        (
            BooleanAtomRef(
                ref.token,
                BooleanPolarity.NEGATIVE if negated else BooleanPolarity.POSITIVE,
            ),
        ),
        returned_atoms=_returned_atom_refs(
            builder.verified.binding_plan, branch_id=branch_id
        ),
        branch_id=branch_id,
        owner_expression_ref=owner_expression_ref,
        use_site=use_site,
    )
    if condition is not None:
        return condition
    constant_id = f"condition.satisfied:{ref.token}"
    return ConstantRef(
        constant_id,
        "semantic-question-contract@1",
        FactValue.literal(
            id=constant_id, literal_type=LiteralType.BOOLEAN, value="true"
        ),
    )


def _compile_aggregate_filter(
    builder: _ProgramBuilder,
    *,
    index: RequestedFactSemanticIndex,
    aggregate_ref: FactLocalRef,
    filter_ref: str,
    branch_id: str | None = None,
    qualification_applied: bool = True,
) -> Expression | None:
    if qualification_applied and index.requested_fact.qualification_ref == filter_ref:
        return None
    requirements = tuple(
        requirement
        for requirement in index.boolean_requirements
        if requirement.owner_expression_ref == aggregate_ref.token
    )
    if requirements and all(
        _requirement_is_invocation_realized(
            builder,
            requirement_ref=requirement.requirement_ref,
        )
        for requirement in requirements
    ):
        return None
    return _compile_scoped_condition(
        builder,
        _semantic_ref(index, filter_ref),
        branch_id=branch_id or _representative_branch(builder),
        owner_expression_ref=aggregate_ref.token,
        use_site=BooleanRequirementUseSite.AGGREGATE_FILTER,
    )


def _requirement_is_invocation_realized(
    builder: _ProgramBuilder,
    *,
    requirement_ref: str,
) -> bool:
    realizations = builder.verified.binding_plan.boolean_bindings.get(
        requirement_ref,
        (),
    )
    return all(
        any(
            mechanic.kind is SourceMechanicKind.INVOCATION_PREDICATE
            for realization in realizations
            if realization.branch_id == branch.branch_id
            for mechanic in realization.mechanics
        )
        for branch in builder.verified.request.strategy.branches
    )


def _aggregate_grain_fields(
    builder: _ProgramBuilder, argument: FactLocalRef, *, branch_id: str | None = None
) -> tuple[str, ...]:
    index = builder.verified.request.index
    if argument.kind is FactLocalKind.SET:
        sets = [argument]
    else:
        domain = index.evaluation_domain_by_ref[argument]
        if isinstance(domain, RowDomain):
            sets = [domain.owner_ref]
        elif isinstance(domain, AssociationExpandedDomain):
            sets = [
                index.fact_local_ref_by_local_id[set_id]
                for association_ref in domain.path_refs
                for association in (index.term_by_ref[association_ref],)
                if isinstance(association, AssociationTerm)
                for set_id in (association.from_set_ref, association.to_set_ref)
            ]
        else:
            sets = []
    fields: list[str] = []
    for set_ref in dict.fromkeys(sets):
        fields.extend(
            _set_scope_keys(
                builder, set_ref.token, branch_id or _representative_branch(builder)
            )
        )
    return tuple(dict.fromkeys(fields))


def _filter_missing_entity_groups(
    builder: _ProgramBuilder,
    *,
    input_relation: str,
    index: RequestedFactSemanticIndex,
) -> str:
    from fervis.lookup.answer_program.operations import FilterSpec

    keys = tuple(
        component.field_id
        for ref in index.grouping_refs
        if (identity := _result_entity_key(builder, ref)) is not None
        for component in identity.components
    )
    if not keys:
        return input_relation
    relation_id = f"{index.requested_fact_id}.entity_groups"
    builder.operations.append(
        Operation(
            id=f"{relation_id}.operation",
            output_relation=relation_id,
            spec=FilterSpec(
                input_relation=input_relation,
                condition=_combine_boolean_expressions(
                    tuple(
                        UnaryExpression(ExpressionUnaryOperator.NOT_NULL, FieldRef(key))
                        for key in dict.fromkeys(keys)
                    ),
                    operator=ExpressionBinaryOperator.AND,
                ),
            ),
        )
    )
    builder.fields_by_relation[relation_id] = builder.fields_by_relation[input_relation]
    builder.grain_by_relation[relation_id] = builder.grain_by_relation[input_relation]
    return relation_id


def _filter_aggregate_input(
    builder: _ProgramBuilder, aggregate: Aggregate, relation: str, *, branch_id: str
) -> str:
    if aggregate.filter_ref is None:
        return relation
    condition = _compile_aggregate_filter(
        builder,
        index=builder.verified.request.index,
        aggregate_ref=_semantic_ref(builder.verified.request.index, aggregate.id),
        filter_ref=aggregate.filter_ref,
        branch_id=branch_id,
    )
    if condition is None:
        return relation
    output = f"{relation}.{aggregate.id}.filtered"
    builder.operations.append(
        Operation(
            id=f"{output}.operation",
            output_relation=output,
            spec=FilterSpec(relation, condition),
        )
    )
    builder.fields_by_relation[output] = builder.fields_by_relation[relation]
    builder.grain_by_relation[output] = builder.grain_by_relation[relation]
    return output


def _declare_independent_population_guarantee(
    builder: _ProgramBuilder, relation_id: str
) -> None:
    """Certify membership without claiming an unrelated subject's qualification."""
    if any(item.relation_id == relation_id for item in builder.relation_guarantees):
        return
    index = builder.verified.request.index
    builder.relation_guarantees.append(
        RelationGuaranteeDeclaration(
            relation_id=relation_id,
            qualification=QualificationGuarantee(
                requested_fact_id=index.requested_fact_id,
                formula=qualification_dnf(index.requested_fact_id, (frozenset(),)),
                atom_proofs=(),
            ),
            subject=builder.verified.subject_guarantee,
        )
    )


def _aggregate_input_relation(
    builder: _ProgramBuilder,
    aggregate: Aggregate,
    *,
    input_relation: str,
    index: RequestedFactSemanticIndex,
) -> str:
    refs = (_semantic_ref(index, aggregate.argument_ref),) + (
        ()
        if aggregate.filter_ref is None
        else (_semantic_ref(index, aggregate.filter_ref),)
    )
    sets = set().union(
        *(value_set_dependencies(builder.verified.request.index, ref) for ref in refs)
    )
    fields = {item.field_id for item in builder.fields_by_relation[input_relation]}
    representative = _representative_branch(builder)
    required_keys = {
        key for ref in sets for key in _set_scope_keys(builder, ref, representative)
    }
    if required_keys <= fields and len(builder.verified.request.strategy.branches) == 1:
        relation = input_relation
        for ref in refs:
            relation = _attach_relational_values(
                builder, relation, ref, branch_id=representative
            )
        return _filter_aggregate_input(
            builder, aggregate, relation, branch_id=representative
        )
    inputs = []
    for branch in builder.verified.request.strategy.branches:
        occurrences = _connected_occurrences(builder, branch.branch_id, sets)
        root = builder.qualified_branches[branch.branch_id]
        root_fields = {item.field_id for item in builder.fields_by_relation[root]}
        scope = builder.occurrence_scopes[branch.branch_id]
        initial = tuple(
            item.id
            for item in scope.occurrences
            if item.id in occurrences
            and all(
                set(_set_scope_keys(builder, ref, branch.branch_id)) <= root_fields
                for ref in item.set_refs
            )
        )
        relation = _compile_relational_scope(
            builder,
            branch_id=branch.branch_id,
            occurrence_refs=occurrences,
            label=f"aggregate_scope_{aggregate.id}",
            initial_relation=root if initial else None,
            initial_occurrences=initial,
        )
        if not initial:
            _declare_independent_population_guarantee(builder, relation)
        for ref in refs:
            relation = _attach_relational_values(
                builder, relation, ref, branch_id=branch.branch_id
            )
        relation = _filter_aggregate_input(
            builder, aggregate, relation, branch_id=branch.branch_id
        )
        if len(builder.verified.request.strategy.branches) > 1:
            relation = _normalize_branch_fields(
                builder, branch_id=branch.branch_id, input_relation=relation
            )
        inputs.append(relation)
    if len(inputs) == 1:
        return inputs[0]
    return _union_relations(
        builder,
        relation_ids=tuple(inputs),
        output_relation=f"{index.requested_fact_id}.aggregate_input_{aggregate.id}",
    )


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
        if isinstance(node, Aggregate)
        and ref in required_expression_refs
        and not isinstance(index.evaluation_domain_by_ref[ref], RowDomain)
    )
    current = input_relation
    current = _project_semantic_values(
        builder,
        input_relation=current,
        refs=index.grouping_refs,
        aggregate_fields={},
        label="group_values",
    )
    current = _filter_missing_entity_groups(
        builder, input_relation=current, index=index
    )
    group_fields = (
        _subject_grain_fields(builder, index=index)
        if isinstance(index.result_grain, SubjectRows)
        else tuple(
            field
            for ref in index.grouping_refs
            for field in _result_key_fields(builder, ref, aggregate_fields={})
        )
    )
    aggregate_fields: dict[FactLocalRef, str] = {
        ref: value.value_field
        for (branch, ref), value in builder.relational_values.items()
        if branch == _representative_branch(builder)
        and isinstance(index.expression_by_ref[ref], Aggregate)
    }
    if aggregates:
        input_groups: dict[str, list[tuple[FactLocalRef, Aggregate]]] = {}
        for position, (ref, aggregate) in enumerate(aggregates, start=1):
            aggregate_fields[ref] = f"aggregate_{position}"
            argument = _semantic_ref(index, aggregate.argument_ref)
            aggregate_input = (
                _aggregate_input_relation(
                    builder, aggregate, input_relation=current, index=index
                )
                if not group_fields
                else current
            )
            input_groups.setdefault(aggregate_input, []).append((ref, aggregate))
        aggregate_relations = []
        for group_position, (aggregate_input, scoped_aggregates) in enumerate(
            input_groups.items(), start=1
        ):
            aggregate_input = _project_semantic_values(
                builder,
                input_relation=aggregate_input,
                refs=tuple(
                    _semantic_ref(index, aggregate.argument_ref)
                    for _, aggregate in scoped_aggregates
                ),
                aggregate_fields={},
                label=f"aggregate_arguments_{group_position}",
            )
            specs = tuple(
                AggregationSpec(
                    function=_aggregation_function(aggregate),
                    output_field=aggregate_fields[ref],
                    input_field=""
                    if (argument := _semantic_ref(index, aggregate.argument_ref)).kind
                    is FactLocalKind.SET
                    else _result_field(builder, argument, aggregate_fields={}),
                    filter=None
                    if aggregate.filter_ref is None or not group_fields
                    else _compile_aggregate_filter(
                        builder,
                        index=index,
                        aggregate_ref=ref,
                        filter_ref=aggregate.filter_ref,
                    ),
                    distinct_argument=aggregate.distinct_argument,
                    grain_fields=_aggregate_grain_fields(builder, argument),
                )
                for ref, aggregate in scoped_aggregates
            )
            suffix = "" if len(input_groups) == 1 else f"_{group_position}"
            output_relation = f"{index.requested_fact_id}.aggregate{suffix}"
            operation_id = f"{index.requested_fact_id}.aggregate_operation{suffix}"
            builder.operations.append(
                Operation(
                    id=operation_id,
                    output_relation=output_relation,
                    spec=AggregateSpec(aggregate_input, group_fields, specs),
                )
            )
            builder.fields_by_relation[output_relation] = tuple(
                RelationField(field_id, (FieldBindingRole.OUTPUT,))
                for field_id in (
                    *group_fields,
                    *(aggregate_fields[ref] for ref, _ in scoped_aggregates),
                )
            )
            builder.grain_by_relation[output_relation] = group_fields
            if not group_fields:
                builder.scalar_aggregate_outputs.update(
                    {
                        ref: NodeOutputRef(operation_id, aggregate_fields[ref])
                        for ref, _ in scoped_aggregates
                    }
                )
            aggregate_relations.append(output_relation)
        current = aggregate_relations[0]
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
    if isinstance(index.result_grain, SubjectRows):
        keys = _subject_grain_fields(builder, index=index)
        retained = tuple(
            dict.fromkeys(
                field
                for ref in result_value_refs
                for field in _result_key_fields(
                    builder, ref, aggregate_fields=aggregate_fields
                )
            )
        )
        output_relation = f"{index.requested_fact_id}.subject_rows"
        builder.operations.append(
            Operation(
                id=f"{output_relation}.operation",
                output_relation=output_relation,
                spec=ProjectToKeySpec(
                    current,
                    keys,
                    tuple(field for field in retained if field not in keys),
                ),
            )
        )
        builder.fields_by_relation[output_relation] = tuple(
            item
            for item in builder.fields_by_relation[current]
            if item.field_id in {*keys, *retained}
        )
        builder.grain_by_relation[output_relation] = keys
        current = output_relation
    if requested_fact.distinct_by:
        output_relation = f"{index.requested_fact_id}.distinct"
        key_fields = tuple(
            field
            for local_id in requested_fact.distinct_by
            for field in _result_key_fields(
                builder,
                _semantic_ref(index, local_id),
                aggregate_fields=aggregate_fields,
            )
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
    relational = builder.relational_values.get((_representative_branch(builder), ref))
    if relational is not None and relational.key_set_ref is None:
        return NodeOutputRef(relational.operation_id, relational.value_field)
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

    return _compile_node(
        builder, node, branch_id=_representative_branch(builder), value_resolver=value
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
    value_resolver: Callable[[str], Expression] | None = None,
) -> Expression:
    index = builder.verified.request.index

    def value(ref: str) -> Expression:
        if value_resolver is not None:
            return value_resolver(ref)
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
        return _temporal_bucket_expression(
            value(node.value_ref),
            grain=node.grain.value,
            constant_id=f"temporal_grain.{node.id}",
        )
    if isinstance(node, (Aggregate, Quantify, RelatedRow, Coverage)):
        value_binding = _lower_relational_value(
            builder, index.fact_local_ref_by_local_id[node.id], branch_id=branch_id
        )
        if value_binding.key_set_ref is None:
            return NodeOutputRef(value_binding.operation_id, value_binding.value_field)
        return FieldRef(value_binding.value_field)
    raise ValueError(
        f"semantic node {type(node).__name__} requires relational lowering"
    )


def _connected_occurrences(
    builder: _ProgramBuilder,
    branch_id: str,
    set_refs: Collection[str],
    *,
    association_refs: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    scope = builder.occurrence_scopes[branch_id]
    needed = {scope.for_set(ref).id for ref in set_refs}
    if not needed:
        raise ValueError("relational scope requires a row domain")
    ordered = [item.id for item in scope.occurrences if item.id in needed]
    connected = {ordered[0]}
    allowed = None if association_refs is None else set(association_refs)
    links = tuple(
        link
        for link in scope.links
        if allowed is None or link.association_ref in allowed
    )
    for target in ordered[1:]:
        queue: list[tuple[str, tuple[str, ...]]] = [(start, ()) for start in connected]
        visited = set(connected)
        path = None
        while queue:
            current, prefix = queue.pop(0)
            if current == target:
                path = prefix
                break
            for link in links:
                other = (
                    link.right_occurrence
                    if link.left_occurrence == current
                    else link.left_occurrence
                    if link.right_occurrence == current
                    else None
                )
                if other is not None and other not in visited:
                    visited.add(other)
                    queue.append((other, (*prefix, other)))
        if path is None:
            raise ValueError("logical row domains lack a declared association path")
        connected.update(path)
    return tuple(item.id for item in scope.occurrences if item.id in connected)


def _set_scope_keys(
    builder: _ProgramBuilder, set_ref: str, branch_id: str
) -> tuple[str, ...]:
    values = builder.verified.binding_plan.set_bindings[set_ref]
    realized = next(value for value in values if value.branch_id == branch_id)
    if realized.identity_ref is None:
        return (
            _occurrence_row_key(
                builder.occurrence_scopes[branch_id].for_set(set_ref).id
            ),
        )
    return tuple(
        component.field_id
        for component in _identity_projection_components(
            builder,
            source_ref=realized.source_ref,
            identity_ref=realized.identity_ref,
            logical_ref=set_ref,
            branch_id=branch_id,
        )
    )


def _attach_relational_values(
    builder: _ProgramBuilder,
    relation_id: str,
    ref: FactLocalRef | str,
    *,
    branch_id: str,
) -> str:
    if isinstance(ref, str):
        return relation_id
    index = builder.verified.request.index
    node = index.expression_by_ref.get(ref)
    if isinstance(node, Aggregate) and not isinstance(
        index.evaluation_domain_by_ref[ref], RowDomain
    ):
        return relation_id
    if isinstance(node, Quantify) and not relational_free_sets(index, node):
        # Global results are lowered after every qualifying branch is available.
        return relation_id
    if isinstance(node, (Aggregate, Quantify, RelatedRow, Coverage)):
        value = _lower_relational_value(builder, ref, branch_id=branch_id)
        if value.key_set_ref is None:
            return relation_id
        cache_key = (relation_id, value.relation_id)
        if cache_key in builder.attached_values:
            return builder.attached_values[cache_key]
        output = f"{relation_id}.with_{ref.local_id}"
        builder.operations.append(
            Operation(
                id=f"{output}.operation",
                output_relation=output,
                spec=JoinSpec(
                    relation_id,
                    value.relation_id,
                    tuple(JoinKey(key, key) for key in value.key_fields),
                    JoinMode.LEFT,
                ),
            )
        )
        builder.fields_by_relation[output] = tuple(
            dict.fromkeys(
                (
                    *builder.fields_by_relation[relation_id],
                    *builder.fields_by_relation[value.relation_id],
                )
            )
        )
        builder.grain_by_relation[output] = builder.grain_by_relation[relation_id]
        builder.attached_values[cache_key] = output
        return output
    for child in sorted(index.direct_dependencies_by_ref.get(ref, ()), key=str):
        relation_id = _attach_relational_values(
            builder, relation_id, child, branch_id=branch_id
        )
    return relation_id


def _quantifier_aggregation(node: Quantify | RelatedRow) -> AggregationFunction:
    return (
        AggregationFunction.BOOL_ALL
        if isinstance(node, Quantify) and node.quantifier is Quantifier.FORALL
        else AggregationFunction.BOOL_ANY
    )


def _lower_global_quantifier(
    builder: _ProgramBuilder, ref: FactLocalRef, node: Quantify
) -> RelationalValue:
    """Fold Boolean observations over all qualifying branches, including empty ones."""
    index = builder.verified.request.index
    base = f"{index.requested_fact_id}.{ref.local_id}.global"
    condition_field, presence_field = f"{base}.condition", f"{base}.present"
    over_ref = index.fact_local_ref_by_local_id[node.over_set_ref].token
    associations = tuple(
        index.fact_local_ref_by_local_id[item].token for item in node.association_refs
    )
    sets = {over_ref}
    for association in associations:
        sets.update(value_set_dependencies(index, FactLocalRef.from_token(association)))
    condition_ref = (
        None if node.condition_ref is None else _semantic_ref(index, node.condition_ref)
    )
    if condition_ref is not None:
        sets.update(value_set_dependencies(index, condition_ref))
    projections = []
    for branch in builder.verified.request.strategy.branches:
        root = builder.qualified_branches[branch.branch_id]
        root_fields = {item.field_id for item in builder.fields_by_relation[root]}
        scope = builder.occurrence_scopes[branch.branch_id]
        occurrences = _connected_occurrences(
            builder, branch.branch_id, sets, association_refs=associations or None
        )
        initial = tuple(
            item.id
            for item in scope.occurrences
            if item.id in occurrences
            and all(
                set(_set_scope_keys(builder, set_ref, branch.branch_id)) <= root_fields
                for set_ref in item.set_refs
            )
        )
        domain = _compile_relational_scope(
            builder,
            branch_id=branch.branch_id,
            occurrence_refs=occurrences,
            association_refs=associations or None,
            label=f"{ref.local_id}.qualified_domain",
            initial_relation=root if initial else None,
            initial_occurrences=initial,
        )
        presence = _combine_boolean_expressions(
            tuple(
                UnaryExpression(ExpressionUnaryOperator.NOT_NULL, FieldRef(key))
                for key in _set_scope_keys(builder, over_ref, branch.branch_id)
            ),
            operator=ExpressionBinaryOperator.AND,
        )
        if condition_ref is None:
            condition = presence
        else:
            domain = _attach_relational_values(
                builder, domain, condition_ref, branch_id=branch.branch_id
            )
            condition = _compile_scoped_condition(
                builder,
                condition_ref,
                branch_id=branch.branch_id,
                owner_expression_ref=ref.token,
                use_site=BooleanRequirementUseSite.QUANTIFIER_CONDITION,
            )
        projected = f"{base}.{branch.branch_id}.observations"
        builder.operations.append(
            Operation(
                id=f"{projected}.operation",
                output_relation=projected,
                spec=ProjectSpec(
                    domain,
                    (
                        NamedExpression(condition_field, condition),
                        NamedExpression(presence_field, presence),
                    ),
                ),
            )
        )
        builder.fields_by_relation[projected] = (
            RelationField(condition_field, (FieldBindingRole.PREDICATE,)),
            RelationField(presence_field, (FieldBindingRole.PREDICATE,)),
        )
        builder.grain_by_relation[projected] = ()
        if not initial:
            _declare_independent_population_guarantee(builder, projected)
        projections.append(projected)
    domain = projections[0]
    if len(projections) > 1:
        domain = f"{base}.observations"
        # ANY and ALL are idempotent: overlapping branches do not need row deduplication.
        builder.operations.append(
            Operation(
                id=f"{domain}.operation",
                output_relation=domain,
                spec=UnionSpec(tuple(projections), (condition_field, presence_field)),
            )
        )
        builder.fields_by_relation[domain] = builder.fields_by_relation[projections[0]]
        builder.grain_by_relation[domain] = ()
    quantified = f"{base}.quantified"
    result_field = f"{base}.truth"
    operation_id = f"{quantified}.operation"
    function = _quantifier_aggregation(node)
    builder.operations.append(
        Operation(
            id=operation_id,
            output_relation=quantified,
            spec=AggregateSpec(
                domain,
                (),
                (
                    AggregationSpec(
                        function,
                        result_field,
                        condition_field,
                        filter=FieldRef(presence_field),
                    ),
                ),
            ),
        )
    )
    builder.fields_by_relation[quantified] = (
        RelationField(result_field, (FieldBindingRole.OUTPUT,)),
    )
    builder.grain_by_relation[quantified] = ()
    if node.quantifier is Quantifier.NOT_EXISTS:
        scalar = f"{base}.not_exists"
        builder.operations.append(
            Operation(
                id=scalar,
                spec=ComputeSpec(
                    UnaryExpression(
                        ExpressionUnaryOperator.NOT,
                        NodeOutputRef(operation_id, result_field),
                    ),
                    scalar,
                ),
            )
        )
        operation_id = result_field = scalar
    return RelationalValue(quantified, None, (), result_field, operation_id)


def _lower_relational_value(
    builder: _ProgramBuilder, ref: FactLocalRef, *, branch_id: str
) -> RelationalValue:
    index = builder.verified.request.index
    node = index.expression_by_ref[ref]
    if (
        isinstance(node, Quantify)
        and not relational_free_sets(index, node)
        and len(builder.qualified_branches)
        == len(builder.verified.request.strategy.branches)
    ):
        if ref not in builder.global_relational_values:
            builder.global_relational_values[ref] = _lower_global_quantifier(
                builder, ref, node
            )
        return builder.global_relational_values[ref]
    cached = builder.relational_values.get((branch_id, ref))
    if cached is not None:
        return cached
    index = builder.verified.request.index
    node = index.expression_by_ref[ref]
    if isinstance(node, Aggregate):
        return _lower_correlated_aggregate(builder, ref, node, branch_id=branch_id)
    if isinstance(node, Coverage):
        return _lower_coverage_value(builder, ref, node, branch_id=branch_id)
    assert isinstance(node, (Quantify, RelatedRow))
    over = node.over_set_ref if isinstance(node, Quantify) else node.set_ref
    free_sets = relational_free_sets(builder.verified.request.index, node)
    if len(free_sets) > 1:
        raise ValueError(
            "relational value requires a declared single outer candidate scope"
        )
    outer = next(iter(free_sets), None)
    sets = {*free_sets, index.fact_local_ref_by_local_id[over].token}
    associations = tuple(
        index.fact_local_ref_by_local_id[item].token for item in node.association_refs
    )
    for association in associations:
        sets.update(
            value_set_dependencies(
                builder.verified.request.index, FactLocalRef.from_token(association)
            )
        )
    if node.condition_ref is not None:
        sets.update(
            value_set_dependencies(
                builder.verified.request.index, _semantic_ref(index, node.condition_ref)
            )
        )
    occurrence_refs = _connected_occurrences(
        builder, branch_id, sets, association_refs=associations or None
    )
    if outer is not None:
        outer_occurrence = builder.occurrence_scopes[branch_id].for_set(outer).id
        occurrence_refs = (
            outer_occurrence,
            *(item for item in occurrence_refs if item != outer_occurrence),
        )
    domain = _compile_relational_scope(
        builder,
        branch_id=branch_id,
        occurrence_refs=occurrence_refs,
        association_refs=associations or None,
        label=f"{ref.local_id}.domain",
    )
    over_ref = index.fact_local_ref_by_local_id[over].token
    presence_keys = _set_scope_keys(builder, over_ref, branch_id)
    presence = _combine_boolean_expressions(
        tuple(
            UnaryExpression(ExpressionUnaryOperator.NOT_NULL, FieldRef(key))
            for key in presence_keys
        ),
        operator=ExpressionBinaryOperator.AND,
    )
    if node.condition_ref is None:
        condition = presence
    else:
        condition_ref = _semantic_ref(index, node.condition_ref)
        domain = _attach_relational_values(
            builder, domain, condition_ref, branch_id=branch_id
        )
        condition = _compile_scoped_condition(
            builder,
            condition_ref,
            branch_id=branch_id,
            owner_expression_ref=ref.token,
            use_site=BooleanRequirementUseSite.QUANTIFIER_CONDITION,
        )
    base = f"{index.requested_fact_id}.{branch_id}.{ref.local_id}"
    condition_field = f"{base}.condition"
    projected = f"{base}.values"
    builder.operations.append(
        Operation(
            id=f"{projected}.operation",
            output_relation=projected,
            spec=ProjectSpec(
                domain,
                (
                    *tuple(
                        NamedExpression(f.field_id, FieldRef(f.field_id))
                        for f in builder.fields_by_relation[domain]
                    ),
                    NamedExpression(condition_field, condition),
                ),
            ),
        )
    )
    builder.fields_by_relation[projected] = (
        *builder.fields_by_relation[domain],
        RelationField(condition_field, (FieldBindingRole.PREDICATE,)),
    )
    builder.grain_by_relation[projected] = builder.grain_by_relation[domain]
    keys = () if outer is None else _set_scope_keys(builder, outer, branch_id)
    quantified = f"{base}.quantified"
    result_field = f"{base}.truth"
    function = _quantifier_aggregation(node)
    builder.operations.append(
        Operation(
            id=f"{quantified}.operation",
            output_relation=quantified,
            spec=AggregateSpec(
                projected,
                keys,
                (
                    AggregationSpec(
                        function, result_field, condition_field, filter=presence
                    ),
                ),
            ),
        )
    )
    builder.fields_by_relation[quantified] = (
        *tuple(f for f in builder.fields_by_relation[projected] if f.field_id in keys),
        RelationField(
            result_field, (FieldBindingRole.OUTPUT, FieldBindingRole.PREDICATE)
        ),
    )
    builder.grain_by_relation[quantified] = keys
    operation_id = f"{quantified}.operation"
    if isinstance(node, Quantify) and node.quantifier is Quantifier.NOT_EXISTS:
        negated = f"{base}.not_exists"
        builder.operations.append(
            Operation(
                id=f"{negated}.operation",
                output_relation=negated,
                spec=ProjectSpec(
                    quantified,
                    (
                        *tuple(NamedExpression(key, FieldRef(key)) for key in keys),
                        NamedExpression(
                            result_field,
                            UnaryExpression(
                                ExpressionUnaryOperator.NOT, FieldRef(result_field)
                            ),
                        ),
                    ),
                ),
            )
        )
        builder.fields_by_relation[negated] = builder.fields_by_relation[quantified]
        builder.grain_by_relation[negated] = keys
        quantified = negated
        if not keys:
            scalar_id = f"{base}.scalar"
            builder.operations.append(
                Operation(
                    id=scalar_id,
                    spec=ComputeSpec(
                        UnaryExpression(
                            ExpressionUnaryOperator.NOT,
                            NodeOutputRef(operation_id, result_field),
                        ),
                        f"{base}.scalar_truth",
                    ),
                )
            )
            operation_id = scalar_id
            result_field = f"{base}.scalar_truth"
    value = RelationalValue(quantified, outer, keys, result_field, operation_id)
    builder.relational_values[branch_id, ref] = value
    return value


def _lower_correlated_aggregate(
    builder: _ProgramBuilder,
    ref: FactLocalRef,
    node: Aggregate,
    *,
    branch_id: str,
) -> RelationalValue:
    index = builder.verified.request.index
    domain = index.evaluation_domain_by_ref[ref]
    if not isinstance(domain, RowDomain):
        raise ValueError("aggregate requires its declared row or result scope")
    outer = domain.owner_ref.token
    argument = _semantic_ref(index, node.argument_ref)
    sets = {outer, *value_set_dependencies(builder.verified.request.index, argument)}
    if node.filter_ref is not None:
        sets.update(
            value_set_dependencies(
                builder.verified.request.index, _semantic_ref(index, node.filter_ref)
            )
        )
    occurrences = _connected_occurrences(builder, branch_id, sets)
    first = builder.occurrence_scopes[branch_id].for_set(outer).id
    occurrences = (first, *(item for item in occurrences if item != first))
    base = f"{index.requested_fact_id}.{branch_id}.{ref.local_id}"
    relation = _compile_relational_scope(
        builder,
        branch_id=branch_id,
        occurrence_refs=occurrences,
        label=f"{ref.local_id}.domain",
    )
    relation = _attach_relational_values(
        builder, relation, argument, branch_id=branch_id
    )
    if node.filter_ref is not None:
        relation = _attach_relational_values(
            builder,
            relation,
            _semantic_ref(index, node.filter_ref),
            branch_id=branch_id,
        )
    input_field = ""
    if argument.kind is not FactLocalKind.SET:
        input_field = f"{base}.argument"
        projected = f"{base}.values"
        builder.operations.append(
            Operation(
                id=f"{projected}.operation",
                output_relation=projected,
                spec=ProjectSpec(
                    relation,
                    (
                        *tuple(
                            NamedExpression(item.field_id, FieldRef(item.field_id))
                            for item in builder.fields_by_relation[relation]
                        ),
                        NamedExpression(
                            input_field,
                            _compile_semantic_ref(
                                builder, argument, branch_id=branch_id
                            ),
                        ),
                    ),
                ),
            )
        )
        builder.fields_by_relation[projected] = (
            *builder.fields_by_relation[relation],
            RelationField(input_field, (FieldBindingRole.OUTPUT,)),
        )
        builder.grain_by_relation[projected] = builder.grain_by_relation[relation]
        relation = projected
    keys = _set_scope_keys(builder, outer, branch_id)
    output = f"{base}.aggregate"
    value_field = f"{base}.value"
    operation_id = f"{output}.operation"
    builder.operations.append(
        Operation(
            id=operation_id,
            output_relation=output,
            spec=AggregateSpec(
                relation,
                keys,
                (
                    AggregationSpec(
                        _aggregation_function(node),
                        value_field,
                        input_field,
                        filter=None
                        if node.filter_ref is None
                        else _compile_aggregate_filter(
                            builder,
                            index=index,
                            aggregate_ref=ref,
                            filter_ref=node.filter_ref,
                            branch_id=branch_id,
                            qualification_applied=False,
                        ),
                        distinct_argument=node.distinct_argument,
                        grain_fields=_aggregate_grain_fields(
                            builder, argument, branch_id=branch_id
                        ),
                    ),
                ),
            ),
        )
    )
    builder.fields_by_relation[output] = (
        *tuple(
            item
            for item in builder.fields_by_relation[relation]
            if item.field_id in keys
        ),
        RelationField(
            value_field, (FieldBindingRole.OUTPUT, FieldBindingRole.PREDICATE)
        ),
    )
    builder.grain_by_relation[output] = keys
    result = RelationalValue(output, outer, keys, value_field, operation_id)
    builder.relational_values[branch_id, ref] = result
    return result


def _aggregation_function(node: Aggregate) -> AggregationFunction:
    return {
        "sum": AggregationFunction.SUM,
        "count": AggregationFunction.COUNT,
        "minimum": AggregationFunction.MIN,
        "maximum": AggregationFunction.MAX,
        "average": AggregationFunction.AVG,
    }[node.function.value]


def _lower_coverage_value(
    builder: _ProgramBuilder, ref: FactLocalRef, node: Coverage, *, branch_id: str
) -> RelationalValue:
    """For each candidate, every required member has some qualifying observation."""
    index = builder.verified.request.index
    scope = builder.occurrence_scopes[branch_id]
    candidate_ref, required_ref, observation_ref = (
        index.fact_local_ref_by_local_id[item].token
        for item in (
            node.candidate_set_ref,
            node.required_dimension_set_ref,
            node.observation_set_ref,
        )
    )
    candidate_occurrence = scope.for_set(candidate_ref).id
    required_occurrence = scope.for_set(required_ref).id
    if candidate_occurrence == required_occurrence:
        raise ValueError(
            "coverage requires independent candidate and required populations"
        )
    candidate = builder.occurrence_relations[branch_id, candidate_occurrence]
    required = builder.occurrence_relations[branch_id, required_occurrence]
    base = f"{index.requested_fact_id}.{branch_id}.{ref.local_id}"
    if node.required_member_condition_ref is not None:
        condition_ref = _semantic_ref(index, node.required_member_condition_ref)
        required = _attach_relational_values(
            builder, required, condition_ref, branch_id=branch_id
        )
        filtered = f"{base}.required"
        builder.operations.append(
            Operation(
                id=f"{filtered}.operation",
                output_relation=filtered,
                spec=FilterSpec(
                    required,
                    _compile_scoped_condition(
                        builder,
                        condition_ref,
                        branch_id=branch_id,
                        owner_expression_ref=ref.token,
                        use_site=BooleanRequirementUseSite.COVERAGE_REQUIRED_MEMBER,
                    ),
                ),
            )
        )
        builder.fields_by_relation[filtered] = builder.fields_by_relation[required]
        builder.grain_by_relation[filtered] = builder.grain_by_relation[required]
        required = filtered
    pairs = f"{base}.pairs"
    builder.operations.append(
        Operation(
            id=f"{pairs}.operation",
            output_relation=pairs,
            spec=CrossJoinSpec(candidate, required),
        )
    )
    builder.fields_by_relation[pairs] = (
        *builder.fields_by_relation[candidate],
        *builder.fields_by_relation[required],
    )
    builder.grain_by_relation[pairs] = (
        *builder.grain_by_relation[candidate],
        *builder.grain_by_relation[required],
    )
    associations = tuple(
        index.fact_local_ref_by_local_id[item].token
        for item in (
            *node.candidate_observation_association_refs,
            *node.dimension_observation_association_refs,
        )
    )
    sets = {candidate_ref, required_ref, observation_ref}
    for association in associations:
        sets.update(
            value_set_dependencies(
                builder.verified.request.index, FactLocalRef.from_token(association)
            )
        )
    occurrences = _connected_occurrences(
        builder, branch_id, sets, association_refs=associations
    )
    domain = _compile_relational_scope(
        builder,
        branch_id=branch_id,
        occurrence_refs=occurrences,
        association_refs=associations,
        label=f"{ref.local_id}.observations",
        initial_relation=pairs,
        initial_occurrences=(candidate_occurrence, required_occurrence),
    )
    condition_ref = _semantic_ref(index, node.condition_ref)
    domain = _attach_relational_values(
        builder, domain, condition_ref, branch_id=branch_id
    )
    condition_field = f"{base}.condition"
    values = f"{base}.values"
    builder.operations.append(
        Operation(
            id=f"{values}.operation",
            output_relation=values,
            spec=ProjectSpec(
                domain,
                (
                    *tuple(
                        NamedExpression(item.field_id, FieldRef(item.field_id))
                        for item in builder.fields_by_relation[domain]
                    ),
                    NamedExpression(
                        condition_field,
                        _compile_scoped_condition(
                            builder,
                            condition_ref,
                            branch_id=branch_id,
                            owner_expression_ref=ref.token,
                            use_site=BooleanRequirementUseSite.COVERAGE_CONDITION,
                        ),
                    ),
                ),
            ),
        )
    )
    builder.fields_by_relation[values] = (
        *builder.fields_by_relation[domain],
        RelationField(condition_field, (FieldBindingRole.PREDICATE,)),
    )
    builder.grain_by_relation[values] = builder.grain_by_relation[domain]
    candidate_keys = _set_scope_keys(builder, candidate_ref, branch_id)
    required_keys = _set_scope_keys(builder, required_ref, branch_id)
    presence = _combine_boolean_expressions(
        tuple(
            UnaryExpression(ExpressionUnaryOperator.NOT_NULL, FieldRef(key))
            for key in _set_scope_keys(builder, observation_ref, branch_id)
        ),
        operator=ExpressionBinaryOperator.AND,
    )
    pair_truth = f"{base}.pair_truth"
    pair_field = f"{base}.has_observation"
    pair_keys = (*candidate_keys, *required_keys)
    builder.operations.append(
        Operation(
            id=f"{pair_truth}.operation",
            output_relation=pair_truth,
            spec=AggregateSpec(
                values,
                pair_keys,
                (
                    AggregationSpec(
                        AggregationFunction.BOOL_ANY,
                        pair_field,
                        condition_field,
                        filter=presence,
                    ),
                ),
            ),
        )
    )
    builder.fields_by_relation[pair_truth] = (
        *tuple(
            item
            for item in builder.fields_by_relation[values]
            if item.field_id in pair_keys
        ),
        RelationField(pair_field, (FieldBindingRole.OUTPUT,)),
    )
    builder.grain_by_relation[pair_truth] = pair_keys
    all_truth = f"{base}.all_truth"
    truth_field = f"{base}.truth"
    count_field = f"{base}.required_count"
    builder.operations.append(
        Operation(
            id=f"{all_truth}.operation",
            output_relation=all_truth,
            spec=AggregateSpec(
                pair_truth,
                candidate_keys,
                (
                    AggregationSpec(
                        AggregationFunction.BOOL_ALL, truth_field, pair_field
                    ),
                    AggregationSpec(AggregationFunction.COUNT, count_field),
                ),
            ),
        )
    )
    builder.fields_by_relation[all_truth] = (
        *tuple(
            item
            for item in builder.fields_by_relation[pair_truth]
            if item.field_id in candidate_keys
        ),
        RelationField(truth_field, (FieldBindingRole.OUTPUT,)),
        RelationField(count_field, (FieldBindingRole.OUTPUT,)),
    )
    builder.grain_by_relation[all_truth] = candidate_keys
    joined = f"{base}.candidates"
    builder.operations.append(
        Operation(
            id=f"{joined}.operation",
            output_relation=joined,
            spec=JoinSpec(
                candidate,
                all_truth,
                tuple(JoinKey(key, key) for key in candidate_keys),
                JoinMode.LEFT,
            ),
        )
    )
    builder.fields_by_relation[joined] = tuple(
        dict.fromkeys(
            (
                *builder.fields_by_relation[candidate],
                *builder.fields_by_relation[all_truth],
            )
        )
    )
    builder.grain_by_relation[joined] = builder.grain_by_relation[candidate]
    output = f"{base}.result"
    # An absent grouped result means no required members. A present unknown truth stays unknown.
    expression = BinaryExpression(
        ExpressionBinaryOperator.OR,
        UnaryExpression(ExpressionUnaryOperator.IS_NULL, FieldRef(count_field)),
        FieldRef(truth_field),
    )
    builder.operations.append(
        Operation(
            id=f"{output}.operation",
            output_relation=output,
            spec=ProjectSpec(
                joined,
                (
                    *tuple(
                        NamedExpression(key, FieldRef(key)) for key in candidate_keys
                    ),
                    NamedExpression(truth_field, expression),
                ),
            ),
        )
    )
    builder.fields_by_relation[output] = (
        *tuple(
            item
            for item in builder.fields_by_relation[candidate]
            if item.field_id in candidate_keys
        ),
        RelationField(
            truth_field, (FieldBindingRole.OUTPUT, FieldBindingRole.PREDICATE)
        ),
    )
    builder.grain_by_relation[output] = candidate_keys
    result = RelationalValue(
        output, candidate_ref, candidate_keys, truth_field, f"{output}.operation"
    )
    builder.relational_values[branch_id, ref] = result
    return result


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


def _temporal_bucket_expression(
    value: Expression,
    *,
    grain: str,
    constant_id: str,
    timezone_ref: str = ANCHOR_TIMEZONE_REF,
) -> FunctionExpression:
    return FunctionExpression(
        ExpressionFunction.TEMPORAL_BUCKET,
        (
            value,
            ConstantRef(
                constant_id=constant_id,
                version_ref="semantic-question-contract@1",
                value=FactValue.literal(
                    id=constant_id, literal_type=LiteralType.STRING, value=grain
                ),
            ),
            EnvironmentRef(key=timezone_ref),
        ),
    )


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
    temporal_value = next(
        (
            item.typed_value.payload
            for item in builder.verified.request.canonical_values
            if item.input_ref == node.right_ref
        ),
        None,
    )
    if not isinstance(temporal_value, TimeValuePayload):
        raise ValueError("within requires a certified temporal scope")
    return _compile_temporal_scope(
        point=value(node.left_ref), temporal_value=temporal_value,
        lower_bound=builder.inputs.expression_for_question_input(node.right_ref, component="start"),
        upper_bound=builder.inputs.expression_for_question_input(node.right_ref, component="end"),
        constant_id=f"within_day.{node.id}",
    )


def _compile_temporal_scope(*, point, temporal_value, lower_bound, upper_bound, constant_id):
    if temporal_value.granularity != "hour":
        point = _temporal_bucket_expression(point, grain="day", constant_id=constant_id,
            timezone_ref=temporal_value.timezone_ref)
    return BinaryExpression(ExpressionBinaryOperator.AND,
        BinaryExpression(ExpressionBinaryOperator.GTE, point, lower_bound),
        BinaryExpression(ExpressionBinaryOperator.LTE, point, upper_bound))


def _occurrence_row_key(occurrence_ref: str) -> str:
    return f"row_occurrence:{occurrence_ref}"


def _source_needs_field_scope(builder: _ProgramBuilder, source_ref: str) -> bool:
    return any(
        scope.for_source(source_ref) and len(scope.occurrences) > 1
        for scope in builder.occurrence_scopes.values()
    )


def _occurrence_for_ref(builder, source_ref, logical_ref, branch_id=None):
    scopes = (
        (builder.occurrence_scopes[branch_id],)
        if branch_id is not None
        else tuple(builder.occurrence_scopes.values())
    )
    ref = (
        FactLocalRef.from_token(logical_ref)
        if isinstance(logical_ref, str)
        else logical_ref
    )
    index = builder.verified.request.index
    if ref is not None and ref.kind is FactLocalKind.FACT:
        term = index.term_by_ref[ref]
        owner = (
            term.value_type.set_ref
            if isinstance(term.value_type, IdentifierType)
            else term.owner_ref
        )
        ref = index.fact_local_ref_by_local_id[owner]
    candidates = {}
    for scope in scopes:
        if ref is not None and ref.kind is FactLocalKind.SET:
            occurrence = scope.for_set(ref.token)
            if occurrence.source_ref == source_ref:
                candidates[occurrence.id] = occurrence
        elif ref is not None and ref.kind is FactLocalKind.ASSOCIATION:
            link = next(
                (link for link in scope.links if link.association_ref == ref.token),
                None,
            )
            if link is not None:
                matches = tuple(
                    item
                    for item in scope.for_source(source_ref)
                    if item.id in {link.left_occurrence, link.right_occurrence}
                )
                if len(matches) > 1:
                    matches = tuple(
                        item for item in matches if item.id == link.left_occurrence
                    )
                candidates.update((item.id, item) for item in matches)
            else:
                term = index.term_by_ref[ref]
                occurrence = scope.for_set(
                    index.fact_local_ref_by_local_id[term.from_set_ref].token
                )
                if occurrence.source_ref == source_ref:
                    candidates[occurrence.id] = occurrence
        else:
            candidates.update((item.id, item) for item in scope.for_source(source_ref))
    if len(candidates) != 1:
        raise ValueError("producer field requires an unambiguous logical occurrence")
    return next(iter(candidates.values()))


def _execution_field_id(
    builder: _ProgramBuilder,
    source_ref: str,
    field_id: str,
    *,
    logical_ref=None,
    branch_id=None,
    occurrence_ref=None,
) -> str:
    if not _source_needs_field_scope(builder, source_ref):
        return field_id
    occurrence_id = (
        occurrence_ref
        or _occurrence_for_ref(builder, source_ref, logical_ref, branch_id).id
    )
    return f"source_field:{occurrence_id}:{field_id}"


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
    return _execution_field_id(
        builder,
        source.id,
        next(
            field.id
            for field in source.fields
            if field.field_ref == realization.field_refs[0]
        ),
        logical_ref=ref,
        branch_id=branch_id,
    )


def _result_key_fields(
    builder: _ProgramBuilder, ref, *, aggregate_fields
) -> tuple[str, ...]:
    identity = _result_entity_key(builder, ref)
    if identity is not None:
        return tuple(component.field_id for component in identity.components)
    return (_result_field(builder, ref, aggregate_fields=aggregate_fields),)


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


def _subject_grain_fields(
    builder: _ProgramBuilder, *, index: RequestedFactSemanticIndex
) -> tuple[str, ...]:
    identity = _result_entity_key(builder, index.subject_obligation.subject_set_ref)
    if identity is None:
        raise ValueError("subject rows require a declared identity")
    return tuple(component.field_id for component in identity.components)


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
    if not realizations or any(item.identity_ref is None for item in realizations):
        raise ValueError("identifier output requires one declared identity authority")
    catalog = builder.verified.request.source_catalog
    authorities = {
        (
            catalog.identity(item.identity_ref).entity_kind,
            catalog.identity(item.identity_ref).key_id,
        )
        for item in realizations
        if item.identity_ref is not None
    }
    if len(authorities) != 1:
        raise ValueError("identifier output requires one declared identity authority")
    representative = next(
        item
        for item in realizations
        if item.branch_id == _representative_branch(builder)
    )
    entity_kind, key_id = next(iter(authorities))
    return EntityKeyProjection(
        entity_kind=entity_kind,
        key_id=key_id,
        components=_identity_projection_components(
            builder,
            source_ref=representative.source_ref,
            identity_ref=representative.identity_ref,
            logical_ref=ref,
            branch_id=representative.branch_id,
        ),
    )


def _identity_projection_components(
    builder: _ProgramBuilder,
    *,
    source_ref: str,
    identity_ref: str | None,
    logical_ref=None,
    branch_id=None,
) -> tuple[EntityKeyProjectionComponent, ...]:
    if identity_ref is None:
        raise ValueError("identifier realization lacks identity authority")
    source = builder.verified.request.source_catalog.source(source_ref)
    evidence = builder.verified.request.source_catalog.identity(identity_ref)
    if evidence.source_ref != source_ref:
        raise ValueError("identity authority belongs to another source")
    candidate_key = next(
        (
            key
            for key in source.candidate_keys
            if identity_ref == f"source_identity:{source.id}:candidate_key:{key.id}"
        ),
        None,
    )
    if candidate_key is not None:
        return tuple(
            EntityKeyProjectionComponent(
                component.id,
                _execution_field_id(
                    builder,
                    source_ref,
                    component.field_id,
                    logical_ref=logical_ref,
                    branch_id=branch_id,
                ),
            )
            for component in candidate_key.components
        )
    entity_reference = next(
        (
            reference
            for reference in source.entity_references
            if identity_ref
            == f"source_identity:{source.id}:entity_reference:{reference.id}"
        ),
        None,
    )
    if entity_reference is None:
        raise ValueError("identifier authority is absent from its declared source")
    return tuple(
        EntityKeyProjectionComponent(
            component.target_component_id,
            _execution_field_id(
                builder,
                source_ref,
                component.local_field_id,
                logical_ref=logical_ref,
                branch_id=branch_id,
            ),
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
    builder.grain_by_relation[output_relation] = builder.grain_by_relation[
        input_relation
    ]
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

    return _compile_node(
        builder, node, branch_id=_representative_branch(builder), value_resolver=value
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
        return (
            AtPosition(position=expression)
            if isinstance(selection, PositionWithTies)
            else Take(limit=expression)
        )
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
    return _compiled_field(builder, ref, branch_id=_representative_branch(builder))


def _representative_branch(builder: _ProgramBuilder) -> str:
    return builder.verified.request.strategy.branches[0].branch_id


__all__ = [
    "compile_verified_source_strategies",
    "compile_verified_source_strategy",
]
