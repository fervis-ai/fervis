from dataclasses import replace

import pytest

from fervis.lookup.available_sources import (
    AvailableSourceCatalog,
    SourceContractSnapshot,
    SourceRelationEvidence,
)
from fervis.lookup.fact_compilation import (
    compile_verified_source_strategies,
    compile_verified_source_strategy,
)
from fervis.lookup.contract_codec import (
    canonical_answer_program_json,
    decode_answer_program,
)
from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
from fervis.lookup.answer_program.inputs import resolve_value_expression
from fervis.lookup.answer_program.invocation import RuntimePorts, invoke_answer_program
from fervis.lookup.answer_rendering import render_fact_result, rendered_fact_text
from fervis.lookup.answer_program.expressions import expression_references
from fervis.lookup.answer_program.operations import FilterSpec
from fervis.lookup.answer_program.values import (
    FactValue,
    LiteralType,
    ValueProjectionKind,
)
from fervis.lookup.relation_catalog.row_sources import (
    RowSource,
    RowSourceKind,
    build_row_source_catalog,
)
from fervis.lookup.relation_catalog.row_sources import (
    RowSourceEntityReference,
    RowSourceEntityReferenceComponent,
    RowSourceCandidateKey,
    RowSourceField,
    RowSourceKeyComponent,
    RowSourceParam,
    RowSourceValueType,
)
from fervis.lookup.grounding.semantic import CanonicalInputValue
from fervis.lookup.canonical_data import EntityKeyComponentValue, EntityKeyValue
from fervis.lookup.source_binding.model import (
    CandidateSourceStrategy,
    SourceStrategyBranch,
)
from fervis.lookup.question_contract.analysis import analyze_requested_fact
from fervis.lookup.question_contract.model import (
    Aggregate,
    AggregateFunction,
    AllResults,
    FirstRankWithTies,
    InstanceInterpretation,
    QuestionContract,
    RequestedFact,
    RequestedOutput,
    Comparison,
    FactTerm,
    InputDenotation,
    InputDenotationKind,
    InputTerm,
    AssociationTerm,
    Arithmetic,
    Ordering,
    OrderingDirection,
    Quantifier,
    Quantify,
    SetTerm,
    Subject,
)
from fervis.lookup.semantic_types import SourceOrigin, SourceOriginKind
from fervis.lookup.semantic_types import (
    CollectionType,
    DateTimeType,
    DecimalType,
    IdentifierType,
    PercentageMeasure,
    TemporalScopeType,
    UnitlessMeasure,
)
from fervis.lookup.expression_operators import ExpressionBinaryOperator
from fervis.lookup.relation_catalog.model import (
    CandidateKey,
    CandidateKeyComponent,
    CatalogField,
    EndpointRead,
    EntityReference,
    EntityReferenceComponent,
    EntityKeyComponentTarget,
    ParamSource,
    RowCardinality,
    RowPath,
)
from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.plan_execution.relations import (
    CompletenessProof,
    CompletenessSourceKind,
    CompletenessStatus,
    RelationRows,
)
from fervis.lookup.plan_execution.errors import VerificationError
from fervis.lookup.memory.projection import LookupMemory
from fervis.lookup.outcomes.model import AnswerResult
from fervis.lookup.source_binding.model import (
    AssociationRealization,
    AssociationRealizationKind,
    SemanticSourceBindingRequest,
    SetRealization,
    BooleanRequirementRealization,
    FactRealization,
    FactRealizationKind,
    SourceMechanic,
    SourceMechanicKind,
    InvocationTargetApplication,
    InvocationValueApplication,
    SourceBindingPlan,
    SubjectChoiceReview,
    SubjectObligationBinding,
    SubjectObligationRealization,
    SubjectSurfaceReview,
)
from fervis.lookup.source_binding.verification import (
    VerifiedSourceStrategy,
    verify_source_strategy,
)


def _strategy(requested_fact_id, *branches):
    return CandidateSourceStrategy(requested_fact_id, tuple(branches))


def _denotation(
    input_term: InputTerm,
    *,
    kind: InputDenotationKind = InputDenotationKind.NON_IDENTITY_SCALAR,
) -> dict[str, InputDenotation]:
    return {
        input_term.id: InputDenotation(
            id=f"denotation_{input_term.id}",
            input_ref=input_term.id,
            operand_meaning="test operand",
            denotation_basis="The test supplies this operand.",
            denoted_instance_kind=(
                "identified instance"
                if kind is InputDenotationKind.IDENTITY_REFERENCE
                else None
            ),
            kind=kind,
        )
    }


def _compile_memory_count(
    rows: tuple[dict[str, str], ...],
    *,
    fact_id: str = "fact_1",
    meaning: str = "event count",
    relation_id: str = "events",
):
    origin = SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, meaning)
    fact = RequestedFact(
        id=fact_id,
        origin=origin,
        sets=(SetTerm("s1", origin),),
        associations=(),
        facts=(),
        expressions=(
            Aggregate(
                id="e1",
                function=AggregateFunction.COUNT,
                argument_ref="s1",
                filter_ref=None,
                distinct_argument=False,
                origin=origin,
            ),
        ),
        subject=Subject("s1", InstanceInterpretation.NORMAL_BUSINESS_INSTANCE),
        qualification_ref=None,
        grouping_refs=(),
        outputs=(RequestedOutput("output_1", "e1", origin),),
        ordering=(),
        selection=AllResults(),
        distinct_by=(),
    )
    contract = QuestionContract(inputs=(), requested_facts=(fact,))
    index = analyze_requested_fact(fact, inputs={}, input_denotations={})
    memory_relation = RelationRows(
        id=relation_id,
        rows=rows,
        completeness=CompletenessProof(
            status=CompletenessStatus.COMPLETE,
            source_kind=CompletenessSourceKind.MEMORY_READ,
            proof_refs=("memory:events",),
        ),
    )
    row_sources = build_row_source_catalog(
        RelationCatalog(), memory_relations=(memory_relation,)
    )
    source = next(
        item for item in row_sources.sources if item.kind is RowSourceKind.MEMORY_READ
    )
    catalog = AvailableSourceCatalog(
        contract_snapshot=SourceContractSnapshot.from_content("{}"),
        sources=(source,),
        relation_evidence=(),
    )
    [clause] = index.qualification.clauses
    branch = SourceStrategyBranch(
        branch_id=f"{fact_id}:source_branch:1",
        source_refs=(source.id,),
        relation_evidence_refs=(),
        qualification_clause_refs=(clause.clause_ref,),
    )
    strategy = _strategy(fact.id, branch)
    request = SemanticSourceBindingRequest(
        index=index,
        strategy=strategy,
        source_catalog=catalog,
        canonical_values=(),
    )
    set_ref = index.subject_obligation.subject_set_ref.token
    plan = SourceBindingPlan(
        strategy=strategy,
        set_bindings={
            set_ref: (
                SetRealization(
                    branch_id=branch.branch_id,
                    mapping_basis="Event rows realize the event set.",
                    source_ref=source.id,
                    identity_ref=None,
                    identity_field_refs=(),
                    contract_evidence_refs=(catalog.contract_snapshot.ref,),
                ),
            )
        },
        fact_bindings={},
        association_bindings={},
        invocation_applications=(),
        boolean_bindings={},
        subject_binding=SubjectObligationBinding(
            subject_ref=set_ref,
            branch_realizations=(
                SubjectObligationRealization(
                    branch_id=branch.branch_id,
                    surface_reviews=(),
                ),
            ),
        ),
    )
    verified = verify_source_strategy(plan, request=request)
    assert isinstance(verified, VerifiedSourceStrategy)
    result = compile_verified_source_strategy(verified)
    execution = invoke_answer_program(
        program=result.answer_program,
        bindings=result.initial_bindings,
        environment=ExecutionEnvironment(
            catalog=RelationCatalog(),
            memory_relations=(memory_relation,),
        ),
        ports=RuntimePorts(
            data_access_port=None,
            memory=LookupMemory(relations=(memory_relation,)),
        ),
    )
    return contract, source, memory_relation, result, execution, verified


def test_verified_count_compiles_directly_to_answer_program() -> None:
    contract, source, memory_relation, result, execution, _ = _compile_memory_count(
        ({"event_id": "event_1"}, {"event_id": "event_2"})
    )

    assert contract.requested_facts == result.answer_program.fact_template
    assert result.initial_bindings.bindings == ()
    assert [
        relation.source.memory_relation_id
        for relation in result.answer_program.relations
    ] == ["events"]
    [output] = result.answer_program.result_projection.relation_outputs
    assert output.relation_id == "fact_1.aggregate"
    assert output.field_id == "aggregate_1"
    assert (
        decode_answer_program(canonical_answer_program_json(result.answer_program))
        == result.answer_program
    )

    assert execution.issue is None
    assert execution.fact_result is not None
    assert isinstance(execution.fact_result.outcome, AnswerResult)
    assert execution.fact_result.outcome.projected_rows[0].values == {
        "fact_1.output_1": 2
    }
    assert rendered_fact_text(render_fact_result(execution.fact_result)) == "2"

    with pytest.raises(
        VerificationError, match="fulfillment result lacks semantic guarantee"
    ):
        invoke_answer_program(
            program=replace(result.answer_program, relation_guarantees=()),
            bindings=result.initial_bindings,
            environment=ExecutionEnvironment(
                catalog=RelationCatalog(),
                memory_relations=(memory_relation,),
            ),
            ports=RuntimePorts(
                data_access_port=None,
                memory=LookupMemory(relations=(memory_relation,)),
            ),
        )


@pytest.mark.parametrize("unused_has_choices", [False, True])
def test_only_bound_sources_become_executable_reads(unused_has_choices) -> None:
    from tests.lookup.source_binding._fixtures import (
        compile_binding_fixture,
        validate_binding_fixture,
    )

    _, source, memory, _, _, existing = _compile_memory_count(({"event_id": "one"},))
    extra = replace(
        source,
        id="unrelated_source",
        memory_ref="never_read",
        fields=(
            RowSourceField(
                "status",
                "status",
                "status",
                RowSourceValueType.CHOICE,
                (),
                choices=("OPEN", "CLOSED"),
            ),
        )
        if unused_has_choices
        else source.fields,
    )
    [branch] = existing.request.strategy.branches
    candidate_strategy = replace(
        existing.request.strategy,
        branches=(replace(branch, source_refs=(source.id, extra.id)),),
    )
    request = replace(
        existing.request,
        strategy=candidate_strategy,
        source_catalog=replace(
            existing.request.source_catalog, sources=(source, extra)
        ),
    )
    subject_ref = request.index.subject_obligation.subject_set_ref.token
    payload = {
        "set_bindings": {
            subject_ref: [
                {
                    "branch_id": branch.branch_id,
                    "mapping_basis": "The event rows alone realize the counted set.",
                    "rows_ref": source.id,
                }
            ]
        },
        "fact_bindings": {},
        "association_bindings": {},
        "resolved_input_applications": {branch.branch_id: []},
        "finite_choice_applications": {branch.branch_id: {}},
        "subject_binding": {
            "subject_ref": subject_ref,
            "branch_realizations": [
                {
                    "branch_id": branch.branch_id,
                    "finite_choice_reviews": {},
                }
            ],
        },
    }
    validate_binding_fixture(payload, request=request)
    plan = compile_binding_fixture(payload, request=request)
    verified = verify_source_strategy(plan, request=request)
    assert isinstance(verified, VerifiedSourceStrategy)
    result = compile_verified_source_strategy(verified)
    assert [r.source.memory_relation_id for r in result.answer_program.relations] == [
        memory.id
    ]
    execution = invoke_answer_program(
        program=result.answer_program,
        bindings=result.initial_bindings,
        environment=ExecutionEnvironment(
            catalog=RelationCatalog(), memory_relations=(memory,)
        ),
        ports=RuntimePorts(
            data_access_port=None, memory=LookupMemory(relations=(memory,))
        ),
    )
    assert execution.issue is None
    assert rendered_fact_text(render_fact_result(execution.fact_result)) == "1"


def test_independent_facts_with_local_output_ids_render_with_their_scopes() -> None:
    _, _, first_relation, _, _, first = _compile_memory_count(
        ({"event_id": "a1"}, {"event_id": "a2"}, {"event_id": "a3"}),
        fact_id="fact_1",
        meaning="sales for staff A",
        relation_id="staff_a_sales",
    )
    _, _, second_relation, _, _, second = _compile_memory_count(
        ({"event_id": "b1"}, {"event_id": "b2"}),
        fact_id="fact_2",
        meaning="sales for staff B",
        relation_id="staff_b_sales",
    )

    compiled = compile_verified_source_strategies((first, second))
    execution = invoke_answer_program(
        program=compiled.answer_program,
        bindings=compiled.initial_bindings,
        environment=ExecutionEnvironment(
            catalog=RelationCatalog(),
            memory_relations=(first_relation, second_relation),
        ),
        ports=RuntimePorts(
            data_access_port=None,
            memory=LookupMemory(relations=(first_relation, second_relation)),
        ),
    )

    assert execution.issue is None
    assert execution.fact_result is not None
    assert rendered_fact_text(render_fact_result(execution.fact_result)) == (
        "sales for staff A: 3\nsales for staff B: 2"
    )


class _EventDataAccess:
    def read(self, *, endpoint_name, args):
        assert endpoint_name == "list_events"
        assert args == {}
        return {
            "responseStatus": 200,
            "responseBody": {
                "data": [
                    {"event_id": "event_1", "status": "ACTIVE"},
                    {"event_id": "event_2", "status": "ACTIVE"},
                    {"event_id": "event_3", "status": "DRAFT"},
                ]
            },
        }


@pytest.mark.parametrize("has_ordinary_state", [True, False])
def test_returned_choice_surface_excludes_nonordinary_rows_before_counting(has_ordinary_state) -> None:
    origin = SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, "event count")
    fact = RequestedFact(
        id="fact_1",
        origin=origin,
        sets=(SetTerm("s1", origin),),
        associations=(),
        facts=(),
        expressions=(
            Aggregate("e1", AggregateFunction.COUNT, "s1", None, False, origin),
        ),
        subject=Subject("s1", InstanceInterpretation.NORMAL_BUSINESS_INSTANCE),
        qualification_ref=None,
        grouping_refs=(),
        outputs=(RequestedOutput("output_1", "e1", origin),),
        ordering=(),
        selection=AllResults(),
        distinct_by=(),
    )
    index = analyze_requested_fact(fact, inputs={}, input_denotations={})
    read = EndpointRead(
        id="list_events",
        endpoint_name="list_events",
        row_paths=(RowPath("data", "data", RowCardinality.MANY),),
        fields=(
            CatalogField(
                ref="event.event_id",
                path="data.event_id",
                row_path_id="data",
                type="string",
            ),
            CatalogField(
                ref="event.status",
                path="data.status",
                row_path_id="data",
                type="choice",
                choices=("ACTIVE", "DRAFT"),
            ),
        ),
    )
    relation_catalog = RelationCatalog(reads=(read,))
    row_sources = build_row_source_catalog(relation_catalog)
    source = next(item for item in row_sources.sources if item.read_id == read.id)
    catalog = AvailableSourceCatalog(
        contract_snapshot=SourceContractSnapshot.from_content("{}"),
        sources=(source,),
        relation_evidence=(),
    )
    [clause] = index.qualification.clauses
    branch = SourceStrategyBranch(
        "fact_1:source_branch:1",
        (source.id,),
        (),
        (clause.clause_ref,),
    )
    strategy = _strategy(fact.id, branch)
    request = SemanticSourceBindingRequest(index, strategy, catalog, ())
    [surface] = catalog.choice_surfaces
    included_choice_ref = next(
        value.value_ref for value in surface.values if value.value == "ACTIVE"
    )
    mechanic = SourceMechanic(
        "Filter returned rows by the reviewed status choices.",
        source.id,
        (),
        (
            catalog.contract_snapshot.ref,
            surface.surface_ref,
            *(value.value_ref for value in surface.values),
        ),
        SourceMechanicKind.RETURNED_ROW_PREDICATE,
    )
    set_ref = index.subject_obligation.subject_set_ref.token
    plan = SourceBindingPlan(
        strategy,
        {
            set_ref: (
                SetRealization(
                    branch.branch_id,
                    "Event rows realize the event set.",
                    source.id,
                    None,
                    (),
                    (catalog.contract_snapshot.ref,),
                ),
            )
        },
        {},
        {},
        (),
        {},
        SubjectObligationBinding(
            set_ref,
            (
                SubjectObligationRealization(
                    branch.branch_id,
                    (
                        SubjectSurfaceReview(
                            owner_set_ref=set_ref,
                            surface_ref=surface.surface_ref,
                            surface_mapping_basis=(
                                "The status surface defines the event population."
                            ),
                            choice_reviews=tuple(
                                SubjectChoiceReview(
                                    choice_ref=value.value_ref,
                                    choice_domain_meaning=(
                                        f"{value.value} event rows."
                                    ),
                                    decision_basis=(
                                        "The choice is an ordinary event state."
                                        if value.value == "ACTIVE"
                                        else "The choice is a canceled event state."
                                    ),
                                    baseline_included=has_ordinary_state and value.value == "ACTIVE",
                                    explicit_user_override_applies=False,
                                )
                                for value in surface.values
                            ),
                            included_choice_refs=(included_choice_ref,) if has_ordinary_state else (),
                            mechanics=(mechanic,),
                        ),
                    ),
                ),
            ),
        ),
    )
    verified = verify_source_strategy(plan, request=request)
    assert isinstance(verified, VerifiedSourceStrategy)
    result = compile_verified_source_strategy(verified)
    execution = invoke_answer_program(
        program=result.answer_program,
        bindings=result.initial_bindings,
        environment=ExecutionEnvironment(catalog=relation_catalog),
        ports=RuntimePorts(
            data_access_port=_EventDataAccess(),
            memory=LookupMemory(),
        ),
    )

    assert execution.issue is None
    assert execution.fact_result is not None
    assert isinstance(execution.fact_result.outcome, AnswerResult)
    assert execution.fact_result.outcome.projected_rows[0].values == {
        "fact_1.output_1": 2 if has_ordinary_state else 0
    }


def _compile_returned_row_predicate(
    *,
    input_term: InputTerm,
    typed_value: FactValue,
    fact_type,
    field_type: RowSourceValueType,
    operator: ExpressionBinaryOperator,
    aggregate_filter: bool = False,
    qualification_and_aggregate_filter: bool = False,
    mechanic_kind: SourceMechanicKind = SourceMechanicKind.RETURNED_ROW_PREDICATE,
):
    origin = SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, "qualified events")
    fact = RequestedFact(
        id="fact_1",
        origin=origin,
        sets=(SetTerm("s1", origin),),
        associations=(),
        facts=(FactTerm("f1", "s1", fact_type, origin),),
        expressions=(
            Comparison(
                "e1",
                operator,
                "f1",
                input_term.id,
                origin,
            ),
            Aggregate(
                "e2",
                AggregateFunction.COUNT,
                "s1",
                "e1" if aggregate_filter else None,
                False,
                origin,
            ),
        ),
        subject=Subject("s1", InstanceInterpretation.NORMAL_BUSINESS_INSTANCE),
        qualification_ref=(
            "e1" if qualification_and_aggregate_filter or not aggregate_filter else None
        ),
        grouping_refs=(),
        outputs=(RequestedOutput("output_1", "e2", origin),),
        ordering=(),
        selection=AllResults(),
        distinct_by=(),
    )
    index = analyze_requested_fact(
        fact,
        inputs={input_term.id: input_term},
        input_denotations=_denotation(input_term),
    )
    value_field = RowSourceField(
        id="value",
        field_ref="field.value",
        label="value",
        type=field_type,
        allowed_roles=(),
    )
    source_params = (
        (
            RowSourceParam(
                id="start_date",
                param_ref="source_events.start_date",
                name="start_date",
                type=RowSourceValueType.DATE,
                source=ParamSource.QUERY,
                required=False,
            ),
            RowSourceParam(
                id="end_date",
                param_ref="source_events.end_date",
                name="end_date",
                type=RowSourceValueType.DATE,
                source=ParamSource.QUERY,
                required=False,
            ),
        )
        if mechanic_kind is SourceMechanicKind.INVOCATION_PREDICATE
        else ()
    )
    source = RowSource(
        id="source_events",
        kind=RowSourceKind.API_READ,
        label="events",
        read_id="list_events",
        fields=(value_field,),
        params=source_params,
    )
    catalog = AvailableSourceCatalog(
        contract_snapshot=SourceContractSnapshot.from_content("{}"),
        sources=(source,),
        relation_evidence=(),
    )
    [clause] = index.qualification.clauses
    branch = SourceStrategyBranch(
        branch_id="fact_1:source_branch:1",
        source_refs=(source.id,),
        relation_evidence_refs=(),
        qualification_clause_refs=(clause.clause_ref,),
    )
    strategy = _strategy(fact.id, branch)
    canonical_value = CanonicalInputValue(
        canonical_value_id=typed_value.id,
        input_ref=input_term.id,
        use_refs=tuple(item.use_ref for item in index.input_use_sites),
        typed_value=typed_value,
        certification_refs=("question_input:i1",),
    )
    request = SemanticSourceBindingRequest(
        index=index,
        strategy=strategy,
        source_catalog=catalog,
        canonical_values=(canonical_value,),
    )
    set_ref = next(
        ref.token for ref in index.source_requirement_refs if ref.kind.value == "set"
    )
    fact_ref = next(
        ref.token for ref in index.source_requirement_refs if ref.kind.value == "fact"
    )
    requirement = next(
        (
            item
            for item in index.boolean_requirements
            if item.owner_expression_ref is None
        ),
        index.boolean_requirements[0],
    )
    invocation_applications = (
        (
            InvocationValueApplication(
                "apply_period",
                branch.branch_id,
                source.id,
                canonical_value.canonical_value_id,
                requirement.requirement_ref,
                (
                    InvocationTargetApplication(
                        "Use the period start.",
                        source_params[0].param_ref,
                        canonical_value.canonical_value_id,
                        ValueProjectionKind.TEMPORAL_START,
                        None,
                    ),
                    InvocationTargetApplication(
                        "Use the period end.",
                        source_params[1].param_ref,
                        canonical_value.canonical_value_id,
                        ValueProjectionKind.TEMPORAL_END,
                        None,
                    ),
                ),
            ),
        )
        if mechanic_kind is SourceMechanicKind.INVOCATION_PREDICATE
        else ()
    )
    mechanic_application_refs = tuple(
        item.application_ref for item in invocation_applications
    )
    plan = SourceBindingPlan(
        strategy=strategy,
        set_bindings={
            set_ref: (
                SetRealization(
                    branch.branch_id,
                    "Event rows.",
                    source.id,
                    None,
                    (),
                    (source.id,),
                ),
            )
        },
        fact_bindings={
            fact_ref: (
                FactRealization(
                    branch.branch_id,
                    "Amount field.",
                    source.id,
                    FactRealizationKind.RETURNED_FIELD,
                    None,
                    (value_field.field_ref,),
                    (source.id, value_field.field_ref),
                ),
            )
        },
        association_bindings={},
        invocation_applications=invocation_applications,
        boolean_bindings={
            requirement.requirement_ref: (
                BooleanRequirementRealization(
                    branch.branch_id,
                    (
                        SourceMechanic(
                            "Realize the predicate at its declared source boundary.",
                            source.id,
                            mechanic_application_refs,
                            (source.id, value_field.field_ref),
                            mechanic_kind,
                        ),
                    ),
                ),
            )
        },
        subject_binding=SubjectObligationBinding(
            set_ref,
            (SubjectObligationRealization(branch.branch_id, ()),),
        ),
    )
    plan = replace(
        plan,
        boolean_bindings={
            **plan.boolean_bindings,
            **{
                item.requirement_ref: (
                    BooleanRequirementRealization(
                        branch.branch_id,
                        (
                            SourceMechanic(
                                "Retain the aggregate's declared condition.",
                                source.id,
                                (),
                                (source.id, value_field.field_ref),
                                SourceMechanicKind.RETURNED_ROW_PREDICATE,
                            ),
                        ),
                    ),
                )
                for item in index.boolean_requirements
                if item.requirement_ref != requirement.requirement_ref
            },
        },
    )
    verified = verify_source_strategy(plan, request=request)
    assert isinstance(verified, VerifiedSourceStrategy)
    return compile_verified_source_strategy(verified), canonical_value


def test_returned_row_threshold_compiles_to_shared_filter_expression() -> None:
    threshold = InputTerm(
        id="i1",
        origin=SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, "threshold"),
        operand="1000",
        value_type=DecimalType(UnitlessMeasure()),
    )
    result, canonical_value = _compile_returned_row_predicate(
        input_term=threshold,
        typed_value=FactValue.literal(
            id="threshold_value",
            known_input_id=threshold.id,
            literal_type=LiteralType.NUMBER,
            value="1000",
            proof_refs=("question_input:i1",),
        ),
        fact_type=DecimalType(UnitlessMeasure()),
        field_type=RowSourceValueType.DECIMAL,
        operator=ExpressionBinaryOperator.GT,
    )

    [parameter] = result.answer_program.parameters
    assert parameter.input_ref == threshold.id
    assert parameter.input_use_refs == canonical_value.use_refs
    assert result.answer_program.fulfillment[0].answer_output_id == "output_1"


def test_returned_row_time_scope_lowers_to_explicit_bounds() -> None:
    period = InputTerm(
        id="i1",
        origin=SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, "March 2026"),
        operand="March 2026",
        value_type=TemporalScopeType(),
    )
    result, _ = _compile_returned_row_predicate(
        input_term=period,
        typed_value=FactValue.time(
            id="period_value",
            known_input_id=period.id,
            expression="March 2026",
            resolved_start="2026-03-01",
            resolved_end="2026-03-31",
            granularity="month",
            proof_refs=("question_input:i1",),
        ),
        fact_type=DateTimeType(),
        field_type=RowSourceValueType.DATETIME,
        operator=ExpressionBinaryOperator.WITHIN,
    )

    filter_spec = next(
        operation.spec
        for operation in result.answer_program.operations
        if isinstance(operation.spec, FilterSpec)
    )
    parameter_refs = expression_references(filter_spec.condition).parameters
    resolved_bounds = tuple(
        resolve_value_expression(item, bindings=result.initial_bindings).value
        for item in parameter_refs
    )

    assert {item.component for item in parameter_refs} == {"start", "end"}
    assert set(resolved_bounds) == {"2026-03-01", "2026-03-31"}


class _StoreAreaDataAccess:
    def read(self, *, endpoint_name, args):
        assert endpoint_name == "list_stores"
        assert args == {}
        return {
            "responseStatus": 200,
            "responseBody": {
                "data": [
                    {"store_id": "store-1", "area_id": "area-a"},
                    {"store_id": "store-2", "area_id": "area-b"},
                    {"store_id": "store-3", "area_id": "area-a"},
                ]
            },
        }


def test_co_resident_exists_filters_subject_rows_before_counting() -> None:
    origin = SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, "stores in one area")
    area_input = InputTerm(
        id="i1",
        origin=SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, "named area"),
        operand="Area A",
        value_type=IdentifierType("s_area"),
    )
    fact = RequestedFact(
        id="fact_1",
        origin=origin,
        sets=(SetTerm("s_store", origin), SetTerm("s_area", origin)),
        associations=(AssociationTerm("a_store_area", "s_store", "s_area", origin),),
        facts=(FactTerm("f_area", "s_area", IdentifierType("s_area"), origin),),
        expressions=(
            Comparison(
                "e_area",
                ExpressionBinaryOperator.EQUALS,
                "f_area",
                area_input.id,
                origin,
            ),
            Quantify(
                "e_exists",
                Quantifier.EXISTS,
                "s_area",
                ("a_store_area",),
                "e_area",
                origin,
            ),
            Aggregate(
                "e_count",
                AggregateFunction.COUNT,
                "s_store",
                None,
                False,
                origin,
            ),
        ),
        subject=Subject("s_store", InstanceInterpretation.NORMAL_BUSINESS_INSTANCE),
        qualification_ref="e_exists",
        grouping_refs=(),
        outputs=(RequestedOutput("output_1", "e_count", origin),),
        ordering=(),
        selection=AllResults(),
        distinct_by=(),
    )
    index = analyze_requested_fact(
        fact,
        inputs={area_input.id: area_input},
        input_denotations=_denotation(
            area_input,
            kind=InputDenotationKind.IDENTITY_REFERENCE,
        ),
    )
    read = EndpointRead(
        id="list_stores",
        endpoint_name="list_stores",
        row_paths=(RowPath("data", "data", RowCardinality.MANY),),
        fields=(
            CatalogField(
                ref="store.store_id",
                path="data.store_id",
                row_path_id="data",
                type="string",
            ),
            CatalogField(
                ref="store.area_id",
                path="data.area_id",
                row_path_id="data",
                type="string",
            ),
        ),
        candidate_keys=(
            CandidateKey(
                id="store_primary_key",
                entity_kind="store",
                components=(CandidateKeyComponent("store_id", "store.store_id"),),
                primary=True,
            ),
        ),
        entity_references=(
            EntityReference(
                id="area_reference",
                target_entity_kind="area",
                target_key_id="primary_key",
                components=(EntityReferenceComponent("area_id", "store.area_id"),),
            ),
        ),
    )
    relation_catalog = RelationCatalog(reads=(read,))
    source = next(
        item
        for item in build_row_source_catalog(relation_catalog).sources
        if item.read_id == read.id and item.row_path == "data"
    )
    area_id = source.field("area_id")
    catalog = AvailableSourceCatalog(
        contract_snapshot=SourceContractSnapshot.from_content("{}"),
        sources=(source,),
        relation_evidence=(),
    )
    [clause] = index.qualification.clauses
    branch = SourceStrategyBranch(
        "fact_1:source_branch:1",
        (source.id,),
        (),
        (clause.clause_ref,),
    )
    strategy = _strategy(fact.id, branch)
    typed_value = FactValue.identity(
        id="area-a",
        known_input_id=area_input.id,
        key=EntityKeyValue(
            "area",
            "primary_key",
            (EntityKeyComponentValue("area_id", "area-a"),),
        ),
        display_value="Area A",
        proof_refs=("resolver:area",),
    )
    canonical = CanonicalInputValue(
        "area-a",
        area_input.id,
        tuple(item.use_ref for item in index.input_use_sites),
        typed_value,
        ("resolver:area",),
    )
    request = SemanticSourceBindingRequest(index, strategy, catalog, (canonical,))
    identities = {item.entity_kind: item for item in catalog.identity_evidence}
    refs = {ref.local_id: ref.token for ref in index.source_requirement_refs}
    returned_mechanic = SourceMechanic(
        "Compare the returned area identity.",
        source.id,
        (),
        (source.id, area_id.field_ref),
        SourceMechanicKind.RETURNED_ROW_PREDICATE,
    )
    plan = SourceBindingPlan(
        strategy=strategy,
        set_bindings={
            refs["s_store"]: (
                SetRealization(
                    branch.branch_id,
                    "Store rows.",
                    source.id,
                    identities["store"].identity_ref,
                    identities["store"].field_refs,
                    (source.id, identities["store"].identity_ref),
                ),
            ),
            refs["s_area"]: (
                SetRealization(
                    branch.branch_id,
                    "Area reference on each store row.",
                    source.id,
                    identities["area"].identity_ref,
                    identities["area"].field_refs,
                    (source.id, identities["area"].identity_ref),
                ),
            ),
        },
        fact_bindings={
            refs["f_area"]: (
                FactRealization(
                    branch.branch_id,
                    "Returned area identity.",
                    source.id,
                    FactRealizationKind.ENTITY_KEY,
                    identities["area"].identity_ref,
                    identities["area"].field_refs,
                    (source.id, identities["area"].identity_ref),
                ),
            )
        },
        association_bindings={
            refs["a_store_area"]: (
                AssociationRealization(
                    branch.branch_id,
                    "The store row carries its area reference.",
                    AssociationRealizationKind.CO_RESIDENT,
                    (source.id,),
                    None,
                    (source.id, identities["area"].identity_ref),
                ),
            )
        },
        invocation_applications=(),
        boolean_bindings={
            requirement.requirement_ref: (
                BooleanRequirementRealization(
                    branch.branch_id,
                    (returned_mechanic,),
                ),
            )
            for requirement in index.boolean_requirements
        },
        subject_binding=SubjectObligationBinding(
            refs["s_store"],
            (SubjectObligationRealization(branch.branch_id, ()),),
        ),
    )
    verified = verify_source_strategy(plan, request=request)
    assert isinstance(verified, VerifiedSourceStrategy)

    result = compile_verified_source_strategy(verified)
    execution = invoke_answer_program(
        program=result.answer_program,
        bindings=result.initial_bindings,
        environment=ExecutionEnvironment(catalog=relation_catalog),
        ports=RuntimePorts(
            data_access_port=_StoreAreaDataAccess(),
            memory=LookupMemory(),
        ),
    )

    assert execution.issue is None
    assert execution.fact_result is not None
    assert isinstance(execution.fact_result.outcome, AnswerResult)
    assert execution.fact_result.outcome.projected_rows[0].values == {
        "fact_1.output_1": 2
    }


@pytest.mark.parametrize("distinct", [False, True])
def test_qualifying_identity_set_compiles_as_entity_output(distinct) -> None:
    origin = SourceOrigin(
        SourceOriginKind.QUESTION_CONTEXT,
        "Which staff member is named Nadia Wanjiku?",
    )
    staff_input = InputTerm(
        id="i1",
        origin=SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, "Nadia Wanjiku"),
        operand="Nadia Wanjiku",
        value_type=IdentifierType("s_staff"),
    )
    fact = RequestedFact(
        id="fact_1",
        origin=origin,
        sets=(SetTerm("s_staff", origin),),
        associations=(),
        facts=(
            FactTerm(
                "f_staff_identity",
                "s_staff",
                IdentifierType("s_staff"),
                origin,
            ),
        ),
        expressions=(
            Comparison(
                "e_named_staff",
                ExpressionBinaryOperator.EQUALS,
                "f_staff_identity",
                staff_input.id,
                origin,
            ),
        ),
        subject=Subject("s_staff", InstanceInterpretation.NORMAL_BUSINESS_INSTANCE),
        qualification_ref="e_named_staff",
        grouping_refs=(),
        outputs=(RequestedOutput("output_1", "s_staff", origin),),
        ordering=(),
        selection=AllResults(),
        distinct_by=("s_staff",) if distinct else (),
    )
    index = analyze_requested_fact(
        fact,
        inputs={staff_input.id: staff_input},
        input_denotations=_denotation(
            staff_input,
            kind=InputDenotationKind.IDENTITY_REFERENCE,
        ),
    )
    read = EndpointRead(
        id="list_staff",
        endpoint_name="list_staff",
        row_paths=(RowPath("data", "data", RowCardinality.MANY),),
        fields=(
            CatalogField(
                ref="staff.staff_id",
                path="data.staff_id",
                row_path_id="data",
                type="uuid",
            ),
        ),
        candidate_keys=(
            CandidateKey(
                id="primary_key",
                entity_kind="staff",
                components=(CandidateKeyComponent("staff_id", "staff.staff_id"),),
                primary=True,
            ),
        ),
    )
    source = next(
        item
        for item in build_row_source_catalog(RelationCatalog(reads=(read,))).sources
        if item.read_id == read.id and item.row_path == "data"
    )
    catalog = AvailableSourceCatalog(
        contract_snapshot=SourceContractSnapshot.from_content("{}"),
        sources=(source,),
        relation_evidence=(),
    )
    [clause] = index.qualification.clauses
    branch = SourceStrategyBranch(
        "fact_1:source_branch:1",
        (source.id,),
        (),
        (clause.clause_ref,),
    )
    strategy = _strategy(fact.id, branch)
    canonical = CanonicalInputValue(
        "nadia",
        staff_input.id,
        tuple(item.use_ref for item in index.input_use_sites),
        FactValue.identity(
            id="nadia",
            known_input_id=staff_input.id,
            key=EntityKeyValue(
                "staff",
                "primary_key",
                (
                    EntityKeyComponentValue(
                        "staff_id",
                        "40404040-0000-0000-0002-000000000001",
                    ),
                ),
            ),
            display_value="Nadia Wanjiku",
            proof_refs=("resolver:staff",),
        ),
        ("resolver:staff",),
    )
    request = SemanticSourceBindingRequest(index, strategy, catalog, (canonical,))
    identity = catalog.identity_evidence[0]
    refs = {ref.local_id: ref.token for ref in index.source_requirement_refs}
    plan = SourceBindingPlan(
        strategy=strategy,
        set_bindings={
            refs["s_staff"]: (
                SetRealization(
                    branch.branch_id,
                    "Staff rows.",
                    source.id,
                    identity.identity_ref,
                    identity.field_refs,
                    (source.id, identity.identity_ref),
                ),
            ),
        },
        fact_bindings={
            refs["f_staff_identity"]: (
                FactRealization(
                    branch.branch_id,
                    "Returned staff identity.",
                    source.id,
                    FactRealizationKind.ENTITY_KEY,
                    identity.identity_ref,
                    identity.field_refs,
                    (source.id, identity.identity_ref),
                ),
            ),
        },
        association_bindings={},
        invocation_applications=(),
        boolean_bindings={
            requirement.requirement_ref: (
                BooleanRequirementRealization(
                    branch.branch_id,
                    (
                        SourceMechanic(
                            "Compare the returned staff identity.",
                            source.id,
                            (),
                            (source.id, identity.identity_ref),
                            SourceMechanicKind.RETURNED_ROW_PREDICATE,
                        ),
                    ),
                ),
            )
            for requirement in index.boolean_requirements
        },
        subject_binding=SubjectObligationBinding(
            refs["s_staff"],
            (SubjectObligationRealization(branch.branch_id, ()),),
        ),
    )
    verified = verify_source_strategy(plan, request=request)
    assert isinstance(verified, VerifiedSourceStrategy)

    result = compile_verified_source_strategy(verified)

    [output] = result.answer_program.result_projection.relation_outputs
    assert output.field_id == ""
    assert output.entity_key is not None
    assert output.entity_key.entity_kind == "staff"
    assert output.entity_key.key_id == "primary_key"
    assert [
        (component.component_id, component.field_id)
        for component in output.entity_key.components
    ] == [("staff_id", "staff_id")]


def test_aggregate_filter_proof_is_not_misclassified_as_population_proof() -> None:
    period = InputTerm(
        id="i1",
        origin=SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, "March 2026"),
        operand="March 2026",
        value_type=TemporalScopeType(),
    )

    result, _ = _compile_returned_row_predicate(
        input_term=period,
        typed_value=FactValue.time(
            id="period_value",
            known_input_id=period.id,
            expression="March 2026",
            resolved_start="2026-03-01",
            resolved_end="2026-03-31",
            granularity="month",
            proof_refs=("question_input:i1",),
        ),
        fact_type=DateTimeType(),
        field_type=RowSourceValueType.DATETIME,
        operator=ExpressionBinaryOperator.WITHIN,
        aggregate_filter=True,
    )

    aggregate = next(
        operation.spec
        for operation in result.answer_program.operations
        if operation.kind.value == "aggregate"
    )
    filters = {
        operation.output_relation
        for operation in result.answer_program.operations
        if operation.kind.value == "filter"
    }
    assert aggregate.input_relation in filters
    assert all(
        not clause.atom_refs
        for declaration in result.answer_program.relation_guarantees
        for clause in declaration.qualification.formula.clauses
    )


def test_invocation_realized_aggregate_filter_is_not_reapplied_to_rows() -> None:
    period = InputTerm(
        id="i1",
        origin=SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, "March 2026"),
        operand="March 2026",
        value_type=TemporalScopeType(),
    )

    result, _ = _compile_returned_row_predicate(
        input_term=period,
        typed_value=FactValue.time(
            id="period_value",
            known_input_id=period.id,
            expression="March 2026",
            resolved_start="2026-03-01",
            resolved_end="2026-03-31",
            granularity="month",
            proof_refs=("question_input:i1",),
        ),
        fact_type=DateTimeType(),
        field_type=RowSourceValueType.DATETIME,
        operator=ExpressionBinaryOperator.WITHIN,
        aggregate_filter=True,
        mechanic_kind=SourceMechanicKind.INVOCATION_PREDICATE,
    )

    aggregate = next(
        operation.spec
        for operation in result.answer_program.operations
        if operation.kind.value == "aggregate"
    )
    assert aggregate.aggregations[0].filter is None


def test_outer_qualification_is_not_reapplied_as_an_aggregate_filter() -> None:
    period = InputTerm(
        id="i1",
        origin=SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, "March 2026"),
        operand="March 2026",
        value_type=TemporalScopeType(),
    )

    result, _ = _compile_returned_row_predicate(
        input_term=period,
        typed_value=FactValue.time(
            id="period_value",
            known_input_id=period.id,
            expression="March 2026",
            resolved_start="2026-03-01",
            resolved_end="2026-03-31",
            granularity="month",
            proof_refs=("question_input:i1",),
        ),
        fact_type=DateTimeType(),
        field_type=RowSourceValueType.DATETIME,
        operator=ExpressionBinaryOperator.WITHIN,
        aggregate_filter=True,
        qualification_and_aggregate_filter=True,
    )

    aggregate = next(
        operation.spec
        for operation in result.answer_program.operations
        if operation.kind.value == "aggregate"
    )
    assert aggregate.aggregations[0].filter is None


class _StaffSalesDataAccess:
    def read(self, *, endpoint_name, args):
        assert endpoint_name == "list_sales"
        assert args == {}
        return {
            "responseStatus": 200,
            "responseBody": {
                "data": [
                    {"sale_id": "sale-1", "staff_id": "staff-1"},
                    {"sale_id": "sale-2", "staff_id": "staff-1"},
                    {"sale_id": "sale-3", "staff_id": "staff-2"},
                ]
            },
        }


def test_candidate_grain_is_preserved_by_related_set_aggregate() -> None:
    origin = SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, "sales per staff member")
    fact = RequestedFact(
        id="fact_1",
        origin=origin,
        sets=(SetTerm("s_staff", origin), SetTerm("s_sale", origin)),
        associations=(AssociationTerm("a_staff_sale", "s_staff", "s_sale", origin),),
        facts=(
            FactTerm(
                "f_staff",
                "s_staff",
                IdentifierType("s_staff"),
                origin,
            ),
        ),
        expressions=(
            Aggregate(
                "e_count",
                AggregateFunction.COUNT,
                "s_sale",
                None,
                False,
                origin,
            ),
        ),
        subject=Subject(
            "s_staff",
            InstanceInterpretation.NORMAL_BUSINESS_INSTANCE,
        ),
        qualification_ref=None,
        grouping_refs=(),
        outputs=(
            RequestedOutput("output_1", "f_staff", origin),
            RequestedOutput("output_2", "e_count", origin),
        ),
        ordering=(),
        selection=AllResults(),
        distinct_by=(),
    )
    index = analyze_requested_fact(fact, inputs={}, input_denotations={})
    read = EndpointRead(
        id="list_sales",
        endpoint_name="list_sales",
        row_paths=(RowPath("data", "data", RowCardinality.MANY),),
        fields=(
            CatalogField(
                ref="sale.sale_id",
                path="data.sale_id",
                row_path_id="data",
                type="string",
            ),
            CatalogField(
                ref="sale.staff_id",
                path="data.staff_id",
                row_path_id="data",
                type="string",
            ),
        ),
        candidate_keys=(
            CandidateKey(
                "primary_key",
                "sale",
                (CandidateKeyComponent("sale_id", "sale.sale_id"),),
                primary=True,
            ),
        ),
        entity_references=(
            EntityReference(
                "staff_reference",
                "staff",
                "primary_key",
                (EntityReferenceComponent("staff_id", "sale.staff_id"),),
            ),
        ),
    )
    relation_catalog = RelationCatalog(reads=(read,))
    source = next(
        item
        for item in build_row_source_catalog(relation_catalog).sources
        if item.read_id == read.id
    )
    catalog = AvailableSourceCatalog(
        SourceContractSnapshot.from_content("{}"),
        (source,),
        (),
    )
    [clause] = index.qualification.clauses
    branch = SourceStrategyBranch(
        "fact_1:source_branch:1",
        (source.id,),
        (),
        (clause.clause_ref,),
    )
    strategy = _strategy(fact.id, branch)
    request = SemanticSourceBindingRequest(index, strategy, catalog, ())
    refs = {ref.local_id: ref.token for ref in index.source_requirement_refs}
    staff_identity = next(
        item for item in catalog.identity_evidence if item.entity_kind == "staff"
    )
    sale_identity = next(
        item for item in catalog.identity_evidence if item.entity_kind == "sale"
    )
    plan = SourceBindingPlan(
        strategy=strategy,
        set_bindings={
            refs["s_staff"]: (
                SetRealization(
                    branch.branch_id,
                    "Staff identity is carried by each sale row.",
                    source.id,
                    staff_identity.identity_ref,
                    staff_identity.field_refs,
                    (source.id, staff_identity.identity_ref),
                ),
            ),
            refs["s_sale"]: (
                SetRealization(
                    branch.branch_id,
                    "Sale rows realize the counted set.",
                    source.id,
                    sale_identity.identity_ref,
                    sale_identity.field_refs,
                    (source.id, sale_identity.identity_ref),
                ),
            ),
        },
        fact_bindings={
            refs["f_staff"]: (
                FactRealization(
                    branch.branch_id,
                    "The staff identifier is returned on each sale row.",
                    source.id,
                    FactRealizationKind.ENTITY_KEY,
                    staff_identity.identity_ref,
                    staff_identity.field_refs,
                    (source.id, staff_identity.identity_ref),
                ),
            ),
        },
        association_bindings={
            refs["a_staff_sale"]: (
                AssociationRealization(
                    branch.branch_id,
                    "Staff and sale are co-resident on each row.",
                    AssociationRealizationKind.CO_RESIDENT,
                    (source.id,),
                    None,
                    (source.id, staff_identity.identity_ref),
                ),
            ),
        },
        invocation_applications=(),
        boolean_bindings={},
        subject_binding=SubjectObligationBinding(
            refs["s_staff"],
            (SubjectObligationRealization(branch.branch_id, ()),),
        ),
    )
    verified = verify_source_strategy(plan, request=request)
    assert isinstance(verified, VerifiedSourceStrategy)

    result = compile_verified_source_strategy(verified)
    execution = invoke_answer_program(
        program=result.answer_program,
        bindings=result.initial_bindings,
        environment=ExecutionEnvironment(catalog=relation_catalog),
        ports=RuntimePorts(
            data_access_port=_StaffSalesDataAccess(),
            memory=LookupMemory(),
        ),
    )

    assert execution.issue is None
    assert execution.fact_result is not None
    assert rendered_fact_text(render_fact_result(execution.fact_result)) == (
        "{'entityKind': 'staff', 'keyId': 'primary_key', "
        "'components': {'staff_id': 'staff-1'}}: 2\n"
        "{'entityKind': 'staff', 'keyId': 'primary_key', "
        "'components': {'staff_id': 'staff-2'}}: 1"
    )


def test_identity_collection_compiles_to_existing_invocation_union() -> None:
    origin = SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, "events for two actors")
    actor_ids = InputTerm(
        id="i1",
        origin=SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, "two actor identifiers"),
        operand=("actor-1", "actor-2"),
        value_type=CollectionType(IdentifierType("s_actor")),
    )
    fact = RequestedFact(
        id="fact_1",
        origin=origin,
        sets=(SetTerm("s_event", origin), SetTerm("s_actor", origin)),
        associations=(AssociationTerm("a_event_actor", "s_event", "s_actor", origin),),
        facts=(FactTerm("f_actor", "s_event", IdentifierType("s_actor"), origin),),
        expressions=(
            Comparison(
                "e_filter",
                ExpressionBinaryOperator.IN,
                "f_actor",
                actor_ids.id,
                origin,
            ),
            Aggregate(
                "e_count",
                AggregateFunction.COUNT,
                "s_event",
                None,
                False,
                origin,
            ),
        ),
        subject=Subject("s_event", InstanceInterpretation.NORMAL_BUSINESS_INSTANCE),
        qualification_ref="e_filter",
        grouping_refs=(),
        outputs=(RequestedOutput("output_1", "e_count", origin),),
        ordering=(),
        selection=AllResults(),
        distinct_by=(),
    )
    index = analyze_requested_fact(
        fact,
        inputs={actor_ids.id: actor_ids},
        input_denotations=_denotation(
            actor_ids,
            kind=InputDenotationKind.IDENTITY_REFERENCE,
        ),
    )
    actor_field = RowSourceField(
        id="actor_id",
        field_ref="field.actor_id",
        label="actor identifier",
        type=RowSourceValueType.STRING,
        allowed_roles=(),
    )
    event_field = RowSourceField(
        id="event_id",
        field_ref="field.event_id",
        label="event identifier",
        type=RowSourceValueType.STRING,
        allowed_roles=(),
    )
    actor_param = RowSourceParam(
        id="actor_id",
        param_ref="source_events.actor_id",
        name="actor_id",
        type=RowSourceValueType.STRING,
        source=ParamSource.QUERY,
        required=True,
        entity_target=EntityKeyComponentTarget(
            entity_kind="actor",
            key_id="primary_key",
            component_id="actor_id",
        ),
    )
    source = RowSource(
        id="source_events",
        kind=RowSourceKind.API_READ,
        label="events",
        read_id="list_actor_events",
        fields=(event_field, actor_field),
        candidate_keys=(
            RowSourceCandidateKey(
                id="primary_key",
                entity_kind="event",
                components=(RowSourceKeyComponent("event_id", "event_id"),),
                primary=True,
            ),
        ),
        entity_references=(
            RowSourceEntityReference(
                id="actor_reference",
                target_entity_kind="actor",
                target_key_id="primary_key",
                components=(RowSourceEntityReferenceComponent("actor_id", "actor_id"),),
            ),
        ),
        params=(actor_param,),
    )
    catalog = AvailableSourceCatalog(
        contract_snapshot=SourceContractSnapshot.from_content("{}"),
        sources=(source,),
        relation_evidence=(),
    )
    [clause] = index.qualification.clauses
    branch = SourceStrategyBranch(
        "fact_1:source_branch:1",
        (source.id,),
        (),
        (clause.clause_ref,),
    )
    strategy = _strategy(fact.id, branch)
    typed = FactValue.identity_set(
        id="actors",
        known_input_id=actor_ids.id,
        keys=(
            EntityKeyValue(
                "actor",
                "primary_key",
                (EntityKeyComponentValue("actor_id", "actor-1"),),
            ),
            EntityKeyValue(
                "actor",
                "primary_key",
                (EntityKeyComponentValue("actor_id", "actor-2"),),
            ),
        ),
        display_value="actor-1, actor-2",
        proof_refs=("resolver:actor",),
    )
    canonical = CanonicalInputValue(
        "actors",
        actor_ids.id,
        tuple(item.use_ref for item in index.input_use_sites),
        typed,
        ("resolver:actor",),
    )
    request = SemanticSourceBindingRequest(
        index,
        strategy,
        catalog,
        (canonical,),
    )
    set_refs = tuple(
        ref.token for ref in index.source_requirement_refs if ref.kind.value == "set"
    )
    fact_ref = next(
        ref.token for ref in index.source_requirement_refs if ref.kind.value == "fact"
    )
    association_ref = next(
        ref.token
        for ref in index.source_requirement_refs
        if ref.kind.value == "association"
    )
    identity = next(
        item for item in catalog.identity_evidence if item.entity_kind == "actor"
    )
    [requirement] = index.boolean_requirements
    application = InvocationValueApplication(
        "apply_actor_ids",
        branch.branch_id,
        source.id,
        canonical.canonical_value_id,
        requirement.requirement_ref,
        (
            InvocationTargetApplication(
                "The actor identifier scopes the event read.",
                actor_param.param_ref,
                canonical.canonical_value_id,
                ValueProjectionKind.IDENTITY_COMPONENT,
                "actor_id",
            ),
        ),
    )
    plan = SourceBindingPlan(
        strategy,
        {
            set_ref: (
                SetRealization(
                    branch.branch_id,
                    "The source carries this required set.",
                    source.id,
                    (identity.identity_ref if set_ref.endswith(":s_actor") else None),
                    (identity.field_refs if set_ref.endswith(":s_actor") else ()),
                    (source.id, *identity.field_refs),
                ),
            )
            for set_ref in set_refs
        },
        {
            fact_ref: (
                FactRealization(
                    branch.branch_id,
                    "Actor identity carried by each event.",
                    source.id,
                    FactRealizationKind.ENTITY_KEY,
                    identity.identity_ref,
                    identity.field_refs,
                    (source.id, identity.identity_ref),
                ),
            )
        },
        {
            association_ref: (
                AssociationRealization(
                    branch.branch_id,
                    "The event source carries the actor reference.",
                    AssociationRealizationKind.CO_RESIDENT,
                    (source.id,),
                    None,
                    (source.id, identity.identity_ref),
                ),
            )
        },
        (application,),
        {
            requirement.requirement_ref: (
                BooleanRequirementRealization(
                    branch.branch_id,
                    (
                        SourceMechanic(
                            "The request parameter scopes one actor at a time.",
                            source.id,
                            (application.application_ref,),
                            (source.id, actor_param.param_ref),
                            SourceMechanicKind.INVOCATION_PREDICATE,
                        ),
                    ),
                ),
            )
        },
        SubjectObligationBinding(
            index.subject_obligation.subject_set_ref.token,
            (SubjectObligationRealization(branch.branch_id, ()),),
        ),
    )
    verified = verify_source_strategy(plan, request=request)
    assert isinstance(verified, VerifiedSourceStrategy)

    result = compile_verified_source_strategy(verified)

    assert [
        relation.source.read_id for relation in result.answer_program.relations
    ] == [
        "list_actor_events",
        "list_actor_events",
    ]
    union = result.answer_program.operations[0].spec
    assert union.identity_fields == ("event_id",)
    parameter_refs = [
        binding.value_expr.item_index
        for relation in result.answer_program.relations
        for binding in relation.source.param_bindings
    ]
    assert parameter_refs == [0, 1]

    source_without_stable_grain = replace(source, candidate_keys=())
    request_without_stable_grain = replace(
        request,
        source_catalog=replace(
            catalog,
            sources=(source_without_stable_grain,),
        ),
    )
    verified_without_stable_grain = verify_source_strategy(
        plan,
        request=request_without_stable_grain,
    )
    assert isinstance(verified_without_stable_grain, VerifiedSourceStrategy)

    with pytest.raises(ValueError, match="union requires stable identity fields"):
        compile_verified_source_strategy(verified_without_stable_grain)


def test_declared_association_compiles_to_existing_join_and_grouped_aggregate() -> None:
    origin = SourceOrigin(
        SourceOriginKind.QUESTION_CONTEXT,
        "total event amount by associated category",
    )
    fact = RequestedFact(
        id="fact_1",
        origin=origin,
        sets=(SetTerm("s_event", origin), SetTerm("s_category", origin)),
        associations=(
            AssociationTerm("a_event_category", "s_event", "s_category", origin),
        ),
        facts=(
            FactTerm("f_amount", "s_event", DecimalType(UnitlessMeasure()), origin),
            FactTerm(
                "f_category",
                "a_event_category",
                IdentifierType("s_category"),
                origin,
            ),
        ),
        expressions=(
            Aggregate(
                "e_total",
                AggregateFunction.SUM,
                "f_amount",
                None,
                False,
                origin,
            ),
        ),
        subject=Subject("s_event", InstanceInterpretation.NORMAL_BUSINESS_INSTANCE),
        qualification_ref=None,
        grouping_refs=("f_category",),
        outputs=(
            RequestedOutput("output_1", "f_category", origin),
            RequestedOutput("output_2", "e_total", origin),
        ),
        ordering=(Ordering("e_total", OrderingDirection.DESCENDING, origin),),
        selection=FirstRankWithTies(),
        distinct_by=(),
    )
    index = analyze_requested_fact(fact, inputs={}, input_denotations={})
    event_id = RowSourceField(
        "event_id",
        "field.event_id",
        "event identifier",
        RowSourceValueType.STRING,
        (),
    )
    category_id = RowSourceField(
        "category_id",
        "field.category_id",
        "category identifier",
        RowSourceValueType.STRING,
        (),
    )
    amount = RowSourceField(
        "amount",
        "field.amount",
        "amount",
        RowSourceValueType.DECIMAL,
        (),
    )
    events = RowSource(
        id="source_events",
        kind=RowSourceKind.API_READ,
        label="events",
        read_id="list_events",
        fields=(event_id, category_id, amount),
        candidate_keys=(
            RowSourceCandidateKey(
                "primary_key",
                "event",
                (RowSourceKeyComponent("event_id", "event_id"),),
                primary=True,
            ),
        ),
        entity_references=(
            RowSourceEntityReference(
                "category_reference",
                "category",
                "primary_key",
                (RowSourceEntityReferenceComponent("category_id", "category_id"),),
            ),
        ),
    )
    category_key = RowSourceField(
        "category_id",
        "field.category_id",
        "category identifier",
        RowSourceValueType.STRING,
        (),
    )
    categories = RowSource(
        id="source_categories",
        kind=RowSourceKind.API_READ,
        label="categories",
        read_id="list_categories",
        fields=(category_key,),
        candidate_keys=(
            RowSourceCandidateKey(
                "primary_key",
                "category",
                (RowSourceKeyComponent("category_id", "category_id"),),
                primary=True,
            ),
        ),
    )
    evidence = SourceRelationEvidence(
        "source_relation:event_category",
        events.id,
        categories.id,
        ("category_id",),
        ("category_id",),
    )
    catalog = AvailableSourceCatalog(
        SourceContractSnapshot.from_content("{}"),
        (events, categories),
        (evidence,),
    )
    [clause] = index.qualification.clauses
    branch = SourceStrategyBranch(
        "fact_1:source_branch:1",
        (events.id, categories.id),
        (evidence.evidence_ref,),
        (clause.clause_ref,),
    )
    strategy = _strategy(fact.id, branch)
    request = SemanticSourceBindingRequest(index, strategy, catalog, ())
    requirement_refs = {
        ref.local_id: ref.token for ref in index.source_requirement_refs
    }
    event_identity = next(
        item
        for item in catalog.identity_evidence
        if item.source_ref == events.id and item.entity_kind == "event"
    )
    category_identity = next(
        item
        for item in catalog.identity_evidence
        if item.source_ref == categories.id and item.entity_kind == "category"
    )
    plan = SourceBindingPlan(
        strategy=strategy,
        set_bindings={
            requirement_refs["s_event"]: (
                SetRealization(
                    branch.branch_id,
                    "Event source.",
                    events.id,
                    event_identity.identity_ref,
                    event_identity.field_refs,
                    (events.id, event_identity.identity_ref),
                ),
            ),
            requirement_refs["s_category"]: (
                SetRealization(
                    branch.branch_id,
                    "Category source.",
                    categories.id,
                    category_identity.identity_ref,
                    category_identity.field_refs,
                    (categories.id, category_identity.identity_ref),
                ),
            ),
        },
        fact_bindings={
            requirement_refs["f_amount"]: (
                FactRealization(
                    branch.branch_id,
                    "Returned event amount.",
                    events.id,
                    FactRealizationKind.RETURNED_FIELD,
                    None,
                    (amount.field_ref,),
                    (events.id, amount.field_ref),
                ),
            ),
            requirement_refs["f_category"]: (
                FactRealization(
                    branch.branch_id,
                    "Returned category identity.",
                    categories.id,
                    FactRealizationKind.ENTITY_KEY,
                    category_identity.identity_ref,
                    category_identity.field_refs,
                    (categories.id, category_identity.identity_ref),
                ),
            ),
        },
        association_bindings={
            requirement_refs["a_event_category"]: (
                AssociationRealization(
                    branch.branch_id,
                    "Declared event-category relation.",
                    AssociationRealizationKind.DECLARED_RELATION,
                    (events.id, categories.id),
                    evidence.evidence_ref,
                    (evidence.evidence_ref,),
                ),
            ),
        },
        invocation_applications=(),
        boolean_bindings={},
        subject_binding=SubjectObligationBinding(
            index.subject_obligation.subject_set_ref.token,
            (SubjectObligationRealization(branch.branch_id, ()),),
        ),
    )
    verified = verify_source_strategy(plan, request=request)
    assert isinstance(verified, VerifiedSourceStrategy)

    result = compile_verified_source_strategy(verified)

    assert [item.kind.value for item in result.answer_program.operations] == [
        "project",
        "project",
        "join",
        "filter",
        "aggregate",
        "order",
    ]
    assert [item.answer_output_id for item in result.answer_program.fulfillment] == [
        "output_1",
        "output_2",
    ]
    identifier_output = result.answer_program.result_projection.relation_outputs[0]
    assert identifier_output.field_id == ""
    assert identifier_output.entity_key is not None
    assert identifier_output.entity_key.entity_kind == "category"
    assert identifier_output.entity_key.key_id == "primary_key"
    assert tuple(
        (component.component_id, component.field_id)
        for component in identifier_output.entity_key.components
    ) == (("category_id", "source_field:source_categories:category_id"),)


def test_aggregate_arithmetic_compiles_to_one_shared_compute_expression() -> None:
    origin = SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, "percentage of total")
    percentage = InputTerm(
        "i1",
        SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, "requested percentage"),
        "10%",
        DecimalType(PercentageMeasure()),
    )
    fact = RequestedFact(
        id="fact_1",
        origin=origin,
        sets=(SetTerm("s1", origin),),
        associations=(),
        facts=(FactTerm("f1", "s1", DecimalType(UnitlessMeasure()), origin),),
        expressions=(
            Aggregate("e1", AggregateFunction.SUM, "f1", None, False, origin),
            Arithmetic(
                "e2",
                ExpressionBinaryOperator.MULTIPLY,
                ("e1", percentage.id),
                origin,
            ),
        ),
        subject=Subject("s1", InstanceInterpretation.NORMAL_BUSINESS_INSTANCE),
        qualification_ref=None,
        grouping_refs=(),
        outputs=(RequestedOutput("output_1", "e2", origin),),
        ordering=(),
        selection=AllResults(),
        distinct_by=(),
    )
    index = analyze_requested_fact(
        fact,
        inputs={percentage.id: percentage},
        input_denotations=_denotation(percentage),
    )
    amount = RowSourceField(
        "amount",
        "field.amount",
        "amount",
        RowSourceValueType.DECIMAL,
        (),
    )
    source = RowSource(
        "source_events",
        RowSourceKind.API_READ,
        "events",
        read_id="list_events",
        fields=(amount,),
    )
    catalog = AvailableSourceCatalog(
        SourceContractSnapshot.from_content("{}"),
        (source,),
        (),
    )
    [clause] = index.qualification.clauses
    branch = SourceStrategyBranch(
        "fact_1:source_branch:1", (source.id,), (), (clause.clause_ref,)
    )
    strategy = _strategy(fact.id, branch)
    canonical = CanonicalInputValue(
        "percentage",
        percentage.id,
        tuple(item.use_ref for item in index.input_use_sites),
        FactValue.literal(
            id="percentage",
            literal_type=LiteralType.NUMBER,
            value="0.1",
            known_input_id=percentage.id,
            proof_refs=("question_input:i1",),
        ),
        ("question_input:i1",),
    )
    request = SemanticSourceBindingRequest(index, strategy, catalog, (canonical,))
    set_ref = next(
        ref.token for ref in index.source_requirement_refs if ref.kind.value == "set"
    )
    fact_ref = next(
        ref.token for ref in index.source_requirement_refs if ref.kind.value == "fact"
    )
    plan = SourceBindingPlan(
        strategy,
        {
            set_ref: (
                SetRealization(
                    branch.branch_id,
                    "Event rows.",
                    source.id,
                    None,
                    (),
                    (source.id,),
                ),
            )
        },
        {
            fact_ref: (
                FactRealization(
                    branch.branch_id,
                    "Returned amount.",
                    source.id,
                    FactRealizationKind.RETURNED_FIELD,
                    None,
                    (amount.field_ref,),
                    (source.id, amount.field_ref),
                ),
            )
        },
        {},
        (),
        {},
        SubjectObligationBinding(
            set_ref,
            (SubjectObligationRealization(branch.branch_id, ()),),
        ),
    )
    verified = verify_source_strategy(plan, request=request)
    assert isinstance(verified, VerifiedSourceStrategy)

    result = compile_verified_source_strategy(verified)

    [output] = result.answer_program.result_projection.scalar_outputs
    assert output.scalar_id == "fact_1.output_1.scalar"
