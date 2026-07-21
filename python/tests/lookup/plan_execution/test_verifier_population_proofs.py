from fervis.lookup.answer_program.expressions import FieldRef
from fervis.lookup.answer_program.model import AnswerProgram
from fervis.lookup.answer_program.operations import (
    FilterSpec,
    Operation,
    Predicate,
    PredicateOperator,
)
from fervis.lookup.answer_program.relations import (
    EndpointParamBinding,
    FieldBindingRole,
    PopulationCoverageClaim,
    PopulationCoverageRole,
    Relation,
    RelationField,
    RelationSource,
    SourceKind,
)
from fervis.lookup.answer_program.values import BindingSet, ConstantRef, FactValue, LiteralType
from fervis.lookup.fact_plan.row_sources import RowSourceCatalog
from fervis.lookup.plan_execution.verification.sources import (
    _verify_source_population_coverage_claims,
)
from fervis.lookup.question_contract import (
    AnswerPopulationMembershipTestKind,
    AnswerPopulationMembershipTestPolarity,
    MembershipTestRef,
    KnownInputSource,
    LiteralInputRole,
    QuestionContract,
    RequestedFact,
    RequestedFactAnswerPopulation,
    RequestedFactAnswerPopulationMembershipTest,
    RequestedFactAnswerOutput,
    RequestedFactLiteralInput,
)


def test_filter_claim_may_depend_on_its_input_source_and_filter_mechanics() -> None:
    test = RequestedFactAnswerPopulationMembershipTest(
        id="requested_state",
        kind=AnswerPopulationMembershipTestKind.EXPLICIT_USER_CONSTRAINT,
        polarity=AnswerPopulationMembershipTestPolarity.MUST_PASS,
        test_question="Does the row have the requested state?",
        owned_question_input_refs=("state_input",),
    )
    claim = PopulationCoverageClaim(
        test_ref=MembershipTestRef("fact_1", test.id),
        role=PopulationCoverageRole.ROW_POPULATION,
        proof_refs=("source_param:state", "returned_field:data.state"),
    )
    state = ConstantRef(
        constant_id="requested_state",
        version_ref="test@1",
        value=FactValue.literal(
            id="requested_state",
            literal_type=LiteralType.STRING,
            value="finished",
        ),
    )
    answer = AnswerProgram(
        relations=(
            Relation(
                id="rows",
                source=RelationSource(
                    kind=SourceKind.API_READ,
                    read_id="list_rows",
                    row_source_id="list_rows.data",
                    param_bindings=(
                        EndpointParamBinding(param_id="state", value_expr=state),
                    ),
                ),
                fields=(
                    RelationField(
                        field_id="data.state",
                        roles=(FieldBindingRole.PREDICATE,),
                    ),
                ),
            ),
        ),
        operations=(
            Operation(
                id="filter_state",
                spec=FilterSpec(
                    input_relation="rows",
                    predicate=Predicate(
                        left=FieldRef("data.state"),
                        operator=PredicateOperator.EQUALS,
                        right=state,
                    ),
                    proof_refs=("returned_field:data.state",),
                    population_coverage_claims=(claim,),
                ),
                output_relation="filtered_rows",
            ),
        ),
    )
    question_contract = QuestionContract(
        requested_facts=(
            RequestedFact(
                id="fact_1",
                description="requested rows",
                answer_outputs=(
                    RequestedFactAnswerOutput(id="answer", role="ANSWER_VALUE"),
                ),
                answer_population=RequestedFactAnswerPopulation(
                    counted_unit="row",
                    membership_tests=(
                        RequestedFactAnswerPopulationMembershipTest(
                            id="subject",
                            kind=AnswerPopulationMembershipTestKind.SUBJECT_IDENTITY,
                            polarity=AnswerPopulationMembershipTestPolarity.MUST_PASS,
                            test_question="Is this a row?",
                        ),
                        test,
                    ),
                ),
                known_inputs=(
                    RequestedFactLiteralInput(
                        id="state_input",
                        source=KnownInputSource.QUESTION_CONTEXT,
                        role=LiteralInputRole.PREDICATE_VALUE,
                        text="finished",
                        resolved_value_text="finished",
                    ),
                ),
                input_refs=("state_input",),
            ),
        ),
    )

    _verify_source_population_coverage_claims(
        answer,
        question_contract=question_contract,
        row_sources=RowSourceCatalog(),
        bindings=BindingSet(),
    )
