import pytest

from fervis.lookup.relation_catalog import (
    CatalogFact,
    CatalogFactAvailability,
    CatalogField,
    CatalogParam,
    EndpointRead,
    ParamSource,
    RelationCatalog,
    RowCardinality,
    RowPath,
)
from fervis.lookup.relation_catalog.selection import (
    CatalogSelectionResult,
    RequestedFactCatalogSelection,
)
from fervis.lookup.plan_execution.errors import VerificationError
from fervis.lookup.plan_execution.verification import (
    verify_fact_plan as verify_fact_plan_impl,
)
from fervis.lookup.grounding.model import GroundedInputUse
from fervis.lookup.answer_program.model import AnswerProgram, FactFulfillment
from fervis.lookup.fact_plan.fact_plan import (
    BlockedFact,
    BlockedFactBasis,
    BlockedFactField,
    FactPlan,
    MissingCatalogChoiceInput,
    MissingCatalogRequiredInput,
    PlanClarification,
    PlanImpossible,
)
from fervis.lookup.answer_program.operations import (
    Operation,
    ProjectField,
    ProjectSpec,
)
from fervis.lookup.answer_program.relations import (
    FieldBindingRole,
    Relation,
    RelationField,
    RelationSource,
    SourceKind,
)
from fervis.lookup.fact_plan.row_sources import (
    api_row_source_id,
    read_evidence_ref,
    read_field_evidence_ref,
)
from fervis.lookup.answer_program.values import FactValue
from fervis.lookup.fact_planning.required_inputs import required_input_id
from fervis.lookup.question_contract import (
    KnownInputSource,
    LiteralInputRole,
    QuestionContract,
    RequestedFact,
    RequestedFactAnswerOutput,
    RequestedFactLiteralInput,
)
from fervis.lookup.answer_program.result_projection import (
    RelationResultOutput,
    ResultProjection,
)


def _result_projection() -> ResultProjection:
    return ResultProjection(
        relation_outputs=(
            RelationResultOutput(id="answer", relation_id="result", field_id="name"),
        )
    )


def _operation() -> Operation:
    return Operation(
        id="op",
        spec=ProjectSpec(
            input_relation="rows",
            fields=(ProjectField(source="name"),),
        ),
        output_relation="result",
    )


def test_fact_plan_accepts_only_one_terminal_shape():
    question_contract = _question_contract()
    plan = FactPlan(
        outcome=AnswerProgram(
            fulfillment=(
                FactFulfillment(
                    requested_fact_id="rf_name",
                    answer_output_id="answer",
                    result_output_id="answer",
                ),
            ),
            relations=(
                Relation(
                    id="rows",
                    source=RelationSource(
                        kind=SourceKind.API_READ,
                        read_id="list_rows",
                    ),
                    fields=(
                        RelationField(
                            field_id="name",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(_operation(),),
            result_projection=_result_projection(),
        )
    )
    assert verify_fact_plan(plan, question_contract=question_contract) is plan


def test_answer_plan_requires_operations_and_result_projection():
    fulfillment = (
        FactFulfillment(
            requested_fact_id="rf_name",
            answer_output_id="answer",
            result_output_id="answer",
        ),
    )
    missing_operations = FactPlan(
        outcome=AnswerProgram(
            fulfillment=fulfillment,
            result_projection=_result_projection(),
        )
    )
    missing_render = FactPlan(
        outcome=AnswerProgram(
            fulfillment=fulfillment,
            relations=(
                Relation(
                    id="rows",
                    source=RelationSource(
                        kind=SourceKind.API_READ,
                        read_id="list_rows",
                    ),
                    fields=(
                        RelationField(
                            field_id="name",
                            roles=(FieldBindingRole.OUTPUT,),
                        ),
                    ),
                ),
            ),
            operations=(_operation(),),
        )
    )

    with pytest.raises(VerificationError, match="at least one operation"):
        verify_fact_plan(missing_operations)
    with pytest.raises(VerificationError, match="result projection"):
        verify_fact_plan(missing_render)


def test_pre_execution_clarification_without_operations_is_valid():
    input_id = required_input_id(
        row_source_id=api_row_source_id("list_rows", "data"),
        param_id="selector",
    )
    verified = verify_fact_plan(
        FactPlan(
            outcome=PlanClarification(
                missing_catalog_inputs=(
                    MissingCatalogChoiceInput(
                        id="metric_choice",
                        requested_fact_id="rf_name",
                        required_catalog_choice_input_id=input_id,
                    ),
                ),
            )
        ),
        catalog=_required_choice_catalog(),
    )
    assert verified.outcome.missing_catalog_inputs == (
        MissingCatalogChoiceInput(
            id="metric_choice",
            requested_fact_id="rf_name",
            required_catalog_choice_input_id=input_id,
        ),
    )


def test_missing_value_clarification_requires_structural_requirement_handle():
    with pytest.raises(
        ValueError,
        match="requires catalog input",
    ):
        MissingCatalogRequiredInput(
            id="ask_metric",
            requested_fact_id="rf_name",
            required_catalog_input_id="",
        )


def test_requested_fact_output_and_known_input_ids_are_disjoint():
    with pytest.raises(
        ValueError,
        match="answer output and known input ids must be disjoint",
    ):
        QuestionContract(
            requested_facts=(
                RequestedFact(
                    id="rf_name",
                    description="name",
                    answer_outputs=(
                        RequestedFactAnswerOutput(id="answer", role="ANSWER_VALUE"),
                    ),
                    known_inputs=(
                        RequestedFactLiteralInput(
                            id="answer",
                            source=KnownInputSource.QUESTION_CONTEXT,
                            text="Alice",
                            role=LiteralInputRole.REFERENCE_VALUE,
                            resolved_value_text="Alice",
                        ),
                    ),
                ),
            )
        )


def test_missing_value_clarification_requires_catalog_required_input():
    with pytest.raises(
        VerificationError,
        match="unknown required input",
    ):
        verify_fact_plan(
            FactPlan(
                outcome=PlanClarification(
                    missing_catalog_inputs=(
                        MissingCatalogRequiredInput(
                            id="ask_timezone",
                            requested_fact_id="rf_name",
                            required_catalog_input_id="anchor_timezone",
                        ),
                    ),
                )
            )
        )


def test_missing_value_clarification_accepts_catalog_required_input():
    source_id = api_row_source_id("list_rows", "data")
    input_id = required_input_id(
        row_source_id=source_id,
        param_id="selector",
    )
    verified = verify_fact_plan(
        FactPlan(
            outcome=PlanClarification(
                missing_catalog_inputs=(
                    MissingCatalogRequiredInput(
                        id="ask_selector",
                        requested_fact_id="rf_name",
                        required_catalog_input_id=input_id,
                    ),
                ),
            )
        ),
        catalog=_required_input_catalog(),
    )
    assert verified.outcome.missing_catalog_inputs == (
        MissingCatalogRequiredInput(
            id="ask_selector",
            requested_fact_id="rf_name",
            required_catalog_input_id=input_id,
        ),
    )


def test_missing_value_clarification_rejects_grounded_required_input():
    source_id = api_row_source_id("list_rows", "data")
    input_id = required_input_id(
        row_source_id=source_id,
        param_id="selector",
    )

    with pytest.raises(VerificationError, match="already satisfied"):
        verify_fact_plan(
            FactPlan(
                outcome=PlanClarification(
                    missing_catalog_inputs=(
                        MissingCatalogRequiredInput(
                            id="ask_selector",
                            requested_fact_id="rf_name",
                            required_catalog_input_id=input_id,
                        ),
                    ),
                )
            ),
            catalog=_required_input_catalog(),
            available_values=(FactValue.named(id="selector_value", text="selector"),),
            available_value_uses=(
                GroundedInputUse(
                    id="grounded_selector",
                    value_id="selector_value",
                    row_source_id=source_id,
                    param_id="selector",
                    requested_fact_id="rf_name",
                ),
            ),
        )


def test_missing_value_clarification_rejects_internal_calendar_required_input():
    with pytest.raises(
        VerificationError,
        match="unknown required input",
    ):
        verify_fact_plan(
            FactPlan(
                outcome=PlanClarification(
                    missing_catalog_inputs=(
                        MissingCatalogRequiredInput(
                            id="ask_date",
                            requested_fact_id="rf_name",
                            required_catalog_input_id="rs_calendar_days.interval_start",
                        ),
                    ),
                )
            )
        )


def test_missing_value_clarification_cannot_target_known_input():
    with pytest.raises(
        VerificationError,
        match="cannot target known input or answer output",
    ):
        verify_fact_plan(
            FactPlan(
                outcome=PlanClarification(
                    missing_catalog_inputs=(
                        MissingCatalogRequiredInput(
                            id="ask_person",
                            requested_fact_id="rf_name",
                            required_catalog_input_id="person_name",
                        ),
                    ),
                )
            ),
            question_contract=_question_contract_with_known_person(),
        )


def test_catalog_choice_clarification_requires_catalog_choice_input():
    with pytest.raises(
        VerificationError,
        match="cannot target known input or answer output",
    ):
        verify_fact_plan(
            FactPlan(
                outcome=PlanClarification(
                    missing_catalog_inputs=(
                        MissingCatalogChoiceInput(
                            id="metric_choice",
                            requested_fact_id="rf_name",
                            required_catalog_choice_input_id="person_name",
                        ),
                    ),
                )
            ),
            question_contract=_question_contract_with_known_person(),
        )


def test_catalog_choice_clarification_accepts_choice_bearing_required_input():
    input_id = required_input_id(
        row_source_id=api_row_source_id("list_rows", "data"),
        param_id="selector",
    )
    verified = verify_fact_plan(
        FactPlan(
            outcome=PlanClarification(
                missing_catalog_inputs=(
                    MissingCatalogChoiceInput(
                        id="metric_choice",
                        requested_fact_id="rf_name",
                        required_catalog_choice_input_id=input_id,
                    ),
                ),
            )
        ),
        question_contract=_question_contract_with_known_person(),
        catalog=_required_choice_catalog(),
    )
    assert verified.outcome.missing_catalog_inputs == (
        MissingCatalogChoiceInput(
            id="metric_choice",
            requested_fact_id="rf_name",
            required_catalog_choice_input_id=input_id,
        ),
    )


def test_catalog_choice_clarification_rejects_non_choice_input():
    input_id = required_input_id(
        row_source_id=api_row_source_id("list_rows", "data"),
        param_id="selector",
    )
    with pytest.raises(
        VerificationError,
        match="choice-bearing required input",
    ):
        verify_fact_plan(
            FactPlan(
                outcome=PlanClarification(
                    missing_catalog_inputs=(
                        MissingCatalogChoiceInput(
                            id="metric_choice",
                            requested_fact_id="rf_name",
                            required_catalog_choice_input_id=input_id,
                        ),
                    ),
                )
            ),
            catalog=_required_input_catalog(),
        )


def test_impossible_rejects_missing_reviewed_reads_without_repair():
    plan = FactPlan(
        outcome=PlanImpossible(
            blocked_facts=(
                BlockedFact(
                    requested_fact_id="rf_name",
                    basis=BlockedFactBasis.CATALOG_ACCESS,
                    evidence_refs=(
                        read_field_evidence_ref(
                            read_id="restricted_read",
                            field_id="masked_name",
                        ),
                    ),
                ),
            )
        )
    )

    with pytest.raises(VerificationError, match="requires reviewed reads"):
        verify_fact_plan(plan, catalog=_blocked_catalog())


def test_impossible_review_scope_uses_requested_fact_catalog_selection():
    plan = FactPlan(
        outcome=PlanImpossible(
            blocked_facts=(
                BlockedFact(
                    requested_fact_id="rf_name",
                    basis=BlockedFactBasis.CATALOG_ACCESS,
                    evidence_refs=(read_evidence_ref("restricted_read"),),
                    reviewed_read_ids=("restricted_read",),
                ),
            )
        )
    )
    selection = CatalogSelectionResult(
        relation_catalog=_blocked_catalog(),
        requested_fact_selections=(
            RequestedFactCatalogSelection(
                requested_fact_id="rf_name",
                query_terms=("name",),
                rankings=(),
                selected_read_ids=("restricted_read",),
            ),
        ),
        selected_read_ids=("restricted_read",),
    )

    verified = verify_fact_plan(
        plan,
        catalog=_blocked_catalog(),
        catalog_selection=selection,
    )

    blocked = verified.outcome.blocked_facts[0]
    assert blocked.reviewed_read_ids == ("restricted_read",)
    assert "other_read" not in blocked.reviewed_read_ids


def test_impossible_preserves_the_model_visible_review_scope_after_narrowing():
    plan = FactPlan(
        outcome=PlanImpossible(
            blocked_facts=(
                BlockedFact(
                    requested_fact_id="rf_name",
                    basis=BlockedFactBasis.CATALOG_ACCESS,
                    evidence_refs=(read_evidence_ref("restricted_read"),),
                    reviewed_read_ids=("restricted_read",),
                ),
            )
        )
    )
    selection = CatalogSelectionResult(
        relation_catalog=_blocked_catalog(),
        requested_fact_selections=(
            RequestedFactCatalogSelection(
                requested_fact_id="rf_name",
                query_terms=("name",),
                rankings=(),
                selected_read_ids=("restricted_read", "other_read"),
            ),
        ),
        selected_read_ids=("restricted_read", "other_read"),
    )

    verified = verify_fact_plan(
        plan,
        catalog=_blocked_catalog(),
        catalog_selection=selection,
    )

    assert verified.outcome.blocked_facts[0].reviewed_read_ids == ("restricted_read",)


def test_impossible_verifies_selected_catalog_surface_with_unselected_diagnostics():
    plan = FactPlan(
        outcome=PlanImpossible(
            blocked_facts=(
                BlockedFact(
                    requested_fact_id="rf_name",
                    basis=BlockedFactBasis.CATALOG_ACCESS,
                    evidence_refs=(read_evidence_ref("restricted_read"),),
                    reviewed_read_ids=("restricted_read",),
                ),
            )
        )
    )
    selection = CatalogSelectionResult(
        relation_catalog=_blocked_catalog(),
        requested_fact_selections=(
            RequestedFactCatalogSelection(
                requested_fact_id="rf_name",
                query_terms=("name",),
                rankings=(),
                selected_read_ids=("restricted_read",),
                unselected_positive_read_ids=("other_read",),
            ),
        ),
        selected_read_ids=("restricted_read",),
    )

    verified = verify_fact_plan(
        plan,
        catalog=_blocked_catalog(),
        catalog_selection=selection,
    )

    blocked = verified.outcome.blocked_facts[0]
    assert blocked.reviewed_read_ids == ("restricted_read",)


def test_impossible_policy_basis_requires_policy_proof():
    plan = FactPlan(
        outcome=PlanImpossible(
            blocked_facts=(
                BlockedFact(
                    requested_fact_id="rf_name",
                    basis=BlockedFactBasis.POLICY_ACCESS,
                    evidence_refs=(
                        read_field_evidence_ref(
                            read_id="restricted_read",
                            field_id="masked_name",
                        ),
                    ),
                    reviewed_read_ids=("restricted_read",),
                ),
            )
        )
    )
    selection = CatalogSelectionResult(
        relation_catalog=_blocked_catalog(),
        requested_fact_selections=(
            RequestedFactCatalogSelection(
                requested_fact_id="rf_name",
                query_terms=("name",),
                rankings=(),
                selected_read_ids=("restricted_read",),
            ),
        ),
        selected_read_ids=("restricted_read",),
    )

    with pytest.raises(
        VerificationError,
        match="policy blocked fact requires policy evidence",
    ):
        verify_fact_plan(
            plan,
            catalog=_blocked_catalog(),
            catalog_selection=selection,
        )


def test_zero_catalog_selection_requires_exact_selection_evidence():
    plan = FactPlan(
        outcome=PlanImpossible(
            blocked_facts=(
                BlockedFact(
                    requested_fact_id="rf_name",
                    basis=BlockedFactBasis.CATALOG_ACCESS,
                    evidence_refs=("catalog_selection:rf_name",),
                ),
            )
        )
    )
    selection = CatalogSelectionResult(
        relation_catalog=RelationCatalog(),
        requested_fact_selections=(
            RequestedFactCatalogSelection(
                requested_fact_id="rf_name",
                query_terms=("name",),
                rankings=(),
                selected_read_ids=(),
                unselected_positive_read_ids=(),
            ),
        ),
        selected_read_ids=(),
    )

    verified = verify_fact_plan(
        plan,
        catalog=RelationCatalog(),
        catalog_selection=selection,
    )

    blocked = verified.outcome.blocked_facts[0]
    assert blocked.evidence_refs == ("catalog_selection:rf_name",)
    assert blocked.reviewed_read_ids == ()
    assert blocked.nearest_fields == ()


def test_zero_selected_catalog_selection_with_unselected_diagnostics_verifies():
    plan = FactPlan(
        outcome=PlanImpossible(
            blocked_facts=(
                BlockedFact(
                    requested_fact_id="rf_name",
                    basis=BlockedFactBasis.CATALOG_ACCESS,
                    evidence_refs=("catalog_selection:rf_name",),
                ),
            )
        )
    )
    selection = CatalogSelectionResult(
        relation_catalog=RelationCatalog(),
        requested_fact_selections=(
            RequestedFactCatalogSelection(
                requested_fact_id="rf_name",
                query_terms=("name",),
                rankings=(),
                selected_read_ids=(),
                unselected_positive_read_ids=("restricted_read",),
            ),
        ),
        selected_read_ids=(),
    )

    verified = verify_fact_plan(
        plan,
        catalog=RelationCatalog(),
        catalog_selection=selection,
    )

    blocked = verified.outcome.blocked_facts[0]
    assert blocked.evidence_refs == ("catalog_selection:rf_name",)
    assert blocked.reviewed_read_ids == ()
    assert blocked.nearest_fields == ()


def _question_contract() -> QuestionContract:
    return QuestionContract(
        requested_facts=(
            RequestedFact(
                id="rf_name",
                description="name",
                answer_outputs=(
                    RequestedFactAnswerOutput(id="answer", role="ANSWER_VALUE"),
                ),
            ),
        )
    )


def _answer_name_evidence_ref() -> str:
    return read_field_evidence_ref(read_id="list_rows", field_id="name")


def _question_contract_with_known_person() -> QuestionContract:
    return QuestionContract(
        requested_facts=(
            RequestedFact(
                id="rf_name",
                description="name",
                answer_outputs=(
                    RequestedFactAnswerOutput(id="answer", role="ANSWER_VALUE"),
                ),
                known_inputs=(
                    RequestedFactLiteralInput(
                        id="person_name",
                        source=KnownInputSource.QUESTION_CONTEXT,
                        text="Alice",
                        role=LiteralInputRole.REFERENCE_VALUE,
                        resolved_value_text="Alice",
                    ),
                ),
            ),
        )
    )


def _blocked_catalog() -> RelationCatalog:
    return RelationCatalog(
        reads=(
            EndpointRead(
                id="restricted_read",
                endpoint_name="list_restricted",
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(
                        ref="field.masked_name",
                        path="data.masked_name",
                        row_path_id="data",
                        type="string",
                    ),
                ),
                facts=(
                    CatalogFact(
                        ref="person.masked_name",
                        availability=CatalogFactAvailability.NOT_READABLE,
                        field_ref="field.masked_name",
                        read_id="restricted_read",
                        proof_refs=("proof:masked_name_only",),
                    ),
                ),
                source_metadata={"description": "Only masked names are readable."},
            ),
            EndpointRead(
                id="other_read",
                endpoint_name="list_other",
                source_metadata={"description": "Unrelated read."},
            ),
        )
    )


def test_impossible_rejects_unknown_catalog_evidence():
    plan = FactPlan(
        outcome=PlanImpossible(
            blocked_facts=(
                BlockedFact(
                    requested_fact_id="rf_name",
                    basis=BlockedFactBasis.CATALOG_ACCESS,
                    evidence_refs=("catalog:unknown",),
                    reviewed_read_ids=("list_rows",),
                    nearest_fields=(
                        BlockedFactField(
                            read_id="list_rows",
                            field_id="name",
                        ),
                    ),
                ),
            )
        )
    )

    with pytest.raises(VerificationError, match="unknown catalog evidence"):
        verify_fact_plan(plan, catalog=_answer_catalog())


def _required_input_catalog() -> RelationCatalog:
    return RelationCatalog(
        reads=(
            EndpointRead(
                id="list_rows",
                endpoint_name="list_rows",
                params=(
                    CatalogParam(
                        ref="list_rows.query.selector",
                        name="selector",
                        source=ParamSource.QUERY,
                        type="string",
                        required=True,
                    ),
                ),
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(
                        ref="field.name",
                        path="data.name",
                        row_path_id="data",
                        type="string",
                    ),
                ),
            ),
        )
    )


def _required_choice_catalog() -> RelationCatalog:
    return RelationCatalog(
        reads=(
            EndpointRead(
                id="list_rows",
                endpoint_name="list_rows",
                params=(
                    CatalogParam(
                        ref="list_rows.query.selector",
                        name="selector",
                        source=ParamSource.QUERY,
                        type="string",
                        required=True,
                        choices=("sales", "units"),
                    ),
                ),
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(
                        ref="field.name",
                        path="data.name",
                        row_path_id="data",
                        type="string",
                    ),
                ),
            ),
        )
    )


def verify_fact_plan(plan: FactPlan, **kwargs):
    question_contract = kwargs.pop("question_contract", _question_contract())
    catalog = kwargs.pop("catalog", _answer_catalog())
    if not isinstance(plan.outcome, AnswerProgram):
        return verify_fact_plan_impl(
            plan,
            question_contract=question_contract,
            catalog=catalog,
            **kwargs,
        )
    from fervis.lookup.answer_program.compilation import compile_answer_program
    from fervis.lookup.answer_program.instantiation import (
        ExecutionEnvironment,
        instantiate_answer_program,
    )

    program, bindings = compile_answer_program(
        plan.outcome,
        question_contract=question_contract,
        catalog=catalog,
        bindings=plan.bindings,
    )
    instantiate_answer_program(
        program,
        bindings,
        ExecutionEnvironment(catalog=catalog),
    )
    return plan


def _answer_catalog() -> RelationCatalog:
    return RelationCatalog(
        reads=(
            EndpointRead(
                id="list_rows",
                endpoint_name="list_rows",
                row_paths=(
                    RowPath(id="data", path="data", cardinality=RowCardinality.MANY),
                ),
                fields=(
                    CatalogField(
                        ref="field.name",
                        path="data.name",
                        row_path_id="data",
                        type="string",
                    ),
                ),
                facts=(
                    CatalogFact(
                        ref="field.name",
                        field_ref="field.name",
                    ),
                ),
            ),
        )
    )
