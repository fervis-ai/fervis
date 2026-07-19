from tests.lookup.grounding._support import (
    CALENDAR_END_PARAM_ID,
    CALENDAR_START_PARAM_ID,
    GroundedValueCertificationMethod,
    GroundingRequest,
    GroundingTurnPrompt,
    KnownInputSource,
    KnownTimeResolutionTask,
    LiteralInputRole,
    QuestionContract,
    RelationCatalog,
    RequestedFact,
    RequestedFactAnswerOutput,
    RequestedFactLiteralInput,
    ResolvedCanonicalIdentity,
    RuntimeValueContext,
    ValidationError,
    _BusinessTimeGroundingModel,
    _CurrentPeriodBusinessResultGroundingModel,
    _NoGroundingModel,
    _compiled_resolution_input,
    _full_period_time_intent,
    _named_quarter_time_intent,
    _point_date_time_intent,
    entity_key_value,
    ground_question_inputs,
    pytest,
    validate,
)
from tests.lookup.grounding._fixtures import (
    _date_sales_catalog,
    _quarter_question_contract,
    _time_question_contract,
)


def test_grounding_time_schema_rejects_relative_word_as_yearless_point_date():
    request = GroundingRequest(
        question="How many shifts do we have today?",
        tasks=(),
        resolver_catalog=RelationCatalog(),
        time_tasks=(
            KnownTimeResolutionTask(
                known_input_id="input_date",
                known_input_text="today",
                requested_fact_id="fact_1",
                time_expression="today",
            ),
        ),
    )
    schema = GroundingTurnPrompt(request).response_contract().provider_schema
    payload = {
        "known_time_resolutions": {
            "input_date": {
                "date_intent": {
                    "expression": "today",
                    "intent": {
                        "time_shape": "point_date",
                        "unit": "day",
                        "mode": "none",
                        "year": 0,
                        "month": 1,
                        "day": 1,
                        "year_policy": "none",
                        "relative_offset": 0,
                        "named_value": 0,
                        "end_year": 0,
                        "end_month": 0,
                        "end_day": 0,
                        "end_year_policy": "none",
                        "count": 0,
                        "direction": "none",
                    },
                }
            }
        },
        "known_input_binding_reviews": {},
    }

    with pytest.raises(ValidationError):
        validate(instance=payload, schema=schema)


def test_grounding_imports_resolved_canonical_identity_without_resolver():
    known = RequestedFactLiteralInput(
        id="input_staff",
        source=KnownInputSource.CONVERSATION_RESOLUTION,
        text="her",
        resolved_value_text="Alice Smith",
        value_meaning_hint="staff member",
        role=LiteralInputRole.REFERENCE_VALUE,
        resolved_input_ref="cr_input_1",
    )
    output = ground_question_inputs(
        question="What were her sales?",
        question_contract=QuestionContract(
            requested_facts=(
                RequestedFact(
                    id="fact_1",
                    description="sales for Alice Smith",
                    answer_outputs=(
                        RequestedFactAnswerOutput(
                            id="sales_total", role="ANSWER_VALUE"
                        ),
                    ),
                    known_inputs=(known,),
                ),
            )
        ),
        full_catalog=RelationCatalog(),
        resolver_selections=(),
        runtime_values=RuntimeValueContext(
            runtime_date="2026-05-09",
            timezone="Africa/London",
        ),
        model_port=_NoGroundingModel(),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
        conversation_resolution=_compiled_resolution_input(
            question="What were her sales?",
            input_ref="cr_input_1",
            source_text="her",
            resolved_text="Alice Smith",
            role=LiteralInputRole.REFERENCE_VALUE,
            value_meaning_hint="staff member",
            canonical_identity=ResolvedCanonicalIdentity(
                key=entity_key_value(
                    "staff",
                    "primary_key",
                    {"staff_id": "51515151-0000-0000-0002-000000000001"},
                ),
                authority_refs=("prior_source_read:staff:list:row_1",),
                lineage_refs=("memory:turn_1.entity.staff.alice",),
            ),
        ),
    )

    assert not output.ledger.issues
    value = output.ledger.values[0]
    assert value.payload.entity_kind == "staff"
    assert value.payload.only_component().component_id == "staff_id"
    assert (
        value.payload.only_component().value == "51515151-0000-0000-0002-000000000001"
    )
    assert value.proof_refs == (
        "known_input:input_staff",
        "resolved_question_input:cr_input_1",
        "prior_source_read:staff:list:row_1",
    )
    assert output.ledger.certifications[0].method == (
        GroundedValueCertificationMethod.IMPORTED_PRIOR_IDENTITY
    )
    assert output.ledger.certifications[0].authority_refs == (
        "prior_source_read:staff:list:row_1",
    )
    assert output.ledger.certifications[0].lineage_refs == (
        "known_input:input_staff",
        "resolved_question_input:cr_input_1",
        "memory:turn_1.entity.staff.alice",
    )


def test_time_grounding_records_known_input_proof_ref():
    output = ground_question_inputs(
        question="How much revenue on February 14, 2026?",
        question_contract=_time_question_contract("February 14, 2026"),
        full_catalog=_date_sales_catalog(),
        resolver_selections=(),
        runtime_values=RuntimeValueContext(
            runtime_date="2026-05-09",
            timezone="Africa/London",
        ),
        model_port=_BusinessTimeGroundingModel(
            intents_by_text={
                "February 14, 2026": _point_date_time_intent(
                    "February 14, 2026",
                    year=2026,
                    month=2,
                    day=14,
                )
            }
        ),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )

    assert not output.ledger.issues
    assert output.ledger.values[0].proof_refs == ("known_input:input_date",)
    assert {
        (use.row_source_id, use.param_id, use.value_component.value)
        for use in output.ledger.uses
    } == {
        ("rs_calendar_days", CALENDAR_START_PARAM_ID, "start"),
        ("rs_calendar_days", CALENDAR_END_PARAM_ID, "end"),
    }


def test_time_grounding_uses_conversation_resolved_value_text():
    time_input = RequestedFactLiteralInput(
        id="input_date",
        source=KnownInputSource.CONVERSATION_RESOLUTION,
        text="that same period",
        resolved_value_text="yesterday",
        role=LiteralInputRole.TIME_VALUE,
        resolved_input_ref="cr_input_time_1",
    )
    contract = QuestionContract(
        question_inputs=(time_input,),
        requested_facts=(
            RequestedFact(
                id="fact_1",
                description="sales total for the resolved period",
                answer_outputs=(
                    RequestedFactAnswerOutput(id="total_sales", role="ANSWER_VALUE"),
                ),
                known_inputs=(time_input,),
                input_refs=("input_date",),
            ),
        ),
    )

    output = ground_question_inputs(
        question="What about that same period?",
        question_contract=contract,
        full_catalog=_date_sales_catalog(),
        resolver_selections=(),
        runtime_values=RuntimeValueContext(
            runtime_date="2026-05-09",
            timezone="Africa/London",
        ),
        model_port=_BusinessTimeGroundingModel(
            intents_by_text={
                "yesterday": _point_date_time_intent(
                    "yesterday",
                    year=2026,
                    month=5,
                    day=8,
                )
            }
        ),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
        conversation_resolution=_compiled_resolution_input(
            question="What about that same period?",
            input_ref="cr_input_time_1",
            source_text="that same period",
            resolved_text="yesterday",
            role=LiteralInputRole.TIME_VALUE,
            value_meaning_hint="time scope",
        ),
    )

    assert not output.ledger.issues
    value = output.ledger.values[0]
    assert value.payload.expression == "yesterday"
    assert value.payload.resolved_start == "2026-05-08"
    assert value.payload.resolved_end == "2026-05-08"


def test_time_grounding_uses_model_authored_quarter_intent_without_year():
    output = ground_question_inputs(
        question="How much sales at ABC Mall in Q1?",
        question_contract=_quarter_question_contract("Q1"),
        full_catalog=_date_sales_catalog(),
        resolver_selections=(),
        runtime_values=RuntimeValueContext(
            runtime_date="2026-05-12",
            timezone="Africa/London",
        ),
        model_port=_BusinessTimeGroundingModel(
            intents_by_text={"Q1": _named_quarter_time_intent("Q1", quarter=1)}
        ),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )

    assert not output.ledger.issues
    value = output.ledger.values[0]
    assert value.payload.resolved_start == "2026-01-01"
    assert value.payload.resolved_end == "2026-03-31"


def test_time_grounding_uses_model_authored_full_month_for_this_month():
    output = ground_question_inputs(
        question="How much cash was deposited this month?",
        question_contract=_time_question_contract("this month"),
        full_catalog=_date_sales_catalog(),
        resolver_selections=(),
        runtime_values=RuntimeValueContext(
            runtime_date="2026-06-01",
            timezone="Africa/London",
        ),
        model_port=_BusinessTimeGroundingModel(
            intents_by_text={"this month": _full_period_time_intent("this month")}
        ),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )

    assert not output.ledger.issues
    value = output.ledger.values[0]
    assert value.payload.resolved_start == "2026-06-01"
    assert value.payload.resolved_end == "2026-06-30"


def test_time_grounding_treats_explicit_current_week_to_date_wording_as_to_date():
    output = ground_question_inputs(
        question="How much revenue did we make this week so far?",
        question_contract=_time_question_contract("this week so far"),
        full_catalog=_date_sales_catalog(),
        resolver_selections=(),
        runtime_values=RuntimeValueContext(
            runtime_date="2026-06-04",
            timezone="Africa/London",
        ),
        model_port=_CurrentPeriodBusinessResultGroundingModel(),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )

    assert not output.ledger.issues
    value = output.ledger.values[0]
    assert value.payload.resolved_start == "2026-06-01"
    assert value.payload.resolved_end == "2026-06-04"
    assert value.payload.intent["mode"] == "to_date"
