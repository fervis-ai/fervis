import json

import pytest
from jsonschema import ValidationError, validate

from fervis.lookup.relation_catalog import (
    CatalogField,
    CatalogParam,
    EndpointRead,
    IdentityMetadata,
    ParamSource,
    RelationCatalog,
    RowCardinality,
    RowPath,
)
from fervis.lookup.conversation_resolution import (
    CompiledConversationResolution,
    ResolvedCanonicalIdentity,
    ResolvedLiteralQuestionInput,
)
from fervis.lookup.conversation_resolution.compilation import CompiledResolvedClause
from fervis.memory.addresses import FactAddress
from fervis.memory.artifacts import (
    build_fact_artifact,
    FactOutcome,
)
from fervis.lookup.grounding.resolution import ground_question_inputs
from fervis.lookup.grounding.model import (
    GroundedValueCertificationMethod,
    GroundingTerminalKind,
)
from fervis.lookup.grounding.model import (
    InputBindingOption,
    GroundingRequest,
    KnownInputBindingTask,
    KnownTimeResolutionTask,
)
from fervis.lookup.grounding.prompt import GroundingTurnPrompt
from fervis.lookup.fact_planning.request import RuntimeValueContext
from fervis.lookup.fact_plan.row_sources import (
    CALENDAR_END_PARAM_ID,
    CALENDAR_START_PARAM_ID,
)
from fervis.lookup.answer_program.values import IdentityValuePayload
from fervis.lookup.turn_prompts import build_turn_prompt_context
from fervis.lookup.question_contract import (
    AnswerPopulationMembershipTestKind,
    AnswerPopulationMembershipTestPolarity,
    KnownInputSource,
    LiteralInputRole,
    QuestionContract,
    RequestedFact,
    RequestedFactAnswerPopulation,
    RequestedFactAnswerPopulationMembershipTest,
    RequestedFactAnswerOutput,
    RequestedFactKnownInput,
    RequestedFactLiteralInput,
)


def _compiled_resolution_input(
    *,
    question: str,
    input_ref: str,
    source_text: str,
    resolved_text: str,
    role: LiteralInputRole,
    value_meaning_hint: str,
    canonical_identity: ResolvedCanonicalIdentity | None = None,
) -> CompiledConversationResolution:
    return CompiledConversationResolution(
        current_question_text=question,
        contextualized_question=question,
        clauses=(
            CompiledResolvedClause(
                current_clause_text=question,
                resolved_text=question,
                retained_frame_parts=(),
                values=(),
            ),
        ),
        inputs=(
            ResolvedLiteralQuestionInput(
                input_ref=input_ref,
                value_source_text=source_text,
                resolved_value_text=resolved_text,
                role=role,
                value_meaning_hint=value_meaning_hint,
                canonical_identity=canonical_identity,
            ),
        ),
        frame_call=None,
        used_source_card_ids=(),
        used_memory_ids=(),
    )


class _DataAccess:
    def __init__(self, body):
        self.body = body
        self.calls = []

    def read(self, *, endpoint_name, args):
        self.calls.append((endpoint_name, dict(args)))
        return self.body


class _GroundingModel:
    def __init__(self, *, known_input_id: str, binding_option_id: str):
        self.known_input_id = known_input_id
        self.binding_option_id = binding_option_id

    def generate(self, **kwargs):
        prompt = str(kwargs.get("prompt") or "")
        return {
            "answer": json.dumps(
                {
                    "tool": "submit_grounding",
                    "arguments": _grounding_review_arguments(
                        prompt,
                        selected_by_input={
                            self.known_input_id: self.binding_option_id,
                        },
                    ),
                }
            ),
            "usage": {},
        }


def _reference_input(
    input_id: str,
    text: str,
    *,
    value_meaning_hint: str = "",
    resolved_value_text: str | None = None,
    field_label_text: str = "",
) -> RequestedFactKnownInput:
    return RequestedFactLiteralInput(
        id=input_id,
        source=KnownInputSource.QUESTION_CONTEXT,
        text=text,
        resolved_value_text=resolved_value_text or text,
        field_label_text=field_label_text,
        value_meaning_hint=value_meaning_hint,
        role=LiteralInputRole.REFERENCE_VALUE,
    )


def _time_input(input_id: str, text: str) -> RequestedFactKnownInput:
    return RequestedFactLiteralInput(
        id=input_id,
        source=KnownInputSource.QUESTION_CONTEXT,
        text=text,
        resolved_value_text=text,
        role=LiteralInputRole.TIME_VALUE,
    )


def _result_limit_input(
    input_id: str,
    text: str,
    *,
    resolved_value_text: str,
) -> RequestedFactKnownInput:
    return RequestedFactLiteralInput(
        id=input_id,
        source=KnownInputSource.QUESTION_CONTEXT,
        text=text,
        resolved_value_text=resolved_value_text,
        role=LiteralInputRole.RESULT_LIMIT,
    )


class _AreaRouteGroundingModel:
    def __init__(self):
        self.prompt = ""

    def generate(self, **kwargs):
        self.prompt = str(kwargs.get("prompt") or "")
        payload = _json_payload_from_prompt_section(self.prompt, "Binding options:")
        selected = ""
        for task in payload["known_input_binding_options"]:
            for option in task["binding_options"]:
                if option.get("read_id") == "list_area_list":
                    selected = option["binding_option_id"]
                    break
            if selected:
                known_input_id = task["known_input_id"]
                break
        else:
            raise AssertionError("grounding prompt did not expose list_area_list route")
        return {
            "answer": json.dumps(
                {
                    "tool": "submit_grounding",
                    "arguments": _grounding_review_arguments(
                        self.prompt,
                        selected_by_input={known_input_id: selected},
                    ),
                }
            ),
            "usage": {},
        }


class _ReadRouteGroundingModel:
    def __init__(self, *, read_id: str):
        self.read_id = read_id
        self.prompt = ""

    def generate(self, **kwargs):
        self.prompt = str(kwargs.get("prompt") or "")
        for task in _json_payload_from_prompt_section(self.prompt, "Binding options:")[
            "known_input_binding_options"
        ]:
            for option in task["binding_options"]:
                if option.get("read_id") == self.read_id:
                    selected_by_input = {
                        task["known_input_id"]: option["binding_option_id"]
                    }
                    return {
                        "answer": json.dumps(
                            {
                                "tool": "submit_grounding",
                                "arguments": _grounding_review_arguments(
                                    self.prompt,
                                    selected_by_input=selected_by_input,
                                ),
                            }
                        ),
                        "usage": {},
                    }
        raise AssertionError(f"grounding prompt did not expose {self.read_id} route")


class _CompatibilityGroundingModel:
    def __init__(self, *, compatible_read_ids: set[str]):
        self.compatible_read_ids = compatible_read_ids
        self.prompt = ""

    def generate(self, **kwargs):
        self.prompt = str(kwargs.get("prompt") or "")
        payload = _json_payload_from_prompt_section(self.prompt, "Binding options:")
        reviews = {}
        for task in payload["known_input_binding_options"]:
            reviews[task["known_input_id"]] = {
                "option_reviews": {
                    option["binding_option_id"]: {
                        "resolver_fit_question": option["resolver_fit_question"],
                        "because": "Reviewed by compatibility test model.",
                        "decision": (
                            "CAN_RESOLVE_LOOKUP_TEXT"
                            if option.get("read_id") in self.compatible_read_ids
                            else "CANNOT_RESOLVE_LOOKUP_TEXT"
                        ),
                    }
                    for option in task["binding_options"]
                }
            }
        return {
            "answer": json.dumps(
                {
                    "tool": "submit_grounding",
                    "arguments": {
                        "known_time_resolutions": {},
                        "known_input_binding_reviews": reviews,
                    },
                }
            ),
            "usage": {},
        }


class _BusinessTimeGroundingModel:
    def __init__(self, *, intents_by_text: dict[str, dict[str, object]]):
        self.intents_by_text = intents_by_text
        self.prompt = ""

    def generate(self, **kwargs):
        self.prompt = str(kwargs.get("prompt") or "")
        time_resolutions = {}
        for task in _json_payload_from_prompt_section(
            self.prompt,
            "Time inputs to resolve:",
        )["known_time_resolution_tasks"]:
            text = task["time_expression"]
            time_resolutions[task["known_input_id"]] = {
                "date_intent": self.intents_by_text[text]
            }
        return {
            "answer": json.dumps(
                {
                    "tool": "submit_grounding",
                    "arguments": {
                        "known_time_resolutions": time_resolutions,
                        "known_input_binding_reviews": {},
                    },
                }
            ),
            "usage": {},
        }


class _CurrentPeriodBusinessResultGroundingModel:
    def __init__(self) -> None:
        self.prompt = ""

    def generate(self, **kwargs):
        self.prompt = str(kwargs.get("prompt") or "")
        time_resolutions = {}
        for task in _json_payload_from_prompt_section(
            self.prompt,
            "Time inputs to resolve:",
        )["known_time_resolution_tasks"]:
            text = task["time_expression"]
            lowered = text.lower()
            use_to_date = "so far" in lowered or "to date" in lowered
            time_resolutions[task["known_input_id"]] = {
                "date_intent": _period_relative_time_intent(
                    text,
                    unit="week",
                    mode="to_date" if use_to_date else "full",
                )
            }
        return {
            "answer": json.dumps(
                {
                    "tool": "submit_grounding",
                    "arguments": {
                        "known_time_resolutions": time_resolutions,
                        "known_input_binding_reviews": {},
                    },
                }
            ),
            "usage": {},
        }


class _NoGroundingModel:
    def generate(self, **kwargs):
        raise AssertionError(
            "grounding model should not be called for deterministic single-route grounding"
        )


class _EndpointDataAccess:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def read(self, *, endpoint_name, args):
        self.calls.append((endpoint_name, dict(args)))
        return _endpoint_result(self.responses[endpoint_name])


class _StaffResolverDataAccess:
    def __init__(self):
        self.calls = []

    def read(self, *, endpoint_name, args):
        self.calls.append((endpoint_name, dict(args)))
        if args == {"list_staff.query.ordering": "Alice"}:
            raise AssertionError("ordering is not a resolver lookup template")
        if args == {"list_staff.query.name": "Alice"}:
            return _endpoint_result(
                {
                    "data": [
                        {
                            "staff_id": "staff_1",
                            "full_name": "Alice Smith",
                            "first_name": "Alice",
                        }
                    ]
                }
            )
        return _endpoint_result({"data": []})


def _json_payload_from_prompt_section(prompt: str, heading: str) -> dict:
    start = prompt.index(heading) + len(heading)
    rest = prompt[start:].lstrip()
    decoder = json.JSONDecoder()
    payload, _ = decoder.raw_decode(rest)
    return payload


def _grounding_review_arguments(
    prompt: str,
    *,
    selected_by_input: dict[str, str],
) -> dict:
    time_resolutions = {}
    for task in _json_payload_from_prompt_section(
        prompt,
        "Time inputs to resolve:",
    )["known_time_resolution_tasks"]:
        time_resolutions[task["known_input_id"]] = {
            "date_intent": _full_period_time_intent(task["time_expression"])
        }
    reviews = {}
    for task in _json_payload_from_prompt_section(
        prompt,
        "Binding options:",
    )["known_input_binding_options"]:
        known_input_id = task["known_input_id"]
        selected = selected_by_input[known_input_id]
        reviews[known_input_id] = {
            "option_reviews": {
                option["binding_option_id"]: {
                    "resolver_fit_question": option["resolver_fit_question"],
                    "because": "Selected by test model.",
                    "decision": (
                        "CAN_RESOLVE_LOOKUP_TEXT"
                        if option["binding_option_id"] == selected
                        else "CANNOT_RESOLVE_LOOKUP_TEXT"
                    ),
                }
                for option in task["binding_options"]
            }
        }
    return {
        "known_time_resolutions": time_resolutions,
        "known_input_binding_reviews": reviews,
    }


def _full_period_time_intent(text: str) -> dict[str, object]:
    return _period_relative_time_intent(text, unit="month", mode="full")


def _period_relative_time_intent(
    text: str,
    *,
    unit: str,
    mode: str,
) -> dict[str, object]:
    return {
        "expression": text,
        "intent": {
            "time_shape": "period_relative",
            "unit": unit,
            "mode": mode,
            "year": 0,
            "month": 0,
            "day": 0,
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


def _point_date_time_intent(
    text: str,
    *,
    year: int,
    month: int,
    day: int,
) -> dict[str, object]:
    return {
        "expression": text,
        "intent": {
            "time_shape": "point_date",
            "unit": "day",
            "mode": "none",
            "year": year,
            "month": month,
            "day": day,
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


def _named_quarter_time_intent(text: str, *, quarter: int) -> dict[str, object]:
    return {
        "expression": text,
        "intent": {
            "time_shape": "period_named",
            "unit": "quarter",
            "mode": "full",
            "year": 0,
            "month": 0,
            "day": 0,
            "year_policy": "most_recent",
            "relative_offset": 0,
            "named_value": quarter,
            "end_year": 0,
            "end_month": 0,
            "end_day": 0,
            "end_year_policy": "none",
            "count": 0,
            "direction": "none",
        },
    }


def test_grounding_prompt_instructs_binding_id_copying_verbatim():
    request = GroundingRequest(
        question="What were sales at ABC Mall?",
        tasks=(
            KnownInputBindingTask(
                known_input_id="input_location",
                known_input_text="ABC Mall",
                known_input_kind="literal_text",
                requested_fact_id="fact_1",
                lookup_text="ABC Mall",
                options=(
                    InputBindingOption(
                        id="bind_input_location_1",
                        known_input_id="input_location",
                        path="Location name -> sales location",
                    ),
                ),
            ),
        ),
    )
    prompt = _grounding_prompt(request)

    assert prompt.index("Known inputs to ground:") < prompt.index("Binding options:")
    schema = GroundingTurnPrompt(request).response_contract().provider_schema
    reviews_schema = schema["properties"]["known_input_binding_reviews"]
    assert reviews_schema["type"] == "object"
    assert reviews_schema["required"] == ["input_location"]
    option_reviews_schema = reviews_schema["properties"]["input_location"][
        "properties"
    ]["option_reviews"]
    assert option_reviews_schema["type"] == "object"
    assert option_reviews_schema["additionalProperties"] is False
    assert option_reviews_schema["required"] == ["bind_input_location_1"]
    item_schema = option_reviews_schema["properties"]["bind_input_location_1"]
    assert item_schema["properties"]["resolver_fit_question"]["enum"] == [
        "Can this resolver search lookup text 'ABC Mall' and return canonical "
        "API identity 'no_returned_identity' for target meaning ''?"
    ]
    assert item_schema["properties"]["decision"]["enum"] == [
        "CAN_RESOLVE_LOOKUP_TEXT",
        "CANNOT_RESOLVE_LOOKUP_TEXT",
    ]


def test_grounding_time_schema_rejects_relative_word_as_yearless_point_date():
    request = GroundingRequest(
        question="How many shifts do we have today?",
        tasks=(),
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
                    answer_outputs=(RequestedFactAnswerOutput(id="sales_total"),),
                    known_inputs=(known,),
                ),
            )
        ),
        full_catalog=RelationCatalog(),
        resolver_catalog=RelationCatalog(),
        data_access_port=_DataAccess({}),
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
                identity_type="staff",
                identity_field="staff_id",
                value="51515151-0000-0000-0002-000000000001",
                authority_refs=("prior_source_read:staff:list:row_1",),
                lineage_refs=("memory:turn_1.entity.staff.alice",),
            ),
        ),
    )

    assert not output.ledger.issues
    value = output.ledger.values[0]
    assert value.payload.identity_type == "staff"
    assert value.payload.identity_field == "staff_id"
    assert value.payload.value == "51515151-0000-0000-0002-000000000001"
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


def test_grounding_imports_canonical_handoff_without_active_memory_check():
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
                    answer_outputs=(RequestedFactAnswerOutput(id="sales_total"),),
                    known_inputs=(known,),
                ),
            )
        ),
        full_catalog=RelationCatalog(),
        resolver_catalog=RelationCatalog(),
        data_access_port=_DataAccess({}),
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
                identity_type="staff",
                identity_field="staff_id",
                value="51515151-0000-0000-0002-000000000001",
                authority_refs=("prior_source_read:staff:list:row_1",),
                lineage_refs=("memory:turn_1.entity.staff.alice",),
            ),
        ),
    )

    assert not output.ledger.issues
    assert output.ledger.values[0].payload.value == (
        "51515151-0000-0000-0002-000000000001"
    )
    assert output.ledger.certifications[0].lineage_refs == (
        "known_input:input_staff",
        "resolved_question_input:cr_input_1",
        "memory:turn_1.entity.staff.alice",
    )


def test_grounding_certifies_separate_current_input_when_prior_lineage_mentions_same_id():
    imported = RequestedFactLiteralInput(
        id="input_staff",
        source=KnownInputSource.CONVERSATION_RESOLUTION,
        text="her",
        resolved_value_text="Alice Smith",
        value_meaning_hint="staff member",
        role=LiteralInputRole.REFERENCE_VALUE,
        resolved_input_ref="cr_input_1",
    )
    current = _reference_input(
        "old_id",
        "Jane Doe",
        value_meaning_hint="staff member",
    )
    data_access = _DataAccess(
        _endpoint_result(
            {
                "data": [
                    {
                        "staff_id": "staff_jane",
                        "full_name": "Jane Doe",
                        "first_name": "Jane",
                        "last_name": "Doe",
                    }
                ]
            }
        )
    )

    output = ground_question_inputs(
        question="Compare her sales with Jane Doe.",
        question_contract=QuestionContract(
            requested_facts=(
                RequestedFact(
                    id="fact_1",
                    description="sales comparison",
                    answer_outputs=(RequestedFactAnswerOutput(id="sales_total"),),
                    known_inputs=(imported, current),
                    input_refs=("input_staff", "old_id"),
                ),
            )
        ),
        full_catalog=_staff_catalog(),
        resolver_catalog=RelationCatalog(reads=(_staff_read(),)),
        data_access_port=data_access,
        runtime_values=RuntimeValueContext(
            runtime_date="2026-05-09",
            timezone="Africa/London",
        ),
        model_port=_GroundingModel(
            known_input_id="old_id",
            binding_option_id="bind_old_id_1",
        ),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
        conversation_resolution=_compiled_resolution_input(
            question="Compare her sales with Jane Doe.",
            input_ref="cr_input_1",
            source_text="her",
            resolved_text="Alice Smith",
            role=LiteralInputRole.REFERENCE_VALUE,
            value_meaning_hint="staff member",
            canonical_identity=ResolvedCanonicalIdentity(
                identity_type="staff",
                identity_field="staff_id",
                value="staff_alice",
                authority_refs=("prior_source_read:staff:list:row_1",),
                lineage_refs=("known_input:old_id",),
            ),
        ),
    )

    assert not output.ledger.issues
    values_by_id = {value.id: value for value in output.ledger.values}
    assert values_by_id["grounded_input_staff"].payload.value == "staff_alice"
    jane_values = [
        value
        for value in output.ledger.values
        if value.proof_refs == ("known_input:old_id",)
    ]
    assert len(jane_values) == 1
    assert jane_values[0].payload.value == "staff_jane"
    assert data_access.calls == [
        ("list_staff_list", {"list_staff_list.query.name": "Jane Doe"}),
    ]


def test_reference_grounding_executes_compatible_routes_and_dedupes_identity():
    data_access = _EndpointDataAccess(
        {
            "list_location_list": {
                "data": [{"location_id": "loc_bbs", "name": "ABC Mall"}]
            },
            "list_location_alias_list": {
                "data": [{"location_id": "loc_bbs", "display_name": "ABC Mall"}]
            },
        }
    )

    output = ground_question_inputs(
        question="What were sales at ABC Mall?",
        question_contract=_question_contract("ABC Mall", description="store"),
        full_catalog=RelationCatalog(reads=(_location_read(), _location_alias_read())),
        resolver_catalog=RelationCatalog(
            reads=(_location_read(), _location_alias_read())
        ),
        data_access_port=data_access,
        runtime_values=RuntimeValueContext(
            runtime_date="2026-05-09",
            timezone="Africa/London",
        ),
        model_port=_CompatibilityGroundingModel(
            compatible_read_ids={"list_location_list", "list_location_alias_list"}
        ),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )

    assert not output.ledger.issues
    assert len(output.ledger.values) == 1
    value = output.ledger.values[0]
    assert isinstance(value.payload, IdentityValuePayload)
    assert value.payload.identity_type == "location"
    assert value.payload.identity_field == "location_id"
    assert value.payload.value == "loc_bbs"
    assert data_access.calls == [
        ("list_location_list", {"list_location_list.query.name": "ABC Mall"}),
        ("list_location_list", {}),
        (
            "list_location_alias_list",
            {"list_location_alias_list.query.display_name": "ABC Mall"},
        ),
        ("list_location_alias_list", {}),
    ]


def test_reference_grounding_ambiguous_when_compatible_routes_find_multiple_identities():
    data_access = _EndpointDataAccess(
        {
            "list_location_list": {
                "data": [{"location_id": "loc_bbs", "name": "ABC Mall"}]
            },
            "list_location_alias_list": {
                "data": [
                    {"location_id": "loc_bbs", "display_name": "ABC Mall"},
                    {"location_id": "loc_other", "display_name": "ABC Mall"},
                ]
            },
        }
    )

    output = ground_question_inputs(
        question="What were sales at ABC Mall?",
        question_contract=_question_contract("ABC Mall", description="store"),
        full_catalog=RelationCatalog(reads=(_location_read(), _location_alias_read())),
        resolver_catalog=RelationCatalog(
            reads=(_location_read(), _location_alias_read())
        ),
        data_access_port=data_access,
        runtime_values=RuntimeValueContext(
            runtime_date="2026-05-09",
            timezone="Africa/London",
        ),
        model_port=_CompatibilityGroundingModel(
            compatible_read_ids={"list_location_list", "list_location_alias_list"}
        ),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )

    assert not output.ledger.values
    assert len(output.ledger.issues) == 1
    issue = output.ledger.issues[0]
    assert issue.kind == GroundingTerminalKind.AMBIGUOUS_REFERENCE
    assert issue.known_input_id == "input_location"
    assert issue.candidates == (
        "location:location_id:loc_bbs",
        "location:location_id:loc_other",
    )
    assert data_access.calls == [
        ("list_location_list", {"list_location_list.query.name": "ABC Mall"}),
        ("list_location_list", {}),
        (
            "list_location_alias_list",
            {"list_location_alias_list.query.display_name": "ABC Mall"},
        ),
        ("list_location_alias_list", {}),
    ]


def test_reference_grounding_city_target_carries_identity_candidates_without_clarifying():
    model = _CompatibilityGroundingModel(
        compatible_read_ids={"list_area_list", "list_location_list"}
    )
    locations = [
        {
            "location_id": f"loc_{index}",
            "name": f"London Store {index}",
            "area": {"area_id": "area_nairobi", "name": "London"},
        }
        for index in range(1, 25)
    ]
    data_access = _EndpointDataAccess(
        {
            "list_location_list": {"data": locations},
            "list_area_list": {"data": [{"area_id": "area_nairobi", "name": "London"}]},
        }
    )

    output = ground_question_inputs(
        question="How many stores are in London?",
        question_contract=_city_question_contract("London"),
        full_catalog=RelationCatalog(reads=(_location_with_area_read(), _area_read())),
        resolver_catalog=RelationCatalog(
            reads=(_location_with_area_read(), _area_read())
        ),
        data_access_port=data_access,
        runtime_values=RuntimeValueContext(
            runtime_date="2026-05-09",
            timezone="Africa/London",
        ),
        model_port=model,
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )

    assert not output.ledger.issues
    identities = {
        (
            value.payload.identity_type,
            value.payload.identity_field,
            value.payload.value,
        )
        for value in output.ledger.values
        if isinstance(value.payload, IdentityValuePayload)
    }
    assert ("area", "area_id", "area_nairobi") in identities
    options = _all_binding_options(model.prompt)
    assert "list_area_list" in model.prompt
    area_option = next(
        option for option in options if option.get("read_id") == "list_area_list"
    )
    assert {
        "param_ref": "list_area_list.query.name",
        "name": "name",
        "type": "string",
    } in area_option["query_params"]
    assert {
        "field_ref": "field.data.area_id",
        "field_path": "data.area_id",
        "type": "string",
        "identity": {
            "entity_ref": "area",
            "identity_field": "area_id",
            "primary_key": True,
        },
    } in area_option["selected_output_fields"]
    location_option = next(
        option for option in options if option.get("read_id") == "list_location_list"
    )
    assert {
        "param_ref": "list_location_list.query.type",
        "name": "type",
        "type": "choice",
        "choices": ["STORE", "WAREHOUSE"],
    } in location_option["query_params"]
    assert {
        "field_ref": "field.data.type",
        "field_path": "data.type",
        "type": "choice",
        "choices": ["STORE", "WAREHOUSE"],
    } in location_option["selected_output_fields"]
    assert {
        "field_ref": "field.data.area.area_id",
        "field_path": "data.area.area_id",
        "type": "string",
        "identity": {
            "entity_ref": "area",
            "identity_field": "area_id",
            "primary_key": True,
        },
    } in location_option["selected_output_fields"]
    assert {
        "field_ref": "field.data.area.name",
        "field_path": "data.area.name",
        "type": "string",
    } in location_option["selected_output_fields"]
    assert (
        "list_area_list",
        {"list_area_list.query.name": "London"},
    ) in data_access.calls


def _all_binding_options(prompt: str) -> list[dict]:
    payload = _json_payload_from_prompt_section(prompt, "Binding options:")
    return [
        option
        for task in payload["known_input_binding_options"]
        for option in task["binding_options"]
    ]


def _grounding_prompt(request: GroundingRequest) -> str:
    return (
        GroundingTurnPrompt(request)
        .to_model_payload(
            build_turn_prompt_context(
                current_question=request.question,
                conversation_context=request.conversation_context,
            )
        )
        .prompt_text
    )


def test_reference_grounding_extracts_canonical_identity_from_exact_lookup_match():
    output = ground_question_inputs(
        question="What were sales at ABC Mall?",
        question_contract=_question_contract("ABC Mall"),
        full_catalog=_catalog(),
        resolver_catalog=RelationCatalog(reads=(_location_read(),)),
        data_access_port=_DataAccess(
            _endpoint_result({"data": [{"location_id": "loc_1", "name": "ABC Mall"}]})
        ),
        runtime_values=RuntimeValueContext(
            runtime_date="2026-05-09",
            timezone="Africa/London",
        ),
        model_port=_GroundingModel(
            known_input_id="input_location",
            binding_option_id="bind_input_location_1",
        ),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )

    assert output.turn is not None
    assert not output.ledger.issues
    assert len(output.ledger.values) == 1
    value = output.ledger.values[0]
    assert isinstance(value.payload, IdentityValuePayload)
    assert value.payload.identity_type == "location"
    assert value.payload.identity_field == "location_id"
    assert value.payload.value == "loc_1"
    assert value.payload.matched_field_ref == "field.data.name"
    assert value.payload.matched_field_path == "data.name"
    assert len(output.ledger.uses) == 1
    assert output.ledger.uses[0].field_id == "location_id"


def test_reference_grounding_preserves_question_input_fact_applicability():
    model = _ReadRouteGroundingModel(read_id="list_staff_list")
    output = ground_question_inputs(
        question=(
            "What was Alice Smith's total sales amount yesterday, and what "
            "store was associated with Alice Smith's sales yesterday?"
        ),
        question_contract=_shared_staff_question_contract("Alice Smith"),
        full_catalog=_staff_catalog(),
        resolver_catalog=RelationCatalog(reads=(_staff_read(),)),
        data_access_port=_DataAccess(
            _endpoint_result(
                {
                    "data": [
                        {
                            "staff_id": "staff_1",
                            "full_name": "Alice Smith",
                            "first_name": "Alice",
                            "last_name": "Smith",
                        }
                    ]
                }
            )
        ),
        runtime_values=RuntimeValueContext(
            runtime_date="2026-05-19",
            timezone="Africa/London",
        ),
        model_port=model,
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )

    assert not output.ledger.issues
    value = output.ledger.values[0]
    assert isinstance(value.payload, IdentityValuePayload)
    assert value.payload.value == "staff_1"
    assert value.applies_to_requested_fact_ids == ("fact_1", "fact_2")
    task = _json_payload_from_prompt_section(model.prompt, "Known inputs to ground:")[
        "known_input_binding_tasks"
    ][0]
    assert [
        item["requested_fact_id"]
        for item in task["question_context"]["requested_facts"]
    ] == ["fact_1", "fact_2"]


def test_reference_grounding_does_not_accept_contains_match_as_exact_match():
    output = ground_question_inputs(
        question="What were sales at ABC?",
        question_contract=_question_contract("ABC"),
        full_catalog=_catalog(),
        resolver_catalog=RelationCatalog(reads=(_location_read(),)),
        data_access_port=_DataAccess(
            _endpoint_result({"data": [{"location_id": "loc_1", "name": "ABC Mall"}]})
        ),
        runtime_values=RuntimeValueContext(
            runtime_date="2026-05-09",
            timezone="Africa/London",
        ),
        model_port=_GroundingModel(
            known_input_id="input_location",
            binding_option_id="bind_input_location_1",
        ),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )

    assert not output.ledger.values
    assert output.ledger.issues[0].kind == GroundingTerminalKind.UNRESOLVED_REFERENCE


def test_reference_grounding_uses_case_and_spacing_normalized_exact_match():
    output = ground_question_inputs(
        question="What were sales at abc mall?",
        question_contract=_question_contract("  abc   mall  "),
        full_catalog=_catalog(),
        resolver_catalog=RelationCatalog(reads=(_location_read(),)),
        data_access_port=_DataAccess(
            _endpoint_result({"data": [{"location_id": "loc_1", "name": "ABC Mall"}]})
        ),
        runtime_values=RuntimeValueContext(
            runtime_date="2026-05-09",
            timezone="Africa/London",
        ),
        model_port=_GroundingModel(
            known_input_id="input_location",
            binding_option_id="bind_input_location_1",
        ),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )

    assert not output.ledger.issues
    assert output.ledger.values[0].payload.value == "loc_1"


def test_reference_grounding_resolver_route_owns_allowed_lookup_fields():
    output = ground_question_inputs(
        question="How much did Azraah make in sales?",
        question_contract=_staff_question_contract("Azraah"),
        full_catalog=_staff_catalog(),
        resolver_catalog=RelationCatalog(reads=(_staff_read(),)),
        data_access_port=_DataAccess(
            _endpoint_result(
                {
                    "data": [
                        {
                            "staff_id": "staff_1",
                            "full_name": "Azraah Fatuma",
                            "first_name": "Azraah",
                            "last_name": "Fatuma",
                        }
                    ]
                }
            )
        ),
        runtime_values=RuntimeValueContext(
            runtime_date="2026-05-09",
            timezone="Africa/London",
        ),
        model_port=_GroundingModel(
            known_input_id="input_staff",
            binding_option_id="bind_input_staff_1",
        ),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )

    assert not output.ledger.issues
    assert output.ledger.values[0].payload.value == "staff_1"
    assert len(output.ledger.uses) == 1
    assert output.ledger.uses[0].field_id == "staff_id"


def test_reference_grounding_uses_resolver_catalog_selected_for_declared_entity_target():
    data_access = _EndpointDataAccess(
        {
            "list_staff_list": {"data": []},
            "list_store_list": {"data": [{"store_id": "store_1", "name": "Nadia"}]},
        }
    )

    output = ground_question_inputs(
        question="How much did Nadia make in sales?",
        question_contract=_staff_question_contract("Nadia", description="staff member"),
        full_catalog=RelationCatalog(reads=(_staff_read(), _store_read())),
        resolver_catalog=RelationCatalog(reads=(_staff_read(),)),
        data_access_port=data_access,
        runtime_values=RuntimeValueContext(
            runtime_date="2026-05-09",
            timezone="Africa/London",
        ),
        model_port=_ReadRouteGroundingModel(read_id="list_staff_list"),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )

    assert output.turn is not None
    assert not output.ledger.values
    assert output.ledger.issues[0].kind == GroundingTerminalKind.UNRESOLVED_REFERENCE
    assert data_access.calls == [
        ("list_staff_list", {"list_staff_list.query.name": "Nadia"}),
    ]


@pytest.mark.parametrize(
    ("field_label_text", "returned_staff_id", "expect_grounded"),
    (
        ("staff_id", "51515151-0000-0000-0002-000000000001", True),
        ("staff_id", "staff_other", False),
    ),
)
def test_reference_grounding_validates_field_labeled_identity_value(
    field_label_text: str,
    returned_staff_id: str,
    expect_grounded: bool,
):
    staff_id = "51515151-0000-0000-0002-000000000001"
    data_access = _DataAccess(
        _endpoint_result(
            {
                "data": [
                    {
                        "staff_id": returned_staff_id,
                        "full_name": "Alice Smith",
                        "first_name": "Alice",
                        "last_name": "Smith",
                    }
                ]
            }
        )
    )

    output = ground_question_inputs(
        question=f"How much did staff_id {staff_id} make today?",
        question_contract=_staff_question_contract(
            f"staff_id {staff_id}",
            description="staff member",
            resolved_value_text=staff_id,
            field_label_text=field_label_text,
        ),
        full_catalog=_staff_catalog(),
        resolver_catalog=RelationCatalog(reads=(_staff_read(),)),
        data_access_port=data_access,
        runtime_values=RuntimeValueContext(
            runtime_date="2026-05-09",
            timezone="Africa/London",
        ),
        model_port=_GroundingModel(
            known_input_id="input_staff",
            binding_option_id="bind_input_staff_1",
        ),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )

    if expect_grounded:
        assert not output.ledger.issues
        value = output.ledger.values[0]
        assert value.payload.identity_type == "staff"
        assert value.payload.identity_field == "staff_id"
        assert value.payload.value == staff_id
    else:
        assert not output.ledger.values
        assert (
            output.ledger.issues[0].kind == GroundingTerminalKind.UNRESOLVED_REFERENCE
        )
    assert data_access.calls == [
        ("list_staff_list", {"list_staff_list.query.name": staff_id})
    ]


def test_reference_grounding_exact_qualifier_does_not_override_ambiguous_evidence():
    staff_id = "51515151-0000-0000-0002-000000000001"
    data_access = _DataAccess(
        _endpoint_result(
            {
                "data": [
                    {
                        "staff_id": staff_id,
                        "full_name": "Alice Smith",
                        "first_name": "Alice",
                        "last_name": "Smith",
                    },
                    {
                        "staff_id": "staff_other",
                        "full_name": staff_id,
                        "first_name": "Other",
                        "last_name": "Person",
                    },
                ]
            }
        )
    )

    output = ground_question_inputs(
        question=f"How much did staff_id {staff_id} make today?",
        question_contract=_staff_question_contract(
            f"staff_id {staff_id}",
            description="staff member",
            resolved_value_text=staff_id,
            field_label_text="staff_id",
        ),
        full_catalog=_staff_catalog(),
        resolver_catalog=RelationCatalog(reads=(_staff_read(),)),
        data_access_port=data_access,
        runtime_values=RuntimeValueContext(
            runtime_date="2026-05-09",
            timezone="Africa/London",
        ),
        model_port=_CompatibilityGroundingModel(
            compatible_read_ids={"list_staff_list"}
        ),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )

    assert not output.ledger.values
    assert output.ledger.issues[0].kind == GroundingTerminalKind.AMBIGUOUS_REFERENCE
    assert output.ledger.issues[0].candidates == (
        "staff:staff_id:51515151-0000-0000-0002-000000000001",
        "staff:staff_id:staff_other",
    )
    assert data_access.calls == [
        ("list_staff_list", {"list_staff_list.query.name": staff_id}),
        ("list_staff_list", {}),
    ]


def test_reference_grounding_verifies_natural_id_qualifier_against_identity_field():
    staff_id = "51515151-0000-0000-0002-000000000001"
    data_access = _EndpointDataAccess(
        {
            "list_staff_list": {
                "data": [
                    {
                        "staff_id": staff_id,
                        "full_name": "Alice Smith",
                        "first_name": "Alice",
                        "last_name": "Smith",
                    },
                ]
            }
        }
    )

    output = ground_question_inputs(
        question=f"How much did the staff member id {staff_id} make today?",
        question_contract=_staff_question_contract(
            staff_id,
            description="staff member",
            resolved_value_text=staff_id,
            field_label_text="staff member id",
        ),
        full_catalog=_staff_catalog(),
        resolver_catalog=RelationCatalog(reads=(_staff_read(),)),
        data_access_port=data_access,
        runtime_values=RuntimeValueContext(
            runtime_date="2026-05-09",
            timezone="Africa/London",
        ),
        model_port=_CompatibilityGroundingModel(
            compatible_read_ids={"list_staff_list"}
        ),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )

    assert not output.ledger.issues
    value = output.ledger.values[0]
    assert value.payload.identity_type == "staff"
    assert value.payload.identity_field == "staff_id"
    assert value.payload.value == staff_id
    assert data_access.calls == [
        ("list_staff_list", {"list_staff_list.query.name": staff_id}),
        ("list_staff_list", {}),
    ]


def test_reference_grounding_proves_field_labeled_id_through_required_detail_read():
    staff_id = "51515151-0000-0000-0002-000000000001"
    data_access = _EndpointDataAccess(
        {
            "get_staff_detail": {
                "data": {
                    "staff_id": staff_id,
                    "full_name": "Alice Smith",
                }
            }
        }
    )

    output = ground_question_inputs(
        question=f"How much did staff_id {staff_id} make today?",
        question_contract=_staff_question_contract(
            f"staff_id {staff_id}",
            description="staff member",
            resolved_value_text=staff_id,
            field_label_text="staff_id",
        ),
        full_catalog=RelationCatalog(reads=(_staff_detail_read(),)),
        resolver_catalog=RelationCatalog(reads=(_staff_detail_read(),)),
        data_access_port=data_access,
        runtime_values=RuntimeValueContext(
            runtime_date="2026-05-09",
            timezone="Africa/London",
        ),
        model_port=_ReadRouteGroundingModel(read_id="get_staff_detail"),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )

    assert output.turn is not None
    assert not output.ledger.issues
    value = output.ledger.values[0]
    assert value.payload.identity_type == "staff"
    assert value.payload.identity_field == "staff_id"
    assert value.payload.value == staff_id
    assert output.ledger.certifications[0].method == (
        GroundedValueCertificationMethod.RESOLVER_SOURCE_READ
    )
    assert data_access.calls == [
        ("get_staff_detail", {"get_staff_detail.path.staff_id": staff_id})
    ]


def test_reference_grounding_natural_id_qualifier_can_use_identity_param_route():
    staff_id = "51515151-0000-0000-0002-000000000001"
    data_access = _EndpointDataAccess(
        {
            "get_staff_detail": {
                "data": {
                    "staff_id": staff_id,
                    "full_name": "Alice Smith",
                }
            }
        }
    )

    output = ground_question_inputs(
        question=f"How much did staff member id {staff_id} make today?",
        question_contract=_staff_question_contract(
            f"staff member id {staff_id}",
            description="staff member",
            resolved_value_text=staff_id,
            field_label_text="staff member id",
        ),
        full_catalog=RelationCatalog(reads=(_staff_detail_read(),)),
        resolver_catalog=RelationCatalog(reads=(_staff_detail_read(),)),
        data_access_port=data_access,
        runtime_values=RuntimeValueContext(
            runtime_date="2026-05-09",
            timezone="Africa/London",
        ),
        model_port=_ReadRouteGroundingModel(read_id="get_staff_detail"),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )

    assert output.turn is not None
    assert not output.ledger.issues
    assert output.ledger.values[0].payload.value == staff_id
    assert data_access.calls == [
        ("get_staff_detail", {"get_staff_detail.path.staff_id": staff_id})
    ]


def test_reference_grounding_direct_identity_route_uses_catalog_identity_metadata():
    staff_id = "staff-5151"
    staff_read = EndpointRead(
        id="get_staff_detail",
        endpoint_name="get_staff_detail",
        resource_names=("staff",),
        params=(
            CatalogParam(
                ref="get_staff_detail.path.staff_id",
                name="staff_id",
                source=ParamSource.PATH,
                type="pk",
                required=True,
                identity=IdentityMetadata(
                    entity_ref="staff",
                    identity_field="staff_id",
                    primary_key=True,
                    stable=True,
                ),
            ),
        ),
        row_paths=(RowPath(id="data", path="data", cardinality=RowCardinality.ONE),),
        fields=(
            CatalogField(
                ref="field.data.staff_id",
                path="data.staff_id",
                row_path_id="data",
                type="pk",
                identity=IdentityMetadata(
                    entity_ref="staff",
                    identity_field="staff_id",
                    primary_key=True,
                    stable=True,
                    display_fields=("field.data.full_name",),
                ),
            ),
            CatalogField(
                ref="field.data.full_name",
                path="data.full_name",
                row_path_id="data",
                type="string",
            ),
        ),
    )
    data_access = _EndpointDataAccess(
        {
            "get_staff_detail": {
                "data": {
                    "staff_id": staff_id,
                    "full_name": "Alice Smith",
                }
            }
        }
    )

    output = ground_question_inputs(
        question=f"How much did staff_id {staff_id} make today?",
        question_contract=_staff_question_contract(
            f"staff_id {staff_id}",
            description="staff member",
            resolved_value_text=staff_id,
            field_label_text="staff_id",
        ),
        full_catalog=RelationCatalog(reads=(staff_read,)),
        resolver_catalog=RelationCatalog(reads=(staff_read,)),
        data_access_port=data_access,
        runtime_values=RuntimeValueContext(
            runtime_date="2026-05-09",
            timezone="Africa/London",
        ),
        model_port=_ReadRouteGroundingModel(read_id="get_staff_detail"),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )

    assert output.turn is not None
    assert not output.ledger.issues
    assert output.ledger.values[0].payload.value == staff_id
    assert data_access.calls == [
        ("get_staff_detail", {"get_staff_detail.path.staff_id": staff_id})
    ]


def test_reference_grounding_generic_id_does_not_auto_execute_wrong_identity_type():
    staff_id = "51515151-0000-0000-0002-000000000001"
    data_access = _EndpointDataAccess(
        {
            "get_staff_detail": {"data": {}},
            "get_location_detail": {
                "data": {
                    "location_id": staff_id,
                    "name": "Wrong Location",
                }
            },
        }
    )

    output = ground_question_inputs(
        question=f"How much did the sales person with id {staff_id} make today?",
        question_contract=_staff_question_contract(
            f"id {staff_id}",
            description="staff member",
            resolved_value_text=staff_id,
            field_label_text="id",
        ),
        full_catalog=RelationCatalog(
            reads=(_staff_detail_read(), _location_detail_read())
        ),
        resolver_catalog=RelationCatalog(
            reads=(_staff_detail_read(), _location_detail_read())
        ),
        data_access_port=data_access,
        runtime_values=RuntimeValueContext(
            runtime_date="2026-05-09",
            timezone="Africa/London",
        ),
        model_port=_ReadRouteGroundingModel(read_id="get_staff_detail"),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )

    assert output.turn is not None
    assert not output.ledger.values
    assert output.ledger.issues[0].kind == GroundingTerminalKind.UNRESOLVED_REFERENCE
    assert data_access.calls == [
        ("get_staff_detail", {"get_staff_detail.path.staff_id": staff_id})
    ]


def test_reference_grounding_uses_explicit_identity_display_fields_not_name_heuristics():
    book_read = EndpointRead(
        id="list_books",
        endpoint_name="list_books",
        params=(
            CatalogParam(
                ref="list_books.query.q",
                name="q",
                source=ParamSource.QUERY,
                type="string",
            ),
        ),
        row_paths=(RowPath(id="data", path="data", cardinality=RowCardinality.MANY),),
        fields=(
            CatalogField(
                ref="field.data.book_id",
                path="data.book_id",
                row_path_id="data",
                type="string",
                identity=IdentityMetadata(
                    entity_ref="book",
                    identity_field="book_id",
                    primary_key=True,
                    stable=True,
                    display_fields=("field.data.title",),
                ),
            ),
            CatalogField(
                ref="field.data.title",
                path="data.title",
                row_path_id="data",
                type="string",
            ),
        ),
    )

    output = ground_question_inputs(
        question="Show revenue for Wealth of Nations.",
        question_contract=QuestionContract(
            requested_facts=(
                RequestedFact(
                    id="fact_1",
                    description="book revenue",
                    answer_outputs=(
                        RequestedFactAnswerOutput(
                            id="answer_1",
                            description="revenue",
                        ),
                    ),
                    known_inputs=(
                        _reference_input(
                            "input_book",
                            "Wealth of Nations",
                            value_meaning_hint="book",
                        ),
                    ),
                ),
            )
        ),
        full_catalog=RelationCatalog(reads=(book_read,)),
        resolver_catalog=RelationCatalog(reads=(book_read,)),
        data_access_port=_DataAccess(
            _endpoint_result(
                {"data": [{"book_id": "book_1", "title": "Wealth of Nations"}]}
            )
        ),
        runtime_values=RuntimeValueContext(
            runtime_date="2026-05-09",
            timezone="Africa/London",
        ),
        model_port=_GroundingModel(
            known_input_id="input_book",
            binding_option_id="bind_input_book_1",
        ),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )

    assert not output.ledger.issues
    assert output.ledger.values[0].payload.value == "book_1"


def test_reference_grounding_does_not_require_identity_display_fields_for_resolver_route():
    staff_read = EndpointRead(
        id="list_staff",
        endpoint_name="list_staff",
        params=(
            CatalogParam(
                ref="list_staff.query.name",
                name="name",
                source=ParamSource.QUERY,
                type="string",
            ),
        ),
        row_paths=(RowPath(id="data", path="data", cardinality=RowCardinality.MANY),),
        fields=(
            CatalogField(
                ref="field.data.staff_id",
                path="data.staff_id",
                row_path_id="data",
                type="uuid",
                identity=IdentityMetadata(
                    entity_ref="staff",
                    identity_field="staff_id",
                    primary_key=True,
                    stable=True,
                ),
            ),
            CatalogField(
                ref="field.data.full_name",
                path="data.full_name",
                row_path_id="data",
                type="string",
            ),
            CatalogField(
                ref="field.data.first_name",
                path="data.first_name",
                row_path_id="data",
                type="string",
            ),
        ),
    )

    output = ground_question_inputs(
        question="Which products did Alice sell today?",
        question_contract=QuestionContract(
            requested_facts=(
                RequestedFact(
                    id="fact_1",
                    description="products Alice sold today",
                    answer_outputs=(
                        RequestedFactAnswerOutput(
                            id="answer_1",
                            description="products",
                        ),
                    ),
                    known_inputs=(
                        _reference_input(
                            "input_staff",
                            "Alice",
                            value_meaning_hint="staff",
                        ),
                    ),
                ),
            )
        ),
        full_catalog=RelationCatalog(reads=(staff_read,)),
        resolver_catalog=RelationCatalog(reads=(staff_read,)),
        data_access_port=_DataAccess(
            _endpoint_result(
                {
                    "data": [
                        {
                            "staff_id": "staff_1",
                            "full_name": "Alice Smith",
                            "first_name": "Alice",
                        }
                    ]
                }
            )
        ),
        runtime_values=RuntimeValueContext(
            runtime_date="2026-05-09",
            timezone="Africa/London",
        ),
        model_port=_ReadRouteGroundingModel(read_id="list_staff"),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )

    assert not output.ledger.issues
    assert output.ledger.values[0].payload.value == "staff_1"
    assert output.ledger.values[0].payload.identity_type == "staff"


def test_reference_grounding_excludes_control_params_from_lookup_templates():
    staff_read = EndpointRead(
        id="list_staff",
        endpoint_name="list_staff",
        params=(
            CatalogParam(
                ref="list_staff.query.ordering",
                name="ordering",
                source=ParamSource.QUERY,
                type="string",
                semantics="response_shape",
            ),
            CatalogParam(
                ref="list_staff.query.name",
                name="name",
                source=ParamSource.QUERY,
                type="string",
            ),
        ),
        row_paths=(RowPath(id="data", path="data", cardinality=RowCardinality.MANY),),
        fields=(
            CatalogField(
                ref="field.data.staff_id",
                path="data.staff_id",
                row_path_id="data",
                type="uuid",
                identity=IdentityMetadata(
                    entity_ref="staff",
                    identity_field="staff_id",
                    primary_key=True,
                    stable=True,
                ),
            ),
            CatalogField(
                ref="field.data.full_name",
                path="data.full_name",
                row_path_id="data",
                type="string",
            ),
            CatalogField(
                ref="field.data.first_name",
                path="data.first_name",
                row_path_id="data",
                type="string",
            ),
        ),
    )
    data_access = _StaffResolverDataAccess()

    output = ground_question_inputs(
        question="Which products did Alice sell today?",
        question_contract=QuestionContract(
            requested_facts=(
                RequestedFact(
                    id="fact_1",
                    description="products Alice sold today",
                    answer_outputs=(
                        RequestedFactAnswerOutput(
                            id="answer_1",
                            description="products",
                        ),
                    ),
                    known_inputs=(
                        _reference_input(
                            "input_staff",
                            "Alice",
                            value_meaning_hint="staff",
                        ),
                    ),
                ),
            )
        ),
        full_catalog=RelationCatalog(reads=(staff_read,)),
        resolver_catalog=RelationCatalog(reads=(staff_read,)),
        data_access_port=data_access,
        runtime_values=RuntimeValueContext(
            runtime_date="2026-05-09",
            timezone="Africa/London",
        ),
        model_port=_ReadRouteGroundingModel(read_id="list_staff"),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )

    assert not output.ledger.issues
    assert output.ledger.values[0].payload.value == "staff_1"
    assert data_access.calls == [
        ("list_staff", {"list_staff.query.name": "Alice"}),
    ]


def test_reference_grounding_executes_selected_route_after_model_selection():
    data_access = _EndpointDataAccess(
        {
            "list_store_list": {"data": []},
            "list_location_list": {
                "data": [{"location_id": "loc_1", "name": "ABC Mall"}]
            },
        }
    )

    output = ground_question_inputs(
        question="What were sales at ABC Mall?",
        question_contract=_question_contract("ABC Mall"),
        full_catalog=RelationCatalog(reads=(_store_read(), _location_read())),
        resolver_catalog=RelationCatalog(reads=(_store_read(), _location_read())),
        data_access_port=data_access,
        runtime_values=RuntimeValueContext(
            runtime_date="2026-05-09",
            timezone="Africa/London",
        ),
        model_port=_ReadRouteGroundingModel(read_id="list_location_list"),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )

    assert output.turn is not None
    assert not output.ledger.issues
    assert output.ledger.values[0].payload.value == "loc_1"
    assert data_access.calls == [
        ("list_location_list", {"list_location_list.query.name": "ABC Mall"}),
    ]


def test_reference_grounding_exposes_field_only_route_alongside_lookup_param():
    model = _CompatibilityGroundingModel(compatible_read_ids=set())

    output = ground_question_inputs(
        question="What were sales at ABC Mall?",
        question_contract=_question_contract("ABC Mall"),
        full_catalog=RelationCatalog(reads=(_location_read(),)),
        resolver_catalog=RelationCatalog(reads=(_location_read(),)),
        data_access_port=_EndpointDataAccess({"list_location_list": {"data": []}}),
        runtime_values=RuntimeValueContext(
            runtime_date="2026-05-09",
            timezone="Africa/London",
        ),
        model_port=model,
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )

    options = [
        option
        for option in _all_binding_options(model.prompt)
        if option.get("read_id") == "list_location_list"
    ]
    surfaces = [option["lookup_surface"] for option in options]

    assert output.turn is not None
    assert {
        "param_ref": "list_location_list.query.name",
        "field_refs": ["field.data.location_id", "field.data.name"],
    } in surfaces
    assert {"field_refs": ["field.data.location_id", "field.data.name"]} in surfaces


def test_reference_grounding_resolves_store_target_from_location_rows_without_lookup_param():
    data_access = _EndpointDataAccess(
        {
            "list_store_list": {"data": []},
            "list_location_list": {
                "data": [{"location_id": "loc_1", "name": "Westlands Beauty Hub"}]
            },
        }
    )

    output = ground_question_inputs(
        question="What were sales at Westlands Beauty Hub?",
        question_contract=_question_contract(
            "Westlands Beauty Hub",
            description="store",
        ),
        full_catalog=RelationCatalog(
            reads=(_store_read(), _location_read_without_lookup_param())
        ),
        resolver_catalog=RelationCatalog(
            reads=(_store_read(), _location_read_without_lookup_param())
        ),
        data_access_port=data_access,
        runtime_values=RuntimeValueContext(
            runtime_date="2026-05-09",
            timezone="Africa/London",
        ),
        model_port=_ReadRouteGroundingModel(read_id="list_location_list"),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )

    assert not output.ledger.issues
    assert output.ledger.values[0].payload.value == "loc_1"
    assert output.ledger.values[0].payload.identity_type == "location"
    assert data_access.calls == [
        ("list_location_list", {}),
    ]


def test_reference_grounding_skips_resolver_when_declared_row_path_is_unavailable():
    data_access = _EndpointDataAccess(
        {
            "list_store_list": {"data": {"stores": []}},
            "list_location_list": {
                "data": [{"location_id": "loc_1", "name": "ABC Mall"}]
            },
        }
    )

    output = ground_question_inputs(
        question="What were sales at ABC Mall?",
        question_contract=_question_contract("ABC Mall"),
        full_catalog=RelationCatalog(reads=(_broken_store_read(), _location_read())),
        resolver_catalog=RelationCatalog(
            reads=(_broken_store_read(), _location_read())
        ),
        data_access_port=data_access,
        runtime_values=RuntimeValueContext(
            runtime_date="2026-05-09",
            timezone="Africa/London",
        ),
        model_port=_ReadRouteGroundingModel(read_id="list_location_list"),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )

    assert not output.ledger.issues
    assert output.ledger.values[0].payload.value == "loc_1"
    assert data_access.calls == [
        ("list_location_list", {"list_location_list.query.name": "ABC Mall"}),
    ]


def test_reference_grounding_uses_live_resolver_for_active_memory_without_cr_handoff():
    artifact = build_fact_artifact(
        artifact_id="turn_1",
        outcome=FactOutcome.ANSWERED,
        addresses=(
            FactAddress.entity(
                address="entity.location.abc",
                resource="location",
                reference_text="ABC Mall",
                identity={"location_id": "loc_1"},
            ),
        ),
    )
    data_access = _EndpointDataAccess(
        {
            "list_location_list": {
                "data": [{"location_id": "loc_live_catalog", "name": "ABC Mall"}]
            },
        }
    )

    output = ground_question_inputs(
        question="What were sales at ABC Mall?",
        question_contract=_question_contract("ABC Mall"),
        full_catalog=_catalog(),
        resolver_catalog=RelationCatalog(reads=(_location_read(),)),
        data_access_port=data_access,
        runtime_values=RuntimeValueContext(
            runtime_date="2026-05-09",
            timezone="Africa/London",
        ),
        conversation_context={"factArtifacts": [artifact.to_dict()]},
        active_memory_ids=frozenset({"turn_1.entity.location.abc"}),
        model_port=_ReadRouteGroundingModel(read_id="list_location_list"),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )

    assert output.turn is not None
    assert not output.ledger.issues
    assert output.ledger.values[0].payload.value == "loc_live_catalog"
    assert output.ledger.certifications[0].method == (
        GroundedValueCertificationMethod.RESOLVER_SOURCE_READ
    )
    assert data_access.calls == [
        ("list_location_list", {"list_location_list.query.name": "ABC Mall"})
    ]


def test_reference_grounding_does_not_import_selected_active_memory_without_cr_handoff():
    other_artifact = build_fact_artifact(
        artifact_id="turn_0",
        outcome=FactOutcome.ANSWERED,
        addresses=(
            FactAddress.entity(
                address="entity.location.other",
                resource="location",
                reference_text="Other Mall",
                identity={"location_id": "loc_1"},
            ),
        ),
    )
    selected_artifact = build_fact_artifact(
        artifact_id="turn_1",
        outcome=FactOutcome.ANSWERED,
        addresses=(
            FactAddress.entity(
                address="entity.location.abc",
                resource="location",
                reference_text="ABC Mall",
                identity={"location_id": "loc_1"},
            ),
        ),
    )
    data_access = _EndpointDataAccess(
        {
            "list_location_list": {
                "data": [{"location_id": "loc_live_catalog", "name": "ABC Mall"}]
            },
        }
    )

    output = ground_question_inputs(
        question="What were sales at ABC Mall?",
        question_contract=_question_contract("ABC Mall"),
        full_catalog=_catalog(),
        resolver_catalog=RelationCatalog(reads=(_location_read(),)),
        data_access_port=data_access,
        runtime_values=RuntimeValueContext(
            runtime_date="2026-05-09",
            timezone="Africa/London",
        ),
        conversation_context={
            "factArtifacts": [other_artifact.to_dict(), selected_artifact.to_dict()]
        },
        active_memory_ids=frozenset(
            {
                "turn_0.entity.location.other",
                "turn_1.entity.location.abc",
            }
        ),
        model_port=_ReadRouteGroundingModel(read_id="list_location_list"),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )

    assert output.turn is not None
    assert not output.ledger.issues
    assert output.ledger.values[0].payload.value == "loc_live_catalog"
    assert output.ledger.certifications[0].method == (
        GroundedValueCertificationMethod.RESOLVER_SOURCE_READ
    )
    assert output.ledger.certifications[0].authority_refs == ("list_location_list",)
    assert data_access.calls == [
        ("list_location_list", {"list_location_list.query.name": "ABC Mall"})
    ]


def test_reference_grounding_uses_live_resolver_when_memory_identity_is_not_active():
    artifact = build_fact_artifact(
        artifact_id="turn_1",
        outcome=FactOutcome.ANSWERED,
        addresses=(
            FactAddress.entity(
                address="entity.location.abc",
                resource="location",
                reference_text="ABC Mall",
                identity={"location_id": "loc_stale_memory"},
            ),
        ),
    )
    data_access = _EndpointDataAccess(
        {
            "list_location_list": {
                "data": [{"location_id": "loc_live_catalog", "name": "ABC Mall"}]
            },
        }
    )

    output = ground_question_inputs(
        question="What were sales at ABC Mall?",
        question_contract=_question_contract("ABC Mall"),
        full_catalog=_catalog(),
        resolver_catalog=RelationCatalog(reads=(_location_read(),)),
        data_access_port=data_access,
        runtime_values=RuntimeValueContext(
            runtime_date="2026-05-09",
            timezone="Africa/London",
        ),
        conversation_context={"factArtifacts": [artifact.to_dict()]},
        model_port=_ReadRouteGroundingModel(read_id="list_location_list"),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )

    assert not output.ledger.issues
    assert output.ledger.values[0].payload.value == "loc_live_catalog"
    assert data_access.calls == [
        ("list_location_list", {"list_location_list.query.name": "ABC Mall"})
    ]


def test_reference_grounding_active_memory_without_cr_handoff_does_not_certify():
    artifact = build_fact_artifact(
        artifact_id="turn_1",
        outcome=FactOutcome.ANSWERED,
        addresses=(
            FactAddress.entity(
                address="entity.location.abc",
                resource="location",
                reference_text="ABC Mall",
                identity={"location_id": "loc_1"},
            ),
        ),
    )

    output = ground_question_inputs(
        question="What were sales at ABC Mall?",
        question_contract=_question_contract("ABC Mall"),
        full_catalog=_catalog(),
        resolver_catalog=RelationCatalog(reads=(_location_read(),)),
        data_access_port=_EndpointDataAccess(
            {
                "list_location_list": {
                    "data": [{"location_id": "loc_other", "name": "Other Mall"}]
                },
            }
        ),
        runtime_values=RuntimeValueContext(
            runtime_date="2026-05-09",
            timezone="Africa/London",
        ),
        conversation_context={"factArtifacts": [artifact.to_dict()]},
        active_memory_ids=frozenset({"turn_1.entity.location.abc"}),
        model_port=_ReadRouteGroundingModel(read_id="list_location_list"),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )

    assert not output.ledger.values
    assert output.ledger.issues[0].kind == GroundingTerminalKind.UNRESOLVED_REFERENCE
    assert output.ledger.uses == ()


def test_reference_grounding_does_not_reuse_memory_identity_for_unmatched_target_text():
    artifact = build_fact_artifact(
        artifact_id="turn_1",
        outcome=FactOutcome.ANSWERED,
        addresses=(
            FactAddress.entity(
                address="entity.location.abc",
                resource="location",
                reference_text="ABC Mall",
                identity={"location_id": "loc_1"},
            ),
        ),
    )
    data_access = _EndpointDataAccess(
        {
            "list_location_list": {
                "data": [{"location_id": "loc_2", "name": "Nextgen Mall"}]
            },
        }
    )

    output = ground_question_inputs(
        question="What were sales at Nextgen Mall?",
        question_contract=_question_contract("Nextgen Mall"),
        full_catalog=_catalog(),
        resolver_catalog=RelationCatalog(reads=(_location_read(),)),
        data_access_port=data_access,
        runtime_values=RuntimeValueContext(
            runtime_date="2026-05-09",
            timezone="Africa/London",
        ),
        conversation_context={"factArtifacts": [artifact.to_dict()]},
        model_port=_ReadRouteGroundingModel(read_id="list_location_list"),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )

    assert not output.ledger.issues
    assert output.ledger.values[0].payload.value == "loc_2"
    assert data_access.calls == [
        ("list_location_list", {"list_location_list.query.name": "Nextgen Mall"})
    ]


def test_reference_grounding_without_exact_match_is_a_grounding_issue():
    data_access = _DataAccess(
        _endpoint_result({"data": [{"location_id": "loc_1", "name": "ABC Mall"}]})
    )

    output = ground_question_inputs(
        question="Do location records expose a date of birth field?",
        question_contract=_question_contract("date of birth"),
        full_catalog=_catalog(),
        resolver_catalog=RelationCatalog(reads=(_location_read(),)),
        data_access_port=data_access,
        runtime_values=RuntimeValueContext(
            runtime_date="2026-05-09",
            timezone="Africa/London",
        ),
        model_port=_ReadRouteGroundingModel(read_id="list_location_list"),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )

    assert not output.ledger.values
    assert not output.ledger.uses
    assert output.ledger.issues[0].kind == GroundingTerminalKind.UNRESOLVED_REFERENCE
    assert output.ledger.issues[0].known_input_id == "input_location"
    assert data_access.calls == [
        ("list_location_list", {"list_location_list.query.name": "date of birth"})
    ]


def test_reference_grounding_without_resolver_route_is_a_grounding_issue():
    output = ground_question_inputs(
        question="What is Jane Doe's staff ID?",
        question_contract=_staff_question_contract("Jane Doe"),
        full_catalog=_staff_catalog(),
        resolver_catalog=RelationCatalog(),
        data_access_port=_DataAccess({}),
        runtime_values=RuntimeValueContext(
            runtime_date="2026-05-09",
            timezone="Africa/London",
        ),
        model_port=object(),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )

    assert not output.ledger.values
    assert output.ledger.issues[0].kind == GroundingTerminalKind.UNSUPPORTED_REFERENCE
    assert output.ledger.issues[0].known_input_id == "input_staff"


def test_reference_grounding_uses_live_resolver_for_repeated_concrete_name_without_activation():
    artifact = build_fact_artifact(
        artifact_id="turn_1",
        outcome=FactOutcome.ANSWERED,
        addresses=(
            FactAddress.entity(
                address="entity.staff.jane",
                resource="staff",
                reference_text="Jane Doe",
                identity={"staff_id": "40404040-0000-0000-0002-000000000001"},
            ),
        ),
    )
    data_access = _DataAccess(
        _endpoint_result(
            {
                "data": [
                    {
                        "staff_id": "different-staff",
                        "full_name": "Jane Doe",
                    }
                ]
            }
        )
    )

    output = ground_question_inputs(
        question="Where did Jane Doe work on her first two shifts?",
        question_contract=_staff_question_contract("Jane Doe"),
        full_catalog=_staff_catalog(),
        resolver_catalog=RelationCatalog(reads=(_staff_read(),)),
        data_access_port=data_access,
        runtime_values=RuntimeValueContext(
            runtime_date="2026-05-09",
            timezone="Africa/London",
        ),
        model_port=_ReadRouteGroundingModel(read_id="list_staff_list"),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
        conversation_context={"factArtifacts": [artifact.to_dict()]},
        active_memory_ids=frozenset(),
    )

    assert not output.ledger.issues
    assert output.ledger.values[0].payload.value == "different-staff"
    assert output.ledger.values[0].payload.identity_type == "staff"
    assert data_access.calls == [
        ("list_staff_list", {"list_staff_list.query.name": "Jane Doe"})
    ]


def test_time_grounding_records_known_input_proof_ref():
    output = ground_question_inputs(
        question="How much revenue on February 14, 2026?",
        question_contract=_time_question_contract("February 14, 2026"),
        full_catalog=_date_sales_catalog(),
        resolver_catalog=RelationCatalog(),
        data_access_port=_DataAccess({}),
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
                answer_outputs=(RequestedFactAnswerOutput(id="total_sales"),),
                known_inputs=(time_input,),
                input_refs=("input_date",),
            ),
        ),
    )

    output = ground_question_inputs(
        question="What about that same period?",
        question_contract=contract,
        full_catalog=_date_sales_catalog(),
        resolver_catalog=RelationCatalog(),
        data_access_port=_DataAccess({}),
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
        resolver_catalog=RelationCatalog(),
        data_access_port=_DataAccess({}),
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
        resolver_catalog=RelationCatalog(),
        data_access_port=_DataAccess({}),
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
        resolver_catalog=RelationCatalog(),
        data_access_port=_DataAccess({}),
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


def test_reference_grounding_uses_literal_resolved_value_text():
    data_access = _DataAccess(
        _endpoint_result(
            {
                "data": [
                    {
                        "staff_id": "staff_1",
                        "full_name": "Jane Doe",
                        "first_name": "Nadia",
                        "last_name": "Doe",
                    }
                ]
            }
        )
    )

    output = ground_question_inputs(
        question="What is Jane Doe's staff ID?",
        question_contract=_staff_question_contract_with_resolved_value_text(
            reference_text="Jane Doe",
            resolved_value_text="Jane Doe",
        ),
        full_catalog=_staff_catalog(),
        resolver_catalog=RelationCatalog(reads=(_staff_read(),)),
        data_access_port=data_access,
        runtime_values=RuntimeValueContext(
            runtime_date="2026-05-09",
            timezone="Africa/London",
        ),
        model_port=_GroundingModel(
            known_input_id="input_staff",
            binding_option_id="bind_input_staff_1",
        ),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )

    assert not output.ledger.issues
    value = output.ledger.values[0]
    assert value.payload.value == "staff_1"
    assert [
        certification.to_payload() for certification in output.ledger.certifications
    ] == [
        {
            "value_id": value.id,
            "method": GroundedValueCertificationMethod.RESOLVER_SOURCE_READ.value,
            "authority_refs": ["list_staff_list"],
            "lineage_refs": ["known_input:input_staff"],
        }
    ]
    assert data_access.calls == [
        ("list_staff_list", {"list_staff_list.query.name": "Jane Doe"}),
    ]


def _endpoint_result(response_body):
    return {
        "responseStatus": 200,
        "responseBody": response_body,
    }


def _question_contract(text: str, *, description: str = "") -> QuestionContract:
    return QuestionContract(
        requested_facts=(
            RequestedFact(
                id="fact_1",
                description="sales total",
                answer_outputs=(
                    RequestedFactAnswerOutput(
                        id="total_sales",
                        description="total sales",
                    ),
                ),
                known_inputs=(
                    _reference_input(
                        "input_location",
                        text,
                        value_meaning_hint=description,
                    ),
                ),
            ),
        )
    )


def _city_question_contract(text: str) -> QuestionContract:
    return QuestionContract(
        requested_facts=(
            RequestedFact(
                id="fact_1",
                description="count of stores in city",
                answer_population=RequestedFactAnswerPopulation(
                    population_label="stores in city",
                    counted_unit="store",
                    membership_tests=(
                        RequestedFactAnswerPopulationMembershipTest(
                            id="test_1",
                            kind=AnswerPopulationMembershipTestKind.SUBJECT_IDENTITY,
                            polarity=AnswerPopulationMembershipTestPolarity.MUST_PASS,
                            test_question="Is the row a store?",
                        ),
                    ),
                ),
                answer_outputs=(
                    RequestedFactAnswerOutput(
                        id="answer_1",
                        description="number of stores",
                    ),
                ),
                known_inputs=(
                    _reference_input(
                        "input_city",
                        text,
                        value_meaning_hint="city",
                    ),
                ),
            ),
        )
    )


def _staff_question_contract(
    text: str,
    *,
    description: str = "",
    resolved_value_text: str | None = None,
    field_label_text: str = "",
) -> QuestionContract:
    return QuestionContract(
        requested_facts=(
            RequestedFact(
                id="fact_1",
                description="staff sales total",
                answer_outputs=(
                    RequestedFactAnswerOutput(
                        id="total_sales",
                        description="total sales",
                    ),
                ),
                known_inputs=(
                    _reference_input(
                        "input_staff",
                        text,
                        value_meaning_hint=description,
                        resolved_value_text=resolved_value_text,
                        field_label_text=field_label_text,
                    ),
                ),
            ),
        )
    )


def _shared_staff_question_contract(text: str) -> QuestionContract:
    staff = _reference_input("input_staff", text)
    fact_1 = RequestedFact(
        id="fact_1",
        description="staff sales total",
        answer_outputs=(
            RequestedFactAnswerOutput(
                id="total_sales",
                description="total sales",
            ),
        ),
        known_inputs=(staff,),
        input_refs=("input_staff",),
    )
    fact_2 = RequestedFact(
        id="fact_2",
        description="store associated with staff sales",
        answer_outputs=(
            RequestedFactAnswerOutput(
                id="store",
                description="store",
            ),
        ),
        known_inputs=(staff,),
        input_refs=("input_staff",),
    )
    return QuestionContract(
        question_inputs=(staff,),
        requested_facts=(fact_1, fact_2),
    )


def _staff_question_contract_with_resolved_value_text(
    *,
    reference_text: str,
    resolved_value_text: str,
) -> QuestionContract:
    return QuestionContract(
        requested_facts=(
            RequestedFact(
                id="fact_1",
                description="staff ID",
                answer_outputs=(
                    RequestedFactAnswerOutput(
                        id="staff_id",
                        description="staff ID",
                    ),
                ),
                known_inputs=(
                    _reference_input(
                        "input_staff",
                        reference_text,
                        resolved_value_text=resolved_value_text,
                    ),
                ),
            ),
        )
    )


def _time_question_contract(text: str) -> QuestionContract:
    return QuestionContract(
        requested_facts=(
            RequestedFact(
                id="fact_1",
                description="sales total",
                answer_outputs=(
                    RequestedFactAnswerOutput(
                        id="total_sales",
                        description="total sales",
                    ),
                ),
                known_inputs=(_time_input("input_date", text),),
            ),
        )
    )


def _quarter_question_contract(text: str) -> QuestionContract:
    return QuestionContract(
        requested_facts=(
            RequestedFact(
                id="fact_1",
                description="sales total",
                answer_outputs=(
                    RequestedFactAnswerOutput(
                        id="total_sales",
                        description="total sales",
                    ),
                ),
                known_inputs=(_time_input("input_date", text),),
            ),
        )
    )


def _catalog() -> RelationCatalog:
    return RelationCatalog(
        reads=(
            _sales_read(),
            _location_read(),
        )
    )


def _staff_catalog() -> RelationCatalog:
    return RelationCatalog(
        reads=(
            _staff_sales_read(),
            _staff_read(),
        )
    )


def _selected_catalog() -> RelationCatalog:
    return RelationCatalog(reads=(_sales_read(),))


def _location_read() -> EndpointRead:
    return EndpointRead(
        id="list_location_list",
        endpoint_name="list_location_list",
        resource_names=("location",),
        params=(
            CatalogParam(
                ref="list_location_list.query.name",
                name="name",
                source=ParamSource.QUERY,
                type="string",
            ),
            CatalogParam(
                ref="list_location_list.query.type",
                name="type",
                source=ParamSource.QUERY,
                type="choice",
                choices=("STORE", "WAREHOUSE"),
            ),
        ),
        row_paths=(
            RowPath(
                id="data",
                path="data",
                cardinality=RowCardinality.MANY,
            ),
        ),
        fields=(
            CatalogField(
                ref="field.data.location_id",
                path="data.location_id",
                row_path_id="data",
                type="string",
                identity=IdentityMetadata(
                    entity_ref="location",
                    identity_field="location_id",
                    primary_key=True,
                    stable=True,
                    display_fields=("field.data.name",),
                ),
            ),
            CatalogField(
                ref="field.data.name",
                path="data.name",
                row_path_id="data",
                type="string",
            ),
        ),
    )


def _location_read_without_lookup_param() -> EndpointRead:
    return EndpointRead(
        id="list_location_list",
        endpoint_name="list_location_list",
        resource_names=("location",),
        params=(
            CatalogParam(
                ref="list_location_list.query.type",
                name="type",
                source=ParamSource.QUERY,
                type="choice",
                choices=("STORE", "WAREHOUSE"),
            ),
        ),
        row_paths=(
            RowPath(
                id="data",
                path="data",
                cardinality=RowCardinality.MANY,
            ),
        ),
        fields=(
            CatalogField(
                ref="field.data.location_id",
                path="data.location_id",
                row_path_id="data",
                type="string",
                identity=IdentityMetadata(
                    entity_ref="location",
                    identity_field="location_id",
                    primary_key=True,
                    stable=True,
                    display_fields=("field.data.name",),
                ),
            ),
            CatalogField(
                ref="field.data.name",
                path="data.name",
                row_path_id="data",
                type="string",
            ),
        ),
    )


def _location_alias_read() -> EndpointRead:
    return EndpointRead(
        id="list_location_alias_list",
        endpoint_name="list_location_alias_list",
        resource_names=("location",),
        params=(
            CatalogParam(
                ref="list_location_alias_list.query.display_name",
                name="display_name",
                source=ParamSource.QUERY,
                type="string",
            ),
        ),
        row_paths=(
            RowPath(
                id="data",
                path="data",
                cardinality=RowCardinality.MANY,
            ),
        ),
        fields=(
            CatalogField(
                ref="field.data.location_id",
                path="data.location_id",
                row_path_id="data",
                type="string",
                identity=IdentityMetadata(
                    entity_ref="location",
                    identity_field="location_id",
                    primary_key=True,
                    stable=True,
                    display_fields=("field.data.display_name",),
                ),
            ),
            CatalogField(
                ref="field.data.display_name",
                path="data.display_name",
                row_path_id="data",
                type="string",
            ),
        ),
    )


def _location_with_area_read() -> EndpointRead:
    return EndpointRead(
        id="list_location_list",
        endpoint_name="list_location_list",
        resource_names=("location",),
        params=(
            CatalogParam(
                ref="list_location_list.query.name",
                name="name",
                source=ParamSource.QUERY,
                type="string",
            ),
            CatalogParam(
                ref="list_location_list.query.type",
                name="type",
                source=ParamSource.QUERY,
                type="choice",
                choices=("STORE", "WAREHOUSE"),
            ),
        ),
        row_paths=(
            RowPath(
                id="data",
                path="data",
                cardinality=RowCardinality.MANY,
            ),
        ),
        fields=(
            CatalogField(
                ref="field.data.location_id",
                path="data.location_id",
                row_path_id="data",
                type="string",
                identity=IdentityMetadata(
                    entity_ref="location",
                    identity_field="location_id",
                    primary_key=True,
                    stable=True,
                ),
            ),
            CatalogField(
                ref="field.data.name",
                path="data.name",
                row_path_id="data",
                type="string",
            ),
            CatalogField(
                ref="field.data.type",
                path="data.type",
                row_path_id="data",
                type="choice",
                choices=("STORE", "WAREHOUSE"),
            ),
            CatalogField(
                ref="field.data.area.area_id",
                path="data.area.area_id",
                row_path_id="data",
                type="string",
                identity=IdentityMetadata(
                    entity_ref="area",
                    identity_field="area_id",
                    primary_key=True,
                    stable=True,
                    display_fields=("field.data.area.name",),
                ),
            ),
            CatalogField(
                ref="field.data.area.name",
                path="data.area.name",
                row_path_id="data",
                type="string",
            ),
        ),
    )


def _area_read() -> EndpointRead:
    return EndpointRead(
        id="list_area_list",
        endpoint_name="list_area_list",
        resource_names=("area",),
        params=(
            CatalogParam(
                ref="list_area_list.query.name",
                name="name",
                source=ParamSource.QUERY,
                type="string",
            ),
        ),
        row_paths=(
            RowPath(
                id="data",
                path="data",
                cardinality=RowCardinality.MANY,
            ),
        ),
        fields=(
            CatalogField(
                ref="field.data.area_id",
                path="data.area_id",
                row_path_id="data",
                type="string",
                identity=IdentityMetadata(
                    entity_ref="area",
                    identity_field="area_id",
                    primary_key=True,
                    stable=True,
                    display_fields=("field.data.name",),
                ),
            ),
            CatalogField(
                ref="field.data.name",
                path="data.name",
                row_path_id="data",
                type="string",
            ),
        ),
    )


def _store_read() -> EndpointRead:
    return EndpointRead(
        id="list_store_list",
        endpoint_name="list_store_list",
        params=(
            CatalogParam(
                ref="list_store_list.query.name",
                name="name",
                source=ParamSource.QUERY,
                type="string",
            ),
        ),
        row_paths=(
            RowPath(
                id="data",
                path="data",
                cardinality=RowCardinality.MANY,
            ),
        ),
        fields=(
            CatalogField(
                ref="field.data.store_id",
                path="data.store_id",
                row_path_id="data",
                type="string",
                identity=IdentityMetadata(
                    entity_ref="store",
                    identity_field="store_id",
                    primary_key=True,
                    stable=True,
                    display_fields=("field.data.name",),
                ),
            ),
            CatalogField(
                ref="field.data.name",
                path="data.name",
                row_path_id="data",
                type="string",
            ),
        ),
    )


def _broken_store_read() -> EndpointRead:
    return EndpointRead(
        id="list_store_list",
        endpoint_name="list_store_list",
        params=(
            CatalogParam(
                ref="list_store_list.query.name",
                name="name",
                source=ParamSource.QUERY,
                type="string",
            ),
        ),
        row_paths=(
            RowPath(
                id="data_deposits",
                path="data.deposits",
                cardinality=RowCardinality.MANY,
            ),
        ),
        fields=(
            CatalogField(
                ref="field.data_deposits.store_id",
                path="data.deposits.store_id",
                row_path_id="data_deposits",
                type="string",
                identity=IdentityMetadata(
                    entity_ref="store",
                    identity_field="store_id",
                    primary_key=True,
                    stable=True,
                    display_fields=("field.data_deposits.name",),
                ),
            ),
            CatalogField(
                ref="field.data_deposits.name",
                path="data.deposits.name",
                row_path_id="data_deposits",
                type="string",
            ),
        ),
    )


def _staff_read() -> EndpointRead:
    return EndpointRead(
        id="list_staff_list",
        endpoint_name="list_staff_list",
        params=(
            CatalogParam(
                ref="list_staff_list.query.name",
                name="name",
                source=ParamSource.QUERY,
                type="string",
            ),
        ),
        row_paths=(
            RowPath(
                id="data",
                path="data",
                cardinality=RowCardinality.MANY,
            ),
        ),
        fields=(
            CatalogField(
                ref="field.data.staff_id",
                path="data.staff_id",
                row_path_id="data",
                type="string",
                identity=IdentityMetadata(
                    entity_ref="staff",
                    identity_field="staff_id",
                    primary_key=True,
                    stable=True,
                    display_fields=(
                        "field.data.full_name",
                        "field.data.first_name",
                        "field.data.last_name",
                    ),
                ),
            ),
            CatalogField(
                ref="field.data.full_name",
                path="data.full_name",
                row_path_id="data",
                type="string",
            ),
            CatalogField(
                ref="field.data.first_name",
                path="data.first_name",
                row_path_id="data",
                type="string",
            ),
            CatalogField(
                ref="field.data.last_name",
                path="data.last_name",
                row_path_id="data",
                type="string",
            ),
        ),
    )


def _staff_detail_read() -> EndpointRead:
    return EndpointRead(
        id="get_staff_detail",
        endpoint_name="get_staff_detail",
        resource_names=("staff",),
        params=(
            CatalogParam(
                ref="get_staff_detail.path.staff_id",
                name="staff_id",
                source=ParamSource.PATH,
                type="string",
                required=True,
                identity=IdentityMetadata(
                    entity_ref="staff",
                    identity_field="staff_id",
                    primary_key=True,
                    stable=True,
                ),
            ),
        ),
        row_paths=(
            RowPath(
                id="data",
                path="data",
                cardinality=RowCardinality.ONE,
            ),
        ),
        fields=(
            CatalogField(
                ref="field.data.staff_id",
                path="data.staff_id",
                row_path_id="data",
                type="string",
                identity=IdentityMetadata(
                    entity_ref="staff",
                    identity_field="staff_id",
                    primary_key=True,
                    stable=True,
                    display_fields=("field.data.full_name",),
                ),
            ),
            CatalogField(
                ref="field.data.full_name",
                path="data.full_name",
                row_path_id="data",
                type="string",
            ),
        ),
    )


def _location_detail_read() -> EndpointRead:
    return EndpointRead(
        id="get_location_detail",
        endpoint_name="get_location_detail",
        resource_names=("location",),
        params=(
            CatalogParam(
                ref="get_location_detail.path.location_id",
                name="location_id",
                source=ParamSource.PATH,
                type="string",
                required=True,
                identity=IdentityMetadata(
                    entity_ref="location",
                    identity_field="location_id",
                    primary_key=True,
                    stable=True,
                ),
            ),
        ),
        row_paths=(
            RowPath(
                id="data",
                path="data",
                cardinality=RowCardinality.ONE,
            ),
        ),
        fields=(
            CatalogField(
                ref="field.data.location_id",
                path="data.location_id",
                row_path_id="data",
                type="string",
                identity=IdentityMetadata(
                    entity_ref="location",
                    identity_field="location_id",
                    primary_key=True,
                    stable=True,
                    display_fields=("field.data.name",),
                ),
            ),
            CatalogField(
                ref="field.data.name",
                path="data.name",
                row_path_id="data",
                type="string",
            ),
        ),
    )


def _staff_sales_read() -> EndpointRead:
    return EndpointRead(
        id="list_sale_list",
        endpoint_name="list_sale_list",
        params=(
            CatalogParam(
                ref="list_sale_list.query.staff_id",
                name="staff_id",
                source=ParamSource.QUERY,
                type="string",
                identity=IdentityMetadata(
                    entity_ref="staff",
                    identity_field="staff_id",
                    primary_key=True,
                    stable=True,
                ),
            ),
        ),
        row_paths=(
            RowPath(
                id="data",
                path="data",
                cardinality=RowCardinality.MANY,
            ),
        ),
        fields=(
            CatalogField(
                ref="field.data.amount",
                path="data.amount",
                row_path_id="data",
                type="decimal",
            ),
        ),
    )


def _date_sales_catalog() -> RelationCatalog:
    return RelationCatalog(
        reads=(
            EndpointRead(
                id="list_sales_list",
                endpoint_name="list_sales_list",
                params=(
                    CatalogParam(
                        ref="list_sales_list.query.start_date",
                        name="start_date",
                        source=ParamSource.QUERY,
                        type="date",
                    ),
                    CatalogParam(
                        ref="list_sales_list.query.end_date",
                        name="end_date",
                        source=ParamSource.QUERY,
                        type="date",
                    ),
                ),
                fields=(
                    CatalogField(
                        ref="field.total_sales",
                        type="decimal",
                    ),
                ),
            ),
        )
    )


def _sales_read() -> EndpointRead:
    return EndpointRead(
        id="list_sales_list",
        endpoint_name="list_sales_list",
        params=(
            CatalogParam(
                ref="list_sales_list.query.location_id",
                name="location_id",
                source=ParamSource.QUERY,
                type="string",
                identity=IdentityMetadata(
                    entity_ref="location",
                    identity_field="location_id",
                    primary_key=True,
                    stable=True,
                ),
            ),
        ),
        row_paths=(
            RowPath(
                id="data",
                path="data",
                cardinality=RowCardinality.MANY,
            ),
        ),
        fields=(
            CatalogField(
                ref="field.data.total_sales",
                path="data.total_sales",
                row_path_id="data",
                type="decimal",
            ),
        ),
    )
