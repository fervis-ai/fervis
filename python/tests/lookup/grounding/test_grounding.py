import json
from dataclasses import replace

import pytest
from jsonschema import ValidationError, validate

from fervis.lookup.relation_catalog import (
    CandidateKey,
    CandidateKeyComponent,
    CatalogField,
    CatalogParam,
    EndpointRead,
    EntityKeyComponentTarget,
    EntityReference,
    EntityReferenceComponent,
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
from fervis.lookup.canonical_data import entity_key_value
from fervis.lookup.grounding.resolution import ground_question_inputs
from fervis.lookup.grounding.model import (
    CompatibleInputBinding,
    GroundingCompatibilityResult,
    GroundingTerminalKind,
    GroundedValueCertificationMethod,
    IdentifierKind,
    InputBindingCompatibility,
    InputBindingOption,
    NO_SHOWN_RESOURCE_TYPE,
    ResourceTypeMatch,
    LookupRequestParameter,
    resolver_fit_question_for_option,
)
from fervis.lookup.grounding.parser import parse_grounding_compatibility
from fervis.lookup.grounding.model import (
    GroundingRequest,
    KnownTimeResolutionTask,
)
from fervis.lookup.grounding.prompt import GroundingTurnPrompt
from fervis.lookup.grounding.surface import resolver_option_surface
from fervis.lookup.fact_planning.request import RuntimeValueContext
from fervis.lookup.fact_plan.row_sources import (
    CALENDAR_END_PARAM_ID,
    CALENDAR_START_PARAM_ID,
    build_row_source_catalog,
)
from fervis.lookup.grounding.resolution.references import (
    execute_compatible_reference_bindings,
    reference_binding_sources_by_known_input,
    reference_input_binding_tasks,
    reference_binding_issue,
)
from fervis.lookup.relation_catalog.selection import (
    CatalogSelectionResult,
    EntityTargetResolverSelection,
    RequestedFactCatalogSelection,
)
from fervis.lookup.read_eligibility.model import (
    CanonicalInputSelection,
    ReadEligibilityRequest,
    ReadEligibilityResult,
)
from fervis.lookup.read_eligibility.resolution import (
    resolve_read_eligibility,
)
from fervis.lookup.read_eligibility.parser import parse_read_eligibility
from fervis.lookup.read_eligibility.prompt import ReadEligibilityTurnPrompt
from fervis.lookup.read_eligibility.surface import (
    read_eligibility_candidate_surface,
)
from fervis.lookup.turn_prompts import build_turn_prompt_context
from fervis.lookup.turn_prompts.projections import ApiReadResponseShapeProjector
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
from tests.lookup.prompt_sections import prompt_section_payload


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


def _compatible_binding(
    catalog: RelationCatalog,
    option: InputBindingOption,
    *,
    lookup_text: str,
    match_paths: tuple[str, ...] | None = None,
) -> CompatibleInputBinding:
    read = catalog.read(option.candidate.resolver_read_id)
    lookup_params = tuple(
        parameter
        for parameter in read.params
        if parameter.name in {"name", "display_name"}
    )
    if match_paths is None:
        match_paths = tuple(
            field.path
            for field in read.fields
            if field.path.split(".")[-1]
            in {"name", "display_name", "first_name", "last_name", "full_name"}
        )
    return CompatibleInputBinding(
        option_id=option.id,
        lookup_value=lookup_text,
        identifier_kind=IdentifierKind.DESCRIPTIVE,
        lookup_request_parameters=tuple(
            LookupRequestParameter(param_ref=parameter.ref, value=lookup_text)
            for parameter in lookup_params
        ),
        returned_identity_verification_field_paths=match_paths,
    )


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
            "grounding model should not be called for deterministic identity grounding"
        )


class _NoCompatibleResolverGroundingModel:
    def generate(self, **kwargs):
        prompt = str(kwargs.get("prompt") or "")
        return {
            "answer": json.dumps(
                {
                    "tool": "submit_grounding",
                    "arguments": _grounding_review_arguments(
                        prompt,
                        selected_by_input={},
                    ),
                }
            ),
            "usage": {},
        }


class _NoShownResourceTypeGroundingModel:
    def generate(self, **kwargs):
        prompt = str(kwargs.get("prompt") or "")
        arguments = _grounding_review_arguments(prompt, selected_by_input={})
        for review in arguments["known_input_binding_reviews"].values():
            review["resource_type_x"] = NO_SHOWN_RESOURCE_TYPE
            for option_review in review["option_reviews"].values():
                option_review["resource_type_match"] = (
                    ResourceTypeMatch.DIFFERENT_RESOURCE_TYPE.value
                )
        return {
            "answer": json.dumps(
                {"tool": "submit_grounding", "arguments": arguments}
            ),
            "usage": {},
        }


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
    if heading.rstrip(":") == "Known input binding tasks":
        return prompt_section_payload(prompt, heading.rstrip(":"))
    start = prompt.index(heading) + len(heading)
    rest = prompt[start:].lstrip()
    decoder = json.JSONDecoder()
    payload, _ = decoder.raw_decode(rest)
    return payload


def _grounding_review_arguments(
    prompt: str,
    *,
    selected_by_input: dict[str, str | tuple[str, ...]],
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
        "Known input binding tasks:",
    )["known_input_binding_tasks"]:
        known_input_id = task["known_input_id"]
        selected = selected_by_input.get(known_input_id, ())
        positive_ids = {selected} if isinstance(selected, str) else set(selected)
        lookup_text = task["lookup_text"]
        options_by_id = {
            option["binding_option_id"]: option
            for option in task["binding_options"]
        }
        selected_resource_types = {
            options_by_id[option_id]["resource_type"] for option_id in positive_ids
        }
        resource_type_x = (
            next(iter(selected_resource_types))
            if len(selected_resource_types) == 1
            else task["shown_resource_types"][0]
        )
        reviews[known_input_id] = {
            "resource_type_basis": "The input identifies the selected resource type.",
            "resource_type_x": resource_type_x,
            "identifier_kind_basis": "The lookup is a descriptive identifier.",
            "identifier_kind": "DESCRIPTIVE",
            "option_reviews": {
                option["binding_option_id"]: {
                    "resource_type": option["resource_type"],
                    "resource_type_match": (
                        "SAME_RESOURCE_TYPE"
                        if option["resource_type"] == resource_type_x
                        else "DIFFERENT_RESOURCE_TYPE"
                    ),
                    "resolver_fit_question": option["resolver_fit_question"],
                    "because": "The read capability was reviewed by the test model.",
                    "resolution": {
                        "decision": (
                            "CAN_RESOLVE_LOOKUP_TEXT"
                            if option["binding_option_id"] in positive_ids
                            else "CANNOT_RESOLVE_LOOKUP_TEXT"
                        ),
                        "lookup_request_params": (
                            [
                                {
                                    "param_ref": option["api_read"]
                                    ["input_params"][0]["param_ref"],
                                    "value": lookup_text,
                                }
                            ]
                            if option["binding_option_id"] in positive_ids
                            and option["api_read"]["input_params"]
                            else []
                        ),
                        "returned_identity_verification_fields": (
                            [
                                field["path"]
                                for row in option["api_read"]["response_rows"]
                                for field in row["fields"]
                                if field["type"] in {"string", "uuid"}
                            ]
                            if option["binding_option_id"] in positive_ids
                            else []
                        ),
                    },
                }
                for option in task["binding_options"]
            }
        }
    return {
        "known_time_resolutions": time_resolutions,
        "known_input_binding_reviews": reviews,
    }


def _lookup_text_by_input(prompt: str) -> dict[str, str]:
    payload = _json_payload_from_prompt_section(prompt, "Known input binding tasks:")
    return {
        task["known_input_id"]: task["lookup_text"]
        for task in payload["known_input_binding_tasks"]
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


def test_grounding_prompt_instructs_binding_id_copying_verbatim():
    [task] = reference_input_binding_tasks(
        _question_contract("Shipment Tracker", description="flow"),
        resolver_catalog=RelationCatalog(reads=(_flow_read(),)),
        resolver_sources_by_known_input={
            "input_location": build_row_source_catalog(
                RelationCatalog(reads=(_flow_read(),))
            )
        },
    )
    request = GroundingRequest(
        question="What does Shipment Tracker do?",
        tasks=(task,),
        resolver_catalog=RelationCatalog(reads=(_flow_read(),)),
    )
    prompt = _grounding_prompt(request)

    assert "Known input binding tasks:" in prompt
    assert "<known_input_binding_tasks>" in prompt
    assert '<known_input id="input_location"' in prompt
    assert "<shown_resource_types>" in prompt
    assert "<resource_type>flow</resource_type>" in prompt
    assert "<binding_option" in prompt
    assert "<api_read" in prompt
    assert '"binding_options":' not in prompt
    assert "Write resource_type_basis first" in prompt
    assert "set resource_type_x to exactly one shown_resource_type" in prompt
    assert "SAME_RESOURCE_TYPE means it exactly equals resource_type_x" in prompt
    assert "Within each known-input review, write fields in this order: resource_type_basis, resource_type_x, identifier_kind_basis, identifier_kind, option_reviews." in prompt
    assert "Within each option review, write fields in this order: resource_type, resource_type_match, resolver_fit_question, because, resolution." in prompt
    assert "returned_identity_verification_fields are returned-resource fields that may exactly equal lookup_text" in prompt
    assert "Include each selected field exactly once." not in prompt
    assert "can/cannot identify the returned" not in prompt
    assert "because briefly explains the capability decision" not in prompt
    schema = GroundingTurnPrompt(request).response_contract().provider_schema
    bindings_schema = schema["properties"]["known_input_binding_reviews"]
    assert bindings_schema["type"] == "object"
    assert bindings_schema["required"] == [task.known_input_id]
    known_input_review = bindings_schema["properties"][task.known_input_id]
    assert list(known_input_review["properties"]) == [
        "resource_type_basis",
        "resource_type_x",
        "identifier_kind_basis",
        "identifier_kind",
        "option_reviews",
    ]
    assert known_input_review["properties"]["resource_type_x"]["enum"] == [
        "flow",
        NO_SHOWN_RESOURCE_TYPE,
    ]
    option_reviews = known_input_review["properties"]["option_reviews"]
    assert option_reviews["required"] == [option.id for option in task.options]
    first_review = option_reviews["properties"][task.options[0].id]
    assert "oneOf" not in first_review
    assert list(first_review["properties"]) == [
        "resource_type",
        "resource_type_match",
        "resolver_fit_question",
        "because",
        "resolution",
    ]
    resolution = first_review["properties"]["resolution"]
    assert resolution["properties"]["decision"]["enum"] == [
        "CANNOT_RESOLVE_LOOKUP_TEXT"
    ]
    assert list(resolution["properties"]) == [
        "decision",
        "lookup_request_params",
        "returned_identity_verification_fields",
    ]


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda review: review.update(resource_type_x="staff"),
            "resource_type_x was not shown",
        ),
        (
            lambda review: next(iter(review["option_reviews"].values())).update(
                resource_type="staff"
            ),
            "option resource_type mismatch",
        ),
        (
            lambda review: next(iter(review["option_reviews"].values())).update(
                resource_type_match=ResourceTypeMatch.DIFFERENT_RESOURCE_TYPE.value
            ),
            "resource_type_match contradicts resource types",
        ),
    ),
)
def test_grounding_parser_rejects_inconsistent_resource_type_contract(
    mutate,
    message: str,
) -> None:
    catalog = RelationCatalog(reads=(_location_read(),))
    [task] = reference_input_binding_tasks(
        _question_contract("Nairobi", description="location"),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={
            "input_location": build_row_source_catalog(catalog),
        },
    )
    request = GroundingRequest(
        question="How many stores are in Nairobi?",
        tasks=(task,),
        resolver_catalog=catalog,
    )
    payload = _grounding_review_arguments(
        _grounding_prompt(request),
        selected_by_input={},
    )
    review = payload["known_input_binding_reviews"][task.known_input_id]
    mutate(review)

    with pytest.raises(ValueError, match=message):
        parse_grounding_compatibility(payload, request=request)


def test_grounding_parser_accepts_no_shown_resource_type_only_with_negative_options(
) -> None:
    catalog = RelationCatalog(reads=(_location_read(),))
    [task] = reference_input_binding_tasks(
        _question_contract("Nairobi", description="place"),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={
            "input_location": build_row_source_catalog(catalog),
        },
    )
    request = GroundingRequest(
        question="How many stores are in Nairobi?",
        tasks=(task,),
        resolver_catalog=catalog,
    )
    payload = _grounding_review_arguments(
        _grounding_prompt(request),
        selected_by_input={},
    )
    review = payload["known_input_binding_reviews"][task.known_input_id]
    review["resource_type_x"] = NO_SHOWN_RESOURCE_TYPE
    for option_review in review["option_reviews"].values():
        option_review["resource_type_match"] = (
            ResourceTypeMatch.DIFFERENT_RESOURCE_TYPE.value
        )

    result = parse_grounding_compatibility(payload, request=request)

    assert result.compatibilities == (
        InputBindingCompatibility(known_input_id=task.known_input_id, bindings=()),
    )


def test_grounding_selects_one_resource_type_before_reviewing_resolver_mechanics(
) -> None:
    catalog = RelationCatalog(
        reads=(_staff_read(), _staff_detail_read(), _location_detail_read())
    )
    [task] = reference_input_binding_tasks(
        _staff_question_contract(
            "staff-1",
            description="staff identifier",
            field_label_text="staff_id",
        ),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={
            "input_staff": build_row_source_catalog(catalog),
        },
    )
    option_by_read = {
        option.candidate.resolver_read_id: option for option in task.options
    }
    request = GroundingRequest(
        question="How many sales did the staff with staff_id staff-1 sell?",
        tasks=(task,),
        resolver_catalog=catalog,
    )
    payload = _grounding_review_arguments(
        _grounding_prompt(request),
        selected_by_input={
            task.known_input_id: option_by_read["get_staff_detail"].id,
        },
    )
    review = payload["known_input_binding_reviews"][task.known_input_id]
    review["identifier_kind_basis"] = "staff-1 is the supplied primary key."
    review["identifier_kind"] = IdentifierKind.PRIMARY_KEY.value

    result = parse_grounding_compatibility(payload, request=request)

    assert review["resource_type_x"] == "staff"
    assert {
        option_by_read[read_id].candidate.entity_kind: option_review[
            "resource_type_match"
        ]
        for read_id, option_review in (
            (
                option.candidate.resolver_read_id,
                review["option_reviews"][option.id],
            )
            for option in task.options
        )
    } == {
        "staff": ResourceTypeMatch.SAME_RESOURCE_TYPE.value,
        "location": ResourceTypeMatch.DIFFERENT_RESOURCE_TYPE.value,
    }
    [binding] = result.compatibilities[0].bindings
    assert binding.option_id == option_by_read["get_staff_detail"].id


@pytest.mark.parametrize(
    ("parameter_type", "lookup_text"),
    (
        ("boolean", "not-a-boolean"),
        ("integer", "not-an-integer"),
        ("uuid", "not-a-uuid"),
    ),
)
def test_grounding_schema_rejects_lookup_value_for_incompatible_parameter_type(
    parameter_type: str,
    lookup_text: str,
) -> None:
    read = _staff_detail_read(param_type=parameter_type)
    catalog = RelationCatalog(reads=(read,))
    [task] = reference_input_binding_tasks(
        _staff_question_contract(lookup_text, description="staff member"),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={
            "input_staff": build_row_source_catalog(catalog),
        },
    )
    [option] = task.options
    request = GroundingRequest(
        question=f"Show the staff member {lookup_text}",
        tasks=(task,),
        resolver_catalog=catalog,
    )
    payload = _grounding_review_arguments(
        _grounding_prompt(request),
        selected_by_input={task.known_input_id: option.id},
    )

    with pytest.raises(ValidationError):
        validate(
            payload,
            GroundingTurnPrompt(request).response_contract().provider_schema,
        )


def test_grounding_schema_rejects_repeated_response_match_field() -> None:
    catalog = RelationCatalog(reads=(_uuid_person_read(),))
    [task] = reference_input_binding_tasks(
        _question_contract(
            "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
            description="person",
        ),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={
            "input_location": build_row_source_catalog(catalog),
        },
    )
    [option] = task.options
    request = GroundingRequest(
        question="Which person?",
        tasks=(task,),
        resolver_catalog=catalog,
    )
    schema = GroundingTurnPrompt(request).response_contract().provider_schema
    option_schema = schema["properties"]["known_input_binding_reviews"][
        "properties"
    ][task.known_input_id]["properties"]["option_reviews"]["properties"][option.id]
    match_schema = option_schema["properties"]["resolution"]["oneOf"][1][
        "properties"
    ]["returned_identity_verification_fields"]

    with pytest.raises(ValidationError):
        validate(["data.person_id", "data.person_id"], match_schema)


def test_grounding_schema_requires_no_default_resolver_parameters() -> None:
    read = _staff_detail_read()
    catalog = RelationCatalog(reads=(read,))
    [task] = reference_input_binding_tasks(
        _staff_question_contract("staff-1", description="staff member"),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={
            "input_staff": build_row_source_catalog(catalog),
        },
    )
    [option] = task.options
    request = GroundingRequest(
        question="Show staff member staff-1",
        tasks=(task,),
        resolver_catalog=catalog,
    )

    schema = GroundingTurnPrompt(request).response_contract().provider_schema
    option_schema = schema["properties"]["known_input_binding_reviews"][
        "properties"
    ]["input_staff"]["properties"]["option_reviews"]["properties"][option.id]
    resolution_schema = option_schema["properties"]["resolution"]
    assert [
        branch["properties"]["decision"]["enum"]
        for branch in resolution_schema["oneOf"]
    ] == [
        ["CANNOT_RESOLVE_LOOKUP_TEXT"],
        ["CAN_RESOLVE_LOOKUP_TEXT"],
    ]
    positive_resolution = resolution_schema["oneOf"][1]
    lookup_params_schema = positive_resolution["properties"][
        "lookup_request_params"
    ]

    assert lookup_params_schema["type"] == "array"
    assert lookup_params_schema["minItems"] == 1
    assert lookup_params_schema["maxItems"] == 1
    assert lookup_params_schema["items"]["properties"]["param_ref"]["enum"] == [
        "get_staff_detail.path.staff_id"
    ]
    verification_fields = positive_resolution["properties"][
        "returned_identity_verification_fields"
    ]
    assert verification_fields["type"] == "array"
    assert verification_fields["maxItems"] == len(
        verification_fields["items"]["enum"]
    )
    assert verification_fields["uniqueItems"] is True


@pytest.mark.parametrize(
    "resolution",
    (
        {
            "decision": "CAN_RESOLVE_LOOKUP_TEXT",
            "lookup_request_params": [],
            "returned_identity_verification_fields": [],
        },
        {
            "decision": "CANNOT_RESOLVE_LOOKUP_TEXT",
            "lookup_request_params": [
                {
                    "param_ref": "get_staff_detail.path.staff_id",
                    "value": "staff-1",
                }
            ],
            "returned_identity_verification_fields": ["data.staff_id"],
        },
    ),
)
def test_grounding_schema_correlates_resolution_decision_with_selected_mechanics(
    resolution: dict[str, object],
) -> None:
    catalog = RelationCatalog(reads=(_staff_detail_read(),))
    [task] = reference_input_binding_tasks(
        _staff_question_contract("staff-1", description="staff member"),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={
            "input_staff": build_row_source_catalog(catalog),
        },
    )
    [option] = task.options
    request = GroundingRequest(
        question="Show staff member staff-1",
        tasks=(task,),
        resolver_catalog=catalog,
    )
    schema = GroundingTurnPrompt(request).response_contract().provider_schema
    resolution_schema = schema["properties"]["known_input_binding_reviews"][
        "properties"
    ]["input_staff"]["properties"]["option_reviews"]["properties"][option.id][
        "properties"
    ]["resolution"]

    with pytest.raises(ValidationError):
        validate(resolution, resolution_schema)


def test_named_reference_options_return_canonical_entity_keys():
    catalog = RelationCatalog(reads=(_flow_read(),))
    [task] = reference_input_binding_tasks(
        _question_contract("operations", description="tag label"),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={
            "input_location": build_row_source_catalog(catalog)
        },
    )

    surfaces = tuple(option.candidate.result_surface for option in task.options)

    assert surfaces
    assert set(surfaces) == {"entity flow:primary_key"}
    assert len(task.options) == 1


def test_grounding_shows_one_shared_endpoint_projection_per_canonical_result() -> None:
    catalog = RelationCatalog(reads=(_staff_read(),))
    [task] = reference_input_binding_tasks(
        _staff_question_contract("Azraah", description="staff member or seller"),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={
            "input_staff": build_row_source_catalog(catalog),
        },
    )
    request = GroundingRequest(
        question="How much did Azraah make in sales?",
        tasks=(task,),
        resolver_catalog=catalog,
    )

    [input_task] = _json_payload_from_prompt_section(
        _grounding_prompt(request),
        "Known input binding tasks:",
    )["known_input_binding_tasks"]
    [option] = input_task["binding_options"]

    assert option["api_read"] == ApiReadResponseShapeProjector(
        catalog.read("list_staff_list")
    ).prompt_payload(row_path_ids=("data",))
    assert option["canonical_result"] == {
        "entity_kind": "staff",
        "key_id": "primary_key",
        "components": [
            {
                "component_id": "staff_id",
                "field_path": "data.staff_id",
            }
        ],
    }


def test_grounding_preserves_shared_parameter_context_and_source_overlay() -> None:
    base_read = _variant_person_read()
    name_param, shape_param = base_read.params
    read = replace(
        base_read,
        params=(
            replace(name_param, description="Match a person by name."),
            replace(
                shape_param,
                description="Select the returned person representation.",
                choice_labels={"SUMMARY": "Summary", "DETAIL": "Detail"},
            ),
        ),
    )
    catalog = RelationCatalog(reads=(read,))
    [task] = reference_input_binding_tasks(
        _question_contract("Nadia", description="person"),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={
            "input_location": build_row_source_catalog(catalog),
        },
    )
    option = task.options[0]
    request = GroundingRequest(
        question="What is Nadia's status?",
        tasks=(task,),
        resolver_catalog=catalog,
    )

    input_params = resolver_option_surface(request, option).prompt_payload()[
        "api_read"
    ]["input_params"]

    assert input_params[0]["description"] == "Match a person by name."
    assert input_params[1]["description"] == (
        "Select the returned person representation."
    )
    assert input_params[1]["choice_labels"] == {
        "SUMMARY": "Summary",
        "DETAIL": "Detail",
    }
    assert input_params[1]["default"] in {"SUMMARY", "DETAIL"}
    assert input_params[1]["default_source"] == "source_variant"
    assert input_params[1]["semantics"] == "response_shape"


def test_shared_endpoint_projection_keeps_distinct_fields_with_same_leaf_name() -> None:
    read = _location_with_area_read()

    [row] = ApiReadResponseShapeProjector(read).prompt_payload(row_path_ids=("data",))[
        "response_rows"
    ]

    assert [field["path"] for field in row["fields"]] == [
        "data.location_id",
        "data.name",
        "data.type",
        "data.area.area_id",
        "data.area.name",
    ]


def test_grounding_does_not_offer_related_resource_fields_as_match_fields() -> None:
    catalog = RelationCatalog(reads=(_location_with_area_read(),))
    [task] = reference_input_binding_tasks(
        _question_contract("Nairobi", description="place"),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={
            "input_location": build_row_source_catalog(catalog),
        },
    )
    [option] = task.options
    request = GroundingRequest(
        question="How many stores are in Nairobi?",
        tasks=(task,),
        resolver_catalog=catalog,
    )

    surface = resolver_option_surface(request, option)
    match_field_paths = {field.path for field in surface.response_match_fields}
    assert "data.name" in match_field_paths
    assert "data.area.area_id" not in match_field_paths
    assert "data.area.name" not in match_field_paths

    schema = GroundingTurnPrompt(request).response_contract().provider_schema
    option_review = schema["properties"]["known_input_binding_reviews"][
        "properties"
    ]["input_location"]["properties"]["option_reviews"]["properties"][
        option.id
    ]
    legal_match_paths = option_review["properties"]["resolution"]["oneOf"][1][
        "properties"
    ]["returned_identity_verification_fields"]["items"]["enum"]
    assert "data.name" in legal_match_paths
    assert "data.area.area_id" not in legal_match_paths
    assert "data.area.name" not in legal_match_paths


def test_selected_staff_lookup_fields_produce_one_canonical_staff_key() -> None:
    catalog = RelationCatalog(reads=(_staff_read(),))
    row_sources = build_row_source_catalog(catalog)
    [task] = reference_input_binding_tasks(
        _staff_question_contract("Azraah", description="staff member or seller"),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={"input_staff": row_sources},
    )
    [option] = task.options
    request = GroundingRequest(
        question="How much did Azraah make in sales?",
        tasks=(task,),
        resolver_catalog=catalog,
    )
    [input_task] = _json_payload_from_prompt_section(
        _grounding_prompt(request),
        "Known input binding tasks:",
    )["known_input_binding_tasks"]
    [prompt_option] = input_task["binding_options"]
    shown_field_paths = {
        field["path"]
        for row in prompt_option["api_read"]["response_rows"]
        for field in row["fields"]
    }
    assert "data.phone_number" in shown_field_paths

    compatibility = parse_grounding_compatibility(
        {
            "known_time_resolutions": {},
            "known_input_binding_reviews": {
                "input_staff": {
                    "resource_type_basis": "Azraah identifies a staff member.",
                    "resource_type_x": "staff",
                    "identifier_kind_basis": "Azraah is a descriptive name.",
                    "identifier_kind": "DESCRIPTIVE",
                    "option_reviews": {
                        option.id: {
                            "resource_type": "staff",
                            "resource_type_match": "SAME_RESOURCE_TYPE",
                            "resolver_fit_question": resolver_fit_question_for_option(
                                task=task,
                                option=option,
                            ),
                            "because": (
                                "The read accepts a name and returns staff-name fields."
                            ),
                            "resolution": {
                                "decision": "CAN_RESOLVE_LOOKUP_TEXT",
                                "lookup_request_params": [
                                    {
                                        "param_ref": "list_staff_list.query.name",
                                        "value": "Azraah",
                                    }
                                ],
                                "returned_identity_verification_fields": [
                                    "data.first_name",
                                    "data.last_name",
                                    "data.full_name",
                                ],
                            },
                        }
                    }
                }
            },
        },
        request=request,
    )
    [binding] = compatibility.compatibilities[0].bindings
    data_access = _EndpointDataAccess(
        {
            "list_staff_list": {
                "data": [
                    {
                        "staff_id": "staff_azraah",
                        "first_name": "Azraah",
                        "last_name": "Ahmed",
                        "full_name": "Azraah Ahmed",
                        "phone_number": "+254700000001",
                    },
                    {
                        "staff_id": "staff_other",
                        "first_name": "Other",
                        "last_name": "Staff",
                        "full_name": "Other Staff",
                        "phone_number": "Azraah",
                    },
                ]
            }
        }
    )

    execution = execute_compatible_reference_bindings(
        tasks=(task,),
        bindings=(binding,),
        source_read_key_prefix="test",
        full_catalog=catalog,
        data_access_port=data_access,
    )[option.id]

    assert binding.returned_identity_verification_field_paths == (
        "data.first_name",
        "data.last_name",
        "data.full_name",
    )
    assert data_access.calls == [
        ("list_staff_list", {"list_staff_list.query.name": "Azraah"})
    ]
    assert execution.ledger is not None
    [value] = execution.ledger.values
    assert value.payload.key.component_values() == {"staff_id": "staff_azraah"}
    assert value.payload.matched_field_path == "data.first_name"
    assert value.payload.matched_value == "Azraah"


def test_two_exact_staff_matches_produce_typed_clarification_options() -> None:
    catalog = RelationCatalog(reads=(_staff_read(),))
    row_sources = build_row_source_catalog(catalog)
    [task] = reference_input_binding_tasks(
        _staff_question_contract("Azraah", description="staff member or seller"),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={"input_staff": row_sources},
    )
    [option] = task.options
    binding = _compatible_binding(
        catalog,
        option,
        lookup_text="Azraah",
        match_paths=("data.first_name", "data.last_name", "data.full_name"),
    )

    execution = execute_compatible_reference_bindings(
        tasks=(task,),
        bindings=(binding,),
        source_read_key_prefix="test",
        full_catalog=catalog,
        data_access_port=_EndpointDataAccess(
            {
                "list_staff_list": {
                    "data": [
                        {
                            "staff_id": "staff_1",
                            "first_name": "Azraah",
                            "last_name": "One",
                            "full_name": "Azraah One",
                        },
                        {
                            "staff_id": "staff_2",
                            "first_name": "Azraah",
                            "last_name": "Two",
                            "full_name": "Azraah Two",
                        },
                    ]
                }
            }
        ),
    )[option.id]

    assert execution.ledger is not None
    issue = reference_binding_issue(
        task,
        candidate=option.candidate,
        values=execution.ledger.values,
    )

    assert issue is not None
    assert issue.kind is GroundingTerminalKind.AMBIGUOUS_REFERENCE
    assert tuple(
        candidate.key.component_values() for candidate in issue.candidate_options
    ) == (
        {"staff_id": "staff_1"},
        {"staff_id": "staff_2"},
    )
    assert tuple(candidate.matched_field for candidate in issue.candidate_options) == (
        "field.data.first_name",
        "field.data.first_name",
    )
    assert tuple(candidate.matched_value for candidate in issue.candidate_options) == (
        "Azraah",
        "Azraah",
    )


def test_resolver_row_source_variants_keep_distinct_identity_and_defaults() -> None:
    catalog = RelationCatalog(reads=(_variant_person_read(),))
    row_sources = build_row_source_catalog(catalog)
    [task] = reference_input_binding_tasks(
        _question_contract("Azraah", description="person"),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={"input_location": row_sources},
    )

    assert len(task.options) == 2
    assert len({option.id for option in task.options}) == 2
    prompt_options = GroundingTurnPrompt(
        GroundingRequest(
            question="Which person is named Azraah?",
            tasks=(task,),
            resolver_catalog=catalog,
        )
    ).known_input_binding_tasks_payload()["known_input_binding_tasks"][0][
        "binding_options"
    ]
    shown_shape_defaults = {
        parameter["default"]
        for option in prompt_options
        for parameter in option["api_read"]["input_params"]
        if parameter["param_ref"] == "list_people.query.shape"
    }
    assert shown_shape_defaults == {"SUMMARY", "DETAIL"}

    bindings = tuple(
        CompatibleInputBinding(
            option_id=option.id,
            lookup_value="Azraah",
            identifier_kind=IdentifierKind.DESCRIPTIVE,
            lookup_request_parameters=(
                LookupRequestParameter(
                    param_ref="list_people.query.name",
                    value="Azraah",
                ),
            ),
            returned_identity_verification_field_paths=("data.name",),
        )
        for option in task.options
    )
    data_access = _EndpointDataAccess(
        {"list_people": {"data": [{"person_id": "person_1", "name": "Azraah"}]}}
    )

    executions = execute_compatible_reference_bindings(
        tasks=(task,),
        bindings=bindings,
        source_read_key_prefix="test",
        full_catalog=catalog,
        data_access_port=data_access,
    )

    assert set(executions) == {option.id for option in task.options}
    assert data_access.calls == [
        (
            "list_people",
            {"list_people.query.name": "Azraah", "list_people.query.shape": "SUMMARY"},
        ),
        (
            "list_people",
            {"list_people.query.name": "Azraah", "list_people.query.shape": "DETAIL"},
        ),
    ]


def test_resolver_verifies_the_typed_request_value_against_typed_response_fields() -> (
    None
):
    uppercase_uuid = "AAAAAAAA-BBBB-4CCC-8DDD-EEEEEEEEEEEE"
    canonical_uuid = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
    catalog = RelationCatalog(reads=(_uuid_person_read(),))
    row_sources = build_row_source_catalog(catalog)
    [task] = reference_input_binding_tasks(
        _question_contract(uppercase_uuid, description="person"),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={"input_location": row_sources},
    )
    [option] = task.options
    request = GroundingRequest(
        question="Which person?",
        tasks=(task,),
        resolver_catalog=catalog,
    )
    compatibility = parse_grounding_compatibility(
        {
            "known_time_resolutions": {},
            "known_input_binding_reviews": {
                "input_location": {
                    "resource_type_basis": "The UUID identifies a person.",
                    "resource_type_x": "person",
                    "identifier_kind_basis": "The lookup is a primary key.",
                    "identifier_kind": "PRIMARY_KEY",
                    "option_reviews": {
                        option.id: {
                            "resource_type": "person",
                            "resource_type_match": "SAME_RESOURCE_TYPE",
                            "resolver_fit_question": resolver_fit_question_for_option(
                                task=task,
                                option=option,
                            ),
                            "because": "The exact UUID identifies the returned person.",
                            "resolution": {
                                "decision": "CAN_RESOLVE_LOOKUP_TEXT",
                                "lookup_request_params": [
                                    {
                                        "param_ref": "get_person.query.person_id",
                                        "value": canonical_uuid,
                                    }
                                ],
                                "returned_identity_verification_fields": [
                                    "data.person_id"
                                ],
                            },
                        }
                    },
                }
            },
        },
        request=request,
    )
    [binding] = compatibility.compatibilities[0].bindings
    data_access = _EndpointDataAccess(
        {"get_person": {"data": {"person_id": canonical_uuid, "name": "Azraah"}}}
    )

    execution = execute_compatible_reference_bindings(
        tasks=(task,),
        bindings=(binding,),
        source_read_key_prefix="test",
        full_catalog=catalog,
        data_access_port=data_access,
    )[option.id]

    assert data_access.calls == [
        ("get_person", {"get_person.query.person_id": canonical_uuid})
    ]
    assert execution.ledger is not None
    [value] = execution.ledger.values
    assert value.payload.key.component_values() == {"person_id": canonical_uuid}


def test_resolver_string_verification_keeps_only_the_exact_case_match() -> None:
    catalog = RelationCatalog(reads=(_location_read(),))
    [task] = reference_input_binding_tasks(
        _question_contract("Nairobi", description="location"),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={
            "input_location": build_row_source_catalog(catalog)
        },
    )
    [option] = task.options
    binding = _compatible_binding(catalog, option, lookup_text="Nairobi")

    execution = execute_compatible_reference_bindings(
        tasks=(task,),
        bindings=(binding,),
        source_read_key_prefix="test",
        full_catalog=catalog,
        data_access_port=_EndpointDataAccess(
            {
                "list_location_list": {
                    "data": [
                        {"location_id": "lowercase", "name": "nairobi"},
                        {"location_id": "exact", "name": "Nairobi"},
                    ]
                }
            }
        ),
    )[option.id]

    assert execution.ledger is not None
    [value] = execution.ledger.values
    assert value.payload.key.component_values() == {"location_id": "exact"}


def test_named_reference_option_does_not_preselect_identity_match_fields() -> None:
    catalog = RelationCatalog(reads=(_location_with_area_read(),))
    [task] = reference_input_binding_tasks(
        _question_contract("Nairobi", description="city"),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={
            "input_location": build_row_source_catalog(catalog)
        },
    )

    assert len(task.options) == 1
    assert task.options[0].candidate.entity_kind == "location"


def test_named_reference_without_a_catalog_resolver_requires_clarification() -> None:
    output = ground_question_inputs(
        question="How many stores are in Nairobi?",
        question_contract=_question_contract("Nairobi", description="area"),
        full_catalog=RelationCatalog(),
        resolver_selections=(),
        runtime_values=RuntimeValueContext(
            runtime_date="2026-07-15",
            timezone="Africa/Nairobi",
        ),
        model_port=_NoGroundingModel(),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )

    [issue] = output.ledger.issues
    assert issue.kind is GroundingTerminalKind.UNSUPPORTED_REFERENCE
    assert issue.known_input_id == "input_location"


def test_named_reference_with_no_positive_resolver_requires_clarification() -> None:
    catalog = RelationCatalog(reads=(_location_read(),))
    output = ground_question_inputs(
        question="How many stores are in Nairobi?",
        question_contract=_question_contract("Nairobi", description="area"),
        full_catalog=catalog,
        resolver_selections=(
            EntityTargetResolverSelection(
                target_id="input_location",
                catalog_search_terms=("location",),
                selected_read_ids=("list_location_list",),
            ),
        ),
        runtime_values=RuntimeValueContext(
            runtime_date="2026-07-15",
            timezone="Africa/Nairobi",
        ),
        model_port=_NoCompatibleResolverGroundingModel(),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )

    [issue] = output.ledger.issues
    assert issue.kind is GroundingTerminalKind.UNSUPPORTED_REFERENCE
    assert issue.known_input_id == "input_location"


def test_no_shown_resource_type_requires_clarification() -> None:
    catalog = RelationCatalog(reads=(_location_read(),))
    output = ground_question_inputs(
        question="How many stores are in Nairobi?",
        question_contract=_question_contract("Nairobi", description="place"),
        full_catalog=catalog,
        resolver_selections=(
            EntityTargetResolverSelection(
                target_id="input_location",
                catalog_search_terms=("location",),
                selected_read_ids=("list_location_list",),
            ),
        ),
        runtime_values=RuntimeValueContext(
            runtime_date="2026-07-15",
            timezone="Africa/Nairobi",
        ),
        model_port=_NoShownResourceTypeGroundingModel(),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
    )

    [issue] = output.ledger.issues
    assert issue.kind is GroundingTerminalKind.UNSUPPORTED_REFERENCE
    assert issue.known_input_id == "input_location"


def test_binding_tasks_use_catalog_for_recalled_and_validation_sources() -> None:
    full_catalog = RelationCatalog(reads=(_staff_read(), _staff_detail_read()))
    binding_sources = reference_binding_sources_by_known_input(
        full_row_sources=build_row_source_catalog(full_catalog),
        resolver_selections=(
            EntityTargetResolverSelection(
                target_id="input_staff",
                catalog_search_terms=("staff",),
                selected_read_ids=("list_staff_list",),
            ),
        ),
    )

    [task] = reference_input_binding_tasks(
        _staff_question_contract("staff_1", description="staff identifier"),
        resolver_catalog=full_catalog,
        resolver_sources_by_known_input=binding_sources,
    )

    assert {option.candidate.resolver_read_id for option in task.options} == {
        "list_staff_list",
        "get_staff_detail",
    }


def test_binding_tasks_preserve_resolver_selection_per_known_input() -> None:
    catalog = RelationCatalog(reads=(_location_read(), _store_read()))
    sources_by_input = reference_binding_sources_by_known_input(
        full_row_sources=build_row_source_catalog(catalog),
        resolver_selections=(
            EntityTargetResolverSelection(
                target_id="input_location",
                catalog_search_terms=("location",),
                selected_read_ids=("list_location_list",),
            ),
            EntityTargetResolverSelection(
                target_id="input_store",
                catalog_search_terms=("store",),
                selected_read_ids=("list_store_list",),
            ),
        ),
    )
    contract = QuestionContract(
        requested_facts=(
            RequestedFact(
                id="fact_1",
                description="sales between two places",
                answer_outputs=(
                    RequestedFactAnswerOutput(
                        id="total_sales",
                        role="ANSWER_VALUE",
                        description="total sales",
                    ),
                ),
                known_inputs=(
                    _reference_input("input_location", "Nairobi"),
                    _reference_input("input_store", "Pivot Mall"),
                ),
            ),
        )
    )

    tasks = reference_input_binding_tasks(
        contract,
        resolver_catalog=catalog,
        resolver_sources_by_known_input=sources_by_input,
    )
    read_ids_by_input = {
        task.known_input_id: {
            option.candidate.resolver_read_id for option in task.options
        }
        for task in tasks
    }

    assert read_ids_by_input == {
        "input_location": {"list_location_list"},
        "input_store": {"list_store_list"},
    }


def test_reference_option_read_failure_remains_scoped_to_its_binding() -> None:
    catalog = RelationCatalog(reads=(_location_read(), _store_read()))
    row_sources = build_row_source_catalog(catalog)
    [task] = reference_input_binding_tasks(
        _question_contract("Pivot Mall", description="place"),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={"input_location": row_sources},
    )
    entity_options = task.options

    class _PartiallyFailingDataAccess:
        def read(self, *, endpoint_name, args):
            del args
            if endpoint_name == "list_store_list":
                return {
                    "responseStatus": 404,
                    "responseBody": {"detail": "not found"},
                }
            return _endpoint_result(
                {"data": [{"location_id": "location_1", "name": "Pivot Mall"}]}
            )

    bindings = tuple(
        _compatible_binding(
            catalog,
            option,
            lookup_text="Pivot Mall",
        )
        for option in entity_options
    )
    executions = execute_compatible_reference_bindings(
        tasks=(task,),
        bindings=bindings,
        source_read_key_prefix="test",
        full_catalog=catalog,
        data_access_port=_PartiallyFailingDataAccess(),
    )
    executions_by_read = {
        option.candidate.resolver_read_id: executions[option.id]
        for option in entity_options
    }

    assert executions_by_read["list_location_list"].ledger is not None
    assert executions_by_read["list_location_list"].failure is None
    assert executions_by_read["list_store_list"].ledger is None
    assert executions_by_read["list_store_list"].failure is not None


def test_selected_reference_option_with_no_exact_match_is_unresolved() -> None:
    catalog = RelationCatalog(reads=(_location_read(),))
    row_sources = build_row_source_catalog(catalog)
    [task] = reference_input_binding_tasks(
        _question_contract("Nairobi", description="place"),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={"input_location": row_sources},
    )
    [option] = task.options
    binding = _compatible_binding(catalog, option, lookup_text="Nairobi")

    execution = execute_compatible_reference_bindings(
        tasks=(task,),
        bindings=(binding,),
        source_read_key_prefix="test",
        full_catalog=catalog,
        data_access_port=_EndpointDataAccess({"list_location_list": {"data": []}}),
    )[option.id]

    assert execution.ledger is not None
    issue = reference_binding_issue(
        task,
        candidate=option.candidate,
        values=execution.ledger.values,
    )
    assert issue is not None
    assert issue.kind is GroundingTerminalKind.UNSUPPORTED_REFERENCE
    assert issue.resolver_read_id == "list_location_list"


def test_truncated_reference_response_cannot_prove_a_unique_name_match() -> None:
    catalog = RelationCatalog(reads=(_location_read(),))
    row_sources = build_row_source_catalog(catalog)
    [task] = reference_input_binding_tasks(
        _question_contract("Nairobi", description="place"),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={"input_location": row_sources},
    )
    [option] = task.options
    binding = _compatible_binding(catalog, option, lookup_text="Nairobi")
    data_access = _DataAccess(
        {
            "responseStatus": 200,
            "responseBody": {
                "data": [{"location_id": "location_1", "name": "Nairobi"}]
            },
            "truncated": True,
        }
    )

    execution = execute_compatible_reference_bindings(
        tasks=(task,),
        bindings=(binding,),
        source_read_key_prefix="test",
        full_catalog=catalog,
        data_access_port=data_access,
    )[option.id]

    assert execution.ledger is not None
    issue = reference_binding_issue(
        task,
        candidate=option.candidate,
        values=execution.ledger.values,
        truncated=execution.truncated,
        matched_field_is_stable_unique=execution.matched_field_is_stable_unique,
    )
    assert issue is not None
    assert issue.kind is GroundingTerminalKind.INCOMPLETE_REFERENCE


def test_truncated_reference_response_can_prove_an_exact_stable_unique_match() -> None:
    base_read = _location_read()
    location_read = replace(
        base_read,
        candidate_keys=(
            *base_read.candidate_keys,
            CandidateKey(
                id="unique_name",
                entity_kind="location",
                components=(
                    CandidateKeyComponent(
                        id="name",
                        field_ref="field.data.name",
                    ),
                ),
                stable=True,
            ),
        ),
    )
    catalog = RelationCatalog(reads=(location_read,))
    row_sources = build_row_source_catalog(catalog)
    [task] = reference_input_binding_tasks(
        _question_contract("Nairobi", description="place"),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={"input_location": row_sources},
    )
    [option] = task.options
    binding = _compatible_binding(catalog, option, lookup_text="Nairobi")
    data_access = _DataAccess(
        {
            "responseStatus": 200,
            "responseBody": {
                "data": [{"location_id": "location_1", "name": "Nairobi"}]
            },
            "truncated": True,
        }
    )

    execution = execute_compatible_reference_bindings(
        tasks=(task,),
        bindings=(binding,),
        source_read_key_prefix="test",
        full_catalog=catalog,
        data_access_port=data_access,
    )[option.id]

    assert execution.matched_field_is_stable_unique is True
    assert execution.ledger is not None
    assert (
        reference_binding_issue(
            task,
            candidate=option.candidate,
            values=execution.ledger.values,
            truncated=execution.truncated,
            matched_field_is_stable_unique=execution.matched_field_is_stable_unique,
        )
        is None
    )


def test_reference_match_requires_case_sensitive_exact_equality() -> None:
    catalog = RelationCatalog(reads=(_location_read(),))
    row_sources = build_row_source_catalog(catalog)
    [task] = reference_input_binding_tasks(
        _question_contract("Nairobi", description="place"),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={"input_location": row_sources},
    )
    [option] = task.options
    binding = _compatible_binding(catalog, option, lookup_text="Nairobi")

    execution = execute_compatible_reference_bindings(
        tasks=(task,),
        bindings=(binding,),
        source_read_key_prefix="test",
        full_catalog=catalog,
        data_access_port=_EndpointDataAccess(
            {
                "list_location_list": {
                    "data": [
                        {"location_id": "location_1", "name": "NAIROBI"},
                        {
                            "location_id": "location_2",
                            "name": "Greater Nairobi",
                        },
                    ]
                }
            }
        ),
    )[option.id]

    assert execution.ledger is not None
    assert execution.ledger.values == ()


def test_reference_match_does_not_use_another_response_field() -> None:
    base_read = _location_read()
    location_read = replace(
        base_read,
        fields=(
            *base_read.fields,
            CatalogField(
                ref="field.data.county",
                path="data.county",
                row_path_id="data",
                type="string",
            ),
        ),
    )
    catalog = RelationCatalog(reads=(location_read,))
    row_sources = build_row_source_catalog(catalog)
    [task] = reference_input_binding_tasks(
        _question_contract("Nairobi", description="place"),
        resolver_catalog=catalog,
        resolver_sources_by_known_input={"input_location": row_sources},
    )
    [option] = task.options
    binding = _compatible_binding(
        catalog,
        option,
        lookup_text="Nairobi",
        match_paths=("data.name",),
    )

    execution = execute_compatible_reference_bindings(
        tasks=(task,),
        bindings=(binding,),
        source_read_key_prefix="test",
        full_catalog=catalog,
        data_access_port=_EndpointDataAccess(
            {
                "list_location_list": {
                    "data": [
                        {
                            "location_id": "location_1",
                            "name": "Goldset Nairobi Store",
                            "county": "Nairobi",
                        }
                    ]
                }
            }
        ),
    )[option.id]

    assert execution.ledger is not None
    assert execution.ledger.values == ()


def test_identical_selected_resolver_requests_execute_once_for_distinct_inputs() -> (
    None
):
    catalog = RelationCatalog(reads=(_location_read(),))
    row_sources = build_row_source_catalog(catalog)
    contract = QuestionContract(
        requested_facts=(
            RequestedFact(
                id="fact_1",
                description="route between two locations",
                answer_outputs=(
                    RequestedFactAnswerOutput(
                        id="distance",
                        role="ANSWER_VALUE",
                        description="distance",
                    ),
                ),
                known_inputs=(
                    _reference_input("origin", "Nairobi"),
                    _reference_input("destination", "Nairobi"),
                ),
            ),
        )
    )
    tasks = reference_input_binding_tasks(
        contract,
        resolver_catalog=catalog,
        resolver_sources_by_known_input={
            "origin": row_sources,
            "destination": row_sources,
        },
    )
    selected_options = tuple(task.options[0] for task in tasks)
    selected_bindings = tuple(
        _compatible_binding(catalog, option, lookup_text="Nairobi")
        for option in selected_options
    )
    data_access = _EndpointDataAccess(
        {
            "list_location_list": {
                "data": [{"location_id": "location_1", "name": "Nairobi"}]
            }
        }
    )

    executions = execute_compatible_reference_bindings(
        tasks=tasks,
        bindings=selected_bindings,
        source_read_key_prefix="test",
        full_catalog=catalog,
        data_access_port=data_access,
    )

    assert data_access.calls == [
        ("list_location_list", {"list_location_list.query.name": "Nairobi"})
    ]
    assert {
        execution.ledger.values[0].known_input_id
        for execution in executions.values()
        if execution.ledger is not None
    } == {"origin", "destination"}


def test_selected_canonical_identity_resolves_without_a_read_target() -> None:
    catalog = RelationCatalog(reads=(_location_read(),))
    contract = _question_contract("Nairobi", description="location")
    tasks = reference_input_binding_tasks(
        contract,
        resolver_catalog=catalog,
        resolver_sources_by_known_input={
            "input_location": build_row_source_catalog(catalog),
        },
    )
    compatibility = GroundingCompatibilityResult(
        compatibilities=(
            InputBindingCompatibility(
                known_input_id="input_location",
                bindings=tuple(
                    _compatible_binding(catalog, option, lookup_text="Nairobi")
                    for option in tasks[0].options
                ),
            ),
        )
    )
    catalog_selection = CatalogSelectionResult(
        relation_catalog=catalog,
        requested_fact_selections=(
            RequestedFactCatalogSelection(
                requested_fact_id="fact_1",
                query_terms=(),
                rankings=(),
                selected_read_ids=(),
            ),
        ),
        selected_read_ids=(),
    )
    request = ReadEligibilityRequest(
        question="How much did we sell in Nairobi?",
        question_contract=contract,
        requested_facts=contract.requested_facts,
        catalog_selection=catalog_selection,
        conversation_context={},
        binding_tasks=tasks,
        compatible_reference_bindings=compatibility.compatibilities[0].bindings,
        resolver_catalog=catalog,
    )
    surface = read_eligibility_candidate_surface(request)
    [canonical_option] = surface.canonical_options
    result = ReadEligibilityResult(
        read_assessments=(),
        canonical_inputs=(
            CanonicalInputSelection(
                option=canonical_option,
                selected_resolver_binding=canonical_option.resolver_bindings[0],
                interpretation_question="Which location?",
                canonical_option_assessments=(
                    (canonical_option.id, "The location read exposes this identity."),
                ),
                because="Nairobi denotes the returned location.",
                resolver_option_assessments=(
                    (
                        canonical_option.resolver_bindings[0].option_id,
                        "The name parameter retrieves the returned location.",
                    ),
                ),
            ),
        ),
    )
    data_access = _EndpointDataAccess(
        {
            "list_location_list": {
                "data": [{"location_id": "location_1", "name": "Nairobi"}],
            }
        }
    )
    resolved = resolve_read_eligibility(
        request=request,
        result=result,
        full_catalog=catalog,
        data_access_port=data_access,
        source_read_key_prefix="test",
    )

    assert data_access.calls == [
        ("list_location_list", {"list_location_list.query.name": "Nairobi"})
    ]
    assert len(resolved.ledger.values) == 1
    [value] = resolved.ledger.values
    assert value.known_input_id == "input_location"
    assert value.payload.matched_field_ref == "field.data.name"
    assert value.payload.matched_field_path == "data.name"
    assert value.payload.matched_value == "Nairobi"
    assert resolved.ledger.uses == ()


def test_read_eligibility_executes_only_the_selected_reference_option() -> None:
    catalog = RelationCatalog(reads=(_location_read(), _location_alias_read()))
    contract = _question_contract("Nairobi", description="location")
    tasks = reference_input_binding_tasks(
        contract,
        resolver_catalog=catalog,
        resolver_sources_by_known_input={
            "input_location": build_row_source_catalog(catalog),
        },
    )
    compatibility = GroundingCompatibilityResult(
        compatibilities=(
            InputBindingCompatibility(
                known_input_id="input_location",
                bindings=tuple(
                    _compatible_binding(catalog, option, lookup_text="Nairobi")
                    for option in tasks[0].options
                ),
            ),
        )
    )
    request = ReadEligibilityRequest(
        question="How much did we sell in Nairobi?",
        question_contract=contract,
        requested_facts=contract.requested_facts,
        catalog_selection=CatalogSelectionResult(
            relation_catalog=catalog,
            requested_fact_selections=(
                RequestedFactCatalogSelection(
                    requested_fact_id="fact_1",
                    query_terms=(),
                    rankings=(),
                    selected_read_ids=(),
                ),
            ),
            selected_read_ids=(),
        ),
        conversation_context={},
        binding_tasks=tasks,
        compatible_reference_bindings=compatibility.compatibilities[0].bindings,
        resolver_catalog=catalog,
    )
    data_access = _EndpointDataAccess(
        {
            "list_location_list": {
                "data": [{"location_id": "location_1", "name": "Nairobi"}],
            },
            "list_location_alias_list": {
                "data": [{"location_id": "location_2", "display_name": "Nairobi"}],
            },
        }
    )

    surface = read_eligibility_candidate_surface(request)

    assert data_access.calls == []
    [canonical_option] = surface.canonical_options
    resolver_options_by_id = {
        option.id: option for task in tasks for option in task.options
    }
    assert {
        binding.option_id for binding in canonical_option.resolver_bindings
    } == set(resolver_options_by_id)
    selected_resolver_binding = next(
        binding
        for binding in canonical_option.resolver_bindings
        if resolver_options_by_id[
            binding.option_id
        ].candidate.resolver_read_id
        == "list_location_list"
    )
    interpretation_question = (
        surface.card_payload["requested_fact_read_candidates"][0]["known_inputs"][0][
            "interpretation_question"
        ]
    )
    payload = {
        "requested_fact_assessments": {
            "fact_1": {
                "read_candidate_reviews": {},
                "canonical_inputs": {
                    canonical_option.known_input_token: {
                        "interpretation_question": interpretation_question,
                        "canonical_option_assessments": {
                            canonical_option.id: (
                                "The candidate reads were assessed under this identity."
                            )
                        },
                        "because": (
                            "Nairobi denotes the location returned by this read."
                        ),
                        "canonical_option_id": canonical_option.id,
                        "resolver_option_assessments": {
                            binding.option_id: (
                                "The shown lookup parameters retrieve the location "
                                "and its returned fields verify that location."
                            )
                            for binding in canonical_option.resolver_bindings
                        },
                        "resolver_option_id": selected_resolver_binding.option_id,
                    }
                },
            }
        }
    }
    schema = ReadEligibilityTurnPrompt(request).response_contract().provider_schema
    validate(instance=payload, schema=schema)
    result = parse_read_eligibility(payload, request=request)
    resolved = resolve_read_eligibility(
        request=request,
        result=result,
        full_catalog=catalog,
        data_access_port=data_access,
        source_read_key_prefix="test",
    )

    assert data_access.calls == [
        ("list_location_list", {"list_location_list.query.name": "Nairobi"})
    ]
    assert len(resolved.ledger.values) == 1


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


def _endpoint_result(response_body):
    return {
        "responseStatus": 200,
        "responseBody": response_body,
    }


def _question_contract(
    text: str,
    *,
    description: str = "",
) -> QuestionContract:
    return QuestionContract(
        requested_facts=(
            RequestedFact(
                id="fact_1",
                description="sales total",
                answer_outputs=(
                    RequestedFactAnswerOutput(
                        id="total_sales",
                        role="ANSWER_VALUE",
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
                        role="ANSWER_VALUE",
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
                        role="ANSWER_VALUE",
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
                role="ANSWER_VALUE",
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
                role="ANSWER_VALUE",
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
                        role="ANSWER_VALUE",
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
                        role="ANSWER_VALUE",
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
                        role="ANSWER_VALUE",
                        description="total sales",
                    ),
                ),
                known_inputs=(_time_input("input_date", text),),
            ),
        )
    )


def _candidate_key(
    entity_kind: str,
    component_id: str,
    field_ref: str,
    *,
    context_field_refs: tuple[str, ...] = (),
) -> tuple[CandidateKey, ...]:
    return (
        CandidateKey(
            id="primary_key",
            entity_kind=entity_kind,
            components=(CandidateKeyComponent(id=component_id, field_ref=field_ref),),
            primary=True,
            context_field_refs=context_field_refs,
        ),
    )


def _entity_target(entity_kind: str, component_id: str) -> EntityKeyComponentTarget:
    return EntityKeyComponentTarget(
        entity_kind=entity_kind,
        key_id="primary_key",
        component_id=component_id,
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


def _flow_read() -> EndpointRead:
    return EndpointRead(
        id="list_flows",
        endpoint_name="list_flows",
        resource_names=("flow",),
        row_paths=(RowPath(id="data", path="data", cardinality=RowCardinality.MANY),),
        fields=(
            CatalogField(
                ref="flow.id",
                path="data.id",
                row_path_id="data",
                type="string",
            ),
            CatalogField(
                ref="flow.name",
                path="data.name",
                row_path_id="data",
                type="string",
            ),
            CatalogField(
                ref="flow.tags",
                path="data.tags",
                row_path_id="data",
                type="array",
            ),
        ),
        candidate_keys=_candidate_key(
            "flow",
            "id",
            "flow.id",
            context_field_refs=("flow.name",),
        ),
    )


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
            ),
            CatalogField(
                ref="field.data.name",
                path="data.name",
                row_path_id="data",
                type="string",
            ),
        ),
        candidate_keys=_candidate_key(
            "location",
            "location_id",
            "field.data.location_id",
            context_field_refs=("field.data.name",),
        ),
    )


def _variant_person_read() -> EndpointRead:
    return EndpointRead(
        id="list_people",
        endpoint_name="list_people",
        resource_names=("person",),
        params=(
            CatalogParam(
                ref="list_people.query.name",
                name="name",
                source=ParamSource.QUERY,
                type="string",
            ),
            CatalogParam(
                ref="list_people.query.shape",
                name="shape",
                source=ParamSource.QUERY,
                type="choice",
                required=True,
                choices=("SUMMARY", "DETAIL"),
                semantics="response_shape",
            ),
        ),
        row_paths=(RowPath(id="data", path="data", cardinality=RowCardinality.MANY),),
        fields=(
            CatalogField(
                ref="person.id",
                path="data.person_id",
                row_path_id="data",
                type="string",
            ),
            CatalogField(
                ref="person.name",
                path="data.name",
                row_path_id="data",
                type="string",
            ),
        ),
        candidate_keys=_candidate_key("person", "person_id", "person.id"),
    )


def _uuid_person_read() -> EndpointRead:
    return EndpointRead(
        id="get_person",
        endpoint_name="get_person",
        resource_names=("person",),
        params=(
            CatalogParam(
                ref="get_person.query.person_id",
                name="person_id",
                source=ParamSource.QUERY,
                type="uuid",
            ),
        ),
        row_paths=(RowPath(id="data", path="data", cardinality=RowCardinality.ONE),),
        fields=(
            CatalogField(
                ref="person.id",
                path="data.person_id",
                row_path_id="data",
                type="uuid",
            ),
            CatalogField(
                ref="person.name",
                path="data.name",
                row_path_id="data",
                type="string",
            ),
        ),
        candidate_keys=_candidate_key("person", "person_id", "person.id"),
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
            ),
            CatalogField(
                ref="field.data.name",
                path="data.name",
                row_path_id="data",
                type="string",
            ),
        ),
        candidate_keys=_candidate_key(
            "location",
            "location_id",
            "field.data.location_id",
            context_field_refs=("field.data.name",),
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
            ),
            CatalogField(
                ref="field.data.display_name",
                path="data.display_name",
                row_path_id="data",
                type="string",
            ),
        ),
        candidate_keys=_candidate_key(
            "location",
            "location_id",
            "field.data.location_id",
            context_field_refs=("field.data.display_name",),
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
            ),
            CatalogField(
                ref="field.data.area.name",
                path="data.area.name",
                row_path_id="data",
                type="string",
            ),
        ),
        candidate_keys=_candidate_key(
            "location",
            "location_id",
            "field.data.location_id",
        ),
        entity_references=(
            EntityReference(
                id="area_reference",
                target_entity_kind="area",
                target_key_id="primary_key",
                components=(
                    EntityReferenceComponent(
                        target_component_id="area_id",
                        local_field_ref="field.data.area.area_id",
                    ),
                ),
                context_field_refs=("field.data.area.name",),
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
            ),
            CatalogField(
                ref="field.data.name",
                path="data.name",
                row_path_id="data",
                type="string",
            ),
        ),
        candidate_keys=_candidate_key(
            "area",
            "area_id",
            "field.data.area_id",
            context_field_refs=("field.data.name",),
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
            ),
            CatalogField(
                ref="field.data.name",
                path="data.name",
                row_path_id="data",
                type="string",
            ),
        ),
        candidate_keys=_candidate_key(
            "store",
            "store_id",
            "field.data.store_id",
            context_field_refs=("field.data.name",),
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
            ),
            CatalogField(
                ref="field.data_deposits.name",
                path="data.deposits.name",
                row_path_id="data_deposits",
                type="string",
            ),
        ),
        candidate_keys=_candidate_key(
            "store",
            "store_id",
            "field.data_deposits.store_id",
            context_field_refs=("field.data_deposits.name",),
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
            CatalogField(
                ref="field.data.phone_number",
                path="data.phone_number",
                row_path_id="data",
                type="string",
            ),
        ),
        candidate_keys=_candidate_key(
            "staff",
            "staff_id",
            "field.data.staff_id",
            context_field_refs=(
                "field.data.full_name",
                "field.data.first_name",
                "field.data.last_name",
            ),
        ),
    )


def _staff_uuid_only_read() -> EndpointRead:
    return EndpointRead(
        id="list_staff_uuid_list",
        endpoint_name="list_staff_uuid_list",
        resource_names=("staff",),
        params=(
            CatalogParam(
                ref="list_staff_uuid_list.query.name",
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
            ),
        ),
        candidate_keys=_candidate_key(
            "staff",
            "staff_id",
            "field.data.staff_id",
        ),
    )


def _staff_detail_read(*, param_type: str = "string") -> EndpointRead:
    return EndpointRead(
        id="get_staff_detail",
        endpoint_name="get_staff_detail",
        resource_names=("staff",),
        params=(
            CatalogParam(
                ref="get_staff_detail.path.staff_id",
                name="staff_id",
                source=ParamSource.PATH,
                type=param_type,
                required=True,
                entity_target=_entity_target("staff", "staff_id"),
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
            ),
            CatalogField(
                ref="field.data.full_name",
                path="data.full_name",
                row_path_id="data",
                type="string",
            ),
        ),
        candidate_keys=_candidate_key(
            "staff",
            "staff_id",
            "field.data.staff_id",
            context_field_refs=("field.data.full_name",),
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
                entity_target=_entity_target("location", "location_id"),
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
            ),
            CatalogField(
                ref="field.data.name",
                path="data.name",
                row_path_id="data",
                type="string",
            ),
        ),
        candidate_keys=_candidate_key(
            "location",
            "location_id",
            "field.data.location_id",
            context_field_refs=("field.data.name",),
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
                entity_target=_entity_target("staff", "staff_id"),
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
                entity_target=_entity_target("location", "location_id"),
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
