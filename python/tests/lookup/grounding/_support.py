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
    ResourceTypeMatch,
    LookupRequestParameter,
    resolver_fit_question_for_option,
)
from fervis.lookup.grounding.parser import parse_grounding_compatibility
from fervis.lookup.grounding.model import (
    GroundingRequest,
    GroundingRequestedFactCard,
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


class _AllDifferentResourceTypesGroundingModel:
    def generate(self, **kwargs):
        prompt = str(kwargs.get("prompt") or "")
        arguments = _grounding_review_arguments(prompt, selected_by_input={})
        for review in arguments["known_input_binding_reviews"].values():
            review["resource_type_compatibility"] = {
                resource_type: ResourceTypeMatch.DIFFERENT_RESOURCE_TYPE.value
                for resource_type in review["resource_type_compatibility"]
            }
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
        reviews[known_input_id] = {
            "resource_type_basis": "The input identifies the selected resource type.",
            "resource_type_compatibility": {
                resource_type: (
                    ResourceTypeMatch.SAME_RESOURCE_TYPE.value
                    if resource_type in selected_resource_types
                    else ResourceTypeMatch.DIFFERENT_RESOURCE_TYPE.value
                )
                for resource_type in task["shown_resource_types"]
            },
            "identifier_kind_basis": "The lookup is a descriptive identifier.",
            "identifier_kind": "DESCRIPTIVE",
            "option_reviews": {
                option["binding_option_id"]: {
                    "resource_type": option["resource_type"],
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

from tests.lookup.grounding._fixtures import _endpoint_result
