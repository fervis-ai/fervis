import re

from tests.lookup.orchestrator._payloads import *  # noqa: F403
from tests.lookup.prompt_sections import prompt_section_payload


@dataclass
class _CatalogPort:
    catalog: RelationCatalog

    def build_relation_catalog(self) -> RelationCatalog:
        return self.catalog


@dataclass
class _DataAccessPort:
    responses: dict[str, Any]
    requests: list[dict[str, Any]] = field(default_factory=list)

    def read(self, *, endpoint_name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.requests.append({"endpointName": endpoint_name, "args": dict(args)})
        return {
            "endpointName": endpoint_name,
            "responseStatus": 200,
            "responseBody": self.responses[endpoint_name],
            "truncated": False,
            "pageCount": 1,
        }


@dataclass
class _StatusDataAccessPort:
    responses: dict[str, dict[str, Any]]
    requests: list[dict[str, Any]] = field(default_factory=list)

    def read(self, *, endpoint_name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.requests.append({"endpointName": endpoint_name, "args": dict(args)})
        return dict(self.responses[endpoint_name])


@dataclass
class _PlannerPort:
    plan: FactPlan
    question_contract: QuestionContract | None = None
    conversation_resolution: Any = None
    query_enrichment: dict[str, Any] | None = None
    read_eligibility_retention_specs: (
        tuple[ReadEligibilityRetentionSpec, ...] | None
    ) = None
    calls: int = 0
    prompts: list[str] = field(default_factory=list)
    system_prompts: list[str] = field(default_factory=list)
    source_binding_selection_prompt: str = ""

    def generate(
        self,
        *,
        provider: str,
        prompt: str,
        max_thinking_tokens: int,
        system_prompt: str = "",
        output_mode: Any = None,
        tool_specs: tuple[Any, ...] = (),
    ) -> dict[str, Any]:
        self.calls += 1
        self.system_prompts.append(system_prompt)
        self.prompts.append(prompt)
        if _offered_conversation_resolution_tool_names(tool_specs):
            arguments = _conversation_resolution_payload_from_response(
                prompt,
                self.conversation_resolution,
            )
            return {
                "answer": json.dumps(
                    {
                        "tool": CONVERSATION_RESOLUTION_TOOL_NAME,
                        "arguments": arguments,
                    },
                    default=str,
                ),
                "usage": {
                    "inputTokens": 1,
                    "outputTokens": 1,
                    "thinkingTokens": 0,
                    "costUsd": 0,
                },
            }
        tool_name = tool_specs[0].name if tool_specs else ""
        question_contract = self.question_contract or _question_contract_for_plan(
            self.plan,
            description=_current_question_from_prompt(prompt) or None,
        )
        question_contract = _question_contract_with_answer_expression_from_fact_plan(
            question_contract,
            _plan_payload(self.plan, question_contract=question_contract),
        )
        if tool_name == "submit_question_contract_outcome":
            return {
                "answer": json.dumps(
                    {
                        "tool": "submit_question_contract_outcome",
                        "arguments": _question_contract_decision(
                            _question_contract_payload(question_contract)
                        ),
                    },
                    default=str,
                ),
                "usage": {
                    "inputTokens": 1,
                    "outputTokens": 1,
                    "thinkingTokens": 0,
                    "costUsd": 0,
                },
            }
        if tool_name == "submit_query_enrichment":
            arguments = self.query_enrichment or _query_enrichment_payload_from_prompt(
                prompt
            )
            return {
                "answer": json.dumps(
                    {
                        "tool": "submit_query_enrichment",
                        "arguments": arguments,
                    },
                    default=str,
                ),
                "usage": {
                    "inputTokens": 1,
                    "outputTokens": 1,
                    "thinkingTokens": 0,
                    "costUsd": 0,
                },
            }
        if tool_name == "submit_read_eligibility":
            if self.read_eligibility_retention_specs is not None:
                return read_eligibility_response_from_prompt(
                    prompt,
                    retention_specs=self.read_eligibility_retention_specs,
                )
            return read_eligibility_response_from_fact_plan(
                prompt,
                _plan_payload(
                    self.plan,
                    question_contract=question_contract,
                ),
            )
        if tool_name == "submit_grounding":
            return {
                "answer": json.dumps(
                    {
                        "tool": "submit_grounding",
                        "arguments": _grounding_payload_from_prompt(prompt),
                    },
                    default=str,
                ),
                "usage": {
                    "inputTokens": 1,
                    "outputTokens": 1,
                    "thinkingTokens": 0,
                    "costUsd": 0,
                },
            }
        plan_payload = _plan_payload(
            self.plan,
            question_contract=question_contract,
        )
        if tool_name == "submit_source_binding":
            self.source_binding_selection_prompt = prompt
            return {
                "answer": json.dumps(
                    {
                        "tool": "submit_source_binding",
                        "arguments": source_binding_payload_from_fact_plan(
                            plan_payload,
                            prompt=prompt,
                        ),
                    },
                    default=str,
                ),
                "usage": {
                    "inputTokens": 1,
                    "outputTokens": 1,
                    "thinkingTokens": 0,
                    "costUsd": 0,
                },
            }
        if tool_name == "submit_source_alignment_reviews":
            return {
                "answer": json.dumps(
                    {
                        "tool": "submit_source_alignment_reviews",
                        "arguments": plan_selection_payload_from_fact_plan(
                            plan_payload,
                            prompt=prompt,
                        ),
                    },
                    default=str,
                ),
                "usage": {
                    "inputTokens": 1,
                    "outputTokens": 1,
                    "thinkingTokens": 0,
                    "costUsd": 0,
                },
            }
        if tool_name != "submit_pattern_fact_plan":
            raise AssertionError(f"unexpected tool: {tool_name}")
        return {
            "answer": json.dumps(
                {
                    "tool": tool_name,
                    "arguments": bound_fact_plan_payload_from_fact_plan(
                        plan_payload,
                        prompt=prompt,
                        provider_schema=(
                            tool_specs[0].input_schema if tool_specs else None
                        ),
                    ),
                },
                default=str,
            ),
            "usage": {
                "inputTokens": 1,
                "outputTokens": 1,
                "thinkingTokens": 0,
                "costUsd": 0,
            },
        }


def _grounding_payload_from_prompt(prompt: str) -> dict[str, Any]:
    tasks = _prompt_json_section(prompt, "Known input binding tasks")[
        "known_input_binding_tasks"
    ]
    reviews: dict[str, dict[str, Any]] = {}
    for task in tasks:
        options = task["binding_options"]
        selected = _select_grounding_route_option(
            options,
            value_meaning_hint=str(task.get("value_meaning_hint") or ""),
        )
        reviews[task["known_input_id"]] = _grounding_review_for_task(
            task,
            compatible_option_ids={str(selected["binding_option_id"])},
        )
    return {
        "known_time_resolutions": _time_resolution_payload_from_prompt(prompt),
        "known_input_binding_reviews": reviews,
    }


def _grounding_review_for_task(
    task: dict[str, Any],
    *,
    compatible_option_ids: set[str],
    request_values_for_option: Any = None,
    match_fields_for_option: Any = None,
) -> dict[str, Any]:
    options = task["binding_options"]
    compatible_options = tuple(
        option
        for option in options
        if str(option["binding_option_id"]) in compatible_option_ids
    )
    if not compatible_options:
        compatible_resource_types: set[str] = set()
        identifier_kind = "DESCRIPTIVE"
    else:
        compatible_resource_types = {
            str(option["resource_type"]) for option in compatible_options
        }
        purposes = {str(option.get("purpose") or "") for option in compatible_options}
        if len(purposes) != 1:
            raise AssertionError("scripted compatible resolvers use different purposes")
        identifier_kind = (
            "PRIMARY_KEY" if purposes == {"identity_validation"} else "DESCRIPTIVE"
        )
    request_values = request_values_for_option or _resolver_request_values
    match_fields = match_fields_for_option or _resolver_match_fields
    lookup_text = str(task["lookup_text"])
    return {
        "resource_type_basis": "The input's resource meaning was reviewed.",
        "resource_type_compatibility": {
            resource_type: (
                "SAME_RESOURCE_TYPE"
                if resource_type in compatible_resource_types
                else "DIFFERENT_RESOURCE_TYPE"
            )
            for resource_type in task["shown_resource_types"]
        },
        "identifier_kind_basis": (
            f"The input uses a {identifier_kind.lower()} identifier."
        ),
        "identifier_kind": identifier_kind,
        "option_reviews": {
            option["binding_option_id"]: _grounding_option_review(
                option,
                lookup_text=lookup_text,
                compatible=str(option["binding_option_id"]) in compatible_option_ids,
                request_values_for_option=request_values,
                match_fields_for_option=match_fields,
            )
            for option in options
        },
    }


def _grounding_option_review(
    option: dict[str, Any],
    *,
    lookup_text: str,
    compatible: bool,
    request_values_for_option: Any,
    match_fields_for_option: Any,
) -> dict[str, Any]:
    request_values = (
        request_values_for_option(option, lookup_text=lookup_text)
        if compatible
        else {}
    )
    return {
        "resource_type": str(option["resource_type"]),
        "resolver_fit_question": option["resolver_fit_question"],
        "because": "The declared route capability was reviewed.",
        "resolution": {
            "decision": (
                "CAN_RESOLVE_LOOKUP_TEXT"
                if compatible
                else "CANNOT_RESOLVE_LOOKUP_TEXT"
            ),
            "lookup_request_params": [
                {"param_ref": param_ref, "value": value}
                for param_ref, value in request_values.items()
            ],
            "returned_identity_verification_fields": (
                match_fields_for_option(option) if compatible else []
            ),
        },
    }


def _time_resolution_payload_from_prompt(prompt: str) -> dict[str, Any]:
    time_tasks = _prompt_json_section(prompt, "Time inputs to resolve")[
        "known_time_resolution_tasks"
    ]
    return {
        task["known_input_id"]: {
            "date_intent": _date_intent_payload(str(task["time_expression"]))
        }
        for task in time_tasks
    }


def _date_intent_payload(text: str) -> dict[str, Any]:
    lowered = text.casefold().strip()
    last_days = re.fullmatch(r"last (\d+) days", lowered)
    if last_days:
        return _window_time_intent(
            text,
            unit="day",
            count=int(last_days.group(1)),
            direction="past",
        )
    if lowered == "today":
        return _point_relative_time_intent(text, offset=0)
    if lowered in {"yesterday", "the day before"}:
        return _point_relative_time_intent(text, offset=-1)
    if lowered == "the day before yesterday":
        return _point_relative_time_intent(text, offset=-2)
    if lowered == "tomorrow":
        return _point_relative_time_intent(text, offset=1)
    if lowered == "this month":
        return _period_relative_time_intent(text, unit="month", offset=0)
    if lowered == "last month":
        return _period_relative_time_intent(text, unit="month", offset=-1)
    if lowered == "this week":
        return _period_relative_time_intent(text, unit="week", offset=0)
    if lowered.startswith("q") and lowered[1:].isdigit():
        return _period_named_time_intent(text, unit="quarter", value=int(lowered[1:]))
    raise AssertionError(f"test grounding model needs date intent for {text!r}")


def _neutral_time_intent_fields() -> dict[str, Any]:
    return {
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
    }


def _point_relative_time_intent(text: str, *, offset: int) -> dict[str, Any]:
    intent = _neutral_time_intent_fields()
    intent.update(
        {
            "time_shape": "point_relative",
            "unit": "day",
            "mode": "none",
            "relative_offset": offset,
        }
    )
    return {"expression": text, "intent": intent}


def _period_relative_time_intent(
    text: str,
    *,
    unit: str,
    offset: int,
) -> dict[str, Any]:
    intent = _neutral_time_intent_fields()
    intent.update(
        {
            "time_shape": "period_relative",
            "unit": unit,
            "mode": "full",
            "relative_offset": offset,
        }
    )
    return {"expression": text, "intent": intent}


def _period_named_time_intent(
    text: str,
    *,
    unit: str,
    value: int,
) -> dict[str, Any]:
    intent = _neutral_time_intent_fields()
    intent.update(
        {
            "time_shape": "period_named",
            "unit": unit,
            "mode": "full",
            "named_value": value,
            "year_policy": "most_recent",
        }
    )
    return {"expression": text, "intent": intent}


def _window_time_intent(
    text: str,
    *,
    unit: str,
    count: int,
    direction: str,
) -> dict[str, Any]:
    intent = _neutral_time_intent_fields()
    intent.update(
        {
            "time_shape": "window",
            "unit": unit,
            "mode": "none",
            "count": count,
            "direction": direction,
        }
    )
    return {"expression": text, "intent": intent}


def _select_grounding_route_option(
    options: list[dict[str, Any]],
    *,
    value_meaning_hint: str,
) -> dict[str, Any]:
    normalized_hint = value_meaning_hint.casefold().strip()
    name_search_options = tuple(
        option
        for option in options
        if any(
            str(parameter.get("name") or "") in {"name", "display_name"}
            for parameter in option["api_read"]["input_params"]
        )
    )
    descriptive_match_options = tuple(
        option
        for option in options
        if not any(
            str(parameter.get("source") or "") == "path"
            for parameter in option["api_read"]["input_params"]
        )
        and any(
            "name" in str(field.get("field_id") or "")
            for row in option["api_read"]["response_rows"]
            for field in row["fields"]
        )
    )
    options_by_lookup_fit = (
        name_search_options or descriptive_match_options or tuple(options)
    )
    for option in options_by_lookup_fit:
        result = option.get("canonical_result") or {}
        entity_kind = str(result.get("entity_kind") or "").casefold()
        if normalized_hint and entity_kind == normalized_hint:
            return option
    for option in options_by_lookup_fit:
        result = option.get("canonical_result") or {}
        entity_kind = str(result.get("entity_kind") or "").casefold()
        if normalized_hint and any(
            word in entity_kind for word in normalized_hint.split()
        ):
            return option
    return options_by_lookup_fit[0]


def _resolver_request_values(
    option: dict[str, Any],
    *,
    lookup_text: str,
) -> dict[str, Any]:
    params = option["api_read"]["input_params"]
    preferred = next(
        (
            parameter
            for parameter in params
            if str(parameter.get("name") or "") in {"name", "display_name"}
        ),
        params[0] if params else None,
    )
    if preferred is None:
        return {}
    return {str(preferred["param_ref"]): lookup_text}


def _resolver_match_fields(option: dict[str, Any]) -> list[str]:
    fields = [
        field for row in option["api_read"]["response_rows"] for field in row["fields"]
    ]
    named_fields = [
        str(field["path"])
        for field in fields
        if str(field["field_id"])
        in {
            "name",
            "display_name",
            "first_name",
            "last_name",
            "full_name",
        }
    ]
    if named_fields:
        return named_fields
    return [
        str(component["field_path"])
        for component in option["canonical_result"]["components"]
    ]


def _prompt_json_section(prompt: str, section_name: str) -> dict[str, Any]:
    return prompt_section_payload(prompt, section_name)


def _ports(
    *,
    plan: FactPlan,
    catalog: RelationCatalog,
    responses: dict[str, Any],
    question_contract: QuestionContract | None = None,
    conversation_resolution: dict[str, Any] | None = None,
    query_enrichment: dict[str, Any] | None = None,
    read_eligibility_retention_specs: (
        tuple[ReadEligibilityRetentionSpec, ...] | None
    ) = None,
) -> LookupRuntimePorts:
    return LookupRuntimePorts(
        relation_catalog_port=_CatalogPort(catalog),
        data_access_port=_DataAccessPort(responses),
        planner_model_port=_PlannerPort(
            plan,
            question_contract=question_contract,
            conversation_resolution=conversation_resolution,
            query_enrichment=query_enrichment,
            read_eligibility_retention_specs=read_eligibility_retention_specs,
        ),
    )


__all__ = tuple(name for name in globals() if not name.startswith("__"))
