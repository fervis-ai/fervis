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
                        "tool": _conversation_resolution_tool_name_for_payload(
                            arguments
                        ),
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
        if tool_name == "submit_answer_request_contract":
            return {
                "answer": json.dumps(
                    {
                        "tool": "submit_answer_request_contract",
                        "arguments": _question_contract_payload(
                            question_contract,
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
    tasks = {
        item["known_input_id"]: item
        for item in _prompt_json_section(prompt, "Known inputs to ground")[
            "known_input_binding_tasks"
        ]
    }
    option_groups = _prompt_json_section(prompt, "Binding options")[
        "known_input_binding_options"
    ]
    reviews: dict[str, dict[str, Any]] = {}
    for group in option_groups:
        options = group["binding_options"]
        selected = _select_grounding_route_option(
            options,
            value_meaning_hint=str(
                tasks.get(group["known_input_id"], {}).get("value_meaning_hint") or ""
            ),
        )
        reviews[group["known_input_id"]] = {
            "option_reviews": {
                option["binding_option_id"]: {
                    "resolver_fit_question": option["resolver_fit_question"],
                    "because": "Selected by deterministic test model.",
                    "decision": (
                        "CAN_RESOLVE_LOOKUP_TEXT"
                        if option["binding_option_id"] == selected["binding_option_id"]
                        else "CANNOT_RESOLVE_LOOKUP_TEXT"
                    ),
                }
                for option in options
            }
        }
    return {
        "known_time_resolutions": _time_resolution_payload_from_prompt(prompt),
        "known_input_binding_reviews": reviews,
    }


def _time_resolution_payload_from_prompt(prompt: str) -> dict[str, Any]:
    time_tasks = _prompt_json_section(prompt, "Time inputs to resolve")[
        "known_time_resolution_tasks"
    ]
    return {
        task["known_input_id"]: {
            "date_intent": _date_intent_payload(str(task["known_input_text"]))
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
    usable = list(options)
    normalized_hint = value_meaning_hint.casefold().strip()
    for option in usable:
        identity_type = str(
            (option.get("returned_identity") or {}).get("identity_type") or ""
        ).casefold()
        if identity_type and normalized_hint.endswith(identity_type):
            return option
    return usable[0] if usable else options[0]


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
