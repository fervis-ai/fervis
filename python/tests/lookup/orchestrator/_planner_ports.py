from tests.lookup.orchestrator._runtime_ports import *  # noqa: F403
from tests.lookup.orchestrator._runtime_ports import (
    _grounding_payload_from_prompt,
)
from tests.lookup.source_binding_helpers import (
    source_binding_payload_from_fact_plan_with_invocation_overrides,
)
from tests.testkit.question_contract_provider import (
    provider_membership_tests,
    provider_question_input_ownership,
)


def _provider_population_without_input_uses(
    *,
    description: str,
    subject_text: str,
) -> dict[str, Any]:
    payload = default_answer_population(
        description=description,
        subject_text=subject_text,
        instance_interpretation=RequestedFactAnswerSubject(
            subject_text=subject_text
        ).instance_interpretation,
    ).to_question_contract_dict()
    payload["membership_tests"] = provider_membership_tests(
        payload["membership_tests"],
        ownership=provider_question_input_ownership(),
    )
    return payload


def _offered_conversation_resolution_tool_names(
    tool_specs: tuple[Any, ...],
) -> tuple[str, ...]:
    return tuple(
        tool.name
        for tool in tool_specs
        if tool.name in CONVERSATION_RESOLUTION_TOOL_NAMES
    )


def _select_conversation_resolution_tool_name(
    tool_specs: tuple[Any, ...],
    *,
    responses: dict[str, dict[str, Any]] | None = None,
) -> str:
    offered = _offered_conversation_resolution_tool_names(tool_specs)
    if not offered:
        return ""
    for name in offered:
        if responses and name in responses:
            return name
    return offered[0]


@dataclass
class _RawPlannerPort:
    arguments: dict[str, Any]
    source_binding_arguments: dict[str, Any] | None = None
    source_binding_invocation_overrides: tuple[dict[str, Any], ...] = ()
    question_contract: QuestionContract | None = None
    query_enrichment: dict[str, Any] | None = None
    conversation_resolution: Any = None
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
        question_contract = self.question_contract or _question_contract_for_arguments(
            self.arguments,
            description=_current_question_from_prompt(prompt),
        )
        question_contract = _question_contract_with_answer_expression_from_fact_plan(
            question_contract,
            self.arguments,
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
            return {
                "answer": json.dumps(
                    {
                        "tool": "submit_query_enrichment",
                        "arguments": self.query_enrichment
                        or _query_enrichment_payload_from_prompt(prompt),
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
        if tool_name == "submit_read_eligibility":
            if self.read_eligibility_retention_specs is not None:
                return read_eligibility_response_from_prompt(
                    prompt,
                    retention_specs=self.read_eligibility_retention_specs,
                )
            return read_eligibility_response_from_fact_plan(
                prompt,
                _canonicalized_plan_payload(
                    self.arguments,
                    question_contract=question_contract,
                ),
            )
        plan_payload = _canonicalized_plan_payload(
            self.arguments,
            question_contract=question_contract,
        )
        if tool_name == "submit_source_binding":
            self.source_binding_selection_prompt = prompt
            source_binding_plan_payload = _canonicalized_plan_payload(
                self.source_binding_arguments or self.arguments,
                question_contract=question_contract,
            )
            source_binding_payload = (
                source_binding_payload_from_fact_plan_with_invocation_overrides(
                    source_binding_plan_payload,
                    prompt=prompt,
                    invocation_overrides=self.source_binding_invocation_overrides,
                )
                if self.source_binding_invocation_overrides
                else source_binding_payload_from_fact_plan(
                    source_binding_plan_payload,
                    prompt=prompt,
                )
            )
            return {
                "answer": json.dumps(
                    {
                        "tool": "submit_source_binding",
                        "arguments": source_binding_payload,
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


@dataclass
class _ToolNamePlannerPort:
    responses: dict[str, dict[str, Any]]
    read_eligibility_retention_specs: (
        tuple[ReadEligibilityRetentionSpec, ...] | None
    ) = None
    source_binding_invocation_overrides: tuple[dict[str, Any], ...] = ()
    calls: int = 0
    prompts: list[str] = field(default_factory=list)
    system_prompts: list[str] = field(default_factory=list)
    tool_names: list[str] = field(default_factory=list)
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
        del provider, max_thinking_tokens, output_mode
        self.calls += 1
        self.system_prompts.append(system_prompt)
        self.prompts.append(prompt)
        tool_name = _select_conversation_resolution_tool_name(
            tool_specs,
            responses=self.responses,
        ) or (tool_specs[0].name if tool_specs else "")
        self.tool_names.append(tool_name)
        if (
            tool_name in CONVERSATION_RESOLUTION_TOOL_NAMES
            and tool_name not in self.responses
        ):
            return {
                "answer": json.dumps(
                    {
                        "tool": CONVERSATION_RESOLUTION_TOOL_NAME,
                        "arguments": _conversation_resolution_payload_from_prompt(
                            prompt
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
        if tool_name == "submit_query_enrichment" and tool_name not in self.responses:
            arguments = _query_enrichment_payload_from_prompt(prompt)
            return {
                "answer": json.dumps(
                    {
                        "tool": tool_name,
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
        if tool_name == "submit_read_eligibility" and tool_name not in self.responses:
            if self.read_eligibility_retention_specs is not None:
                return read_eligibility_response_from_prompt(
                    prompt,
                    retention_specs=self.read_eligibility_retention_specs,
                )
            return read_eligibility_response_from_fact_plan(
                prompt,
                self.responses.get("submit_pattern_fact_plan", {}),
            )
        if tool_name == "submit_grounding" and tool_name not in self.responses:
            arguments = _grounding_payload_from_prompt(prompt)
            return {
                "answer": json.dumps(
                    {
                        "tool": tool_name,
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
        if tool_name == "submit_source_binding" and tool_name not in self.responses:
            self.source_binding_selection_prompt = prompt
            plan_selection_response = self.responses.get(
                "submit_source_alignment_reviews",
                {},
            )
            if (
                isinstance(plan_selection_response, dict)
                and (plan_selection_response.get("outcome") or {}).get("kind")
                == "impossible"
            ):
                arguments = plan_selection_response
            else:
                fact_plan = self.responses.get("submit_pattern_fact_plan", {})
                if self.source_binding_invocation_overrides:
                    arguments = (
                        source_binding_payload_from_fact_plan_with_invocation_overrides(
                            fact_plan,
                            prompt=prompt,
                            invocation_overrides=(
                                self.source_binding_invocation_overrides
                            ),
                        )
                    )
                else:
                    arguments = source_binding_payload_from_fact_plan(
                        fact_plan,
                        prompt=prompt,
                    )
            return {
                "answer": json.dumps(
                    {
                        "tool": tool_name,
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
        if (
            tool_name == "submit_source_alignment_reviews"
            and tool_name not in self.responses
        ):
            arguments = plan_selection_payload_from_fact_plan(
                self.responses.get("submit_pattern_fact_plan", {}),
                prompt=prompt,
            )
            return {
                "answer": json.dumps(
                    {
                        "tool": tool_name,
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
        if tool_name not in self.responses:
            raise AssertionError(f"unexpected tool: {tool_name}")
        arguments = _tool_name_planner_arguments(
            self.responses[tool_name],
            prompt=prompt,
            tool_specs=tool_specs,
        )
        if tool_name in CONVERSATION_RESOLUTION_TOOL_NAMES:
            arguments = _conversation_resolution_payload_from_response(
                prompt,
                arguments,
            )
        if tool_name == "submit_question_contract_outcome":
            arguments = _question_contract_decision(arguments)
        if tool_name == "submit_source_binding":
            self.source_binding_selection_prompt = prompt
            arguments = source_binding_payload_for_one_call(arguments, prompt=prompt)
        if tool_name == "submit_source_alignment_reviews":
            arguments = plan_selection_payload_from_fact_plan(arguments, prompt=prompt)
        if tool_name == "submit_pattern_fact_plan":
            arguments = bound_fact_plan_payload_from_fact_plan(
                arguments,
                prompt=prompt,
                provider_schema=tool_specs[0].input_schema if tool_specs else None,
            )
        return {
            "answer": json.dumps(
                {
                    "tool": tool_name,
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


def _tool_name_planner_arguments(
    response: Any,
    *,
    prompt: str,
    tool_specs: tuple[Any, ...],
) -> dict[str, Any]:
    if callable(response):
        return response(prompt=prompt, tool_specs=tool_specs)
    return response


@dataclass
class _PromptSurfacePlannerPort:
    calls: int = 0
    prompts: list[str] = field(default_factory=list)
    fact_plan_field_id: str = ""
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
        del provider, max_thinking_tokens, system_prompt, output_mode
        self.calls += 1
        self.prompts.append(prompt)
        tool_name = tool_specs[0].name if tool_specs else ""
        if tool_name == "submit_question_contract_outcome":
            arguments = {
                "kind": "question_contract",
                "answer_requests_count": 1,
                "question_inputs": [],
                "answer_requests": [
                    {
                        "answer_fact": "salespeople with sales",
                        "answer_expression": {"family": "list_rows"},
                        "answer_subject": _answer_subject_payload("salespeople"),
                        "answer_population": _provider_population_without_input_uses(
                            description="salespeople with sales",
                            subject_text="salespeople",
                        ),
                        "answer_outputs": [
                            {"description": "staff name", "role": "ANSWER_VALUE"}
                        ],
                        "question_input_uses": [],
                    }
                ],
                "question_input_inventory_check": {
                    "all_input_like_phrases_declared": True,
                },
            }
        elif tool_name in CONVERSATION_RESOLUTION_TOOL_NAMES:
            arguments = _conversation_resolution_payload_from_prompt(prompt)
        elif tool_name == "submit_query_enrichment":
            arguments = _query_enrichment_payload_from_prompt(prompt)
        elif tool_name == "submit_read_eligibility":
            return read_eligibility_response_for_retained_fields(
                prompt,
                answer_value_fields=("staff_name",),
            )
        elif tool_name == "submit_source_binding":
            self.source_binding_selection_prompt = prompt
            candidate = source_candidate_with_fields(
                prompt,
                requested_fact_id="fact_1",
                required=("staff_name",),
            )
            binding_target_id = source_binding_target_id_for_candidate(
                prompt,
                requested_fact_id="fact_1",
                source_candidate_id=str(candidate["source_candidate_id"]),
                plan_shape="list_rows",
            )
            arguments = {
                "outcome": {
                    "kind": "source_bindings",
                    "bindings_for_fact_1": {
                        "plan_shape": "list_rows",
                        "primary": {
                            "binding_target_id": binding_target_id,
                            "answer_population": source_candidate_answer_population(
                                prompt,
                                binding_target_id=binding_target_id,
                            ),
                            "fulfillment_decisions": source_fulfills_for_candidate(
                                candidate,
                                field_ids=("staff_name",),
                            ),
                            "param_decisions": {},
                            "finite_choice_param_reviews": {},
                        },
                    },
                }
            }
            arguments = source_binding_payload_for_one_call(arguments, prompt=prompt)
        elif tool_name == "submit_source_alignment_reviews":
            arguments = plan_selection_payload_from_fact_plan(
                {
                    "outcome": {
                        "kind": "fact_plan",
                        "answers": [
                            {
                                "requested_fact_id": "fact_1",
                                "pattern": "list_rows",
                                "source": {"kind": "read", "read_id": "sales"},
                                "output_fields": [{"field_id": "staff_name"}],
                            }
                        ],
                    }
                },
                prompt=prompt,
            )
        elif tool_name == "submit_pattern_fact_plan":
            self.fact_plan_field_id = (
                "staff_name" if '"field_id": "staff_name"' in prompt else "sale_id"
            )
            arguments = {
                "outcome": {
                    "kind": "fact_plan",
                    "answers": [
                        {
                            "requested_fact_id": "fact_1",
                            "answer_output_ids": ["answer_1"],
                            "pattern": "list_rows",
                            "source_binding_id": "sb_1",
                            "output_fields": [{"field_id": self.fact_plan_field_id}],
                        }
                    ],
                }
            }
        else:
            raise AssertionError(f"unexpected tool: {tool_name}")
        if tool_name == "submit_question_contract_outcome":
            arguments = _question_contract_decision(arguments)
        return {
            "answer": json.dumps(
                {"tool": tool_name, "arguments": arguments},
                default=str,
            ),
            "usage": {
                "inputTokens": 1,
                "outputTokens": 1,
                "thinkingTokens": 0,
                "costUsd": 0,
            },
        }


def _source_candidate_with_fulfillment_field(
    payload: dict[str, Any],
    *,
    field_id: str,
) -> dict[str, Any]:
    for fact_sources in payload.get("requested_fact_sources") or ():
        if not isinstance(fact_sources, dict):
            continue
        for context in fact_sources.get("source_contexts") or ():
            if not isinstance(context, dict):
                continue
            for candidate in context.get("source_options") or ():
                if not isinstance(candidate, dict):
                    continue
                if field_id in _source_candidate_fulfillment_fields(candidate):
                    return candidate
    raise AssertionError(f"source candidate missing fulfillment field: {field_id}")


def _source_candidate_fulfillment_fields(candidate: dict[str, Any]) -> set[str]:
    fields = {
        str(item.get("field_id") or "")
        for support_set in candidate.get("fulfillment_choices") or ()
        if isinstance(support_set, dict)
        for slot in support_set.get("fulfillment_slots") or ()
        if isinstance(slot, dict)
        for key in (
            "metric_measure_evidence",
            "row_count_basis_evidence",
            "scope_evidence",
            "group_key_evidence",
        )
        for item in slot.get(key) or ()
        if isinstance(item, dict)
    }
    for row in candidate.get("response_rows") or ():
        if not isinstance(row, dict):
            continue
        for item in row.get("fields") or ():
            if isinstance(item, dict):
                fields.add(str(item.get("field_id") or ""))
    for item in candidate.get("fields") or ():
        if isinstance(item, dict):
            fields.add(str(item.get("field_id") or ""))
    return fields


def _source_candidate_population_binding_id(candidate: dict[str, Any]) -> str:
    for binding in candidate.get("population_bindings") or ():
        if not isinstance(binding, dict):
            continue
        binding_id = str(binding.get("population_binding_id") or "")
        if binding_id:
            return binding_id
    raise AssertionError("source candidate missing population binding")


@dataclass
class _QuestionIntentAwarePlannerPort:
    plan: FactPlan
    prior_reference_id: str = "run_prior_total.value.sales_total"
    required_prompt_fragments: tuple[str, ...] = ()
    calls: int = 0
    prompts: list[str] = field(default_factory=list)
    system_prompts: list[str] = field(default_factory=list)
    tool_names: list[str] = field(default_factory=list)
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
        del provider, max_thinking_tokens, output_mode
        self.calls += 1
        self.prompts.append(prompt)
        self.system_prompts.append(system_prompt)
        tool_name = _select_conversation_resolution_tool_name(
            tool_specs,
        ) or (tool_specs[0].name if tool_specs else "")
        self.tool_names.append(tool_name)
        question_contract = _question_contract_for_plan(
            self.plan,
            description="total for the prior referenced sales amount",
        )
        if tool_name in CONVERSATION_RESOLUTION_TOOL_NAMES:
            arguments = _conversation_resolution_payload_using_memory(
                prompt,
                contextualized_question="How much is the prior referenced sales amount in total?",
                actual_text="that",
                retained_part_ids=("output:1",),
            )
            return {
                "answer": json.dumps(
                    {"tool": tool_name, "arguments": arguments},
                    default=str,
                ),
                "usage": {
                    "inputTokens": 1,
                    "outputTokens": 1,
                    "thinkingTokens": 0,
                    "costUsd": 0,
                },
            }
        if tool_name == "submit_question_contract_outcome":
            if self.required_prompt_fragments and any(
                fragment not in prompt for fragment in self.required_prompt_fragments
            ):
                raise AssertionError("question intent context missing from prompt")
            else:
                arguments = {
                    "kind": "question_contract",
                    "answer_requests_count": 1,
                    "question_inputs": [],
                    "answer_requests": [
                        {
                            "answer_fact": "total for the prior referenced sales amount",
                            "answer_expression": {"family": "list_rows"},
                            "answer_subject": _answer_subject_payload("total"),
                            "answer_population": _provider_population_without_input_uses(
                                description="total for the prior referenced sales amount",
                                subject_text="total",
                            ),
                            "answer_outputs": [
                                {
                                    "description": "metric_total",
                                    "role": "ANSWER_VALUE",
                                }
                            ],
                            "question_input_uses": [],
                        }
                    ],
                    "question_input_inventory_check": {
                        "all_input_like_phrases_declared": True,
                    },
                }
        elif tool_name == "submit_query_enrichment":
            arguments = _query_enrichment_payload_from_prompt(prompt)
        elif tool_name == "submit_read_eligibility":
            plan_payload = _plan_payload(self.plan, question_contract=question_contract)
            return read_eligibility_response_from_fact_plan(prompt, plan_payload)
        elif tool_name == "submit_source_binding":
            self.source_binding_selection_prompt = prompt
            plan_payload = _plan_payload(self.plan, question_contract=question_contract)
            arguments = source_binding_payload_from_fact_plan(
                plan_payload,
                prompt=prompt,
            )
        elif tool_name == "submit_source_alignment_reviews":
            plan_payload = _plan_payload(self.plan, question_contract=question_contract)
            arguments = plan_selection_payload_from_fact_plan(
                plan_payload,
                prompt=prompt,
            )
        elif tool_name == "submit_pattern_fact_plan":
            plan_payload = _plan_payload(self.plan, question_contract=question_contract)
            arguments = bound_fact_plan_payload_from_fact_plan(
                plan_payload,
                prompt=prompt,
                provider_schema=tool_specs[0].input_schema if tool_specs else None,
            )
        else:
            raise AssertionError(f"unexpected tool: {tool_name}")
        if tool_name == "submit_question_contract_outcome":
            arguments = _question_contract_decision(arguments)
        return {
            "answer": json.dumps(
                {"tool": tool_name, "arguments": arguments},
                default=str,
            ),
            "usage": {
                "inputTokens": 1,
                "outputTokens": 1,
                "thinkingTokens": 0,
                "costUsd": 0,
            },
        }


@dataclass
class _ClarificationBiasedPlannerPort:
    plan: FactPlan
    calls: int = 0
    tool_names: list[str] = field(default_factory=list)
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
        del provider, system_prompt, max_thinking_tokens, output_mode
        self.calls += 1
        offered = {tool.name: tool for tool in tool_specs}
        conversation_resolution_payload = _conversation_resolution_payload_from_prompt(
            prompt
        )
        conversation_tool_name = _select_conversation_resolution_tool_name(
            tool_specs,
        )
        if conversation_tool_name:
            tool_name = conversation_tool_name
            arguments = conversation_resolution_payload
        elif "submit_question_contract_outcome" in offered:
            tool_name = "submit_question_contract_outcome"
            question_contract = _question_contract_for_plan(
                self.plan,
                description=_current_question_from_prompt(prompt) or None,
            )
            arguments = _question_contract_decision(
                _question_contract_payload(question_contract)
            )
        elif "submit_query_enrichment" in offered:
            tool_name = "submit_query_enrichment"
            arguments = _query_enrichment_payload_from_prompt(prompt)
        elif "submit_read_eligibility" in offered:
            self.tool_names.append("submit_read_eligibility")
            question_contract = _question_contract_for_plan(
                self.plan,
                description=_current_question_from_prompt(prompt) or None,
            )
            return read_eligibility_response_from_fact_plan(
                prompt,
                _plan_payload(self.plan, question_contract=question_contract),
            )
        elif "submit_source_binding" in offered:
            tool_name = "submit_source_binding"
            question_contract = _question_contract_for_plan(
                self.plan,
                description=_current_question_from_prompt(prompt) or None,
            )
            self.source_binding_selection_prompt = prompt
            arguments = source_binding_payload_from_fact_plan(
                _plan_payload(self.plan, question_contract=question_contract),
                prompt=prompt,
            )
        elif "submit_source_alignment_reviews" in offered:
            tool_name = "submit_source_alignment_reviews"
            question_contract = _question_contract_for_plan(
                self.plan,
                description=_current_question_from_prompt(prompt) or None,
            )
            arguments = plan_selection_payload_from_fact_plan(
                _plan_payload(self.plan, question_contract=question_contract),
                prompt=prompt,
            )
        else:
            tool_name = "submit_pattern_fact_plan"
            question_contract = _question_contract_for_plan(
                self.plan,
                description=_current_question_from_prompt(prompt) or None,
            )
            arguments = bound_fact_plan_payload_from_fact_plan(
                _plan_payload(self.plan, question_contract=question_contract),
                prompt=prompt,
                provider_schema=tool_specs[0].input_schema if tool_specs else None,
            )
        self.tool_names.append(tool_name)
        return {
            "answer": json.dumps(
                {"tool": tool_name, "arguments": arguments},
                default=str,
            ),
            "usage": {
                "inputTokens": 1,
                "outputTokens": 1,
                "thinkingTokens": 0,
                "costUsd": 0,
            },
        }


@dataclass
class _FailingPlannerPort:
    calls: int = 0

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
        del system_prompt
        self.calls += 1
        raise RuntimeError("provider unavailable")


@dataclass
class _TimeoutPlannerPort:
    calls: int = 0

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
        del system_prompt, prompt, max_thinking_tokens, output_mode, tool_specs
        self.calls += 1
        raise api_errors.Unavailable.llm_api_timeout(
            provider=provider,
            reason="provider timed out",
            error_class="APITimeoutError",
        )


def _fact_plan_prompt(planner: Any) -> str:
    for prompt in planner.prompts:
        if "We are currently on the pattern fact planning step." in prompt:
            return prompt
    raise AssertionError("fact planning prompt was not captured")


def _source_binding_prompt(planner: Any) -> str:
    for prompt in planner.prompts:
        if "Candidate evidence sources:" in prompt:
            return prompt
    raise AssertionError("source binding prompt was not captured")


def _plan_selection_prompt(planner: Any) -> str:
    for prompt in planner.prompts:
        if "Source alignment reviews:" in prompt:
            return prompt
    raise AssertionError("plan selection prompt was not captured")


def _query_enrichment_prompt(planner: Any) -> str:
    for prompt in planner.prompts:
        if "We are currently on the query enrichment step." in prompt:
            return prompt
    raise AssertionError("query enrichment prompt was not captured")


__all__ = tuple(name for name in globals() if not name.startswith("__"))
