"""Model-facing prompt for pre-plan grounding."""

from __future__ import annotations

from fervis.lookup.grounding.model import (
    GroundingRequest,
    InputBindingOption,
    KnownInputBindingTask,
    KnownTimeResolutionTask,
    ResolverOutputFieldCard,
    ResolverQueryParamCard,
)
from fervis.lookup.grounding.schema import build_grounding_schema
from fervis.lookup.question_inputs import KnownInputKind, LiteralInputRole
from fervis.lookup.turn_prompts import (
    ProviderResponseContract,
    ProviderToolContract,
    PromptSection,
    TurnPromptBase,
    TurnPromptBuilder,
)
from fervis.model_io.structured_output.specs import required_tool_spec


GROUNDING_TOOL_NAME = "submit_grounding"


class GroundingTurnPrompt(TurnPromptBase):
    turn_name = "grounding"
    turn_task = "resolve known reference text to canonical IDs"

    def __init__(self, request: GroundingRequest) -> None:
        self.request = request

    def prompt_sections(
        self,
        builder: TurnPromptBuilder,
    ) -> tuple[PromptSection, ...]:
        sections = [
            builder.json_section(
                "Known inputs to ground:", self.known_inputs_payload()
            ),
            builder.json_section(
                "Time inputs to resolve:",
                self.time_inputs_payload(),
            ),
            builder.json_section("Binding options:", self.binding_options_payload()),
            self.grounding_objective_section(builder),
            self.time_resolution_section(builder),
            self.resolver_selection_section(builder),
            self.copying_and_validity_section(builder),
            self.output_section(builder),
        ]
        responses = self.request.clarification_responses
        if responses:
            sections.insert(
                0,
                builder.json_section(
                    "Attributed clarification responses:",
                    {
                        "responses": [
                            {
                                "response_id": response.source.response_id,
                                "clarification_id": response.source.clarification_id,
                                "exact_user_text": response.source.exact_user_text,
                                "known_input_id": response.known_input_id,
                            }
                            for response in responses
                        ]
                    },
                ),
            )
        return tuple(sections)

    def grounding_objective_section(
        self,
        builder: TurnPromptBuilder,
    ) -> PromptSection:
        return builder.instruction_block(
            "Grounding Objective",
            (
                "Resolve known question inputs before source binding and planning.",
                "For time inputs, author a typed time intent from the shown question span.",
                "For each reference input, select the one binding option that best resolves it.",
                "Do not decide how the identity will be used by the final answer source.",
                "The selected route will validate or ground the input through a source read.",
                "known_input_text is the original question span; lookup_text and time_expression are the resolved values to search, match, or compile.",
            ),
        )

    def time_resolution_section(
        self,
        builder: TurnPromptBuilder,
    ) -> PromptSection:
        return builder.instruction_block(
            "Time Resolution",
            (
                "For each time input, copy time_expression into date_intent.expression.",
                "date_intent.intent is flat: choose one time_shape, then fill every intent field.",
                "Use time_shape=point_date for exact dates, point_relative for relative dates, period_relative for relative periods, period_named for named periods, range for closed ranges, open_range for open ranges, and window for rolling windows.",
                "Use point_relative for today, yesterday, tomorrow, or any relative day word.",
                "Use mode=full when the user asks for a complete period.",
                "Use period_relative for relative period wording such as this month, last month, this week, or last year.",
                "Use period_named for named calendar periods such as April, Q3, or 2026.",
                "Use mode=to_date only when the user says so far, to date, week-to-date, month-to-date, year-to-date, or equivalent explicit to-date wording.",
                "Fill the fields required by the chosen time_shape with real values.",
                "Use 0 or none only for fields that the chosen time_shape does not use.",
                "For named month or quarter periods, use year when supplied; otherwise use year_policy=most_recent.",
                "Use separate time inputs when the user asks for separate dates or periods. Use one range input when the user asks for one combined range.",
                "Every time input must be represented by exactly one supported time_shape.",
            ),
        )

    def resolver_selection_section(
        self,
        builder: TurnPromptBuilder,
    ) -> PromptSection:
        return builder.instruction_block(
            "Resolver Selection",
            (
                "Select exactly one binding option for this known input, or none when no shown option fits.",
                "A related owner, container, area, location, or source is not the same entity unless the question uses it with that meaning.",
                "Select identity_validation when the endpoint's declared path key fits field_label_text and value_meaning_hint and lookup_text contains a value of that key's declared type.",
                "lookup_text itself need not consist only of that value.",
                "For identity_validation, input_value is the exact value span converted to the declared type, without its field label or surrounding grammar.",
                "Select reference_grounding when the endpoint can search or match the descriptive text.",
                "For reference_grounding with result_kind=canonical_identity, copy lookup_text verbatim as input_value.",
                "For reference_grounding with result_kind=matched_value, select matched_field_ref and set input_value to that field's concrete value expressed by lookup_text; remove subject words and grammar that state how it constrains the subject.",
                "A matched input_value must be possible content of matched_field_ref, not a phrase describing the subject or its relationship to that field.",
                "For reference_grounding, use canonical_identity when lookup_text names the entity returned by the selected route.",
                "Use matched_value when lookup_text supplies a scalar value of one returned field rather than naming the returned entity.",
                "A canonical identity may still constrain another population; that use does not make it a scalar matched value.",
                "Identity validation always uses result_kind=canonical_identity.",
                "Do not reject a resolver because you are unsure how the final answer source will use the grounded result.",
                "Use question_context.question when interpreting value_meaning_hint.",
                "Use question_context.requested_facts only as local context for what the lookup text refers to; do not decide final source use from it.",
                "Use the known input's value_meaning_hint and each option's description, resource_names, returned_identity, lookup_surface, query_params, and selected_output_fields to decide whether the resolver can resolve the lookup text.",
                "Use field_label_text as the catalog-blind attribute approximation supplied by question interpretation; consider it together with value_meaning_hint, but do not treat it as an authoritative catalog field name.",
                "lookup_surface.param_ref means the resolver can search its own resource directly using the lookup text.",
                "lookup_surface.field_refs lists returned fields that can exactly match the lookup text.",
                "query_params shows endpoint filters available on the resolver resource.",
                "selected_output_fields shows identity, display, and choice fields from the resolver response that are useful for distinguishing what object the resolver returns.",
                "Use value_meaning_hint and field_label_text to choose among the options; neither is an authoritative catalog identifier.",
            ),
        )

    def copying_and_validity_section(
        self,
        builder: TurnPromptBuilder,
    ) -> PromptSection:
        return builder.instruction_block(
            "Copying And Validity",
            (
                "Return known_time_resolutions as an object keyed by known_input_id; include every time input key exactly once.",
                "Return known_input_bindings as an object keyed by known_input_id; include every reference input key exactly once.",
                "Copy the selected binding_option_id as selected_option_id, choose its result_kind, and write input_value according to that option's declared purpose.",
                "When no option fits, use selected_option_id=none, result_kind=none, and input_value as an empty string.",
                "Briefly state why the selected option fits in selection_basis.",
                "Do not rewrite, normalize, abbreviate, or invent identity keys.",
            ),
        )

    def output_section(
        self,
        builder: TurnPromptBuilder,
    ) -> PromptSection:
        return builder.instruction_block(
            "Output",
            ("Return the submit_grounding tool call only.",),
        )

    def response_contract(self) -> ProviderResponseContract:
        return ProviderResponseContract(
            provider_schema=build_grounding_schema(self.request)
        )

    def tool_contract(self) -> ProviderToolContract:
        return ProviderToolContract(
            tool_specs=(
                required_tool_spec(
                    tool_name=GROUNDING_TOOL_NAME,
                    tool_description=(
                        "Submit grounding resolver compatibility reviews."
                    ),
                    input_schema=build_grounding_schema(self.request),
                ),
            )
        )

    def known_inputs_payload(self) -> dict[str, object]:
        return {
            "known_input_binding_tasks": [
                {
                    "known_input_id": task.known_input_id,
                    "known_input_text": task.known_input_text,
                    "lookup_text": task.lookup_text,
                    "known_input_kind": task.known_input_kind,
                    "field_label_text": task.field_label_text,
                    "value_meaning_hint": task.known_input_description,
                    "question_context": self._question_context_payload(task),
                }
                for task in self.request.tasks
            ]
        }

    def time_inputs_payload(self) -> dict[str, object]:
        return {
            "known_time_resolution_tasks": [
                {
                    "known_input_id": task.known_input_id,
                    "known_input_text": task.known_input_text,
                    "time_expression": task.time_expression,
                    "known_input_kind": KnownInputKind.LITERAL.value,
                    "known_input_role": LiteralInputRole.TIME_VALUE.value,
                    "question_context": self._time_question_context_payload(task),
                }
                for task in self.request.time_tasks
            ]
        }

    def _time_question_context_payload(
        self,
        task: KnownTimeResolutionTask,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "question": self.request.question,
        }
        if task.requested_facts:
            payload["requested_facts"] = [
                {
                    "requested_fact_id": fact.requested_fact_id,
                    "answer_fact": fact.answer_fact,
                    "answer_population": {
                        "population_label": fact.answer_population_label,
                        "counted_unit": fact.answer_population_counted_unit,
                    },
                    "answer_outputs": list(fact.answer_outputs),
                }
                for fact in task.requested_facts
            ]
        return payload

    def _question_context_payload(
        self,
        task: KnownInputBindingTask,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "question": self.request.question,
        }
        if task.requested_facts:
            payload["requested_facts"] = [
                {
                    "requested_fact_id": fact.requested_fact_id,
                    "answer_fact": fact.answer_fact,
                    "answer_population": {
                        "population_label": fact.answer_population_label,
                        "counted_unit": fact.answer_population_counted_unit,
                    },
                    "answer_outputs": list(fact.answer_outputs),
                }
                for fact in task.requested_facts
            ]
        return payload

    def binding_options_payload(self) -> dict[str, object]:
        return {
            "known_input_binding_options": [
                {
                    "known_input_id": task.known_input_id,
                    "binding_options": [
                        self._binding_option_payload(option)
                        for option in task.options
                    ],
                }
                for task in self.request.tasks
            ]
        }

    def _binding_option_payload(
        self,
        option: InputBindingOption,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "binding_option_id": option.id,
            "purpose": option.purpose.value,
        }
        route = option.route
        if route is None:
            return payload
        payload.update(
            {
                "read_id": route.resolver_read_id,
                "endpoint_name": route.resolver_endpoint_name,
                "description": route.resolver_description,
                "resource_names": list(route.resolver_resource_names),
                "returned_identity": {
                    "entity_kind": route.entity_kind,
                    "key_id": route.key_id,
                    "components": [
                        {
                            "component_id": component.component_id,
                            "field_ref": component.field_ref,
                        }
                        for component in route.key_components
                    ],
                },
                "lookup_surface": {
                    **(
                        {
                            "param_ref": route.lookup_param_ref,
                            "param_type": route.lookup_param_type,
                        }
                        if route.lookup_param_ref
                        else {}
                    ),
                    "field_refs": list(route.lookup_field_refs),
                },
                "query_params": [
                    self._route_param_payload(param) for param in route.query_params
                ],
                "selected_output_fields": [
                    self._route_field_payload(field)
                    for field in route.selected_output_fields
                ],
            }
        )
        return payload

    def _route_param_payload(self, param: ResolverQueryParamCard) -> dict[str, object]:
        payload: dict[str, object] = {
            "param_ref": param.param_ref,
            "name": param.name,
            "type": param.type,
        }
        if param.choices:
            payload["choices"] = list(param.choices)
        return payload

    def _route_field_payload(self, field: ResolverOutputFieldCard) -> dict[str, object]:
        payload: dict[str, object] = {
            "field_ref": field.field_ref,
            "field_path": field.field_path,
            "type": field.type,
        }
        if field.choices:
            payload["choices"] = list(field.choices)
        return payload
