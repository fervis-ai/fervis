"""Model-facing prompt for pre-plan grounding."""

from __future__ import annotations

from fervis.lookup.grounding.model import (
    GroundingRequest,
    InputBindingOption,
    KnownInputBindingTask,
    KnownTimeResolutionTask,
    resolver_fit_question_for_option,
)
from fervis.lookup.grounding.schema import build_grounding_schema
from fervis.lookup.grounding.surface import resolver_option_surface
from fervis.lookup.question_inputs import KnownInputKind, LiteralInputRole
from fervis.lookup.turn_prompts import (
    ProviderResponseContract,
    ProviderToolContract,
    PromptSection,
    TurnPromptBase,
    TurnPromptBuilder,
)
from fervis.lookup.turn_prompts.projections import grounding_binding_tasks_xml
from fervis.model_io.structured_output.specs import required_tool_spec


GROUNDING_TOOL_NAME = "submit_grounding"


class GroundingTurnPrompt(TurnPromptBase):
    turn_name = "grounding"
    turn_task = "resolve time inputs and review named-reference resolver options"

    def __init__(self, request: GroundingRequest) -> None:
        self.request = request

    def prompt_sections(
        self,
        builder: TurnPromptBuilder,
    ) -> tuple[PromptSection, ...]:
        return (
            builder.text_section(
                "Known input binding tasks:",
                grounding_binding_tasks_xml(
                    self.known_input_binding_tasks_payload()
                ),
            ),
            builder.json_section(
                "Time inputs to resolve:",
                self.time_inputs_payload(),
            ),
            self.grounding_objective_section(builder),
            self.time_resolution_section(builder),
            self.resolver_compatibility_section(builder),
            self.output_shape_section(builder),
            self.output_section(builder),
        )

    def grounding_objective_section(
        self,
        builder: TurnPromptBuilder,
    ) -> PromptSection:
        return builder.instruction_block(
            "Grounding Objective",
            (
                "Resolve time inputs and review the resolver options available for each reference input.",
                "Review every binding option independently. More than one option may be positive.",
                "A positive review means only that the read can validate or match the supplied lookup text and produce its declared canonical result through the selected request values and exact-match fields.",
                "Do not choose one resolver. Do not decide which answer read will consume the result. Do not execute a resolver read during this turn.",
                "known_input_text is the original question span. lookup_text and time_expression are the resolved values to search, match, or compile.",
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

    def resolver_compatibility_section(
        self,
        builder: TurnPromptBuilder,
    ) -> PromptSection:
        return builder.instruction_block(
            "Resolver Compatibility",
            (
                "Use CAN_RESOLVE_LOOKUP_TEXT when the read can use selected declared request parameters and exact returned-resource fields to validate or match lookup_text. Use CANNOT_RESOLVE_LOOKUP_TEXT otherwise.",
                "For every option, answer the shown resolver_fit_question.",
                "A positive option must use the lookup text to identify the returned resource itself. An exact match in a field that describes another entity, category, or surrounding context does not identify the returned resource.",
                "Use field_label_text and value_meaning_hint together to understand what the supplied text means. Both are catalog-blind approximations, not authoritative catalog names.",
                "For a positive review, include every required request parameter with no default. Include an optional request parameter only when it performs the lookup. Key request_values by param_ref. Select every returned-resource field that may exactly equal lookup_text.",
                "response_match_alternatives has OR semantics: an exact match in any selected field verifies the returned resource. Do not select fields that describe another entity, category, or surrounding context.",
                "canonical_result identifies the returned resource's complete canonical key. Match fields establish which resource was named, but they never become computation values. Do not substitute a related resource's key.",
                "Use question_text to interpret field_label_text and value_meaning_hint. Do not infer or decide final source use.",
                "Judge business meaning, not word equality between the input hint and the returned entity kind.",
                "Do not reject an option because its result might not fit the final answer source. Later stages own that decision.",
            ),
        )

    def output_shape_section(
        self,
        builder: TurnPromptBuilder,
    ) -> PromptSection:
        return builder.instruction_block(
            "Output Shape",
            (
                "Return known_time_resolutions as an object keyed by known_input_id and include every shown time input exactly once.",
                "Return known_input_binding_reviews as an object keyed by known_input_id. Within each review, return option_reviews keyed by binding_option_id and include every shown binding option exactly once.",
                "Copy all IDs and each resolver_fit_question exactly.",
                'For every option review, write the because field as: "{lookup_text} can/cannot identify the returned {resource} because {selected response fields} describe {field owner}, and the route returns {canonical result}." Replace every template term with concrete text from the option.',
                "Write decision after because. Use CAN_RESOLVE_LOOKUP_TEXT only when the stated field owner is the returned resource; otherwise use CANNOT_RESOLVE_LOOKUP_TEXT.",
                "For CAN_RESOLVE_LOOKUP_TEXT, return request_values keyed by param_ref and at least one response_match_alternative.",
                "For CANNOT_RESOLVE_LOOKUP_TEXT, return an empty request_values object and an empty response_match_alternatives array.",
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
                    tool_description="Submit grounding resolver compatibility reviews.",
                    input_schema=build_grounding_schema(self.request),
                ),
            )
        )

    def known_input_binding_tasks_payload(self) -> dict[str, object]:
        return {
            "known_input_binding_tasks": [
                {
                    "known_input_id": task.known_input_id,
                    "known_input_text": task.known_input_text,
                    "lookup_text": task.lookup_text,
                    "field_label_text": task.field_label_text,
                    "value_meaning_hint": task.known_input_description,
                    "question_text": self.request.question,
                    "binding_options": [
                        self._binding_option_payload(option, task=task)
                        for option in task.options
                    ],
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
        payload: dict[str, object] = {"question": self.request.question}
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

    def _binding_option_payload(
        self,
        option: InputBindingOption,
        *,
        task: KnownInputBindingTask,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "binding_option_id": option.id,
            "resolver_fit_question": resolver_fit_question_for_option(
                task=task,
                option=option,
            ),
        }
        payload.update(resolver_option_surface(self.request, option).prompt_payload())
        return payload
