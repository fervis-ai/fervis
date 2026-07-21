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
    turn_task = "resolve time inputs and review identity resolver options"

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
                "Use question_text, field_label_text, and value_meaning_hint as catalog-blind semantic evidence. Write resource_type_basis first, stating what kind of resource instance lookup_text may identify. Then write resource_type_compatibility for every shown_resource_type.",
                "SAME_RESOURCE_TYPE means an instance returned by the resolver could itself be what lookup_text denotes in the question. Assess each shown type independently; several may be SAME_RESOURCE_TYPE. A resource that merely references, contains information about, records activity for, or otherwise relates to that referent is DIFFERENT_RESOURCE_TYPE.",
                "Before option_reviews, write identifier_kind_basis and then identifier_kind once for the known input. Use PRIMARY_KEY when lookup_text is intended as the complete primary-key value of a SAME_RESOURCE_TYPE. Use DESCRIPTIVE when lookup_text is a non-primary-key value used to refer to a SAME_RESOURCE_TYPE.",
                "DIFFERENT_RESOURCE_TYPE always requires CANNOT_RESOLVE_LOOKUP_TEXT. Only SAME_RESOURCE_TYPE proceeds to resolver_fit_question and route-mechanics assessment.",
                "For every option, answer the shown resolver_fit_question.",
                "An exact match in a field that describes another entity, category, or surrounding context does not identify the returned resource.",
                "For a positive review, include every required request parameter with no default. Include an optional request parameter only when it performs the lookup. For PRIMARY_KEY, a positive route must take lookup_text as the canonical_result component. A route that takes only other request parameters is negative, even if its response returns the canonical key.",
                "canonical_result identifies the returned resource's complete canonical key. Returned identity verification fields never become computation values. Do not substitute a related resource's key.",
                "Use question_text to interpret field_label_text and value_meaning_hint. Do not infer or decide final source use.",
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
                "Within each known-input review, write fields in this order: resource_type_basis, resource_type_compatibility, identifier_kind_basis, identifier_kind, option_reviews.",
                "Within each option review, write fields in this order: resource_type, resolver_fit_question, because, resolution. Within resolution, write decision, lookup_request_params, then returned_identity_verification_fields.",
                'For CAN_RESOLVE_LOOKUP_TEXT, write because as: "{resource_type} is SAME_RESOURCE_TYPE. With identifier_kind={identifier_kind}, this route can resolve {lookup_text} because {route evidence}."',
                'For CANNOT_RESOLVE_LOOKUP_TEXT, write because as either: "{resource_type} is DIFFERENT_RESOURCE_TYPE." or "{resource_type} is SAME_RESOURCE_TYPE, but with identifier_kind={identifier_kind}, this route cannot resolve {lookup_text} because {route evidence}."',
                "Write decision after because.",
                "lookup_request_params answers: Which shown request parameter or parameters exactly match lookup_text's identifier meaning for this SAME_RESOURCE_TYPE? Return those parameter-value pairs, or an empty array when none match.",
                "returned_identity_verification_fields are returned-resource fields that may exactly equal lookup_text. For PRIMARY_KEY, only fields declared by canonical_result.components are valid. For DESCRIPTIVE, a field is valid only when its declared type and choices accept lookup_text and it describes the returned resource itself.",
                "For CAN_RESOLVE_LOOKUP_TEXT, resolution must contain at least one lookup_request_param and at least one returned_identity_verification_field.",
                "For CANNOT_RESOLVE_LOOKUP_TEXT, resolution must contain empty lookup_request_params and returned_identity_verification_fields arrays.",
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
                    "shown_resource_types": list(task.shown_resource_types),
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
