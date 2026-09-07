"""Model-facing semantic identity-grounding contract."""

from __future__ import annotations

from fervis.lookup.grounding.semantic import SemanticGroundingRequest
from fervis.lookup.grounding.semantic_schema import build_semantic_grounding_schema
from fervis.lookup.grounding.surface import resolver_option_surface_from_catalog
from fervis.lookup.turn_prompts import (
    ProviderResponseContract,
    ProviderToolContract,
    PromptSection,
    TurnPromptBase,
    TurnPromptBuilder,
)
from fervis.lookup.turn_prompts.projections.response_shape import (
    semantic_grounding_tasks_xml,
)
from fervis.model_io.structured_output.specs import required_tool_spec


SEMANTIC_GROUNDING_TOOL_NAME = "submit_grounding"


class SemanticGroundingTurnPrompt(TurnPromptBase):
    turn_name = "grounding"
    turn_task = "resolve time inputs and review identity resolver routes"

    def __init__(self, request: SemanticGroundingRequest) -> None:
        self.request = request

    def data_sections(self, builder: TurnPromptBuilder) -> tuple[PromptSection, ...]:
        return (
            builder.json_section(
                "Time resolution tasks:", self._time_payload(), indent=2
            ),
            builder.text_section(
                "Reference grounding tasks:",
                semantic_grounding_tasks_xml(self._task_payload()),
            ),
        )

    def instruction_sections(
        self, builder: TurnPromptBuilder
    ) -> tuple[PromptSection, ...]:
        return (
            builder.instruction_block(
                "Time resolution",
                (
                    "For each time task, copy expression and choose one time_shape, then fill every intent field.",
                    "The chosen shape determines which fields carry values; every other field uses its neutral value.",
                    "Use point_relative for today, yesterday, tomorrow, or another relative day. Use period_relative for relative periods such as this month, last month, this week, or last year. Use period_named for named calendar periods such as April, Q3, or 2026.",
                    "Use mode=full for a complete period. Use mode=to_date only when the question says so far, to date, week-to-date, month-to-date, year-to-date, or equivalent explicit to-date wording. Resolve relative periods from the runtime date and timezone shown with the tasks.",
                ),
            ),
            builder.instruction_block(
                "Identity meaning",
                (
                    "For each identity task, write identifier_kind_basis, then identifier_kind, then purpose before reviewing resolver routes.",
                    "PRIMARY_KEY means the operand is the complete primary-key value of the denoted resource. DESCRIPTIVE means another value refers to that resource.",
                    "Select identity_validation when a shown route's declared path key is the operand's complete primary key. Otherwise, select reference_grounding when a shown route can search the descriptive operand.",
                    "Review only resolver routes whose shown purpose equals the selected purpose.",
                    "Within each resource-type review, write compatibility_basis before compatibility.",
                    "denoted_instance_kind states the kind of instance named by the operand. POSSIBLE_DENOTED_KIND means a returned instance could plausibly be the denoted kind under the question and shown API contract; several API resource types may be possible. Keep an ambiguous mapping possible here; Read Eligibility selects the final canonical meaning. UNRELATED_RESOURCE_KIND means a returned instance could not be the denoted instance.",
                ),
            ),
            builder.instruction_block(
                "Resolver mechanics",
                (
                    "After compatibility, review every resolver route of that type.",
                    "assessment_basis states whether the route returns a POSSIBLE_DENOTED_KIND and whether its shown request parameters can look up the operand under identifier_kind.",
                    "CAN_RESOLVE_LOOKUP_TEXT selects every caller-supplied request parameter used for that lookup and every returned-resource field used to verify the returned identity.",
                    "ENUMERATE_COMPLETE_SOURCE uses a complete source traversal and compares the supplied text against the chosen returned verification fields. It uses empty lookup_request_params; a search parameter is not required when the complete rows can be read. This branch is available only when access is established.",
                    "CANNOT_RESOLVE_LOOKUP_TEXT selects empty lookup_request_params and returned_identity_verification_fields.",
                    "For PRIMARY_KEY, the selected request parameter accepts the canonical key and returned verification fields are canonical_result components. For DESCRIPTIVE, the selected request parameter searches the descriptive value and returned verification fields describe the returned resource itself.",
                ),
            ),
            builder.instruction_block(
                "Output",
                (
                    "Return time_resolutions first, with one entry for every shown time task.",
                    "Return one reference_reviews entry for every reference task_ref.",
                    "Write fields in this order: identifier_kind_basis, identifier_kind, purpose, resource_type_reviews.",
                    "Within each resource-type review, write compatibility_basis, compatibility, then route_reviews. Within each route review, write assessment_basis, then resolution. Within resolution, write decision, lookup_request_params, then returned_identity_verification_fields.",
                    "Return exactly one submit_grounding tool call.",
                ),
            ),
        )

    def response_contract(self) -> ProviderResponseContract:
        return ProviderResponseContract(
            provider_schema={
                SEMANTIC_GROUNDING_TOOL_NAME: build_semantic_grounding_schema(
                    self.request
                )
            }
        )

    def tool_contract(self) -> ProviderToolContract:
        return ProviderToolContract(
            tool_specs=(
                required_tool_spec(
                    tool_name=SEMANTIC_GROUNDING_TOOL_NAME,
                    tool_description="Submit identity resolver compatibility reviews.",
                    input_schema=build_semantic_grounding_schema(self.request),
                ),
            )
        )

    def _task_payload(self) -> dict[str, object]:
        tasks: list[dict[str, object]] = []
        for task in self.request.tasks:
            input_term = self.request.input(task.input_ref)
            routes: list[dict[str, object]] = []
            for option in task.options:
                surface = resolver_option_surface_from_catalog(
                    self.request.resolver_catalog,
                    option,
                    read_access=self.request.read_access,
                )
                route = surface.prompt_payload()
                route["route_ref"] = route.pop("binding_option_id")
                routes.append(route)
            tasks.append(
                {
                    "task_ref": task.task_ref,
                    "input_ref": task.input_ref,
                    "use_refs": list(task.use_refs),
                    "input_text": input_term.origin.meaning,
                    "operand": input_term.operand,
                    "operand_meaning": task.operand_meaning,
                    "denoted_instance_kind": task.denoted_instance_kind,
                    "reference_fact_ref": (
                        task.reference_fact_ref.token
                        if task.reference_fact_ref is not None
                        else None
                    ),
                    "expected_set_ref": (
                        task.expected_set_ref.token
                        if task.expected_set_ref is not None
                        else None
                    ),
                    "expected_set_meaning": (
                        self.request.set_origins[task.expected_set_ref].meaning
                        if task.expected_set_ref is not None
                        else None
                    ),
                    "resolver_routes": routes,
                }
            )
        return {
            "question": self.request.question,
            "reference_grounding_tasks": tasks,
        }

    def _time_payload(self) -> dict[str, object]:
        return {
            "runtime_date": self.request.runtime_date,
            "timezone": self.request.timezone,
            "tasks": [
                {
                    "task_ref": task.task_ref,
                    "input_ref": task.input_ref,
                    "use_refs": list(task.use_refs),
                    "expression": task.expression,
                    "input_text": self.request.input(task.input_ref).origin.meaning,
                    "operand_meaning": task.operand_meaning,
                }
                for task in self.request.time_tasks
            ],
        }


__all__ = ["SEMANTIC_GROUNDING_TOOL_NAME", "SemanticGroundingTurnPrompt"]
