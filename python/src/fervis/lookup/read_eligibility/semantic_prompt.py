"""Model-facing semantic Read Eligibility contract."""

from __future__ import annotations

from fervis.lookup.relation_catalog.row_sources import RowSourceKind
from fervis.lookup.relation_catalog.row_sources import row_source_relation_evidence
from fervis.lookup.read_eligibility.semantic import SemanticReadEligibilityRequest
from fervis.lookup.grounding import IdentityResolutionTask
from fervis.lookup.grounding.surface import resolver_option_surface_from_catalog
from fervis.lookup.read_eligibility.semantic_schema import (
    build_semantic_read_eligibility_schema,
)
from fervis.lookup.turn_prompts import (
    ProviderResponseContract,
    ProviderToolContract,
    PromptSection,
    TurnPromptBase,
    TurnPromptBuilder,
)
from fervis.lookup.turn_prompts.projections.response_shape import (
    semantic_canonical_identity_uses_xml,
    semantic_identity_resolution_tasks_xml,
    semantic_read_relations_xml,
    semantic_read_sources_xml,
)
from fervis.lookup.turn_prompts.projections.response_shape import (
    ApiReadResponseShapeProjector,
)
from fervis.lookup.turn_prompts.projections.semantic_requirements import (
    semantic_requirements_prompt_payload,
)
from fervis.model_io.structured_output.specs import required_tool_spec


SEMANTIC_READ_ELIGIBILITY_TOOL_NAME = "submit_read_eligibility"


class SemanticReadEligibilityTurnPrompt(TurnPromptBase):
    turn_name = "read eligibility"
    turn_task = "assess answer reads, then select identity meanings and resolver routes"

    def __init__(self, request: SemanticReadEligibilityRequest) -> None:
        self.request = request

    def data_sections(self, builder: TurnPromptBuilder) -> tuple[PromptSection, ...]:
        return (
            builder.json_section(
                "Semantic requirements:",
                self._requirements_payload(),
                indent=2,
            ),
            builder.text_section(
                "Available identity resolver routes (separate from answer reads):",
                semantic_identity_resolution_tasks_xml(self._identity_tasks_payload()),
            ),
            builder.text_section(
                "Declared canonical identity uses in answer reads:",
                semantic_canonical_identity_uses_xml(
                    self._canonical_identity_uses_payload()
                ),
            ),
            builder.text_section(
                "Answer read candidates:",
                semantic_read_sources_xml(self._sources_payload()),
            ),
            builder.text_section(
                "Declared cross-read relations:",
                semantic_read_relations_xml(self._relations_payload()),
            ),
        )

    def instruction_sections(
        self, builder: TurnPromptBuilder
    ) -> tuple[PromptSection, ...]:
        return (
            builder.instruction_block(
                "Identity selection",
                (
                    "Assess each identity task before assessing answer reads.",
                    "For each identity task, assess every canonical option with assessment then FITS or DOES_NOT_FIT before writing canonical_option_basis and canonical_option_id.",
                    "Declared canonical identity uses show the exact answer-read request targets and returned identity fields for each option. A canonical option fits when its declared uses belong to retained reads that contribute the semantic requirements without an undeclared identity conversion.",
                    "After choosing a canonical option, assess every shown resolver route with assessment then FITS or DOES_NOT_FIT before writing resolver_route_basis and resolver_route_id. Routes owned by another canonical meaning are DOES_NOT_FIT for the selected meaning.",
                    "A resolver route with purpose=identity_validation fits when it validates the chosen canonical identity and its result can be used by the retained answer reads. A route with purpose=reference_grounding fits when it obtains the chosen canonical identity from a descriptive input and its result can be used by the retained answer reads.",
                    "NO_CANONICAL_INTERPRETATION uses null option and route IDs after every canonical option and resolver route is DOES_NOT_FIT. NO_RESOLVER_ROUTE selects one FITS canonical option, uses a null route ID, and assesses every resolver route as DOES_NOT_FIT.",
                ),
            ),
            builder.instruction_block(
                "Read assessment",
                (
                    "Assess every answer read candidate separately for every requested fact under the selected identity interpretation.",
                    "RETAIN means the read's rows or returned values could contribute candidate rows, identity, qualification, association, grouping, ordering, or a requested value to a complete answer strategy.",
                    "Retain row-level reads that expose rows the requested fact may count, list, filter, group, rank, or aggregate.",
                    "A read remains eligible when it contributes those rows or fields, regardless of whether another read could also answer the fact.",
                    "Do not reject a read only because its endpoint resource name is broader than the requested subject.",
                    "For example, location rows with a type field may remain useful for a store question.",
                    "DROP means the read contributes none of those. Drop a read when it is only related context, audit or detail data, telemetry, receipt data, verification evidence, or an indirect helper. A read used only to resolve, validate, explain, audit, enrich, or provide context for another answer-bearing read belongs to the identity outcome rather than an answer-read RETAIN decision.",
                    "For RETAIN, assessment_basis describes the rows and fields that make the read relevant to the requested fact. It must not rate, rank, or compare retained reads.",
                    "relevant_field_refs is the answer-changing computation support for this fact. Include a field only when its value can change the answer by establishing row grain or identity, qualification, grouping, ordering, association, deduplication, or a requested measure. Fields that only describe, decorate, audit, explain, or provide context are excluded. Exact requirement-to-field realization belongs to Source Binding after Plan Selection.",
                    "Write assessment_basis before relevant_field_refs and decision.",
                ),
            ),
            builder.instruction_block(
                "Output",
                (
                    "Return one read_assessments_by_requested_fact entry for every requested fact, then one identity_outcomes entry for every task ref.",
                    "Write each identity outcome as canonical_option_assessments, canonical_option_basis, canonical_option_id, resolver_route_assessments, resolver_route_basis, resolver_route_id, evidence_refs, then outcome.",
                    "Within each requested fact, assess every shown answer read candidate exactly once.",
                    "Each read assessment contains assessment_basis, relevant_field_refs, then decision. DROP uses an empty relevant_field_refs array.",
                    "Return exactly one submit_read_eligibility tool call.",
                ),
            ),
        )

    def response_contract(self) -> ProviderResponseContract:
        return ProviderResponseContract(
            provider_schema={
                SEMANTIC_READ_ELIGIBILITY_TOOL_NAME: (
                    build_semantic_read_eligibility_schema(self.request)
                )
            }
        )

    def tool_contract(self) -> ProviderToolContract:
        return ProviderToolContract(
            tool_specs=(
                required_tool_spec(
                    tool_name=SEMANTIC_READ_ELIGIBILITY_TOOL_NAME,
                    tool_description="Submit semantic read and identity decisions.",
                    input_schema=build_semantic_read_eligibility_schema(self.request),
                ),
            )
        )

    def _requirements_payload(self) -> dict[str, object]:
        return {
            "requested_facts": [
                semantic_requirements_prompt_payload(index)
                for index in self.request.indexes
            ]
        }

    def _sources_payload(self) -> dict[str, object]:
        sources: list[dict[str, object]] = []
        for candidate in self.request.read_candidates:
            source = candidate.sources[0]
            if source.kind is RowSourceKind.API_READ:
                read = self.request.answer_catalog.read(candidate.read_id)
                sources.append(
                    {
                        "source_ref": candidate.candidate_ref,
                        "api_read": ApiReadResponseShapeProjector(read).prompt_payload(
                            row_path_ids=tuple(
                                item.row_path_id or "root" for item in candidate.sources
                            ),
                        ),
                        "identity_evidence": [
                            {
                                "identity_ref": evidence.identity_ref,
                                "entity_kind": evidence.entity_kind,
                                "key_id": evidence.key_id,
                                "field_refs": evidence.field_refs,
                            }
                            for item in candidate.sources
                            for evidence in item.identity_evidence
                        ],
                    }
                )
                continue
            sources.append(
                {
                    "source_ref": candidate.candidate_ref,
                    "row_source": {
                        "kind": source.kind.value,
                        "label": source.label,
                        "description": source.description,
                        "fields": [
                            {
                                "field_ref": field.field_ref,
                                "label": field.label,
                                "type": field.type.value,
                            }
                            for field in source.fields
                        ],
                    },
                }
            )
        return {"sources": sources}

    def _relations_payload(self) -> dict[str, object]:
        candidates_by_source_ref = {
            source_ref: candidate.candidate_ref
            for candidate in self.request.read_candidates
            for source_ref in candidate.source_refs
        }
        sources_by_ref = {
            source.id: source for source in self.request.source_catalog.sources
        }
        relations = []
        for evidence in row_source_relation_evidence(
            self.request.source_catalog.sources
        ):
            left_candidate_ref = candidates_by_source_ref[evidence.left_source_ref]
            right_candidate_ref = candidates_by_source_ref[evidence.right_source_ref]
            if left_candidate_ref == right_candidate_ref:
                continue
            left_source = sources_by_ref[evidence.left_source_ref]
            right_source = sources_by_ref[evidence.right_source_ref]
            relations.append(
                {
                    "left_source_ref": left_candidate_ref,
                    "left_field_refs": [
                        left_source.field(field_id).field_ref
                        for field_id in evidence.left_field_refs
                    ],
                    "right_source_ref": right_candidate_ref,
                    "right_field_refs": [
                        right_source.field(field_id).field_ref
                        for field_id in evidence.right_field_refs
                    ],
                }
            )
        return {"relations": relations}

    def _identity_tasks_payload(self) -> dict[str, object]:
        return {
            "identity_resolution_tasks": [
                {
                    "task_ref": task.task_ref,
                    "input_ref": task.input_ref,
                    "expected_set_ref": (
                        task.expected_set_ref.token
                        if task.expected_set_ref is not None
                        else None
                    ),
                    "canonical_options": [
                        {
                            "canonical_option_id": option.canonical_option_id,
                            "identity_ref": option.identity_ref,
                            "identifier_kind": self._option_identifier_kind(
                                task,
                                option.resolver_route_refs,
                            ),
                            "resolver_routes": [
                                self._route_payload(task, route_ref)
                                for route_ref in option.resolver_route_refs
                            ],
                        }
                        for option in task.canonical_options
                    ],
                }
                for task in self.request.identity_tasks
            ]
        }

    def _canonical_identity_uses_payload(self) -> dict[str, object]:
        return {
            "canonical_options": [
                {
                    "canonical_option_id": option.canonical_option_id,
                    "identity_ref": option.identity_ref,
                    "answer_reads": [
                        {
                            "read_id": answer_read.read_id,
                            "request_param_refs": answer_read.request_param_refs,
                            "returned_identities": [
                                {
                                    "identity_ref": identity.identity_ref,
                                    "field_refs": identity.field_refs,
                                }
                                for identity in answer_read.returned_identities
                            ],
                        }
                        for answer_read in option.answer_reads
                    ],
                }
                for task in self.request.identity_tasks
                for option in self.request.canonical_identity_answer_uses(task)
            ]
        }

    def _route_payload(
        self, task: IdentityResolutionTask, route_ref: str
    ) -> dict[str, object]:
        route = next(
            item for item in task.resolver_routes if item.route_ref == route_ref
        )
        binding = route.compatibility
        surface = resolver_option_surface_from_catalog(
            self.request.resolver_catalog,
            route.option,
        )
        payload = surface.prompt_payload()
        parameters_by_ref = {
            parameter.param_ref: parameter
            for parameter in surface.request_parameters
        }
        input_term = self.request.input_term(task.input_ref)
        return {
            "binding_option_id": route.route_ref,
            "purpose": route.option.purpose.value,
            "api_read": payload["api_read"],
            "canonical_result": payload["canonical_result"],
            "lookup_request_parameters": [
                {
                    "param_ref": param_ref,
                    "source": _enum_value(parameters_by_ref[param_ref].source),
                    "value": input_term.operand,
                }
                for param_ref in binding.lookup_request_param_refs
            ],
            "returned_identity_verification_fields": list(
                binding.returned_identity_verification_field_paths
            ),
        }

    @staticmethod
    def _option_identifier_kind(
        task: IdentityResolutionTask,
        route_refs: tuple[str, ...],
    ) -> str | None:
        kinds = {
            route.compatibility.identifier_kind.value
            for route in task.resolver_routes
            if route.route_ref in route_refs
        }
        if len(kinds) != 1:
            return None
        return next(iter(kinds))


def _enum_value(value: object) -> object:
    return getattr(value, "value", value)


__all__ = [
    "SEMANTIC_READ_ELIGIBILITY_TOOL_NAME",
    "SemanticReadEligibilityTurnPrompt",
]
