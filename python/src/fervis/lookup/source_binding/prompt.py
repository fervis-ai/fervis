"""Model-facing semantic Source Binding contract."""

from __future__ import annotations

from fervis.lookup.source_binding.model import SemanticSourceBindingRequest
from fervis.lookup.source_binding.schema import (
    build_semantic_source_binding_schema,
)
from fervis.lookup.source_binding.subject_obligations import (
    NORMAL_INSTANCE_EXCLUDED_STATE_ROLES,
)
from fervis.lookup.question_contract import (
    FactTerm,
    RawDataRecord,
    RequestedFactSemanticIndex,
)
from fervis.lookup.semantic_types import IdentifierType
from fervis.lookup.turn_prompts import (
    ProviderResponseContract,
    ProviderToolContract,
    PromptSection,
    TurnPromptBase,
    TurnPromptBuilder,
)
from fervis.lookup.turn_prompts.projections.semantic_requirements import (
    boolean_requirement_prompt_items,
)
from fervis.lookup.turn_prompts.projections.canonical_values import (
    canonical_values_prompt_items,
    prompt_value,
)
from fervis.lookup.turn_prompts.projections.source_identities import (
    source_identity_evidence_prompt_items,
)
from fervis.model_io.structured_output.specs import required_tool_spec


SEMANTIC_SOURCE_BINDING_TOOL_NAME = "submit_source_binding"


class SemanticSourceBindingTurnPrompt(TurnPromptBase):
    turn_name = "source binding"
    turn_task = "bind semantic requirements to the selected source strategy"

    def __init__(self, request: SemanticSourceBindingRequest) -> None:
        self.request = request

    def data_sections(self, builder: TurnPromptBuilder) -> tuple[PromptSection, ...]:
        return (
            builder.json_section(
                "Semantic requirements:", self._requirements_payload(), indent=2
            ),
            builder.json_section(
                "Selected strategy and sources:", self._sources_payload(), indent=2
            ),
            builder.json_section(
                "Canonical input values:", self._values_payload(), indent=2
            ),
        )

    def instruction_sections(
        self, builder: TurnPromptBuilder
    ) -> tuple[PromptSection, ...]:
        return (
            builder.instruction_block(
                "Semantic bindings",
                (
                    "Bind every shown set and association in every strategy branch.",
                    "A set realization selects the source rows representing that set and one declared identity_ref when available.",
                    "After writing invocation applications, bind a fact only when its value must still be observed in returned rows. An identifier fact uses the identity contract selected once for its identified set and selects only the source that realizes that contract. Every other returned fact selects one returned field_ref.",
                    "For each association, select one realization_ref. Select a source_ref when the connected realizations use that same source. Select a relation_evidence_ref when the association connects realizations from different selected sources.",
                    "Write mapping_basis before source or realization choices.",
                ),
            ),
            builder.instruction_block(
                "Resolved input application",
                (
                    "resolved_input_applications has one shown branch key. Each item applies one resolved input value and one of its shown compatible components to one request target in that branch.",
                    "Write mapping_basis before owner_ref, value_ref, value_component, and target_ref.",
                    "Each selected branch must apply every fact-local resolved input that owns an explicit population constraint and exposes a compatible component and request target.",
                    "The same resolved value may be applied to more than one target when the selected computation requires each application.",
                    "Use each target at most once.",
                    "A Boolean requirement without resolved input applications is evaluated from its returned fact bindings.",
                ),
            ),
            builder.instruction_block(
                "Finite-choice requirement application",
                (
                    "finite_choice_applications is keyed first by branch and then by each shown requirement owner.",
                    "For each owner, write application_basis before surface_ref and selected_choice_values.",
                    "application_basis states why the selected request surface and choices realize that requirement.",
                    "surface_ref is the one request parameter whose declared meaning expresses the requirement.",
                    "selected_choice_values contains every choice on that surface that satisfies the requirement.",
                    "Required-source owners select the choices required to invoke that source.",
                    "Normal-instance reviews do not own requirement application.",
                ),
            ),
            builder.instruction_block(
                "Subject obligation",
                (
                    "For a normal business instance, review every shown finite-choice subject surface in each branch.",
                    "finite_choice_reviews is keyed by each shown surface ref. Within each surface, write surface_mapping_basis, then review every shown choice value.",
                    "surface_mapping_basis states what population property the surface controls for the requested subject.",
                    "For each choice, write choice_domain_meaning, role_match_basis, matched_excluded_role, choice_inclusion_basis, then choice_inclusion.",
                    "choice_domain_meaning states what the source returns when this choice value is applied, read against the source description and the requested subject.",
                    "role_match_basis compares the choice domain meaning with every shown excluded subject-state role.",
                    "matched_excluded_role classifies the source choice independently of whether the question requests that choice.",
                    "A subject-state axis classifies the state or kind of each answer-subject instance. Review each choice on such an axis against the excluded subject-state roles.",
                    "A request parameter that filters rows by a subject-state axis remains a subject-state axis; corresponding request-parameter and returned-field choices have the same classification.",
                    "A presentation, ordering, or response-shape axis does not classify subject instances; every choice on that axis has matched_excluded_role=NONE.",
                    "On a subject-state axis, matched_excluded_role is one shown excluded role when the choice matches that role and NONE only after considering every shown role and finding none applies.",
                    "A choice selected for a shown Boolean requirement is explicitly requested. Selecting such a choice with a matched excluded role is the explicit user override for that choice.",
                    "A required-source owner supplies invocation configuration. choice_inclusion remains the authority for whether that choice belongs to the requested subject.",
                    "choice_inclusion_basis explains whether rows with this choice belong in the ordinary subject population before applying explicit requirement selections.",
                    "choice_inclusion records that baseline membership. Assess each choice independently. Multiple ordinary values may be INCLUDE.",
                    "Boolean-requirement applications determine effective membership for their selected choices; choice_inclusion determines effective membership for every other choice.",
                    "Do not narrow the requested population for an unstated preference for cleaner, safer, validated, finalized, or higher-quality evidence.",
                    "For a raw data record, every branch has an empty finite_choice_reviews object.",
                ),
            ),
            builder.instruction_block(
                "Output",
                (
                    "Return set_bindings, resolved_input_applications, finite_choice_applications, fact_bindings, association_bindings, then subject_binding with every shown key exactly once.",
                    "Return exactly one submit_source_binding tool call.",
                ),
            ),
        )

    def response_contract(self) -> ProviderResponseContract:
        return ProviderResponseContract(
            provider_schema={
                SEMANTIC_SOURCE_BINDING_TOOL_NAME: (
                    build_semantic_source_binding_schema(self.request)
                )
            }
        )

    def tool_contract(self) -> ProviderToolContract:
        return ProviderToolContract(
            tool_specs=(
                required_tool_spec(
                    tool_name=SEMANTIC_SOURCE_BINDING_TOOL_NAME,
                    tool_description="Submit semantic source bindings.",
                    input_schema=build_semantic_source_binding_schema(self.request),
                ),
            )
        )

    def _requirements_payload(self) -> dict[str, object]:
        index = self.request.index
        return {
            "requested_fact_id": index.requested_fact_id,
            "terms": [
                {
                    "requirement_ref": ref.token,
                    "kind": ref.kind.value,
                    "meaning": index.term_by_ref[ref].origin.meaning,
                    **_identifier_target(index.term_by_ref[ref], index=index),
                }
                for ref in sorted(index.source_requirement_refs)
            ],
            "associations": [
                {
                    "requirement_ref": ref.token,
                    "meaning": index.term_by_ref[ref].origin.meaning,
                }
                for ref in sorted(index.association_requirement_refs)
            ],
            "boolean_requirements": list(boolean_requirement_prompt_items(index)),
            "subject_obligation": type(index.subject_obligation).__name__,
            "subject_obligation_contract": {
                "subject_ref": index.subject_obligation.subject_set_ref.token,
                "kind": (
                    "RAW_DATA_RECORD"
                    if isinstance(index.subject_obligation, RawDataRecord)
                    else "NORMAL_BUSINESS_INSTANCE"
                ),
                "excluded_state_roles": (
                    []
                    if isinstance(index.subject_obligation, RawDataRecord)
                    else [
                        {"role": item.role.value, "definition": item.definition}
                        for item in NORMAL_INSTANCE_EXCLUDED_STATE_ROLES
                    ]
                ),
            },
        }

    def _sources_payload(self) -> dict[str, object]:
        return {
            "contract_snapshot_ref": self.request.source_catalog.contract_snapshot.ref,
            "branches": [
                {
                    "branch_id": branch.branch_id,
                    "source_refs": list(branch.source_refs),
                    "relation_evidence_refs": list(branch.relation_evidence_refs),
                }
                for branch in self.request.strategy.branches
            ],
            "sources": [
                {
                    "source_ref": source.id,
                    "label": source.label,
                    "description": source.description,
                    "fields": [
                        {
                            "field_ref": field.field_ref,
                            "label": field.label,
                            "type": field.type.value,
                            "choices": [
                                {
                                    "value_ref": value.value_ref,
                                    "value": value.value,
                                    "label": value.label,
                                }
                                for value in self.request.source_catalog.choice_values
                                if value.surface_ref == field.field_ref
                            ],
                        }
                        for field in source.fields
                    ],
                    "candidate_keys": [
                        {
                            "entity_kind": key.entity_kind,
                            "key_id": key.id,
                            "field_refs": [
                                source.field(component.field_id).field_ref
                                for component in key.components
                            ],
                        }
                        for key in source.candidate_keys
                    ],
                    "parameters": [
                        {
                            "parameter_ref": item.param_ref,
                            "name": item.name,
                            "description": item.description,
                            "type": item.type.value,
                            "required": item.required,
                            "choices": [
                                {
                                    "value_ref": value.value_ref,
                                    "value": value.value,
                                    "label": value.label,
                                }
                                for value in self.request.source_catalog.choice_values
                                if value.surface_ref == item.param_ref
                            ],
                            "identity_target": (
                                None
                                if item.entity_target is None
                                else {
                                    "entity_kind": item.entity_target.entity_kind,
                                    "key_id": item.entity_target.key_id,
                                    "component_id": item.entity_target.component_id,
                                }
                            ),
                        }
                        for item in source.params
                    ],
                }
                for source in self.request.source_catalog.sources
            ],
            "relation_evidence": [
                {
                    "evidence_ref": item.evidence_ref,
                    "left_source_ref": item.left_source_ref,
                    "right_source_ref": item.right_source_ref,
                    "left_field_refs": list(item.left_field_refs),
                    "right_field_refs": list(item.right_field_refs),
                }
                for item in self.request.source_catalog.relation_evidence
            ],
            "identity_evidence": [
                *source_identity_evidence_prompt_items(self.request.source_catalog)
            ],
            "invocation_projection_options": [
                {
                    "projection_option_ref": item.option_ref,
                    "source_ref": item.source_ref,
                    "target_ref": item.target_ref,
                    "value_ref": item.value_ref,
                    "projection": item.projection.value,
                    "component_ref": item.component_ref,
                }
                for item in self.request.authored_invocation_projection_options
            ],
        }

    def _values_payload(self) -> dict[str, object]:
        return {
            "values": [*canonical_values_prompt_items(self.request.canonical_values)],
            "catalog_values": [
                {
                    "catalog_input_ref": item.catalog_input_ref,
                    "target_ref": item.target_ref,
                    "value_id": item.value_id,
                    "kind": item.typed_value.kind.value,
                    "label": item.typed_value.label,
                    "value": prompt_value(item.typed_value.payload.canonical_value()),
                    "certification_refs": list(item.certification_refs),
                }
                for item in self.request.catalog_values
            ],
        }


def _identifier_target(
    term: object, *, index: RequestedFactSemanticIndex
) -> dict[str, str]:
    if not isinstance(term, FactTerm) or not isinstance(
        term.value_type, IdentifierType
    ):
        return {}
    return {
        "identifier_of_set_ref": index.fact_local_ref_by_local_id[
            term.value_type.set_ref
        ].token
    }


__all__ = [
    "SEMANTIC_SOURCE_BINDING_TOOL_NAME",
    "SemanticSourceBindingTurnPrompt",
]
