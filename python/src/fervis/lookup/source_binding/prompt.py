"""Model-facing semantic Source Binding contract."""

from __future__ import annotations

from fervis.lookup.source_binding.model import SemanticSourceBindingRequest
from fervis.lookup.source_binding.membership import SourceMembership
from fervis.lookup.available_sources import SourceChoiceSurfaceKind, SourceFieldBinding
from fervis.lookup.source_binding.schema import (
    build_semantic_source_binding_schema,
    build_semantic_source_realization_schema,
)
from fervis.lookup.source_binding.subject_obligations import (
    NORMAL_INSTANCE_EXCLUDED_STATE_ROLES,
)
from fervis.lookup.question_contract import (
    FactTerm,
    FactLocalRef,
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


class _SourcePromptBase(TurnPromptBase):
    turn_name = "source binding"
    turn_task = "select source rows and bind the semantic requirements"

    def __init__(self, request: SemanticSourceBindingRequest) -> None:
        self.request = request

    def data_sections(self, builder: TurnPromptBuilder) -> tuple[PromptSection, ...]:
        return (
            builder.json_section(
                "Semantic requirements:", self._requirements_payload(), indent=2
            ),
            builder.json_section(
                "Available binding scope and sources:", self._sources_payload(), indent=2
            ),
            builder.json_section(
                "Canonical input values:", self._values_payload(), indent=2
            ),
        )

    def _choice_surface_payload(self, source_ref, target_ref, kind):
        surface = self.request.source_catalog.choice_surface_at(source_ref, target_ref, kind)
        return {
            "choice_surface_ref": surface.surface_ref if surface is not None else None,
            "choices": [
                {"value_ref": value.value_ref, "value": value.value, "label": value.label,
                 "explicit_subject_requirement_refs": list(dict.fromkeys(
                     ref for branch in self.request.strategy.branches
                     for ref in self.request.explicit_subject_requirement_refs(value, branch_id=branch.branch_id)
                 ))}
                for value in (surface.values if surface is not None else ())
            ],
        }

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
                    "row_path": source.row_path,
                    "row_cardinality": source.row_cardinality.value,
                    "parent_row_path": source.parent_row_path,
                    "fields": [
                        {
                            "field_ref": SourceFieldBinding(source.id, field).ref,
                            "response_path": field.response_path,
                            "label": field.label,
                            "type": field.type.value,
                            **({"fixed_value":field.declared_entity_kind} if field.declared_entity_kind else {}),
                            **self._choice_surface_payload(
                                source.id, field.field_ref, SourceChoiceSurfaceKind.RETURNED_FIELD,
                            ),
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
                            **self._choice_surface_payload(
                                source.id, item.param_ref, SourceChoiceSurfaceKind.REQUEST_PARAMETER,
                            ),
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



class SemanticSourceRealizationTurnPrompt(_SourcePromptBase):
    turn_name = "source realization"
    turn_task = "select source rows, fields, and declared relationships for the semantic contract"

    def instruction_sections(self, builder):
        return (builder.instruction_block("Source realization", (
            "The sources are candidates. Select the rows that represent each declared set and the fields that realize its facts.",
            "Each association chooses from_rows_ref, to_rows_ref, and realization_ref together from its shown structurally compatible options. These bind the association's declared from_set_ref and to_set_ref.",
            "A set shared by multiple associations must use the same rows_ref in every association. set_bindings contains only isolated sets without associations; their rows_ref is selected directly.",
            "Each identity identifies its declared entity kind. A primary key identifies the source row; an entity-reference key identifies the referenced entity.",
            "Every observed fact requires one field_ref. A fact about a set uses that set's chosen rows. A referenced identity cannot supply scalar fields of the referenced entity.",
            "Bind a qualification-only fact whenever a returned field expresses it. Leave it unbound only when source predicate mechanics can realize it. Input application and ordinary-instance membership are bound in the next step.",
            "Return set_bindings, fact_bindings, and association_bindings. Do not write request inputs or subject-state reviews.",
        )),)

    def _requirements_payload(self):
        payload = super()._requirements_payload()
        for item in payload["terms"]:
            ref = FactLocalRef.from_token(item["requirement_ref"])
            term = self.request.index.term_by_ref[ref]
            if isinstance(term, FactTerm):
                item["owner_ref"] = self.request.index.fact_local_ref_by_local_id[term.owner_ref].token
                item["observed_value_required"] = ref in self.request.index.observed_fact_refs
            elif ref.kind.value == "set":
                item["rows_refs"] = list(self.request.row_references_for_set(ref.token))
        from fervis.lookup.source_binding.association_choices import association_endpoints

        for item in payload["associations"]:
            item["from_set_ref"], item["to_set_ref"] = association_endpoints(
                self.request, item["requirement_ref"]
            )
        return payload

    def response_contract(self):
        return ProviderResponseContract(provider_schema={
            "submit_source_realization": build_semantic_source_realization_schema(self.request),
        })

    def tool_contract(self):
        return ProviderToolContract(tool_specs=(required_tool_spec(
            tool_name="submit_source_realization", tool_description="Submit source row and field realizations.",
            input_schema=build_semantic_source_realization_schema(self.request),
        ),))


class SemanticSourceBindingTurnPrompt(_SourcePromptBase):
    turn_name = "source binding"
    turn_task = "bind inputs and population controls to the fixed source realization"

    def __init__(self, membership: SourceMembership):
        super().__init__(membership.realization.request)
        self.realization = membership.realization
        self.membership = membership

    def data_sections(self, builder):
        fixed = {
            "sets": {ref: [{"source_ref": value.source_ref, "identity_ref": value.identity_ref}
                            for value in values] for ref, values in self.realization.set_bindings.items()},
            "facts": {ref: [{"source_ref": value.source_ref, "field_refs": list(value.field_refs)}
                             for value in values] for ref, values in self.realization.fact_bindings.items()},
            "associations": {ref: [{"source_refs": list(value.source_refs), "relation_evidence_ref": value.relation_evidence_ref}
                                    for value in values] for ref, values in self.realization.association_bindings.items()},
        }
        return (*super().data_sections(builder), builder.json_section("Fixed realizations:", fixed, indent=2),
                builder.json_section("Fixed ordinary membership:", self.membership.reviews, indent=2))

    def instruction_sections(
        self, builder: TurnPromptBuilder
    ) -> tuple[PromptSection, ...]:
        return (
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
                    "finite_choice_applications is keyed first by branch and then by each shown requirement owner. Use null for a Boolean owner when no request-parameter choice implements its exact predicate; returned facts or returned choices must then realize it.",
                    "For each owner, write application_basis before surface_ref and selected_choice_values.",
                    "application_basis states why the selected request surface and choices realize that requirement.",
                    "surface_ref is the one request parameter whose declared meaning expresses the requirement.",
                    "selected_choice_values contains every choice on that surface that satisfies the requirement.",
                    "Required-source owners select the choices required to invoke that source.",
                    "For a raw-record subject, subject_scope owners reconsider defaults that may hide requested records. Select the declared choices that expose the requested raw-record scope.",
                    "Normal-instance reviews do not own requirement application.",
                ),
            ),
            builder.instruction_block(
                "Choice requirement correspondence",
                (
                    "Ordinary membership is fixed evidence. It was classified without explicit question qualifications and cannot be changed here.",
                    "choice_requirement_applications is keyed by branch, surface, and choice. It states which exact choices satisfy each shown predicate, whether or not that predicate can be pushed into an invocation.",
                    "For each choice, write mapping_basis explaining which shown requirement predicates that exact value satisfies, then selected_by_requirements listing those requirement refs. Use an empty list when this field or choice does not implement a shown predicate.",
                    "A requirement ref means the complete signed predicate. Its presence selects that exact choice as satisfying the predicate; it does not merely say that the choice exists in the same source.",
                    "An explicit predicate may select a choice excluded from ordinary membership. The runtime combines the fixed baseline with these explicit applications and preserves the full Boolean qualification.",
                ),
            ),
            builder.instruction_block(
                "Output",
                (
                    "Return resolved_input_applications, finite_choice_applications, and choice_requirement_applications with every shown key exactly once. The row, field, and association realizations are fixed inputs.",
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
    "SemanticSourceRealizationTurnPrompt",
]
