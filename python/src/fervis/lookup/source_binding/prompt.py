"""Model-facing semantic Source Binding contract."""

from __future__ import annotations


from fervis.lookup.source_binding.model import SemanticSourceBindingRequest
from fervis.lookup.source_binding.model import SourceRealization
from fervis.lookup.available_sources import SourceChoiceSurfaceKind, SourceFieldBinding
from fervis.lookup.source_binding.schema import (
    build_semantic_source_binding_schema,
    build_semantic_source_realization_schema,
    build_unavailable_source_realization_schema,
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
                "Available binding scope and sources:",
                self._sources_payload(),
                indent=2,
            ),
            builder.json_section(
                "Canonical input values:", self._values_payload(), indent=2
            ),
        )

    def _choice_surface_payload(self, source_ref, target_ref, kind):
        surface = self.request.source_catalog.choice_surface_at(
            source_ref, target_ref, kind
        )
        return {
            "choice_surface_ref": surface.surface_ref if surface is not None else None,
            "choices": [
                {
                    "value_ref": value.value_ref,
                    "value": value.value,
                    "label": value.label,
                    "explicit_subject_requirement_refs": list(
                        dict.fromkeys(
                            ref
                            for branch in self.request.strategy.branches
                            for ref in self.request.explicit_subject_requirement_refs(
                                value, branch_id=branch.branch_id
                            )
                        )
                    ),
                }
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
                    else "RESOURCE_POPULATION"
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
                    "request_parameters_supplied_by_complete_traversal": sorted(self.request.source_catalog.read_access.supplied_parameters(source)),
                    "parent_row_path": source.parent_row_path,
                    "fields": [
                        {
                            "field_ref": SourceFieldBinding(source.id, field).ref,
                            "response_path": field.response_path,
                            "label": field.label,
                            "description": field.description,
                            "type": field.type.value,
                            **(
                                {"fixed_value": field.declared_entity_kind}
                                if field.declared_entity_kind
                                else {}
                            ),
                            **self._choice_surface_payload(
                                source.id,
                                field.field_ref,
                                SourceChoiceSurfaceKind.RETURNED_FIELD,
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
                    "union_identity_field_refs": list(source.stable_grain_field_refs),
                    "parameters": [
                        {
                            "parameter_ref": item.param_ref,
                            "name": item.name,
                            "description": item.description,
                            "type": item.type.value,
                            "required": item.required,
                            "default": item.default,
                            "default_is_known": item.default_is_known,
                            "complete_read_arguments": list(self.request.complete_read_arguments(source.id, item.param_ref)),
                            "row_admission": (
                                population.to_public_dict()
                                if (population := self.request.parameter_population(source.id, item.param_ref)) is not None else None
                            ),
                            **self._choice_surface_payload(
                                source.id,
                                item.param_ref,
                                SourceChoiceSurfaceKind.REQUEST_PARAMETER,
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


SOURCE_REALIZATION_OUTCOME_INSTRUCTION = (
    "Use submit_source_realization only when the shown rows, fields, and relationships semantically realize every required set and fact. "
    "If they do not, use report_unavailable_source_realization and identify the unmet requirement refs. "
    "Candidate rows may cover a broader population only when the shown fields can define the logical set. An independent population interpretation must establish executable membership before binding or execution. "
    "Do not substitute a different population or field merely because it is the only structurally compatible choice. "
    "A count or other summary on parent rows does not make those rows instances of the summarized population."
)


def unavailable_source_realization_tool_spec(requirement_refs: tuple[str, ...]):
    return required_tool_spec(
        tool_name="report_unavailable_source_realization",
        tool_description="Report semantic requirements that the available source candidates cannot realize.",
        input_schema=build_unavailable_source_realization_schema(requirement_refs),
    )


class SemanticSourceRealizationTurnPrompt(_SourcePromptBase):
    turn_name = "source realization"
    turn_task = "select source rows, fields, and declared relationships for the semantic contract"

    def instruction_sections(self, builder):
        return (
            builder.instruction_block(
                "Source realization",
                (
                    "The sources are candidates. Select the rows that represent each declared set and the fields that realize its facts.",
                    "Sources describe logical row populations. complete_read_arguments supplies verified controls for retrieving every row with respect to that parameter. When union_identity_field_refs is nonempty, multiple arguments are alternative requests whose results the engine unions using that identity; one unrestricted HTTP request is not required. Population and invocation coverage are verified before execution.",
                    "request_parameters_supplied_by_complete_traversal are operational inputs obtained from other source rows by the execution engine. They do not require user-supplied identities. Select the resulting logical population on its declared row meaning and fields; its prerequisite reads are already part of the access plan.",
                    "Each logical set must have an explicit set_binding. Its mapping_basis explains why the selected rows or referenced identities represent that set's instance kind; structural link compatibility does not establish this meaning.",
                    "Each association selects only relationship evidence connecting its already assigned endpoints. It cannot choose or change either endpoint population. Shared sets have exactly one assignment per branch.",
                    "Each identity identifies its declared entity kind. A primary key identifies the source row; an entity-reference key identifies the referenced entity.",
                    "A fact used as a returned value or arithmetic operand requires a field_ref on its chosen rows. A referenced identity cannot supply scalar fields of the referenced entity.",
                    "For a qualification-only fact, bind a returned field only when it expresses that fact. When a request predicate supplies the qualification and no returned field expresses it, use an empty fact_bindings array. Do not attach another field as a placeholder for an invocation predicate. Input application and exact request predicates are bound in the next step.",
                    SOURCE_REALIZATION_OUTCOME_INSTRUCTION,
                ),
            ),
        )

    def _requirements_payload(self):
        payload = super()._requirements_payload()
        for item in payload["terms"]:
            ref = FactLocalRef.from_token(item["requirement_ref"])
            term = self.request.index.term_by_ref[ref]
            if isinstance(term, FactTerm):
                item["owner_ref"] = self.request.index.fact_local_ref_by_local_id[
                    term.owner_ref
                ].token
                item["observed_value_required"] = (
                    ref in self.request.index.observed_fact_refs
                )
            elif ref.kind.value == "set":
                item["rows_refs"] = list(self.request.row_references_for_set(ref.token))
        from fervis.lookup.source_binding.association_choices import (
            association_endpoints,
        )

        for item in payload["associations"]:
            item["from_set_ref"], item["to_set_ref"] = association_endpoints(
                self.request, item["requirement_ref"]
            )
        return payload

    def response_contract(self):
        return ProviderResponseContract(
            provider_schema={
                spec.name: spec.input_schema for spec in self.tool_contract().tool_specs
            }
        )

    def tool_contract(self):
        return ProviderToolContract(
            tool_specs=(
                required_tool_spec(
                    tool_name="submit_source_realization",
                    tool_description="Submit source row and field realizations.",
                    input_schema=build_semantic_source_realization_schema(self.request),
                ),
                unavailable_source_realization_tool_spec(
                    tuple(
                        ref.token
                        for ref in sorted(self.request.index.source_requirement_refs)
                    )
                ),
            )
        )


SOURCE_INPUT_APPLICATION_INSTRUCTION = (
    "Shown component and target options establish structural compatibility, not semantic meaning. "
    "Apply a resolved input only to a request target whose declared meaning implements its owned predicate. "
    "Leave it unapplied when returned facts or returned choices realize that predicate; "
    "do not add an unrelated request filter merely to consume an input. "
    "Choose kind=no_request_application when the value needs no direct API application, or kind=request_application for a semantic request-target binding. "
    "Choose this kind before writing mapping_basis. Returned-fact evaluation and finite-choice applications remain available."
)


class SemanticSourceBindingTurnPrompt(_SourcePromptBase):
    turn_name = "source binding"
    turn_task = "bind inputs and population controls to the fixed source realization"

    def __init__(self, realization: SourceRealization):
        super().__init__(realization.request)
        self.realization = realization

    def data_sections(self, builder):
        fixed = {
            "sets": {
                ref: [
                    {"source_ref": value.source_ref, "identity_ref": value.identity_ref}
                    for value in values
                ]
                for ref, values in self.realization.set_bindings.items()
            },
            "facts": {
                ref: [
                    {
                        "source_ref": value.source_ref,
                        "field_refs": list(value.field_refs),
                    }
                    for value in values
                ]
                for ref, values in self.realization.fact_bindings.items()
            },
            "associations": {
                ref: [
                    {
                        "source_refs": list(value.source_refs),
                        "relation_evidence_ref": value.relation_evidence_ref,
                    }
                    for value in values
                ]
                for ref, values in self.realization.association_bindings.items()
            },
        }
        return (
            *super().data_sections(builder),
            builder.json_section("Fixed realizations:", fixed, indent=2),
        )

    def instruction_sections(
        self, builder: TurnPromptBuilder
    ) -> tuple[PromptSection, ...]:
        return (
            builder.instruction_block(
                "Resolved input application",
                (
                    "resolved_input_applications has one shown branch key. Each item chooses whether one resolved input value needs a direct request application in that branch.",
                    "Write kind first, then mapping_basis, owner_ref, and value_ref. Only request_application includes value_component and target_ref.",
                    SOURCE_INPUT_APPLICATION_INSTRUCTION,
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
                    "Required-source owners select the choices required to invoke that source. A finite parameter domain does not prove its result sets cover the requested population; do not invent a population restriction.",
                    "Every population restriction must implement a shown question requirement or declared invocation requirement.",
                ),
            ),
            builder.instruction_block(
                "Choice requirement correspondence",
                (
                    "Preserve the API resource population unless a shown requirement narrows it. Lifecycle state alone is not authority to exclude a resource.",
                    "choice_requirement_applications is keyed by branch, surface, and choice. It states which exact choices satisfy each shown predicate, whether or not that predicate can be pushed into an invocation.",
                    "For each choice, write mapping_basis explaining which shown requirement predicates that exact value satisfies, then selected_by_requirements listing those requirement refs. Use an empty list when this field or choice does not implement a shown predicate.",
                    "A requirement ref means the complete signed predicate. Its presence selects that exact choice as satisfying the predicate; it does not merely say that the choice exists in the same source.",
                    "The runtime applies these choices only within the Boolean requirement that owns them, preserving the full qualification expression.",
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
