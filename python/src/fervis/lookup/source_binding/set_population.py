"""Interpret each logical set independently of its consumers' qualifications."""

from dataclasses import dataclass, replace

from fervis.lookup.provider_contract import ProviderOutput, ProviderObject
from fervis.lookup.question_contract import FactLocalRef
from fervis.lookup.turn_prompts import (
    TurnPromptBase,
    ProviderResponseContract,
    ProviderToolContract,
)
from fervis.model_io.structured_output.specs import required_tool_spec
from fervis.lookup.source_binding.membership import (
    membership_schema,
    membership_definitions,
    parse_membership,
)
from fervis.lookup.source_binding.prompt import unavailable_source_realization_tool_spec
from fervis.lookup.available_sources import SourceFieldBinding, SourceChoiceSurfaceKind
from fervis.lookup.source_binding.parser import compile_source_realization
from fervis.lookup.source_binding.model import SourceRealizationUnavailable
from fervis.lookup.answer_program.expressions import expression_references


@dataclass(frozen=True)
class SetPopulationDecisionOutput(ProviderOutput):
    branch_id: str
    logical_set_meaning: str
    mapping_basis: str
    population: ProviderObject


@dataclass(frozen=True)
class SetPopulationsOutput(ProviderOutput):
    populations: dict[str, tuple[SetPopulationDecisionOutput, ...]]


def population_request(realization):
    # Supplied predicates belong to their existing logical owners. Exposing their
    # operands here lets an interpretation of one set borrow a consumer's filter.
    return replace(realization.request, canonical_values=(), catalog_values=())


def set_populations_schema(realization):
    request = population_request(realization)
    fields = {}
    for ref, items in realization.set_bindings.items():
        fields[ref] = {
            "type": "array",
            "minItems": len(items),
            "maxItems": len(items),
            "items": SetPopulationDecisionOutput.schema(
                {
                    "branch_id": {"enum": [item.branch_id for item in items]},
                    "logical_set_meaning": {
                        "enum": [
                            request.index.term_by_ref[
                                FactLocalRef.from_token(ref)
                            ].origin.meaning
                        ]
                    },
                    "mapping_basis": {"type": "string", "minLength": 1},
                    "population": membership_schema(request),
                }
            ),
        }
    result = SetPopulationsOutput.schema(
        {
            "populations": {
                "type": "object",
                "properties": fields,
                "required": list(fields),
                "additionalProperties": False,
            }
        }
    )
    result["$defs"] = membership_definitions(request)
    return result


def apply_set_populations(payload, *, realization):
    request = population_request(realization)
    if payload.get("kind") == "unavailable_source_realization":
        outcome = compile_source_realization(payload, request=request)
        assert isinstance(outcome, SourceRealizationUnavailable)
        if not set(outcome.unmet_requirement_refs) <= set(realization.set_bindings):
            raise ValueError("population failure must identify a logical set")
        return outcome
    parsed = SetPopulationsOutput.parse(payload)
    if set(parsed.populations) != set(realization.set_bindings):
        raise ValueError("population decisions must cover the exact logical set scope")
    bindings = {}
    for ref, values in realization.set_bindings.items():
        decisions = parsed.populations[ref]
        if len(decisions) != len(values) or {item.branch_id for item in decisions} != {
            item.branch_id for item in values
        }:
            raise ValueError(
                "population decisions must cover each selected branch exactly once"
            )
        selected = []
        for value in values:
            decision = next(
                item for item in decisions if item.branch_id == value.branch_id
            )
            if (
                decision.logical_set_meaning
                != request.index.term_by_ref[
                    FactLocalRef.from_token(ref)
                ].origin.meaning
            ):
                raise ValueError(
                    "population decision belongs to another logical set meaning"
                )
            if not decision.mapping_basis.strip():
                raise ValueError(
                    "population decision requires its source meaning basis"
                )
            source = request.source_catalog.source(value.source_ref)
            membership = parse_membership(
                decision.population, source=source, request=request
            )
            refs = expression_references(membership) if membership is not None else None
            evidence = (
                *value.contract_evidence_refs,
                *(
                    (source.field(item.field_id).field_ref for item in refs.fields)
                    if refs is not None
                    else ()
                ),
                *(
                    (item.parameter_id for item in refs.parameters)
                    if refs is not None
                    else ()
                ),
            )
            selected.append(
                replace(
                    value,
                    membership=membership,
                    population_basis=decision.mapping_basis,
                    contract_evidence_refs=tuple(dict.fromkeys(evidence)),
                )
            )
        bindings[ref] = tuple(selected)
    return replace(realization, set_bindings=bindings)


class SetPopulationTurnPrompt(TurnPromptBase):
    turn_name = "logical set population"
    turn_task = "establish exact membership of each selected logical set"
    include_current_question = False

    def __init__(self, realization):
        self.realization = realization
        self.request = population_request(realization)

    def data_sections(self, builder):
        return (
            builder.json_section(
                "Logical set meanings and selected source rows:",
                {
                    "sets": [
                        {
                            "set_ref": ref,
                            "meaning": self.request.index.term_by_ref[
                                FactLocalRef.from_token(ref)
                            ].origin.meaning,
                            "selected_rows": [
                                {
                                    "branch_id": item.branch_id,
                                    "source_ref": item.source_ref,
                                    "identity_ref": item.identity_ref,
                                }
                                for item in values
                            ],
                        }
                        for ref, values in self.realization.set_bindings.items()
                    ],
                    "source_contracts": [
                        {
                            "source_ref": source.id,
                            "label": source.label,
                            "description": source.description,
                            "row_path": source.row_path,
                            "request_parameters_supplied_by_complete_traversal": sorted(self.request.source_catalog.read_access.supplied_parameters(source)),
                            "complete_read_controls": [
                                {
                                    "parameter_ref": param.param_ref,
                                    "description": param.description,
                                    "default": param.default,
                                    "arguments": list(
                                        self.request.complete_read_arguments(
                                            source.id, param.param_ref
                                        )
                                    ),
                                }
                                for param in source.params
                            ],
                            "fields": [
                                {
                                    "field_ref": SourceFieldBinding(
                                        source.id, field
                                    ).ref,
                                    "label": field.label,
                                    "description": field.description,
                                    "type": field.type.value,
                                    "nullable": field.nullable,
                                    "choices": [
                                        {
                                            "value_ref": value.value_ref,
                                            "value": value.value,
                                            "label": value.label,
                                        }
                                        for surface in self.request.source_catalog.choice_surfaces
                                        if surface.source_ref == source.id
                                        and surface.target_ref == field.field_ref
                                        and surface.kind
                                        is SourceChoiceSurfaceKind.RETURNED_FIELD
                                        for value in surface.values
                                    ],
                                }
                                for field in source.fields
                            ],
                        }
                        for source in self.request.source_catalog.sources
                    ],
                },
                indent=2,
            ),
        )

    def instruction_sections(self, builder):
        return (
            builder.text_section(
                "Population interpretation rules:",
                (
                    "Write the stated logical_set_meaning first. Then compare that meaning with the complete source population in mapping_basis before choosing exact or restricted. Complete source retrieval alone does not establish this comparison. Do not infer a consumer's desired filter or normal lifecycle policy.",
                    "Complete-read controls restore the source population across parameter defaults. Interpret that complete row population, not merely the default HTTP response.",
                    "Use exact_population when the selected rows already denote that set. A source containing multiple states still exactly represents an unqualified resource set.",
                    "Use restricted_population only when the logical set itself denotes a subset of the selected source. Supply an executable Boolean condition using the declared fields and values.",
                    "Do not invent thresholds, operands, or membership defaults. Unknown field types are not Boolean authority. If the required subset cannot be represented by the shown expression contract, report the set unavailable.",
                    "The runtime applies membership independently at each set occurrence before consumers join, aggregate, or quantify those rows. Consumer-specific predicates are handled separately and are not shown here.",
                ),
            ),
        )

    def tool_contract(self):
        return ProviderToolContract(
            tool_specs=(
                required_tool_spec(
                    tool_name="submit_set_populations",
                    tool_description="Define exact membership of the selected logical sets.",
                    input_schema=set_populations_schema(self.realization),
                ),
                unavailable_source_realization_tool_spec(
                    tuple(self.realization.set_bindings)
                ),
            )
        )

    def response_contract(self):
        return ProviderResponseContract(
            provider_schema={
                spec.name: spec.input_schema for spec in self.tool_contract().tool_specs
            }
        )
