"""Bind an observed response probe to original, typed question operands."""

from dataclasses import dataclass

from fervis.lookup.question_contract.model import InputTerm, QuestionContract
from fervis.lookup.relation_catalog.model import (
    EndpointRead, RelationCatalog, requires_caller_supplied_input,
)
from fervis.lookup.relation_catalog.parameter_values import parse_catalog_parameter_text
from fervis.lookup.source_reads.representation import can_inspect_representation
from fervis.lookup.turn_prompts import (
    TurnPromptBase, ProviderToolContract, ProviderResponseContract,
)
from fervis.model_io.structured_output.specs import required_tool_spec


@dataclass(frozen=True)
class InspectionInputTarget:
    read: EndpointRead
    options: dict[str, tuple[InputTerm, ...]]


@dataclass(frozen=True)
class InspectionInputRequest:
    targets: tuple[InspectionInputTarget, ...]
    meanings: dict[str, str]


def inspection_input_request(
    *, catalog: RelationCatalog, read_ids: tuple[str, ...],
    contract: QuestionContract, indexes, fact_selections, certified_values,
) -> InspectionInputRequest:
    """Offer only original question operands used by the selected fact."""
    certified = {
        value.input_ref for value in certified_values
        if f"question_input:{value.input_ref}" in value.certification_refs
    }
    inputs = {item.id: item for item in contract.inputs}
    by_fact = {
        index.requested_fact_id: {use.input_ref for use in index.input_use_sites}
        for index in indexes
    }
    selected = {
        read_id: {
            input_ref for fact in fact_selections
            if read_id in fact.selected_read_ids
            for input_ref in by_fact.get(fact.requested_fact_id, ())
        }
        for read_id in read_ids
    }
    targets = []
    for read in catalog.reads:
        if (
            read.id not in selected
            or can_inspect_representation(read)
            or read.fields
            or (read.source_metadata or {}).get("representation_status") in {
                "unavailable", "read_failed"
            }
            or not any(requires_caller_supplied_input(param) for param in read.params)
        ):
            continue
        options = {}
        for param in read.params:
            if not requires_caller_supplied_input(param):
                continue
            candidates = []
            for input_ref in selected[read.id] & certified:
                term = inputs[input_ref]
                if not isinstance(term.operand, str):
                    continue
                try:
                    parse_catalog_parameter_text(
                        term.operand, type_name=param.type, choices=param.choices
                    )
                except ValueError:
                    continue
                candidates.append(term)
            options[param.ref] = tuple(sorted(candidates, key=lambda item: item.id))
        if options and all(options.values()):
            targets.append(InspectionInputTarget(read, options))
    return InspectionInputRequest(tuple(targets), {
        denotation.input_ref: denotation.operand_meaning
        for denotation in contract.input_denotations
    })


def inspection_input_schema(request: InspectionInputRequest):
    def obj(properties):
        return {"type": "object", "properties": properties,
                "required": list(properties), "additionalProperties": False}

    return obj({"reads": obj({target.read.id: {
        "oneOf": [
            obj({"kind": {"enum": ["unsupported"]},
                 "reason": {"type": "string", "minLength": 1}}),
            obj({"kind": {"enum": ["supplied_input"]},
                 "mapping_basis": {"type": "string", "minLength": 1},
                 "parameter_inputs": obj({ref: {"enum": [item.id for item in options]}
                                          for ref, options in target.options.items()})}),
        ]
    } for target in request.targets})})


@dataclass(frozen=True)
class InspectionInputTurnPrompt(TurnPromptBase):
    request: InspectionInputRequest
    turn_name: str = "inspection input grounding"
    turn_task: str = "map original question values to required read addresses"

    def instruction_sections(self, builder):
        return (builder.instruction_block("Response inspection address", (
            "Choose only an original supplied input whose meaning matches the required API parameter.",
            "Matching syntax or scalar type alone does not establish that an input names the addressed resource.",
            "Return unsupported when the question does not supply every required address or meanings differ.",
            "This choice only inspects current response structure; later source realization must bind the actual read arguments independently.",
        )),)

    def data_sections(self, builder):
        return (builder.json_section("Selected API reads and supplied inputs:", [
            {
                "read_id": target.read.id,
                "path": target.read.path,
                "description": target.read.description,
                "parameters": [
                    {"parameter_ref": param.ref, "name": param.name,
                     "type": param.type, "description": param.description,
                     "input_options": [
                         {"input_ref": item.id, "operand": item.operand,
                          "meaning": self.request.meanings.get(item.id, "")}
                         for item in target.options[param.ref]
                     ]}
                    for param in target.read.params if param.ref in target.options
                ],
            }
            for target in self.request.targets
        ], indent=2),)

    def tool_contract(self):
        return ProviderToolContract(tool_specs=(required_tool_spec(
            tool_name="submit_inspection_inputs",
            tool_description="Choose supplied address values for structural inspection.",
            input_schema=inspection_input_schema(self.request),
        ),))

    def response_contract(self):
        return ProviderResponseContract(provider_schema={
            "submit_inspection_inputs": inspection_input_schema(self.request)
        })


def parse_inspection_inputs(payload, *, request: InspectionInputRequest):
    proposals = payload["reads"]
    targets = {target.read.id: target for target in request.targets}
    if set(proposals) != set(targets):
        raise ValueError("inspection grounding omits or invents selected reads")
    bound = {}
    for read_id, proposal in proposals.items():
        if proposal["kind"] == "unsupported":
            if not str(proposal.get("reason") or "").strip():
                raise ValueError("unsupported inspection route requires a reason")
            continue
        if proposal["kind"] != "supplied_input":
            raise ValueError("inspection grounding has unknown decision")
        target = targets[read_id]
        chosen = proposal["parameter_inputs"]
        if set(chosen) != set(target.options) or not proposal["mapping_basis"].strip():
            raise ValueError("inspection grounding has incomplete address")
        params = {param.ref: param for param in target.read.params}
        args = {}
        for ref, input_ref in chosen.items():
            term = next((item for item in target.options[ref] if item.id == input_ref), None)
            if term is None or not isinstance(term.operand, str):
                raise ValueError("inspection address is not an eligible original input")
            args[ref] = parse_catalog_parameter_text(
                term.operand, type_name=params[ref].type, choices=params[ref].choices
            )
        bound[read_id] = args
    return bound
