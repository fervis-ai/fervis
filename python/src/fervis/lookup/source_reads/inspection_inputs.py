"""Bind an observed response probe to original, typed question operands."""

from dataclasses import dataclass

from fervis.lookup.question_contract.model import InputTerm, QuestionContract
from fervis.lookup.relation_catalog.model import (
    EndpointRead, RelationCatalog, requires_caller_supplied_input,
)
from fervis.lookup.relation_catalog.parameter_values import parse_catalog_parameter_text
from fervis.lookup.turn_prompts import (
    TurnPromptBase, ProviderToolContract, ProviderResponseContract,
)
from fervis.model_io.structured_output.specs import required_tool_spec


@dataclass(frozen=True)
class InspectionInputTarget:
    read: EndpointRead
    options: dict[str, tuple[InputTerm, ...]]
    choice_values: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class InspectionInputRequest:
    targets: tuple[InspectionInputTarget, ...]
    meanings: dict[str, str]


def _valid_declared_choices(param) -> tuple[str, ...]:
    declared = param.choices or (
        ("false", "true") if param.type == "boolean" else ()
    )
    for choice in declared:
        try:
            parse_catalog_parameter_text(
                choice, type_name=param.type
            )
        except ValueError:
            return ()
    return declared


def inspection_input_request(
    *, catalog: RelationCatalog, read_ids: tuple[str, ...],
    contract: QuestionContract, indexes, fact_selections, certified_values,
) -> InspectionInputRequest:
    """Offer fact-owned original inputs and API-declared finite choices."""
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
            or read.fields
            or (read.source_metadata or {}).get("representation_status") in {
                "unavailable", "read_failed"
            }
            or (read.source_metadata or {}).get("representation_authority") == "observed_response"
        ):
            continue
        options = {}
        choice_values = {}
        missing_required = False
        for param in read.params:
            choices = _valid_declared_choices(param)
            candidates = []
            for input_ref in selected[read.id] & certified:
                term = inputs[input_ref]
                if not isinstance(term.operand, str):
                    continue
                if param.choices and not choices:
                    continue
                try:
                    parse_catalog_parameter_text(
                        term.operand, type_name=param.type,
                        choices=choices or param.choices,
                    )
                except ValueError:
                    continue
                candidates.append(term)
            if requires_caller_supplied_input(param) and not (candidates or choices):
                missing_required = True
            if candidates or choices:
                options[param.ref] = tuple(sorted(candidates, key=lambda item: item.id))
                choice_values[param.ref] = choices
        if options and not missing_required:
            targets.append(InspectionInputTarget(read, options, choice_values))
    return InspectionInputRequest(tuple(targets), {
        denotation.input_ref: denotation.operand_meaning
        for denotation in contract.input_denotations
    })


def inspection_input_schema(request: InspectionInputRequest):
    def obj(properties):
        return {"type": "object", "properties": properties,
                "required": list(properties), "additionalProperties": False}

    def parameter_inputs(target):
        params = {param.ref: param for param in target.read.params}
        return obj({ref: {"enum": [
            *(item.id for item in options),
            *(f"choice:{choice}" for choice in target.choice_values[ref]),
            *([] if requires_caller_supplied_input(params[ref]) else ["omit"]),
        ]} for ref, options in target.options.items()})

    return obj({"reads": obj({target.read.id: {
        "oneOf": [
            obj({"kind": {"enum": ["unsupported"]},
                 "reason": {"type": "string", "minLength": 1}}),
            obj({"kind": {"enum": ["bound_arguments"]},
                 "mapping_basis": {"type": "string", "minLength": 1},
                 "parameter_inputs": parameter_inputs(target)}),
        ]
    } for target in request.targets})})


@dataclass(frozen=True)
class InspectionInputTurnPrompt(TurnPromptBase):
    request: InspectionInputRequest
    turn_name: str = "inspection input grounding"
    turn_task: str = "bind selected read arguments before response inspection"

    def instruction_sections(self, builder):
        return (builder.instruction_block("Response inspection address", (
            "Choose only an original supplied input or a declared finite API choice whose meaning matches the API parameter.",
            "A declared choice is executable only because the API contract lists it; do not infer choices from sampled response values.",
            "For an optional parameter, choose omit unless the question actually supplies that qualifier or representation.",
            "Matching syntax or scalar type alone does not establish that an input names the addressed resource.",
            "Return unsupported when a required parameter has no question-owned input or question-justified declared choice, or meanings differ.",
            "This choice only inspects current response structure; later source realization must bind the actual read arguments independently.",
        )),)

    def data_sections(self, builder):
        return (builder.json_section("Selected API reads and certified argument options:", [
            {
                "read_id": target.read.id,
                "path": target.read.path,
                "description": target.read.description,
                "parameters": [
                    {"parameter_ref": param.ref, "name": param.name,
                     "type": param.type, "required": requires_caller_supplied_input(param),
                     "description": param.description,
                     "input_options": [
                         {"input_ref": item.id, "operand": item.operand,
                          "meaning": self.request.meanings.get(item.id, "")}
                         for item in target.options[param.ref]
                     ],
                     "declared_choice_options": [
                         {"value": choice,
                          "label": (param.choice_labels or {}).get(choice, choice)}
                         for choice in target.choice_values[param.ref]
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
        if proposal["kind"] != "bound_arguments":
            raise ValueError("inspection grounding has unknown decision")
        target = targets[read_id]
        chosen = proposal["parameter_inputs"]
        if set(chosen) != set(target.options) or not proposal["mapping_basis"].strip():
            raise ValueError("inspection grounding has incomplete address")
        params = {param.ref: param for param in target.read.params}
        args = {}
        for ref, input_ref in chosen.items():
            if input_ref == "omit":
                if requires_caller_supplied_input(params[ref]):
                    raise ValueError("required inspection address cannot be omitted")
                continue
            if input_ref.startswith("choice:"):
                choice = input_ref.removeprefix("choice:")
                if choice not in target.choice_values[ref]:
                    raise ValueError("inspection choice is not a declared inspection choice")
                args[ref] = parse_catalog_parameter_text(
                    choice, type_name=params[ref].type,
                    choices=target.choice_values[ref],
                )
                continue
            term = next((item for item in target.options[ref] if item.id == input_ref), None)
            if term is None or not isinstance(term.operand, str):
                raise ValueError("inspection address is not an eligible original input")
            args[ref] = parse_catalog_parameter_text(
                term.operand, type_name=params[ref].type, choices=params[ref].choices
            )
        bound[read_id] = args
    return bound
