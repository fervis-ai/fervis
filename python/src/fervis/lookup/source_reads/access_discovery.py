"""Question-independent proposals for complete REST prerequisite traversals."""

from dataclasses import dataclass, asdict
import re
from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.relation_catalog.row_sources import (
    RowSource,
    RowSourceCatalog,
    RowSourceKind,
)
from fervis.lookup.relation_catalog.model import (
    RowCardinality,
    requires_caller_supplied_input,
)
from fervis.lookup.plan_execution.declared_values import (
    declared_kind,
    DeclaredValueKind,
)
from fervis.lookup.turn_prompts import (
    TurnPromptBase,
    ProviderToolContract,
    ProviderResponseContract,
)
from fervis.model_io.structured_output.specs import required_tool_spec
from fervis.lookup.api_arguments import compatible_argument, row_source_argument_description
from .access_model import ReadAccessCatalog, ReadDependency, AccessArgument


def _compatible(field, param, parent):
    return declared_kind(field.type.value) is not DeclaredValueKind.RUNTIME and compatible_argument(
        asdict(param), row_source_argument_description(parent, field.field_ref, param.entity_target))


def access_candidates(
    source: RowSource, *, sources: RowSourceCatalog, catalog: RelationCatalog
):
    required = tuple(
        param for param in source.params if requires_caller_supplied_input(param)
    )
    if not required:
        return ()
    path = re.sub(r"{[^}]+}", "{}", catalog.read(source.read_id).path).rstrip("/")
    candidates = []
    for parent in sources.sources:
        if (
            parent.id == source.id
            or parent.kind is not RowSourceKind.API_READ
            or parent.row_cardinality is not RowCardinality.MANY
        ):
            continue
        fields = (*parent.fields, *parent.request_argument_fields)
        if all(
            any(
                not field.declared_entity_kind and _compatible(field, param, parent)
                for field in fields
            )
            for param in required
        ):
            parent_path = re.sub(
                r"{[^}]+}", "{}", catalog.read(parent.read_id).path
            ).rstrip("/")
            is_prefix = bool(parent_path) and path.startswith(parent_path + "/")
            candidates.append(
                (
                    not is_prefix,
                    -len(parent_path) if is_prefix else 0,
                    parent.id,
                    parent,
                )
            )
    candidates.sort(key=lambda item: item[:3])
    return tuple(item[3] for item in candidates)


@dataclass(frozen=True)
class AccessDiscoveryRequest:
    catalog: RelationCatalog
    sources: RowSourceCatalog
    target_sources: tuple[RowSource, ...]
    candidate_offset: int = 0
    candidate_limit: int = 12

    def candidates(self, source):
        return access_candidates(source, sources=self.sources, catalog=self.catalog)[
            self.candidate_offset : self.candidate_offset + self.candidate_limit
        ]


def _object(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def access_schema(request: AccessDiscoveryRequest):
    targets = {}
    for source in request.target_sources:
        parents = {}
        for parent in request.candidates(source):
            fields = (*parent.fields, *parent.request_argument_fields)
            mappings = {
                param.param_ref: {
                    "enum": [
                        field.field_ref
                        for field in fields
                        if not field.declared_entity_kind and _compatible(field, param, parent)
                    ]
                }
                for param in source.params
                if requires_caller_supplied_input(param)
            }
            parents[parent.id] = {
                "oneOf": [
                    _object(
                        {
                            "coverage_basis": {"type": "string", "minLength": 1},
                            "kind": {"enum": ["not_a_complete_traversal"]},
                        }
                    ),
                    _object(
                        {
                            "coverage_basis": {"type": "string", "minLength": 1},
                            "kind": {"enum": ["complete_traversal"]},
                            "argument_fields": _object(mappings),
                        }
                    ),
                ]
            }
        targets[source.id] = _object(parents)
    return _object({"source_traversals": _object(targets)})


@dataclass(frozen=True)
class ReadAccessTurnPrompt(TurnPromptBase):
    request: AccessDiscoveryRequest
    turn_name: str = "source access"
    turn_task: str = (
        "review complete request traversals from the supplied API contracts"
    )
    include_current_question: bool = False

    def instruction_sections(self, builder):
        return (
            builder.instruction_block(
                "Read access",
                (
                    "Review the API contracts independently of any user question or supplied business operand.",
                    "A complete traversal reads every parent row, then invokes the target read with the mapped values from that same row. Every target row must be obtainable through those requests.",
                    "A path prefix identifies candidates only; it is not proof of argument meaning or complete coverage. Verify both against the source descriptions and field meanings.",
                    "All required target arguments must come from one parent row. Request-context fields retain that parent row's actual path arguments; they are not fields returned in its JSON.",
                    "Do not substitute an unrelated identifier, a label for a code, or a numeric measurement for a request limit or threshold. Matching scalar types alone does not establish a traversal.",
                    "A parent restricted to a subset cannot cover children outside that subset. Do not assume undocumented populations, transformations, or identifiers.",
                    "Return not_a_complete_traversal when the contracts do not establish the mapping and coverage. Review every shown parent candidate for every target.",
                ),
            ),
        )

    def data_sections(self, builder):
        payload = []
        for source in self.request.target_sources:

            def describe(item):
                read = self.request.catalog.read(item.read_id)
                return {
                    "source_ref": item.id,
                    "path": read.path,
                    "description": item.description,
                    "parameters": [
                        {
                            "ref": p.param_ref,
                            "name": p.name,
                            "type": p.type.value,
                            "required": p.required,
                            "default": p.default,
                            "description": p.description,
                            "entity_target": asdict(p.entity_target) if p.entity_target else None,
                        }
                        for p in item.params
                    ],
                    "candidate_keys": [asdict(key) for key in item.candidate_keys],
                    "entity_references": [asdict(key) for key in item.entity_references],
                    "fields": [
                        {
                            "ref": f.field_ref,
                            "id": f.id,
                            "name": f.label,
                            "type": f.type.value,
                            "description": f.description,
                            "request_parameter_ref": f.request_parameter_ref,
                        }
                        for f in (*item.fields, *item.request_argument_fields)
                        if not f.declared_entity_kind
                    ],
                }

            payload.append(
                {
                    "target": describe(source),
                    "parent_candidates": [
                        describe(parent) for parent in self.request.candidates(source)
                    ],
                }
            )
        return (builder.json_section("API traversal candidates:", payload, indent=2),)

    def tool_contract(self):
        return ProviderToolContract(
            tool_specs=(
                required_tool_spec(
                    tool_name="submit_read_access",
                    tool_description="Record contract-backed complete read traversals.",
                    input_schema=access_schema(self.request),
                ),
            )
        )

    def response_contract(self):
        return ProviderResponseContract(
            provider_schema={"submit_read_access": access_schema(self.request)}
        )


def parse_read_access(payload, *, request: AccessDiscoveryRequest) -> ReadAccessCatalog:
    dependencies = []
    expected = {source.id for source in request.target_sources}
    proposals = payload["source_traversals"]
    if set(proposals) != expected:
        raise ValueError("access proposal omits or invents target sources")
    for source in request.target_sources:
        parents = {parent.id for parent in request.candidates(source)}
        if set(proposals[source.id]) != parents:
            raise ValueError("access proposal omits or invents parent candidates")
        for parent, proposal in proposals[source.id].items():
            if proposal["kind"] == "not_a_complete_traversal":
                continue
            if proposal["kind"] != "complete_traversal":
                raise ValueError("unknown read access decision")
            required = {
                p.param_ref for p in source.params if requires_caller_supplied_input(p)
            }
            if set(proposal["argument_fields"]) != required:
                raise ValueError("access proposal does not supply the required tuple")
            dependencies.append(
                ReadDependency(
                    source.id,
                    parent,
                    tuple(
                        AccessArgument(param, field)
                        for param, field in proposal["argument_fields"].items()
                    ),
                    proposal["coverage_basis"],
                    (f"read_access:{source.id}:{parent}",),
                )
            )
    result = ReadAccessCatalog(request.sources.sources, tuple(dependencies))
    result.validate()
    return result
