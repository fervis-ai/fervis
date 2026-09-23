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


def _singular_name(name: str) -> str:
    if name.endswith("ies") and len(name) > 3:
        return name[:-3] + "y"
    if name.endswith("s") and not name.endswith("ss") and len(name) > 1:
        return name[:-1]
    return name


def _name_parts(name: str) -> tuple[str, ...]:
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return tuple(_singular_name(part) for part in re.findall(r"[a-z0-9]+", separated.casefold()))


def _argument_name_affinity(parent: RowSource, required) -> int:
    """Order compatible candidates; names never certify a traversal or key."""
    resources = {
        part for name in parent.resource_names for part in _name_parts(name)
    }
    fields = (*parent.fields, *parent.request_argument_fields)
    score = 0
    for param in required:
        param_parts = _name_parts(param.name or param.param_ref)
        if not param_parts:
            continue
        identity_parts = tuple(part for part in param_parts if part not in {"id", "uuid", "pk"})
        best = 0
        for field in fields:
            if field.declared_entity_kind or not _compatible(field, param, parent):
                continue
            field_parts = _name_parts(field.id)
            if field_parts == param_parts:
                best = max(best, 6)
            elif identity_parts and field_parts == identity_parts:
                best = max(best, 5)
            elif field_parts in {("id",), ("uuid",), ("pk",)} and identity_parts and all(
                part in resources for part in identity_parts
            ):
                best = max(best, 4)
            elif identity_parts and any(part in field_parts for part in identity_parts):
                best = max(best, 2)
        score += best
    return score


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
                    -_argument_name_affinity(parent, required),
                    parent.id,
                    parent,
                )
            )
    candidates.sort(key=lambda item: item[:4])
    return tuple(item[4] for item in candidates)


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

            def describe(item, *, field_ids):
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
                        if f.id in field_ids and not f.declared_entity_kind
                    ],
                }

            payload.append(
                {
                    "target": describe(source, field_ids=_identity_field_ids(source)),
                    "parent_candidates": [
                        describe(
                            parent,
                            field_ids=_compatible_parent_field_ids(source, parent),
                        )
                        for parent in self.request.candidates(source)
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


def _compatible_parent_field_ids(target: RowSource, parent: RowSource) -> set[str]:
    required = tuple(
        param for param in target.params if requires_caller_supplied_input(param)
    )
    return {
        field.id
        for field in (*parent.fields, *parent.request_argument_fields)
        if not field.declared_entity_kind
        and any(_compatible(field, param, parent) for param in required)
    }


def _identity_field_ids(source: RowSource) -> set[str]:
    return {
        field_id
        for key in source.candidate_keys
        for field_id in (
            *(component.field_id for component in key.components),
            *key.context_field_ids,
        )
    } | {
        field_id
        for reference in source.entity_references
        for field_id in (
            *(component.local_field_id for component in reference.components),
            *reference.context_field_ids,
        )
    }


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
