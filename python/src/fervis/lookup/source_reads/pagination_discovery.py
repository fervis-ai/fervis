"""Question-independent pagination interpretation over bounded API references."""

from dataclasses import dataclass, replace

from fervis.host_api.contracts import PaginationContract, PaginationKind
from fervis.lookup.provider_contract import ProviderObject, ProviderOutput
from fervis.lookup.relation_catalog.model import (
    RelationCatalog,
    RowCardinality,
    PaginationMode,
)
from fervis.lookup.turn_prompts import (
    TurnPromptBase,
    ProviderToolContract,
    ProviderResponseContract,
)
from fervis.model_io.structured_output.specs import required_tool_spec
from .pagination import bound_pagination_read
from fervis.host_api.compilation.pagination_binding import inside_collection


@dataclass(frozen=True)
class PaginationDiscoveryRequest:
    catalog: RelationCatalog
    read_ids: tuple[str, ...]

    @property
    def targets(self):
        return tuple(
            read
            for read in self.catalog.reads
            if read.id in self.read_ids
            and read.pagination_binding is None
            and not read.complete_single_response
            and (read.pagination is None or read.pagination.mode is PaginationMode.NONE)
            and _needs_traversal_interpretation(read)
            and any(
                path.cardinality is RowCardinality.MANY
                for path in read.row_paths
            )
        )


def _needs_traversal_interpretation(read) -> bool:
    query = tuple(param for param in read.params if param.source == "query")
    if any(param.type == "integer" for param in query):
        return True
    if (
        read.response_envelope.count_path
        or read.response_envelope.has_more_path
        or read.response_envelope.next_path
    ):
        return True
    if not query:
        return False
    arrays = tuple(
        path.path for path in read.row_paths
        if path.cardinality is RowCardinality.MANY
    )
    return any(
        not field.requirements
        and field.type in {"integer", "boolean", "string"}
        and not inside_collection(field.path, arrays)
        for field in read.fields
    )


@dataclass(frozen=True)
class PaginationDiscoveryOutput(ProviderOutput):
    reads: dict[str, ProviderObject]


@dataclass(frozen=True)
class PaginationMappingOutput(ProviderOutput):
    mapping_basis: str
    mode: str
    row_path_ref: str | None
    position_parameter_ref: str | None
    size_parameter_ref: str | None
    total_field_ref: str | None
    continuation_field_ref: str | None


def _position_parameters(read):
    return tuple(
        param
        for param in read.params
        if param.source == "query" and param.type == "integer"
    )


def _collection_paths(read):
    return tuple(
        path
        for path in read.row_paths
        if path.cardinality is RowCardinality.MANY and path.path
    )


def _object(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def pagination_schema(request):
    reads = {}
    for read in request.targets:
        params = [param.ref for param in _position_parameters(read)]
        rows = _collection_paths(read)
        arrays = tuple(
            path.path
            for path in read.row_paths
            if path.cardinality is RowCardinality.MANY
        )
        completion = tuple(
            field
            for field in read.fields
            if not field.requirements and not inside_collection(field.path, arrays)
        )
        can_bind_page = bool(
            params
            and rows
            and any(field.type in {"integer", "boolean", "string"} for field in completion)
        )
        properties = {
            "mapping_basis": {"type": "string", "minLength": 1},
            "mode": {"enum": ["single_response", "unknown"]},
            "row_path_ref": {"enum": [None]},
            "position_parameter_ref": {"enum": [None]},
            "size_parameter_ref": {"enum": [None]},
            "total_field_ref": {"enum": [None]},
            "continuation_field_ref": {"enum": [None]},
        }
        nonpaged = PaginationMappingOutput.schema(properties)
        if can_bind_page:
            paged_properties = {
                **properties,
                "mode": {"enum": ["page_number", "offset"]},
                "row_path_ref": {"enum": [path.id for path in rows]},
                "position_parameter_ref": {"enum": params},
                "size_parameter_ref": {"enum": [None, *params]},
                "total_field_ref": {"enum": [
                    None,
                    *(field.ref for field in completion if field.type == "integer"),
                ]},
                "continuation_field_ref": {"enum": [
                    None,
                    *(field.ref for field in completion if field.type in {"boolean", "string"}),
                ]},
            }
            reads[read.id] = {"oneOf": [
                nonpaged, PaginationMappingOutput.schema(paged_properties)
            ]}
        else:
            reads[read.id] = nonpaged
    return _object({"reads": _object(reads)})


def parse_pagination_discovery(payload, *, request):
    parsed = PaginationDiscoveryOutput.parse(payload)
    targets = {read.id: read for read in request.targets}
    if set(parsed.reads) != set(targets):
        raise ValueError("Pagination interpretation must cover the exact API scope")
    replacements = {}
    unavailable = set()
    for read_id, raw in parsed.reads.items():
        value = raw.parse_as(PaginationMappingOutput)
        if not value.mapping_basis.strip():
            raise ValueError("Pagination interpretation requires its evidence basis")
        if value.mode in {"single_response", "unknown"}:
            if any(
                ref is not None
                for ref in (
                    value.row_path_ref,
                    value.position_parameter_ref,
                    value.size_parameter_ref,
                    value.total_field_ref,
                    value.continuation_field_ref,
                )
            ):
                raise ValueError(
                    "Single-response or unknown mode cannot declare traversal references"
                )
            if value.mode == "unknown":
                unavailable.add(read_id)
            continue
        if (
            value.mode not in {"page_number", "offset"}
            or value.row_path_ref is None
            or value.position_parameter_ref is None
        ):
            raise ValueError("Pagination interpretation requires an explicit mapping")
        read = targets[read_id]
        params = {param.ref: param for param in _position_parameters(read)}
        rows = {path.id: path for path in _collection_paths(read)}
        fields = {field.ref: field for field in read.fields}
        refs = (
            value.position_parameter_ref,
            *(
                (value.size_parameter_ref,)
                if value.size_parameter_ref is not None
                else ()
            ),
        )
        if any(ref not in params for ref in refs) or value.row_path_ref not in rows:
            raise ValueError("Pagination mapping selects a foreign API reference")
        if any(
            ref is not None and ref not in fields
            for ref in (value.total_field_ref, value.continuation_field_ref)
        ):
            raise ValueError("Pagination mapping selects a foreign response field")
        size_param = (
            params.get(value.size_parameter_ref)
            if value.size_parameter_ref is not None
            else None
        )
        default = size_param.default if size_param is not None else 1
        size = default if type(default) is int and default > 0 else 1
        pagination = PaginationContract(
            PaginationKind(value.mode),
            params[value.position_parameter_ref].name,
            size_param.name if size_param is not None else "",
            rows[value.row_path_ref].path,
            size,
            size,
            total_path=fields[value.total_field_ref].path
            if value.total_field_ref is not None
            else "",
            continuation_path=fields[value.continuation_field_ref].path
            if value.continuation_field_ref is not None
            else "",
        )
        replacements[read_id] = bound_pagination_read(
            replace(read, pagination_binding=pagination)
        )
    return replace(
        request.catalog,
        reads=tuple(
            replacements.get(read.id, read)
            for read in request.catalog.reads
            if read.id not in unavailable
        ),
    )


@dataclass(frozen=True)
class PaginationDiscoveryPrompt(TurnPromptBase):
    request: PaginationDiscoveryRequest
    turn_name: str = "pagination contract"
    turn_task: str = "identify complete collection traversal from API metadata"
    include_current_question: bool = False

    def instruction_sections(self, builder):
        return (
            builder.instruction_block(
                "Pagination contracts",
                (
                    "Review each API independently of any user question. Pagination changes which slice of one population is returned; it does not select a business subset.",
                    "Choose mode page_number or offset only when the descriptions and response structure establish a position argument, collection rows, and a total count or continuation signal for that same collection. The selected mode is the traversal decision; do not make a separate pagination classification.",
                    "Select the position mode and exact references. Page-number mode starts at one and advances by one; offset mode starts at zero and advances by the number of returned rows.",
                    "A size argument controls rows per page. Its declared positive default becomes the requested size; otherwise execution uses one. Do not treat thresholds, category identifiers, or business limits as page controls.",
                    "Total and continuation fields must describe the collection as a whole, outside its row objects. Do not invent fields, completeness guarantees, or pagination from scalar types alone.",
                    "Use mode single_response when the declared query parameters are business filters or selectors and the contract exposes no way to request another slice of the same collection. The absence of paging controls alone does not make such a one-response collection unknown. Use mode unknown when a parameter or response field describes a page or continuation mechanism that cannot be bound to a supported complete traversal. For either mode, set every reference field to null.",
                ),
            ),
        )

    def data_sections(self, builder):
        return (
            builder.json_section(
                "API contracts:",
                [
                    {
                        "read_ref": read.id,
                        "path": read.path,
                        "description": read.description,
                        "parameters": [
                            {
                                "ref": param.ref,
                                "name": param.name,
                                "source": param.source,
                                "type": param.type,
                                "default": param.default,
                                "description": param.description,
                            }
                            for param in read.params
                        ],
                        "row_paths": [
                            {
                                "ref": path.id,
                                "path": path.path,
                                "cardinality": path.cardinality.value,
                            }
                            for path in read.row_paths
                        ],
                        "response_fields": [
                            {
                                "ref": field.ref,
                                "path": field.path,
                                "type": field.type,
                                "description": (field.metadata or {}).get(
                                    "description", ""
                                ),
                            }
                            for field in read.fields
                        ],
                    }
                    for read in self.request.targets
                ],
                indent=2,
            ),
        )

    def tool_contract(self):
        return ProviderToolContract(
            tool_specs=(
                required_tool_spec(
                    tool_name="submit_pagination_contracts",
                    tool_description="Bind complete pagination to declared API structure.",
                    input_schema=pagination_schema(self.request),
                ),
            )
        )

    def response_contract(self):
        return ProviderResponseContract(
            provider_schema={
                "submit_pagination_contracts": pagination_schema(self.request)
            }
        )
