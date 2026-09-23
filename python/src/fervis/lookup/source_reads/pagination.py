"""Project explicit pagination bindings into logical reads and replay contracts."""

from dataclasses import replace
from typing import Any
from fervis.host_api.contracts.response_envelope import TOTAL_COUNT_FIELD

from fervis.host_api.contracts import (
    EndpointContract,
    ParameterContract,
    ResponseFieldContract,
    PaginationContract,
    PaginationKind,
)
from fervis.host_api.compilation.pagination_binding import validate_pagination_binding
from fervis.lookup.relation_catalog.model import (
    EndpointRead,
    RelationCatalog,
    PaginationMetadata,
    PaginationMode,
    CompletenessPolicy,
    RowCardinality,
)


def bound_pagination_read(read: EndpointRead) -> EndpointRead:
    binding = read.pagination_binding
    if binding is None:
        return read
    # Keep the original declarations in EndpointRead; RowSource projection owns
    # hiding transport arguments from semantic binding.
    fields = tuple(
        ResponseFieldContract(
            field.ref,
            field.type,
            field.path,
            requires={item.param_ref: item.value for item in field.requirements},
        )
        for field in read.fields
    )
    paths = {field.path for field in fields}
    arrays = tuple(
        ResponseFieldContract(path.id, "array", path.path)
        for path in read.row_paths
        if path.cardinality is RowCardinality.MANY
        and path.path
        and path.path not in paths
    )
    host = EndpointContract(
        read.endpoint_name,
        read.endpoint_name,
        read.method,
        read.path,
        read.description,
        "",
        query_params=tuple(
            ParameterContract(param.name, param.type, source=param.source)
            for param in read.params
            if param.source == "query"
        ),
        response_fields=(*fields, *arrays),
    )
    validate_pagination_binding(host, binding)
    if read.pagination is not None and read.pagination.mode not in {
        PaginationMode.NONE,
        PaginationMode.LIMIT_OFFSET
        if binding.kind is PaginationKind.OFFSET
        else PaginationMode.PAGE_NUMBER,
    }:
        raise ValueError("Pagination binding contradicts the current read contract")
    return replace(
        read,
        pagination=PaginationMetadata(
            mode=PaginationMode.LIMIT_OFFSET
            if binding.kind is PaginationKind.OFFSET
            else PaginationMode.PAGE_NUMBER,
            default_page_size=binding.page_size,
            max_page_size=binding.max_page_size,
            completeness_policy=CompletenessPolicy.ALL_PAGES,
        ),
    )


def pagination_catalog_for_program(
    program, catalog: RelationCatalog
) -> RelationCatalog:
    bindings: dict[str, PaginationContract] = {}
    for relation in program.relations:
        binding = relation.source.pagination_binding
        if binding is None:
            continue
        read_id = relation.source.read_id
        if not read_id or read_id in bindings and bindings[read_id] != binding:
            raise ValueError(
                "One API read cannot carry conflicting pagination bindings"
            )
        bindings[read_id] = binding
    known = {read.id for read in catalog.reads}
    if not set(bindings) <= known:
        raise ValueError("Pagination binding references an unknown API read")
    return replace(
        catalog,
        reads=tuple(
            bound_pagination_read(
                replace(
                    read,
                    pagination_binding=bindings.get(read.id, read.pagination_binding),
                )
            )
            for read in catalog.reads
        ),
    )


def pagination_field_is_available(path: str, binding: PaginationContract) -> bool:
    return (
        path == binding.total_path
        or path == binding.results_path
        or path.startswith(binding.results_path + ".")
    )


def restore_bound_response_paths(result, binding: PaginationContract):
    """Expose only the collected rows and declared total in the original schema."""
    if not 200 <= result.get("responseStatus", 0) < 300:
        return result
    body = result.get("responseBody")
    if not isinstance(body, dict) or not isinstance(body.get("data"), list):
        raise ValueError("Bound pagination did not return a normalized collection")
    restored: dict[str, Any] = {}

    def assign(path, value):
        target = restored
        parts = path.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value

    assign(binding.results_path, body["data"])
    if binding.total_path:
        assign(binding.total_path, body["pagination"][TOTAL_COUNT_FIELD])
    return {**result, "responseBody": restored}
