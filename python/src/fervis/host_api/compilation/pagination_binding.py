"""Bind an explicit traversal to the endpoint's current declared structure."""

from dataclasses import replace
from typing import Any, Mapping

from fervis.host_api.contracts import EndpointContract, PaginationContract
from fervis.host_api.contracts.ports import EndpointExecutionError


def bind_pagination_policy(
    contract: EndpointContract,
    page_policy: Mapping[str, Any] | None,
) -> EndpointContract:
    """Runtime selection may add traversal mechanics, never routes or authority."""
    raw = (page_policy or {}).get("pagination_contract")
    if raw is None:
        return contract
    if not isinstance(raw, dict):
        raise EndpointExecutionError("Pagination binding must be a declared contract")
    try:
        pagination = PaginationContract.from_public_dict(raw)
    except (ValueError, TypeError) as exc:
        raise EndpointExecutionError(str(exc)) from exc
    validate_pagination_binding(contract, pagination)
    return replace(contract, pagination=pagination)


def inside_collection(path: str, array_paths: tuple[str, ...]) -> bool:
    """Collection completion metadata cannot be a property of an individual row."""
    return any(
        not array or path == array or path.startswith(array + ".")
        for array in array_paths
    )


def validate_pagination_binding(
    contract: EndpointContract, pagination: PaginationContract
) -> None:
    if contract.pagination is not None:
        if contract.pagination != pagination:
            raise EndpointExecutionError(
                "Pagination binding conflicts with declared traversal"
            )
        return
    parameters = {parameter.name: parameter for parameter in contract.query_params}
    names = tuple(
        name
        for name in (pagination.position_query_param, pagination.page_size_query_param)
        if name
    )
    if len(names) != len(set(names)) or any(
        name not in parameters or parameters[name].type != "integer" for name in names
    ):
        raise EndpointExecutionError(
            "Pagination controls must be distinct declared integer query parameters"
        )
    fields = {field.path: field for field in contract.response_fields}
    rows = fields.get(pagination.results_path)
    if rows is None or rows.type != "array" or rows.requires:
        raise EndpointExecutionError(
            "Pagination rows must be a declared unconditional array"
        )
    arrays = tuple(field.path for field in fields.values() if field.type == "array")
    for path, types in (
        (pagination.total_path, {"integer"}),
        (pagination.continuation_path, {"boolean", "string"}),
    ):
        if not path:
            continue
        field = fields.get(path)
        if (
            field is None
            or field.type not in types
            or field.requires
            or inside_collection(path, arrays)
        ):
            raise EndpointExecutionError(
                "Pagination completion evidence must be a declared scalar outside row arrays"
            )
