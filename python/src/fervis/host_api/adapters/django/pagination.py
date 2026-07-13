"""DRF pagination contract introspection."""

from __future__ import annotations

from rest_framework.mixins import ListModelMixin
from rest_framework.pagination import LimitOffsetPagination, PageNumberPagination

from fervis.host_api.contracts import PaginationContract, PaginationKind


def pagination_contract(view_class: type) -> PaginationContract | None:
    try:
        if not issubclass(view_class, ListModelMixin):
            return None
    except TypeError:
        return None
    pagination_class = getattr(view_class, "pagination_class", None)
    if pagination_class is None:
        return None
    paginator = pagination_class()
    response_schema, item_schema = _pagination_schema(paginator)
    results_path = _path_to_object(response_schema, target=item_schema)
    if not results_path:
        return None
    if isinstance(paginator, LimitOffsetPagination):
        return _limit_offset_contract(
            paginator,
            results_path=results_path,
            response_schema=response_schema,
        )
    if isinstance(paginator, PageNumberPagination):
        return _page_number_contract(
            paginator,
            results_path=results_path,
            response_schema=response_schema,
        )
    return None


def _limit_offset_contract(
    paginator: LimitOffsetPagination,
    *,
    results_path: str,
    response_schema: dict[str, object],
) -> PaginationContract | None:
    default_page_size = paginator.default_limit
    if default_page_size is None or default_page_size < 1:
        return None
    total_path, continuation_path = _completeness_paths(response_schema)
    if not total_path and not continuation_path:
        return None
    return PaginationContract(
        kind=PaginationKind.OFFSET,
        position_query_param=str(paginator.offset_query_param),
        page_size_query_param=str(paginator.limit_query_param),
        results_path=results_path,
        page_size=default_page_size,
        max_page_size=paginator.max_limit or default_page_size,
        total_path=total_path,
        continuation_path=continuation_path,
    )


def _page_number_contract(
    paginator: PageNumberPagination,
    *,
    results_path: str,
    response_schema: dict[str, object],
) -> PaginationContract | None:
    page_size_query_param = str(paginator.page_size_query_param or "")
    page_size = int(paginator.page_size or 0)
    if page_size < 1:
        return None
    total_path, continuation_path = _completeness_paths(response_schema)
    if not total_path and not continuation_path:
        return None
    return PaginationContract(
        kind=PaginationKind.PAGE_NUMBER,
        position_query_param=str(paginator.page_query_param),
        page_size_query_param=page_size_query_param,
        results_path=results_path,
        page_size=page_size,
        max_page_size=int(paginator.max_page_size or page_size),
        total_path=total_path,
        continuation_path=continuation_path,
    )


def _pagination_schema(
    paginator: object,
) -> tuple[dict[str, object], object | None]:
    schema_builder = getattr(paginator, "get_paginated_response_schema", None)
    if not callable(schema_builder):
        return {}, None
    item_schema: dict[str, object] = {"type": "array"}
    response_schema = schema_builder(item_schema)
    if not isinstance(response_schema, dict):
        return {}, None
    return response_schema, item_schema


def _completeness_paths(response_schema: dict[str, object]) -> tuple[str, str]:
    boolean_paths = _schema_paths_with_type(response_schema, expected_type="boolean")
    integer_paths = _schema_paths_with_type(response_schema, expected_type="integer")
    total_path = integer_paths[0] if len(integer_paths) == 1 else ""
    continuation_path = boolean_paths[0] if len(boolean_paths) == 1 else ""
    return total_path, continuation_path


def _schema_paths_with_type(
    value: object,
    *,
    expected_type: str,
    prefix: tuple[str, ...] = (),
) -> tuple[str, ...]:
    if not isinstance(value, dict):
        return ()
    if value.get("type") == expected_type:
        return (".".join(prefix),)
    paths: list[str] = []
    for key, child in value.items():
        child_prefix = prefix if key == "properties" else (*prefix, str(key))
        paths.extend(
            _schema_paths_with_type(
                child,
                expected_type=expected_type,
                prefix=child_prefix,
            )
        )
    return tuple(path for path in paths if path)


def _path_to_object(
    value: object,
    *,
    target: object,
    prefix: tuple[str, ...] = (),
) -> str:
    if value is target:
        return ".".join(prefix)
    if not isinstance(value, dict):
        return ""
    for key, child in value.items():
        child_prefix = prefix if key == "properties" else (*prefix, str(key))
        path = _path_to_object(child, target=target, prefix=child_prefix)
        if path:
            return path
    return ""
