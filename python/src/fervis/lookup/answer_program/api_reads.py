"""One cached, paginated, lineage-recorded API read path for program execution."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from copy import deepcopy
from fervis.lookup.canonical_data import canonical_runtime_json
from fervis.lookup.relation_catalog import CatalogEndpointMetadata, EndpointRead
from fervis.lookup.relation_catalog.row_sources import RowSource
from fervis.lookup.plan_execution.errors import RelationEngineError, VerificationError
from fervis.lookup.source_reads.response import (
    EndpointResponseError,
    extract_source_read_rows,
    observe_source_read_response,
    response_body_hash,
    source_read_completeness,
)
from fervis.lookup.lineage.source_reads import (
    SourceReadLineageScope,
    record_source_read_observation,
    record_source_read_error,
    require_catalog_endpoint_for_lineage,
    source_read_key_from_index,
)


@dataclass(frozen=True)
class ApiReadResult:
    result: dict[str, Any]
    rows: tuple[dict[str, Any], ...] = ()
    error: str = ""
    source_read_refs: tuple[str, ...] = ()
    response_hash: str = ""
    invocation_count: int = 1
    row_arguments: tuple[dict[str, Any], ...] = ()


@dataclass
class ApiReadSession:
    data_access_port: Any
    source_read_lineage: SourceReadLineageScope | None = None
    request_cache: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    lineage_ref_cache: dict[tuple[str, str], tuple[str, ...]] = field(
        default_factory=dict
    )
    source_read_index: int = 0
    source_read_key_prefix: str = ""

    def read(
        self,
        read: EndpointRead,
        row_source: RowSource,
        arg_sets: tuple[dict[str, Any], ...],
    ) -> ApiReadResult:
        # Request equality is over complete argument tuples, never separate columns.
        arg_sets = tuple(
            {canonical_runtime_json(args): args for args in arg_sets}.values()
        )
        common = dict(
            data_access_port=self.data_access_port,
            request_cache=self.request_cache,
            lineage_ref_cache=self.lineage_ref_cache,
            source_read_lineage=self.source_read_lineage,
            source_read_index=self.source_read_index,
            row_source=row_source,
            source_read_key_prefix=self.source_read_key_prefix,
        )
        if len(arg_sets) == 1:
            args = arg_sets[0]
            result, self.source_read_index = _cached_api_read(
                cache_key=(read.endpoint_name, canonical_runtime_json(args)),
                endpoint_name=read.endpoint_name,
                args=args,
                catalog_endpoint=read.catalog_endpoint,
                **common,
            )
        else:
            result, self.source_read_index = _fanout_api_result(
                read=read, arg_sets=arg_sets, **common
            )
        return result


def _fanout_endpoint_args(
    args: dict[str, Any],
    *,
    row_source: RowSource,
) -> tuple[dict[str, Any], ...]:
    sequence_items = [
        (key, value)
        for key, value in args.items()
        if isinstance(value, tuple) and value
    ]
    if not sequence_items:
        return ()
    if len(sequence_items) > 1:
        raise VerificationError("identity-set fanout supports one param per source")
    key, values = sequence_items[0]
    if _row_source_param_accepts_sequence(row_source, param_ref=key):
        return ()
    return tuple({**args, key: value} for value in values)


def _row_source_param_accepts_sequence(
    row_source: RowSource,
    *,
    param_ref: str,
) -> bool:
    for param in row_source.params:
        if param.param_ref != param_ref:
            continue
        return param.type in {"array", "list"}
    return False


def _fanout_api_result(
    *,
    read: Any,
    row_source: RowSource,
    arg_sets: tuple[dict[str, Any], ...],
    data_access_port: Any,
    request_cache: dict[tuple[str, str], dict[str, Any]],
    lineage_ref_cache: dict[tuple[str, str], tuple[str, ...]],
    source_read_lineage: SourceReadLineageScope | None,
    source_read_index: int,
    source_read_key_prefix: str = "",
) -> tuple[ApiReadResult, int]:
    rows: list[dict[str, Any]] = []
    row_arguments: list[dict[str, Any]] = []
    source_read_refs: list[str] = []
    truncated = False
    for args in arg_sets:
        cache_key = (read.endpoint_name, canonical_runtime_json(args))
        api_read, source_read_index = _cached_api_read(
            cache_key=cache_key,
            endpoint_name=read.endpoint_name,
            args=args,
            row_source=row_source,
            data_access_port=data_access_port,
            request_cache=request_cache,
            lineage_ref_cache=lineage_ref_cache,
            source_read_lineage=source_read_lineage,
            source_read_index=source_read_index,
            catalog_endpoint=read.catalog_endpoint,
            source_read_key_prefix=source_read_key_prefix,
        )
        source_read_refs.extend(api_read.source_read_refs)
        if api_read.error:
            raise RelationEngineError(api_read.error)
        truncated = truncated or bool(api_read.result.get("truncated") is True)
        rows.extend(api_read.rows)
        row_arguments.extend(api_read.row_arguments)
    result = {
        "responseStatus": 200,
        "responseBody": _fanout_response_body(row_source=row_source, rows=tuple(rows)),
        "truncated": truncated,
    }
    return (
        ApiReadResult(
            result=result,
            rows=tuple(rows),
            source_read_refs=tuple(dict.fromkeys(source_read_refs)),
            response_hash=response_body_hash(result),
            invocation_count=len(arg_sets),
            row_arguments=tuple(row_arguments),
        ),
        source_read_index,
    )


def _cached_api_read(
    *,
    cache_key: tuple[str, str],
    endpoint_name: str,
    args: dict[str, Any],
    row_source: RowSource,
    data_access_port: Any,
    request_cache: dict[tuple[str, str], dict[str, Any]],
    lineage_ref_cache: dict[tuple[str, str], tuple[str, ...]],
    source_read_lineage: SourceReadLineageScope | None,
    source_read_index: int,
    catalog_endpoint: CatalogEndpointMetadata | None,
    source_read_key_prefix: str = "",
) -> tuple[ApiReadResult, int]:
    require_catalog_endpoint_for_lineage(
        source_read_lineage=source_read_lineage,
        endpoint_name=endpoint_name,
        catalog_endpoint=catalog_endpoint,
    )
    result = request_cache.get(cache_key)
    if result is None:
        try:
            result = data_access_port.read(
                endpoint_name=endpoint_name, args=deepcopy(args)
            )
        except Exception as exc:
            source_read_index += 1
            record_source_read_error(
                source_read_lineage,
                source_read_key=(
                    source_read_key_prefix + ":" if source_read_key_prefix else ""
                )
                + source_read_key_from_index(source_read_index),
                endpoint_name=endpoint_name,
                catalog_endpoint=catalog_endpoint,
                args=args,
                error_json={"error": str(exc), "errorType": type(exc).__name__},
            )
            raise
        request_cache[cache_key] = result
    observation = observe_source_read_response(result, endpoint_name=endpoint_name)
    source_read_refs = lineage_ref_cache.get(cache_key)
    if source_read_refs is None:
        source_read_index += 1
        source_read_id = record_source_read_observation(
            source_read_lineage,
            source_read_key=(
                source_read_key_prefix + ":" if source_read_key_prefix else ""
            )
            + source_read_key_from_index(source_read_index),
            endpoint_name=endpoint_name,
            catalog_endpoint=catalog_endpoint,
            args=args,
            observation=observation,
            response_body=result.get("responseBody"),
            completeness_json=source_read_completeness(result),
        )
        source_read_refs = (f"source_read:{source_read_id}",) if source_read_id else ()
        lineage_ref_cache[cache_key] = source_read_refs
    rows: tuple[dict[str, Any], ...] = ()
    row_error = _source_read_error_message(observation.error_json)
    if not row_error:
        try:
            rows = extract_source_read_rows(
                result,
                endpoint_name=endpoint_name,
                row_source=row_source,
            )
        except EndpointResponseError as exc:
            row_error = str(exc)
    api_read = ApiReadResult(
        result=result,
        rows=rows,
        error=row_error,
        source_read_refs=source_read_refs,
        response_hash=observation.response_hash,
        row_arguments=tuple(deepcopy(args) for _ in rows),
    )
    return api_read, source_read_index


def _source_read_error_message(error_json: dict[str, Any]) -> str:
    if not error_json:
        return ""
    error = str(error_json.get("error") or "")
    if error:
        return error
    status = error_json.get("responseStatus")
    return f"source read failed with response status {status}"


def _fanout_response_body(
    *,
    row_source: RowSource,
    rows: tuple[dict[str, Any], ...],
) -> Any:
    if not row_source.row_path:
        return list(rows)
    current: Any = list(rows)
    for part in reversed(row_source.row_path.split(".")):
        current = {part: current}
    return current
