"""Verify, compile, and execute typed Lookup fact plans."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import date
from typing import TYPE_CHECKING, Any

from fervis.lookup.relation_catalog import (
    CatalogEndpointMetadata,
    RelationCatalog,
)
from fervis.lookup.relation_catalog.selection import CatalogSelectionResult
from fervis.lookup.plan_execution.errors import (
    RelationEngineError,
    VerificationError,
)
from fervis.lookup.plan_execution.authorized_sources import (
    AuthorizedExecutionSources,
)
from fervis.lookup.source_reads.response import (
    EndpointResponseError,
    extract_source_read_rows,
    observe_source_read_response,
    relative_response_path,
    required_response_path_value,
    source_read_completeness,
)
from fervis.lookup.plan_execution.generated_relations import (
    GeneratedCalendarRelationSource,
    generate_calendar_relation,
)
from fervis.lookup.plan_execution.operation_engine import execute_operations
from fervis.lookup.plan_execution.compiled_execution import (
    ExecutionProofGraph,
    compile_fact_execution,
)
from fervis.lookup.plan_execution.operation_runtime import (
    RelationEngineInput,
    ScalarInput,
)
from fervis.lookup.plan_execution.relations import (
    RelationRows,
    Row,
    RowContextStore,
    api_read_completeness_proof,
)
from fervis.lookup.lineage.source_reads import (
    SourceReadLineageScope,
    record_source_read_observation,
    record_source_read_error,
    require_catalog_endpoint_for_lineage,
    source_read_key_from_index,
)
from fervis.lookup.plan_execution.value_compiler import CompiledRowFilter
from fervis.lookup.plan_execution.verification import verify_fact_plan
from fervis.lookup.outcomes.model import FactResult
from fervis.lookup.outcomes.classification import classify_answer_result
from fervis.lookup.outcomes.errors import ExecutionIssue
from fervis.lookup.fact_plan.row_sources import (
    CALENDAR_DATE_FIELD_ID,
    CALENDAR_END_PARAM_REF,
    CALENDAR_MAX_ROWS,
    CALENDAR_START_PARAM_REF,
    RowSource,
    RowSourceCatalog,
    RowSourceKind,
    build_row_source_catalog,
    row_source_for_relation,
)
from fervis.lookup.fact_plan.fact_plan import AnswerPlan, FactPlan
from fervis.lookup.question_contract import QuestionContract
from fervis.lookup.fact_plan.relations import (
    Relation,
    RelationSource,
    SourceKind,
)
from fervis.lookup.fact_plan.values import (
    FactValue,
    ValueFilterOperator,
)
from fervis.lookup.fact_planning.value_validation import verify_value_contract

if TYPE_CHECKING:
    from fervis.lookup.memory.projection import LookupMemory


@dataclass(frozen=True)
class FactPlanExecutionResult:
    fact_result: FactResult | None
    issue: ExecutionIssue | None = None
    answer_plan: AnswerPlan | None = None
    proof_node_refs_by_render_output_id: dict[str, tuple[str, ...]] = field(
        default_factory=dict
    )
    relations: tuple[RelationRows, ...] = ()
    proof_refs: tuple[str, ...] = ()
    proof_graph: ExecutionProofGraph = field(default_factory=ExecutionProofGraph)
    row_context: RowContextStore = field(default_factory=RowContextStore)


@dataclass(frozen=True)
class _RelationExecutionRows:
    relation: RelationRows
    row_context: tuple[Row, ...] = ()


@dataclass(frozen=True)
class _ApiReadResult:
    result: dict[str, Any]
    rows: tuple[dict[str, Any], ...] = ()
    error: str = ""
    source_read_refs: tuple[str, ...] = ()


def execute_fact_plan(
    *,
    plan: FactPlan,
    question_contract: QuestionContract,
    catalog: RelationCatalog,
    data_access_port: Any,
    memory: LookupMemory,
    catalog_selection: CatalogSelectionResult | None = None,
    available_values: tuple[FactValue, ...] = (),
    available_value_uses: tuple[Any, ...] = (),
    authorized_sources: AuthorizedExecutionSources | None = None,
    source_read_lineage: SourceReadLineageScope | None = None,
) -> FactPlanExecutionResult:
    catalog = _execution_catalog(catalog, authorized_sources)
    available_values = _merge_available_values(available_values)
    if isinstance(plan.outcome, AnswerPlan):
        verify_value_contract(
            values=plan.outcome.values,
            value_uses=plan.outcome.value_uses,
            available_values=available_values,
        )
    row_sources = build_row_source_catalog(
        catalog,
        memory_relations=memory.relations,
    )
    verified = verify_fact_plan(
        plan,
        question_contract=question_contract,
        catalog=catalog,
        available_values=available_values,
        available_value_uses=available_value_uses,
        memory_relations=memory.relations,
        catalog_selection=catalog_selection,
        authorized_sources=authorized_sources,
    )
    if not isinstance(verified.outcome, AnswerPlan):
        raise ValueError("execute_fact_plan requires an answer plan")
    answer = verified.outcome
    compiled = compile_fact_execution(
        answer=answer,
        catalog=catalog,
        row_sources=row_sources,
        available_values=available_values,
        available_value_uses=available_value_uses,
    )
    value_uses = compiled.value_uses
    endpoint_args = _endpoint_args_by_relation(value_uses.endpoint_args)
    endpoint_arg_proofs = _endpoint_arg_proofs_by_relation(value_uses.endpoint_args)
    endpoint_arg_proofs_by_param = _endpoint_arg_proofs_by_relation_param(
        value_uses.endpoint_args
    )
    api_reads = _api_results_by_relation(
        answer.relations,
        catalog=catalog,
        row_sources=row_sources,
        data_access_port=data_access_port,
        endpoint_args=endpoint_args,
        source_read_lineage=source_read_lineage,
    )
    relation_results = tuple(
        _relation_rows(
            relation,
            row_sources=row_sources,
            catalog=catalog,
            api_reads=api_reads,
            memory=memory,
            endpoint_args=endpoint_args,
            endpoint_arg_proofs=endpoint_arg_proofs,
            endpoint_arg_proofs_by_param=endpoint_arg_proofs_by_param,
            row_filters=tuple(
                item
                for item in value_uses.row_filters
                if item.relation_id == relation.id
            ),
        )
        for relation in answer.relations
    )
    relations = tuple(item.relation for item in relation_results)
    row_context = RowContextStore(
        {
            item.relation.id: item.row_context
            for item in relation_results
            if item.row_context
        }
    )
    scalar_inputs = tuple(
        ScalarInput(
            id=item.input_id,
            value=item.value,
            proof_refs=item.proof_refs,
        )
        for item in value_uses.scalar_inputs
    )
    engine_output = execute_operations(
        RelationEngineInput(
            relations=relations,
            operations=answer.operations,
            scalar_inputs=scalar_inputs,
            operation_proof_refs=compiled.operation_proof_refs,
        )
    )
    classified = classify_answer_result(
        answer,
        engine_output=engine_output,
    )
    if isinstance(classified, ExecutionIssue):
        return FactPlanExecutionResult(
            fact_result=None,
            issue=classified,
            answer_plan=answer,
            proof_node_refs_by_render_output_id=(
                compiled.proof_node_refs_by_render_output_id
            ),
            relations=engine_output.relations,
            proof_refs=_proof_refs(engine_output.relations),
            proof_graph=compiled.proof_graph.with_executed_relations(
                engine_output.relations
            ),
            row_context=row_context,
        )
    return FactPlanExecutionResult(
        fact_result=classified,
        answer_plan=answer,
        proof_node_refs_by_render_output_id=compiled.proof_node_refs_by_render_output_id,
        relations=engine_output.relations,
        proof_refs=_proof_refs(engine_output.relations),
        proof_graph=compiled.proof_graph.with_executed_relations(
            engine_output.relations
        ),
        row_context=row_context,
    )


def _merge_available_values(values: tuple[FactValue, ...]) -> tuple[FactValue, ...]:
    output: list[FactValue] = []
    seen: set[str] = set()
    for value in values:
        if value.id in seen:
            continue
        seen.add(value.id)
        output.append(value)
    return tuple(output)


def _relation_rows(
    relation: Relation,
    *,
    row_sources: RowSourceCatalog,
    catalog: RelationCatalog,
    api_reads: dict[str, _ApiReadResult],
    memory: LookupMemory,
    endpoint_args: dict[str, dict[str, Any]],
    endpoint_arg_proofs: dict[str, tuple[str, ...]],
    endpoint_arg_proofs_by_param: dict[str, dict[str, tuple[str, ...]]],
    row_filters: tuple[CompiledRowFilter, ...],
) -> _RelationExecutionRows:
    rows = _source_relation_rows(
        relation.source,
        relation=relation,
        row_sources=row_sources,
        catalog=catalog,
        api_reads=api_reads,
        memory=memory,
        endpoint_args=endpoint_args,
        endpoint_arg_proofs=endpoint_arg_proofs,
        endpoint_arg_proofs_by_param=endpoint_arg_proofs_by_param,
    )
    if row_filters:
        filter_proof_refs = _row_filter_proof_refs(row_filters)
        kept = tuple(
            (row, context)
            for row, context in _rows_with_context(rows)
            if all(_row_filter_matches(row, relation, item) for item in row_filters)
        )
        rows = _RelationExecutionRows(
            relation=RelationRows(
                id=rows.relation.id,
                rows=tuple(row for row, _context in kept),
                grain_keys=rows.relation.grain_keys,
                field_types=rows.relation.field_types,
                field_answer_output_ids=rows.relation.field_answer_output_ids,
                identity_type=rows.relation.identity_type,
                completeness=replace(
                    rows.relation.completeness,
                    proof_refs=_dedupe_refs(
                        (*rows.relation.completeness.proof_refs, *filter_proof_refs)
                    ),
                    scope_fingerprint=_scope_fingerprint(
                        endpoint_args.get(relation.id) or {},
                        endpoint_arg_proofs_by_param.get(relation.id, {}),
                        row_filters,
                    ),
                ),
            ),
            row_context=tuple(context for _row, context in kept),
        )
    return rows


def _rows_with_context(
    rows: _RelationExecutionRows,
) -> tuple[tuple[Row, Row], ...]:
    if not rows.row_context:
        return tuple((row, {}) for row in rows.relation.rows)
    return tuple(zip(rows.relation.rows, rows.row_context))


def _source_relation_rows(
    source: RelationSource,
    *,
    relation: Relation,
    row_sources: RowSourceCatalog,
    catalog: RelationCatalog,
    api_reads: dict[str, _ApiReadResult],
    memory: LookupMemory,
    endpoint_args: dict[str, dict[str, Any]],
    endpoint_arg_proofs: dict[str, tuple[str, ...]],
    endpoint_arg_proofs_by_param: dict[str, dict[str, tuple[str, ...]]],
) -> _RelationExecutionRows:
    if source.kind in {
        SourceKind.API_READ,
        SourceKind.GENERATED_CALENDAR,
        SourceKind.MEMORY_READ,
    }:
        row_source = _row_source_for_relation(relation, row_sources=row_sources)
        if row_source.kind == RowSourceKind.MEMORY_READ:
            memory_relation = memory.relation(row_source.memory_ref)
            rows = (
                tuple(
                    _bound_memory_row(row, relation=relation)
                    for row in memory_relation.rows
                )
                if relation.fields
                else memory_relation.rows
            )
            return _RelationExecutionRows(
                RelationRows(
                    id=relation.id,
                    rows=rows,
                    grain_keys=relation.grain_keys or memory_relation.grain_keys,
                    field_types=memory_relation.field_types,
                    field_answer_output_ids=memory_relation.field_answer_output_ids,
                    completeness=replace(
                        memory_relation.completeness,
                        proof_refs=_dedupe_refs(
                            (
                                *memory_relation.completeness.proof_refs,
                                *source.proof_refs,
                            )
                        ),
                    ),
                )
            )
        if row_source.kind == RowSourceKind.GENERATED_CALENDAR:
            generated = _generated_calendar_rows(
                relation,
                endpoint_args=endpoint_args.get(relation.id) or {},
            )
            if not relation.fields:
                return _RelationExecutionRows(generated)
            rows = tuple(
                _bound_memory_row(row, relation=relation) for row in generated.rows
            )
            return _RelationExecutionRows(
                RelationRows(
                    id=relation.id,
                    rows=rows,
                    grain_keys=relation.grain_keys,
                    field_types=generated.field_types,
                    field_answer_output_ids=generated.field_answer_output_ids,
                    completeness=replace(generated.completeness, row_count=len(rows)),
                )
            )
        if row_source.kind != RowSourceKind.API_READ:
            raise ValueError(f"unsupported row source kind: {row_source.kind.value}")
        read = catalog.read(row_source.read_id)
        api_read = api_reads[relation.id]
        if api_read.error:
            raise RelationEngineError(api_read.error)
        rows = tuple(
            _bound_row(
                item,
                relation=relation,
                catalog=catalog,
                row_source=row_source,
            )
            for item in api_read.rows
        )
        return _RelationExecutionRows(
            relation=RelationRows(
                id=relation.id,
                rows=rows,
                grain_keys=relation.grain_keys,
                field_types=_row_source_field_types(row_source),
                field_answer_output_ids=_row_source_field_answer_output_ids(row_source),
                identity_type=_relation_identity_type(relation, row_source=row_source),
                completeness=api_read_completeness_proof(
                    read,
                    row_count=len(rows),
                    scope_fingerprint=_scope_fingerprint(
                        endpoint_args.get(relation.id) or {},
                        endpoint_arg_proofs_by_param.get(relation.id, {}),
                        (),
                    ),
                    reached_terminal_page=True,
                    truncated=bool(api_read.result.get("truncated") is True),
                    proof_refs=_dedupe_refs(
                        (
                            f"read:{read.endpoint_name}",
                            *api_read.source_read_refs,
                            *endpoint_arg_proofs.get(relation.id, ()),
                        )
                    ),
                ),
            ),
            row_context=tuple(dict(row) for row in api_read.rows),
        )
    raise ValueError(f"unsupported relation source kind: {source.kind.value}")


def _row_source_field_types(row_source: RowSource) -> dict[str, str]:
    return {
        field.id: field.type for field in row_source.fields if field.id and field.type
    }


def _row_source_field_answer_output_ids(
    row_source: RowSource,
) -> dict[str, tuple[str, ...]]:
    return {
        field.id: field.answer_output_ids
        for field in row_source.fields
        if field.id and field.answer_output_ids
    }


def _generated_calendar_rows(
    relation: Relation,
    *,
    endpoint_args: dict[str, Any],
) -> RelationRows:
    return generate_calendar_relation(
        GeneratedCalendarRelationSource(
            id=relation.id,
            start=_calendar_arg_date(endpoint_args, CALENDAR_START_PARAM_REF),
            end=_calendar_arg_date(endpoint_args, CALENDAR_END_PARAM_REF),
            output_date_field=CALENDAR_DATE_FIELD_ID,
            max_rows=CALENDAR_MAX_ROWS,
        )
    )


def _calendar_arg_date(endpoint_args: dict[str, Any], param_ref: str) -> date:
    raw = endpoint_args.get(param_ref)
    if raw in (None, ""):
        raise ValueError(f"generated calendar missing param {param_ref}")
    return date.fromisoformat(str(raw))


def _api_results_by_relation(
    relations: tuple[Relation, ...],
    *,
    catalog: RelationCatalog,
    row_sources: RowSourceCatalog,
    data_access_port: Any,
    endpoint_args: dict[str, dict[str, Any]],
    source_read_lineage: SourceReadLineageScope | None = None,
) -> dict[str, _ApiReadResult]:
    results: dict[str, _ApiReadResult] = {}
    request_cache: dict[tuple[str, str], dict[str, Any]] = {}
    lineage_ref_cache: dict[tuple[str, str], tuple[str, ...]] = {}
    source_read_index = 0
    for relation in relations:
        source = relation.source
        if source.kind != SourceKind.API_READ:
            continue
        row_source = _row_source_for_relation(relation, row_sources=row_sources)
        if row_source.kind != RowSourceKind.API_READ:
            continue
        read = catalog.read(row_source.read_id)
        args = dict(endpoint_args.get(relation.id) or {})
        fanout = _fanout_endpoint_args(args, row_source=row_source)
        if fanout:
            api_read, source_read_index = _fanout_api_result(
                read=read,
                row_source=row_source,
                arg_sets=fanout,
                data_access_port=data_access_port,
                request_cache=request_cache,
                lineage_ref_cache=lineage_ref_cache,
                source_read_lineage=source_read_lineage,
                source_read_index=source_read_index,
            )
        else:
            cache_key = (
                read.endpoint_name,
                json.dumps(args, sort_keys=True, default=str),
            )
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
            )
        results[relation.id] = api_read
    return results


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
) -> tuple[_ApiReadResult, int]:
    rows: list[dict[str, Any]] = []
    source_read_refs: list[str] = []
    truncated = False
    for args in arg_sets:
        cache_key = (read.endpoint_name, json.dumps(args, sort_keys=True, default=str))
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
        )
        source_read_refs.extend(api_read.source_read_refs)
        if api_read.error:
            raise RelationEngineError(api_read.error)
        truncated = truncated or bool(api_read.result.get("truncated") is True)
        rows.extend(api_read.rows)
    return (
        _ApiReadResult(
            result={
                "responseStatus": 200,
                "responseBody": _fanout_response_body(
                    row_source=row_source, rows=tuple(rows)
                ),
                "truncated": truncated,
            },
            rows=tuple(rows),
            source_read_refs=_dedupe_refs(tuple(source_read_refs)),
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
) -> tuple[_ApiReadResult, int]:
    require_catalog_endpoint_for_lineage(
        source_read_lineage=source_read_lineage,
        endpoint_name=endpoint_name,
        catalog_endpoint=catalog_endpoint,
    )
    result = request_cache.get(cache_key)
    if result is None:
        try:
            result = data_access_port.read(endpoint_name=endpoint_name, args=args)
        except Exception as exc:
            source_read_index += 1
            record_source_read_error(
                source_read_lineage,
                source_read_key=source_read_key_from_index(source_read_index),
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
            source_read_key=source_read_key_from_index(source_read_index),
            endpoint_name=endpoint_name,
            catalog_endpoint=catalog_endpoint,
            args=args,
            observation=observation,
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
    api_read = _ApiReadResult(
        result=result,
        rows=rows,
        error=row_error,
        source_read_refs=source_read_refs,
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


def _endpoint_args_by_relation(
    endpoint_args: tuple[Any, ...],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for arg in endpoint_args:
        grouped.setdefault(arg.relation_id, {})[arg.param_ref] = arg.value
    return grouped


def _endpoint_arg_proofs_by_relation(
    endpoint_args: tuple[Any, ...],
) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for arg in endpoint_args:
        grouped.setdefault(arg.relation_id, []).extend(arg.proof_refs)
    return {
        relation_id: _dedupe_refs(tuple(refs)) for relation_id, refs in grouped.items()
    }


def _endpoint_arg_proofs_by_relation_param(
    endpoint_args: tuple[Any, ...],
) -> dict[str, dict[str, tuple[str, ...]]]:
    grouped: dict[str, dict[str, tuple[str, ...]]] = {}
    for arg in endpoint_args:
        grouped.setdefault(arg.relation_id, {})[arg.param_ref] = tuple(arg.proof_refs)
    return grouped


def _row_filter_proof_refs(
    row_filters: tuple[CompiledRowFilter, ...],
) -> tuple[str, ...]:
    return _dedupe_refs(tuple(ref for item in row_filters for ref in item.proof_refs))


def _scope_fingerprint(
    endpoint_args: dict[str, Any],
    endpoint_arg_proof_refs: dict[str, tuple[str, ...]],
    row_filters: tuple[CompiledRowFilter, ...],
) -> str:
    scope = {
        "endpointArgs": endpoint_args,
        "endpointArgProofRefs": {
            param_ref: list(proof_refs)
            for param_ref, proof_refs in endpoint_arg_proof_refs.items()
            if param_ref in endpoint_args and proof_refs
        },
        "rowFilters": [
            {
                "fieldId": item.field_id,
                "operator": item.operator.value,
                "value": item.value,
            }
            for item in row_filters
        ],
    }
    return json.dumps(scope, sort_keys=True, default=str)


def _dedupe_refs(refs: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(ref for ref in refs if ref))


def _execution_catalog(
    catalog: RelationCatalog,
    authorized_sources: AuthorizedExecutionSources | None,
) -> RelationCatalog:
    if authorized_sources is None:
        return catalog
    return authorized_sources.relation_catalog


def _relation_identity_type(relation: Relation, *, row_source: RowSource) -> str:
    identity_types: set[str] = set()
    for relation_field in relation.fields:
        if relation_field.field_id not in set(relation.grain_keys):
            continue
        row_source_field = row_source.field(relation_field.field_id)
        identity = row_source_field.identity
        if identity is not None and identity.entity_ref:
            identity_types.add(identity.entity_ref)
    if len(identity_types) == 1:
        return next(iter(identity_types))
    return ""


def _bound_row(
    raw: dict[str, Any],
    *,
    relation: Relation,
    catalog: RelationCatalog,
    row_source: RowSource,
) -> dict[str, object]:
    read = catalog.read(row_source.read_id)
    fields_by_ref = {field.ref: field for field in read.fields}
    output: dict[str, object] = {}
    for relation_field in relation.fields:
        row_source_field = row_source.field(relation_field.field_id)
        catalog_field = fields_by_ref.get(row_source_field.field_ref)
        if catalog_field is None:
            raise ValueError(f"unknown relation field {relation_field.field_id}")
        output[relation_field.field_id] = required_response_path_value(
            raw,
            row_source_field.response_path
            or relative_response_path(catalog_field.path, row_source.row_path),
        )
    return output


def _bound_memory_row(
    row: dict[str, object], *, relation: Relation
) -> dict[str, object]:
    output: dict[str, object] = {}
    for relation_field in relation.fields:
        if relation_field.field_id in row:
            output[relation_field.field_id] = row[relation_field.field_id]
        else:
            raise ValueError(f"unknown memory relation field {relation_field.field_id}")
    return output


def _row_filter_matches(
    row: dict[str, object],
    relation: Relation,
    row_filter: CompiledRowFilter,
) -> bool:
    value = row.get(row_filter.field_id)
    if row_filter.operator == ValueFilterOperator.EQUALS:
        return str(value) == str(row_filter.value)
    if row_filter.operator == ValueFilterOperator.IN:
        return str(value).lower() in {
            str(item).lower() for item in row_filter.value or ()
        }
    raise ValueError(f"unsupported row filter operator: {row_filter.operator.value}")


def _row_source_for_relation(
    relation: Relation,
    *,
    row_sources: RowSourceCatalog,
) -> RowSource:
    try:
        return row_source_for_relation(relation, row_sources=row_sources)
    except KeyError as exc:
        raise VerificationError(f"unknown relation source {relation.source}") from exc


def _proof_refs(relations: tuple[RelationRows, ...]) -> tuple[str, ...]:
    refs: list[str] = []
    for relation in relations:
        for ref in relation.completeness.proof_refs:
            if ref not in refs:
                refs.append(ref)
    return tuple(refs)
