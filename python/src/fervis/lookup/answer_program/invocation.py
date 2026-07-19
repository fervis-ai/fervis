"""The single deterministic invocation kernel for answer programs."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from typing import TYPE_CHECKING, Any

from fervis.lookup.relation_catalog import (
    CatalogEndpointMetadata,
    RelationCatalog,
)
from fervis.lookup.canonical_data import (
    RuntimeValue,
    canonical_runtime_json,
)
from fervis.lookup.plan_execution.declared_values import (
    parse_declared_value,
)
from fervis.lookup.plan_execution.errors import (
    RelationEngineError,
    VerificationError,
)
from fervis.lookup.source_reads.response import (
    EndpointResponseError,
    extract_source_read_rows,
    observe_source_read_response,
    response_body_hash,
    relative_response_path,
    required_response_path_value,
    source_read_completeness,
)
from fervis.lookup.plan_execution.generated_relations import (
    GeneratedCalendarRelationSource,
    generate_calendar_relation,
)
from fervis.lookup.plan_execution.operation_engine import execute_operations
from fervis.lookup.answer_program.instantiation import (
    ExecutionEnvironment,
    ExecutionProofGraph,
    VerifiedExecution,
    instantiate_answer_program,
)
from fervis.lookup.plan_execution.operation_runtime import (
    RelationEngineInput,
    ScalarInput,
)
from fervis.lookup.plan_execution.relations import (
    RelationEvidence,
    RelationRows,
    Row,
    RowContextStore,
    api_read_completeness_proof,
    relation_snapshot_hash,
)
from fervis.lookup.lineage.source_reads import (
    SourceReadLineageScope,
    record_source_read_observation,
    record_source_read_error,
    require_catalog_endpoint_for_lineage,
    source_read_key_from_index,
)
from fervis.lookup.answer_program.expression_instantiation import (
    ResolvedPopulationChoice,
)
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
    row_source_for_relation,
)
from fervis.lookup.answer_program.model import AnswerProgram
from fervis.lookup.answer_program.relations import (
    Relation,
    RelationSource,
    SourceKind,
)
from fervis.lookup.answer_program.values import BindingSet
from fervis.lookup.answer_program.codec import answer_program_id
from fervis.lookup.answer_program.persistence import ProgramInvocationBinding
from fervis.lineage.enums import ProgramInvocationKind
from fervis.lookup.question_contract import RequestedFact

if TYPE_CHECKING:
    from fervis.lookup.memory.projection import LookupMemory


@dataclass(frozen=True)
class AnswerExecution:
    fact_result: FactResult | None
    issue: ExecutionIssue | None = None
    program: AnswerProgram | None = None
    program_id: str = ""
    invocation_id: str = ""
    proof_node_refs_by_result_output_id: dict[str, tuple[str, ...]] = field(
        default_factory=dict
    )
    relations: tuple[RelationRows, ...] = ()
    proof_refs: tuple[str, ...] = ()
    proof_graph: ExecutionProofGraph = field(default_factory=ExecutionProofGraph)
    effective_requested_facts: tuple[RequestedFact, ...] = ()
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
    response_hash: str = ""


@dataclass(frozen=True)
class RuntimePorts:
    data_access_port: Any
    memory: LookupMemory
    source_read_lineage: SourceReadLineageScope | None = None
    invocation_binding: ProgramInvocationBinding | None = None
    invocation_kind: ProgramInvocationKind = ProgramInvocationKind.COMPILED_QUESTION
    base_invocation_id: str | None = None


def invoke_answer_program(
    *,
    program: AnswerProgram,
    bindings: BindingSet,
    environment: ExecutionEnvironment,
    ports: RuntimePorts,
) -> AnswerExecution:
    execution = instantiate_answer_program(program, bindings, environment)
    invocation_id = ""
    if ports.invocation_binding is not None:
        invocation_id = ports.invocation_binding.bind(
            execution,
            kind=ports.invocation_kind,
            base_invocation_id=ports.base_invocation_id,
        ).invocation_id
    return replace(
        execute_verified_program(execution, ports),
        invocation_id=invocation_id,
    )


def execute_verified_program(
    execution: VerifiedExecution,
    ports: RuntimePorts,
) -> AnswerExecution:
    answer = execution.answer
    instantiated_inputs = execution.instantiated_inputs
    catalog = execution.catalog
    row_sources = execution.row_sources
    endpoint_args = _endpoint_args_by_relation(instantiated_inputs.endpoint_args)
    endpoint_arg_proofs = _endpoint_arg_proofs_by_relation(
        instantiated_inputs.endpoint_args
    )
    endpoint_arg_proofs_by_param = _endpoint_arg_proofs_by_relation_param(
        instantiated_inputs.endpoint_args
    )
    api_reads = _api_results_by_relation(
        answer.relations,
        catalog=catalog,
        row_sources=row_sources,
        data_access_port=ports.data_access_port,
        endpoint_args=endpoint_args,
        source_read_lineage=ports.source_read_lineage,
    )
    relation_results = tuple(
        _relation_rows(
            relation,
            row_sources=row_sources,
            catalog=catalog,
            api_reads=api_reads,
            memory=ports.memory,
            endpoint_args=endpoint_args,
            endpoint_arg_proofs=endpoint_arg_proofs,
            endpoint_arg_proofs_by_param=endpoint_arg_proofs_by_param,
            authority_ref=execution.authority_ref,
            population_choices=tuple(
                item
                for item in instantiated_inputs.population_choices
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
    engine_output = execute_operations(
        RelationEngineInput(
            relations=relations,
            operations=execution.operations,
            scalar_inputs=_operation_scalar_inputs(execution),
            operation_proof_refs=execution.operation_proof_refs,
        )
    )
    classified = classify_answer_result(
        answer,
        engine_output=engine_output,
    )
    if isinstance(classified, ExecutionIssue):
        return AnswerExecution(
            fact_result=None,
            issue=classified,
            program=answer,
            program_id=answer_program_id(answer),
            proof_node_refs_by_result_output_id=(
                execution.proof_node_refs_by_result_output_id
            ),
            relations=engine_output.relations,
            proof_refs=_proof_refs(engine_output.relations),
            proof_graph=execution.proof_graph.with_executed_relations(
                engine_output.relations
            ),
            effective_requested_facts=execution.effective_requested_facts,
            row_context=row_context,
        )
    return AnswerExecution(
        fact_result=classified,
        program=answer,
        program_id=answer_program_id(answer),
        proof_node_refs_by_result_output_id=(
            execution.proof_node_refs_by_result_output_id
        ),
        relations=engine_output.relations,
        proof_refs=_proof_refs(engine_output.relations),
        proof_graph=execution.proof_graph.with_executed_relations(
            engine_output.relations
        ),
        effective_requested_facts=execution.effective_requested_facts,
        row_context=row_context,
    )


def _operation_scalar_inputs(execution: VerifiedExecution) -> tuple[ScalarInput, ...]:
    by_id: dict[str, ScalarInput] = {}
    for item in execution.operation_inputs:
        candidate = ScalarInput(
            id=item.input_id,
            value=item.value,
            value_type=item.value_type,
            proof_refs=item.proof_refs,
        )
        existing = by_id.get(candidate.id)
        if existing is not None and (
            existing.value != candidate.value
            or existing.value_type != candidate.value_type
        ):
            raise VerificationError(f"conflicting operation input {candidate.id}")
        if existing is None:
            by_id[candidate.id] = candidate
            continue
        by_id[candidate.id] = ScalarInput(
            id=candidate.id,
            value=candidate.value,
            value_type=candidate.value_type,
            proof_refs=tuple(
                dict.fromkeys((*existing.proof_refs, *candidate.proof_refs))
            ),
        )
    return tuple(by_id.values())


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
    authority_ref: str,
    population_choices: tuple[ResolvedPopulationChoice, ...],
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
        authority_ref=authority_ref,
    )
    scope_fingerprint = _scope_fingerprint(
        endpoint_args.get(relation.id) or {},
        endpoint_arg_proofs_by_param.get(relation.id, {}),
        population_choices,
    )
    if population_choices:
        rows = _RelationExecutionRows(
            relation=rows.relation.with_scope(
                proof_refs=_population_choice_proof_refs(population_choices),
                scope_fingerprint=scope_fingerprint,
            ),
            row_context=rows.row_context,
        )
    return rows


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
    authority_ref: str,
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
                    evidence=RelationEvidence(
                        source_refs=(row_source.id,),
                        authority_refs=((authority_ref,) if authority_ref else ()),
                        snapshot_hash=relation_snapshot_hash(tuple(rows)),
                        proof_refs=_dedupe_refs(
                            (
                                *memory_relation.evidence.proof_refs,
                                *source.proof_refs,
                            )
                        ),
                    ),
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
                return _RelationExecutionRows(
                    replace(
                        generated,
                        evidence=RelationEvidence(
                            source_refs=(row_source.id,),
                            authority_refs=((authority_ref,) if authority_ref else ()),
                            snapshot_hash=relation_snapshot_hash(generated.rows),
                            proof_refs=generated.completeness.proof_refs,
                        ),
                    )
                )
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
                    evidence=RelationEvidence(
                        source_refs=(row_source.id,),
                        authority_refs=((authority_ref,) if authority_ref else ()),
                        snapshot_hash=relation_snapshot_hash(tuple(rows)),
                        proof_refs=generated.completeness.proof_refs,
                    ),
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
                evidence=RelationEvidence(
                    source_refs=(row_source.id,),
                    read_refs=(read.id,),
                    authority_refs=((authority_ref,) if authority_ref else ()),
                    snapshot_hash=relation_snapshot_hash(rows),
                    proof_refs=_dedupe_refs(
                        (
                            f"read:{read.endpoint_name}",
                            *api_read.source_read_refs,
                            *endpoint_arg_proofs.get(relation.id, ()),
                        )
                    ),
                ),
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
                canonical_runtime_json(args),
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
        )
        source_read_refs.extend(api_read.source_read_refs)
        if api_read.error:
            raise RelationEngineError(api_read.error)
        truncated = truncated or bool(api_read.result.get("truncated") is True)
        rows.extend(api_read.rows)
    result = {
        "responseStatus": 200,
        "responseBody": _fanout_response_body(row_source=row_source, rows=tuple(rows)),
        "truncated": truncated,
    }
    return (
        _ApiReadResult(
            result=result,
            rows=tuple(rows),
            source_read_refs=_dedupe_refs(tuple(source_read_refs)),
            response_hash=response_body_hash(result),
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
    api_read = _ApiReadResult(
        result=result,
        rows=rows,
        error=row_error,
        source_read_refs=source_read_refs,
        response_hash=observation.response_hash,
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


def _population_choice_proof_refs(
    population_choices: tuple[ResolvedPopulationChoice, ...],
) -> tuple[str, ...]:
    return _dedupe_refs(
        tuple(ref for item in population_choices for ref in item.proof_refs)
    )


def _scope_fingerprint(
    endpoint_args: dict[str, Any],
    endpoint_arg_proof_refs: dict[str, tuple[str, ...]],
    population_choices: tuple[ResolvedPopulationChoice, ...] = (),
) -> str:
    scope: dict[str, Any] = {
        "endpointArgs": endpoint_args,
        "endpointArgProofRefs": {
            param_ref: list(proof_refs)
            for param_ref, proof_refs in endpoint_arg_proof_refs.items()
            if param_ref in endpoint_args and proof_refs
        },
    }
    if population_choices:
        scope["populationChoices"] = [
            {
                "controllerKind": item.controller_kind.value,
                "controllerId": item.controller_id,
                "fieldId": item.field_id,
                "requestedFactIds": list(item.requested_fact_ids),
                "semanticControlRef": item.semantic_control_ref,
                "includedValues": list(item.included_values),
                "excludedValues": list(item.excluded_values),
            }
            for item in population_choices
        ]
    return canonical_runtime_json(scope)


def _dedupe_refs(refs: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(ref for ref in refs if ref))


def _bound_row(
    raw: dict[str, Any],
    *,
    relation: Relation,
    catalog: RelationCatalog,
    row_source: RowSource,
) -> dict[str, RuntimeValue]:
    read = catalog.read(row_source.read_id)
    fields_by_ref = {field.ref: field for field in read.fields}
    output: dict[str, RuntimeValue] = {}
    for relation_field in relation.fields:
        row_source_field = row_source.field(relation_field.field_id)
        catalog_field = fields_by_ref.get(row_source_field.field_ref)
        if catalog_field is None:
            raise ValueError(f"unknown relation field {relation_field.field_id}")
        raw_value = required_response_path_value(
            raw,
            row_source_field.response_path
            or relative_response_path(catalog_field.path, row_source.row_path),
        )
        output[relation_field.field_id] = parse_declared_value(
            raw_value,
            row_source_field.type.value,
        )
    return output


def _bound_memory_row(row: Row, *, relation: Relation) -> dict[str, RuntimeValue]:
    output: dict[str, RuntimeValue] = {}
    for relation_field in relation.fields:
        if relation_field.field_id in row:
            output[relation_field.field_id] = row[relation_field.field_id]
        else:
            raise ValueError(f"unknown memory relation field {relation_field.field_id}")
    return output


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
