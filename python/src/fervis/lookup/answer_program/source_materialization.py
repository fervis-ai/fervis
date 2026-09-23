"""Materialize a verified physical source graph through one request executor."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from typing import TYPE_CHECKING, Any, Callable

from fervis.lookup.relation_catalog import (
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
    relative_response_path,
    required_response_path_value,
)
from fervis.lookup.plan_execution.generated_relations import (
    GeneratedCalendarRelationSource,
    generate_calendar_relation,
)
from fervis.lookup.plan_execution.relations import (
    RelationEvidence,
    RelationRows,
    Row,
    RowContextStore,
    api_read_completeness_proof,
    relation_snapshot_hash,
)
from fervis.lookup.relation_catalog.row_sources import (
    CALENDAR_DATE_FIELD_ID,
    CALENDAR_END_PARAM_REF,
    CALENDAR_MAX_ROWS,
    CALENDAR_START_PARAM_REF,
    RowSource,
    RowSourceCatalog,
    RowSourceKind,
    row_source_for_relation,
)
from fervis.lookup.answer_program.model import RelationProgram
from fervis.lookup.answer_program.instantiation import MaterializedRelationInputs
from fervis.lookup.plan_execution.operation_runtime import (
    RelationEngineInput,
    RelationEngineOutput,
    scalar_inputs_for_operations,
)
from fervis.lookup.plan_execution.operation_engine import execute_operations
from fervis.lookup.answer_program.api_reads import (
    ApiReadResult,
    ApiReadSession,
    _fanout_endpoint_args,
)
from fervis.lookup.answer_program.relations import (
    Relation,
    RelationSource,
    SourceKind,
)

if TYPE_CHECKING:
    from fervis.lookup.memory.projection import LookupMemory


@dataclass(frozen=True)
class _RelationExecutionRows:
    relation: RelationRows
    row_context: tuple[Row, ...] = ()


def _relation_rows(
    relation: Relation,
    *,
    row_sources: RowSourceCatalog,
    catalog: RelationCatalog,
    api_reads: dict[str, ApiReadResult],
    memory: LookupMemory,
    endpoint_args: dict[str, dict[str, Any]],
    endpoint_arg_proofs: dict[str, tuple[str, ...]],
    endpoint_arg_proofs_by_param: dict[str, dict[str, tuple[str, ...]]],
    authority_ref: str,
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
    )
    return _RelationExecutionRows(
        relation=rows.relation.with_scope(
            proof_refs=endpoint_arg_proofs.get(relation.id, ()),
            scope_fingerprint=scope_fingerprint,
        ),
        row_context=rows.row_context,
    )


def _source_relation_rows(
    source: RelationSource,
    *,
    relation: Relation,
    row_sources: RowSourceCatalog,
    catalog: RelationCatalog,
    api_reads: dict[str, ApiReadResult],
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
                request_args=api_read.row_arguments[index],
            )
            for index, item in enumerate(api_read.rows)
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
                    read_refs=(read.id,) if api_read.invocation_count else (),
                    authority_refs=((authority_ref,) if authority_ref else ()),
                    snapshot_hash=relation_snapshot_hash(rows),
                    proof_refs=_dedupe_refs(
                        (
                            *(
                                (f"read:{read.endpoint_name}",)
                                if api_read.invocation_count
                                else ()
                            ),
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
                    ),
                    reached_terminal_page=True,
                    truncated=bool(api_read.result.get("truncated") is True),
                    proof_refs=_dedupe_refs(
                        (
                            *(
                                (f"read:{read.endpoint_name}",)
                                if api_read.invocation_count
                                else ()
                            ),
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
        field.id: field.type
        for field in (*row_source.fields, *row_source.request_argument_fields)
        if field.id and field.type
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


def _scope_fingerprint(
    endpoint_args: dict[str, Any],
    endpoint_arg_proof_refs: dict[str, tuple[str, ...]],
) -> str:
    scope: dict[str, Any] = {
        "endpointArgs": endpoint_args,
        "endpointArgProofRefs": {
            param_ref: list(proof_refs)
            for param_ref, proof_refs in endpoint_arg_proof_refs.items()
            if param_ref in endpoint_args and proof_refs
        },
    }
    return canonical_runtime_json(scope)


def _dedupe_refs(refs: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(ref for ref in refs if ref))


def _bound_row(
    raw: dict[str, Any],
    *,
    relation: Relation,
    catalog: RelationCatalog,
    row_source: RowSource,
    request_args: dict[str, Any],
) -> dict[str, RuntimeValue]:
    read = catalog.read(row_source.read_id)
    fields_by_ref = {field.ref: field for field in read.fields}
    output: dict[str, RuntimeValue] = {}
    for relation_field in relation.fields:
        row_source_field = row_source.field(relation_field.field_id)
        if row_source_field.request_parameter_ref:
            if row_source_field.request_parameter_ref not in request_args:
                raise RelationEngineError(
                    "request context field has no supplied argument"
                )
            output[relation_field.field_id] = parse_declared_value(
                request_args[row_source_field.request_parameter_ref],
                row_source_field.type.value,
            )
            continue
        if row_source_field.declared_entity_kind:
            output[relation_field.field_id] = row_source_field.declared_entity_kind
            continue
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


@dataclass
class SourceMaterializer:
    relations: tuple[Relation, ...]
    row_sources: RowSourceCatalog
    catalog: RelationCatalog
    memory: LookupMemory
    read_session: ApiReadSession
    endpoint_args: dict[str, dict[str, Any]] = field(default_factory=dict)
    endpoint_arg_proofs: dict[str, tuple[str, ...]] = field(default_factory=dict)
    endpoint_arg_proofs_by_param: dict[str, dict[str, tuple[str, ...]]] = field(
        default_factory=dict
    )
    authority_ref: str = ""
    _results: dict[str, _RelationExecutionRows] = field(
        default_factory=dict, init=False
    )
    _sources: dict[str, Relation] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._sources = {relation.id: relation for relation in self.relations}
        if len(self._sources) != len(self.relations):
            raise VerificationError("source graph repeats a relation")

    def row_context(self) -> RowContextStore:
        return RowContextStore(
            {
                item.relation.id: item.row_context
                for item in self._results.values()
                if item.row_context
            }
        )

    def load(
        self, relation_id: str, require_relation: Callable[[str], RelationRows]
    ) -> RelationRows:
        relation = self._sources[relation_id]
        source = _row_source_for_relation(relation, row_sources=self.row_sources)
        args = self.endpoint_args.get(relation.id) or {}
        parent = None
        api_reads = {}
        arg_sets: tuple[dict[str, Any], ...] = ()
        if relation.source.argument_relation_id:
            from fervis.lookup.answer_program.dependent_arguments import argument_rows

            parent = require_relation(relation.source.argument_relation_id)
            arg_sets = argument_rows(relation, source, parent, args)
        if source.kind is RowSourceKind.API_READ:
            if parent is None:
                arg_sets = _fanout_endpoint_args(args, row_source=source) or (args,)
            api_reads[relation.id] = self.read_session.read(
                self.catalog.read(source.read_id), source, arg_sets
            )
        rows = _relation_rows(
            relation,
            row_sources=self.row_sources,
            catalog=self.catalog,
            api_reads=api_reads,
            memory=self.memory,
            endpoint_args=self.endpoint_args,
            endpoint_arg_proofs=self.endpoint_arg_proofs,
            endpoint_arg_proofs_by_param=self.endpoint_arg_proofs_by_param,
            authority_ref=self.authority_ref,
        )
        if parent is not None:
            if not parent.rows:
                from fervis.lookup.plan_execution.relations import (
                    CompletenessStatus,
                    CompletenessSourceKind,
                    PaginationCompleteness,
                )

                rows = replace(
                    rows,
                    relation=replace(
                        rows.relation,
                        completeness=replace(
                            rows.relation.completeness,
                            status=CompletenessStatus.COMPLETE,
                            source_kind=CompletenessSourceKind.OPERATION_OUTPUT,
                            pagination=PaginationCompleteness.NOT_PAGINATED,
                        ),
                    ),
                )
            rows = replace(
                rows,
                relation=rows.relation.with_scope(
                    proof_refs=(
                        *parent.evidence.proof_refs,
                        *parent.completeness.proof_refs,
                    ),
                    scope_fingerprint=_scope_fingerprint(
                        {
                            "parent_snapshot": relation_snapshot_hash(parent.rows),
                            "argument_rows": arg_sets,
                        },
                        {},
                    ),
                ),
            )
        self._results[relation.id] = rows
        return rows.relation


@dataclass(frozen=True)
class RelationExecutionResult:
    engine_output: RelationEngineOutput
    row_context: RowContextStore


def execute_relation_inputs(
    program: RelationProgram,
    materialized: MaterializedRelationInputs,
    *,
    catalog: RelationCatalog,
    row_sources: RowSourceCatalog,
    memory: LookupMemory,
    read_session: ApiReadSession,
    authority_ref: str = "",
) -> RelationExecutionResult:
    inputs = materialized.instantiated_inputs
    materializer = SourceMaterializer(
        program.relations,
        row_sources,
        catalog,
        memory,
        read_session,
        inputs.values_by_relation,
        inputs.proofs_by_relation,
        inputs.proofs_by_parameter,
        authority_ref,
    )
    engine_output = execute_operations(
        RelationEngineInput(
            operations=materialized.operations,
            scalar_inputs=scalar_inputs_for_operations(materialized.operation_inputs),
            environment_values=materialized.expression_values,
            environment_types=materialized.expression_types,
            operation_proof_refs=materialized.operation_proof_refs,
            source_relation_ids=tuple(relation.id for relation in program.relations),
            relation_loader=materializer.load,
        )
    )
    return RelationExecutionResult(engine_output, materializer.row_context())
