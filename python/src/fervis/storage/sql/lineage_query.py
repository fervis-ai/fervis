"""SQL-backed lineage query adapter."""

from __future__ import annotations

from typing import Any, Callable

import sqlalchemy as sa
from sqlalchemy.engine import Engine

from fervis.lineage.enums import (
    AnswerValueKind,
    FactResultKind,
    MemoryArtifactSourceKind,
    PresentationClientKey,
    PresentationKind,
    RunResultKind,
    RunStepKey,
    RunStepKind,
    RunTriggerKind,
    QuestionRunKind,
    RuntimeErrorKind,
    SourceReadStatus,
)
from fervis.lineage.memory_artifacts import MemoryArtifactRow
from fervis.lookup.clarification import ClarificationNeed, ClarificationReason
from fervis.lineage.views.query import (
    AnswerProgramRow,
    AnswerOutputRow,
    AnswerPresentationRow,
    AnswerRow,
    CatalogEndpointRow,
    ClarificationRequestRow,
    ClarificationResponseRow,
    ConversationRow,
    BindingPatchRow,
    FactResultRow,
    LineageQueryPort,
    LineageRows,
    ProofGraphRow,
    ProgramInvocationRow,
    ProgramRevisionRow,
    QuestionRow,
    RequestedFactRow,
    RunResultRow,
    RunRow,
    RuntimeErrorRow,
    SourceReadRow,
    StepRow,
)
from fervis.project.persistence.schema import metadata

from .rows import row_mapping, row_mappings
from .transaction import sql_connection


class SQLLineageQuery(LineageQueryPort):
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def run_id_for_answer(self, answer_id: str) -> str | None:
        answer = metadata.tables["fervis_answer"]
        with sql_connection(self.engine) as connection:
            value = connection.execute(
                sa.select(answer.c.run_id).where(answer.c.answer_id == answer_id)
            ).scalar()
        return str(value) if value is not None else None

    def run_by_id(self, run_id: str) -> RunRow | None:
        run = metadata.tables["fervis_question_run"]
        with sql_connection(self.engine) as connection:
            row = connection.execute(
                sa.select(run).where(run.c.run_id == run_id)
            ).first()
        return _run_row(row_mapping(row)) if row is not None else None

    def run_ids_for_run(self, run_id: str) -> tuple[str, ...]:
        run = metadata.tables["fervis_question_run"]
        return _string_column(
            self.engine,
            sa.select(run.c.run_id).where(run.c.run_id == run_id),
        )

    def run_ids_for_question(self, question_id: str) -> tuple[str, ...]:
        run = metadata.tables["fervis_question_run"]
        return _string_column(
            self.engine,
            sa.select(run.c.run_id)
            .where(run.c.question_id == question_id)
            .order_by(run.c.run_number),
        )

    def run_ids_for_conversation(self, conversation_id: str) -> tuple[str, ...]:
        question = metadata.tables["fervis_question"]
        run = metadata.tables["fervis_question_run"]
        return _string_column(
            self.engine,
            sa.select(run.c.run_id)
            .select_from(
                run.join(question, run.c.question_id == question.c.question_id)
            )
            .where(question.c.conversation_id == conversation_id)
            .order_by(question.c.conversation_sequence, run.c.run_number),
        )

    def lineage_rows_for_run_ids(self, run_ids: tuple[str, ...]) -> LineageRows:
        if not run_ids:
            return LineageRows()
        run_ids_set = frozenset(run_ids)
        rows = _LineageRowsForRuns(self.engine, run_ids_set)
        runs = rows.fetch("fervis_question_run", _run_row, "question_id", "run_number")
        program_invocations = rows.fetch(
            "fervis_program_invocation",
            _program_invocation_row,
            "run_id",
        )
        revision_ids = frozenset(
            item.revision_id
            for item in program_invocations
            if item.revision_id is not None
        )
        program_revisions = rows.fetch_by_values(
            "fervis_program_revision",
            "revision_id",
            revision_ids,
            _program_revision_row,
            "revision_id",
        )
        program_ids = frozenset(
            {
                *(item.program_id for item in program_invocations),
                *(item.base_program_id for item in program_revisions),
                *(item.revised_program_id for item in program_revisions),
            }
        )
        question_ids = frozenset(run.question_id for run in runs)
        questions = rows.fetch_questions(question_ids)
        conversation_ids = frozenset(question.conversation_id for question in questions)
        return LineageRows(
            conversations=rows.fetch_conversations(conversation_ids),
            questions=questions,
            runs=runs,
            answer_programs=rows.fetch_by_values(
                "fervis_answer_program",
                "program_id",
                program_ids,
                _answer_program_row,
                "program_id",
            ),
            program_invocations=program_invocations,
            program_revisions=program_revisions,
            steps=rows.fetch("fervis_run_step", _step_row, "run_id", "sequence"),
            run_results=rows.fetch("fervis_run_result", _run_result_row, "run_id"),
            runtime_errors=rows.fetch(
                "fervis_runtime_error_detail",
                _runtime_error_row,
                "run_id",
                "runtime_error_detail_id",
            ),
            clarification_requests=rows.fetch(
                "fervis_clarification_request",
                _clarification_request_row,
                "run_id",
                "clarification_id",
            ),
            clarification_responses=rows.fetch(
                "fervis_clarification_response",
                _clarification_response_row,
                "run_id",
                "response_id",
            ),
            requested_facts=rows.fetch(
                "fervis_requested_fact", _requested_fact_row, "run_id", "fact_key"
            ),
            fact_results=rows.fetch(
                "fervis_fact_result", _fact_result_row, "run_id", "fact_result_id"
            ),
            memory_artifacts=rows.fetch(
                "fervis_memory_artifact",
                _memory_artifact_row,
                "run_id",
                "memory_artifact_id",
            ),
            answers=rows.fetch("fervis_answer", _answer_row, "run_id", "answer_id"),
            answer_outputs=rows.fetch(
                "fervis_answer_output",
                _answer_output_row,
                "run_id",
                "answer_id",
                "output_key",
            ),
            answer_presentations=rows.fetch(
                "fervis_answer_presentation",
                _answer_presentation_row,
                "run_id",
                "answer_id",
                "client_key",
                "locale",
            ),
            catalog_endpoints=rows.fetch(
                "fervis_catalog_endpoint",
                _catalog_endpoint_row,
                "run_id",
                "catalog_endpoint_key",
            ),
            source_reads=rows.fetch(
                "fervis_source_read", _source_read_row, "run_id", "source_read_id"
            ),
            proof_graphs=rows.fetch(
                "fervis_execution_proof_graph",
                _proof_graph_row,
                "run_id",
                "proof_graph_id",
            ),
        )

    def memory_artifact_rows_for_run_ids(
        self,
        run_ids: tuple[str, ...],
    ) -> tuple[MemoryArtifactRow, ...]:
        if not run_ids:
            return ()
        memory = metadata.tables["fervis_memory_artifact"]
        run = metadata.tables["fervis_question_run"]
        question = metadata.tables["fervis_question"]
        with sql_connection(self.engine) as connection:
            rows = connection.execute(
                sa.select(memory)
                .select_from(
                    memory.join(run, memory.c.run_id == run.c.run_id).join(
                        question,
                        run.c.question_id == question.c.question_id,
                    )
                )
                .where(memory.c.run_id.in_(run_ids))
                .order_by(
                    question.c.conversation_sequence,
                    run.c.run_number,
                    memory.c.created_at,
                    memory.c.memory_artifact_id,
                )
            ).all()
        return tuple(_memory_artifact_row(row) for row in row_mappings(rows))


class _LineageRowsForRuns:
    def __init__(self, engine: Engine, run_ids: frozenset[str]) -> None:
        self.engine = engine
        self.run_ids = run_ids

    def fetch(
        self,
        table_name: str,
        mapper: Callable[[dict[str, Any]], Any],
        *order_by: str,
    ) -> tuple[Any, ...]:
        table = metadata.tables[table_name]
        with sql_connection(self.engine) as connection:
            rows = connection.execute(
                sa.select(table)
                .where(table.c.run_id.in_(self.run_ids))
                .order_by(*(table.c[name] for name in order_by))
            ).all()
        return tuple(mapper(row) for row in row_mappings(rows))

    def fetch_questions(self, question_ids: frozenset[str]) -> tuple[QuestionRow, ...]:
        if not question_ids:
            return ()
        question = metadata.tables["fervis_question"]
        with sql_connection(self.engine) as connection:
            rows = connection.execute(
                sa.select(question)
                .where(question.c.question_id.in_(question_ids))
                .order_by(question.c.conversation_sequence)
            ).all()
        return tuple(_question_row(row) for row in row_mappings(rows))

    def fetch_by_values(
        self,
        table_name: str,
        column_name: str,
        values: frozenset[str],
        mapper: Callable[[dict[str, Any]], Any],
        *order_by: str,
    ) -> tuple[Any, ...]:
        if not values:
            return ()
        table = metadata.tables[table_name]
        with sql_connection(self.engine) as connection:
            rows = connection.execute(
                sa.select(table)
                .where(table.c[column_name].in_(values))
                .order_by(*(table.c[name] for name in order_by))
            ).all()
        return tuple(mapper(row) for row in row_mappings(rows))

    def fetch_conversations(
        self, conversation_ids: frozenset[str]
    ) -> tuple[ConversationRow, ...]:
        if not conversation_ids:
            return ()
        conversation = metadata.tables["fervis_conversation"]
        with sql_connection(self.engine) as connection:
            rows = connection.execute(
                sa.select(conversation)
                .where(conversation.c.conversation_id.in_(conversation_ids))
                .order_by(conversation.c.created_at)
            ).all()
        return tuple(_conversation_row(row) for row in row_mappings(rows))


def _string_column(engine: Engine, statement) -> tuple[str, ...]:
    with sql_connection(engine) as connection:
        return tuple(str(value) for value in connection.execute(statement).scalars())


def _tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    return tuple(value)


def _dict(value: Any) -> dict[str, object]:
    return dict(value or {})


def _conversation_row(row: dict[str, Any]) -> ConversationRow:
    return ConversationRow(
        conversation_id=str(row["conversation_id"]),
        tenant_id=str(row["tenant_id"]),
    )


def _question_row(row: dict[str, Any]) -> QuestionRow:
    return QuestionRow(
        question_id=str(row["question_id"]),
        conversation_id=str(row["conversation_id"]),
        conversation_sequence=int(row["conversation_sequence"]),
        original_question=str(row["original_question"]),
    )


def _run_row(row: dict[str, Any]) -> RunRow:
    return RunRow(
        run_id=str(row["run_id"]),
        question_id=str(row["question_id"]),
        run_number=int(row["run_number"]),
        kind=QuestionRunKind(row["kind"]),
        trigger_kind=RunTriggerKind(row["trigger_kind"]),
        base_run_id=row["base_run_id"],
        trigger_clarification_response_id=row["trigger_clarification_response_id"]
        or None,
    )


def _answer_program_row(row: dict[str, Any]) -> AnswerProgramRow:
    return AnswerProgramRow(
        program_id=str(row["program_id"]),
        schema_revision=int(row["schema_revision"]),
        canonical_json=str(row["canonical_json"]),
    )


def _program_invocation_row(row: dict[str, Any]) -> ProgramInvocationRow:
    patch_id = row["patch_id"]
    patch_json = row["binding_patch_json"]
    patch = None
    if patch_id is not None:
        if patch_json is None:
            raise ValueError(
                f"program invocation {row['invocation_id']} has no binding patch"
            )
        patch = BindingPatchRow(
            patch_id=str(patch_id),
            canonical_json=str(patch_json),
        )
    return ProgramInvocationRow(
        invocation_id=str(row["invocation_id"]),
        run_id=str(row["run_id"]),
        program_id=str(row["program_id"]),
        kind=str(row["kind"]),
        base_invocation_id=(
            str(row["base_invocation_id"])
            if row["base_invocation_id"] is not None
            else None
        ),
        bindings_json=str(row["bindings_json"]),
        patch=patch,
        revision_id=(
            str(row["revision_id"]) if row["revision_id"] is not None else None
        ),
    )


def _program_revision_row(row: dict[str, Any]) -> ProgramRevisionRow:
    return ProgramRevisionRow(
        revision_id=str(row["revision_id"]),
        base_program_id=str(row["base_program_id"]),
        revised_program_id=str(row["revised_program_id"]),
        capability_id=str(row["capability_id"]),
        application_json=str(row["application_json"]),
    )


def _step_row(row: dict[str, Any]) -> StepRow:
    return StepRow(
        step_id=str(row["step_id"]),
        run_id=str(row["run_id"]),
        sequence=int(row["sequence"]),
        step_key=RunStepKey(row["step_key"]),
        kind=RunStepKind(row["kind"]),
        input_summary_json=_dict(row["input_summary_json"]),
        output_summary_json=_dict(row["output_summary_json"]),
        error_json=_dict(row["error_json"]),
    )


def _run_result_row(row: dict[str, Any]) -> RunResultRow:
    return RunResultRow(
        run_result_id=str(row["run_result_id"]),
        run_id=str(row["run_id"]),
        result_kind=RunResultKind(row["result_kind"]),
    )


def _runtime_error_row(row: dict[str, Any]) -> RuntimeErrorRow:
    return RuntimeErrorRow(
        runtime_error_detail_id=str(row["runtime_error_detail_id"]),
        run_id=str(row["run_id"]),
        run_result_id=str(row["run_result_id"]),
        failed_step_id=row["failed_step_id"],
        error_kind=RuntimeErrorKind(row["error_kind"]),
        message=str(row["message"]),
    )


def _clarification_request_row(row: dict[str, Any]) -> ClarificationRequestRow:
    return ClarificationRequestRow(
        clarification_id=str(row["clarification_id"]),
        run_id=str(row["run_id"]),
        need=ClarificationNeed(row["need"]),
        reason=ClarificationReason(row["reason"]),
        payload_json=_dict(row["payload_json"]),
        fact_result_id=row["fact_result_id"],
        step_id=row["step_id"],
    )


def _clarification_response_row(row: dict[str, Any]) -> ClarificationResponseRow:
    return ClarificationResponseRow(
        response_id=str(row["response_id"]),
        run_id=str(row["run_id"]),
        clarification_id=str(row["clarification_id"]),
        evidence_ref=str(row["evidence_ref"]),
        source_message_ref=str(row["source_message_ref"]),
        selected_option_id=str(row["selected_option_id"]),
        response_text=str(row["response_text"]),
    )


def _requested_fact_row(row: dict[str, Any]) -> RequestedFactRow:
    return RequestedFactRow(
        requested_fact_id=str(row["requested_fact_id"]),
        run_id=str(row["run_id"]),
        produced_by_step_id=str(row["produced_by_step_id"]),
        fact_key=str(row["fact_key"]),
        description=str(row["description"]),
        answer_expression_family=str(row["answer_expression_family"]),
        requested_fact_json=_dict(row["requested_fact_json"]),
        answer_requests_json=_dict(row["answer_requests_json"]),
    )


def _fact_result_row(row: dict[str, Any]) -> FactResultRow:
    return FactResultRow(
        fact_result_id=str(row["fact_result_id"]),
        run_id=str(row["run_id"]),
        requested_fact_id=str(row["requested_fact_id"]),
        produced_by_step_id=str(row["produced_by_step_id"]),
        result_kind=FactResultKind(row["result_kind"]),
        evidence_refs_json=tuple(
            str(item) for item in _tuple(row["evidence_refs_json"])
        ),
        payload_schema=str(row["payload_schema"]),
        payload_schema_rev=row["payload_schema_rev"],
        payload_json=row["payload_json"],
    )


def _memory_artifact_row(row: dict[str, Any]) -> MemoryArtifactRow:
    return MemoryArtifactRow(
        memory_artifact_id=str(row["memory_artifact_id"]),
        run_id=str(row["run_id"]),
        produced_by_step_id=str(row["produced_by_step_id"]),
        source_kind=MemoryArtifactSourceKind(row["source_kind"]),
        payload_schema=str(row["payload_schema"]),
        payload_schema_rev=int(row["payload_schema_rev"]),
        payload_json=_dict(row["payload_json"]),
        requested_fact_id=row["requested_fact_id"],
        fact_result_id=row["fact_result_id"],
    )


def _answer_row(row: dict[str, Any]) -> AnswerRow:
    return AnswerRow(
        answer_id=str(row["answer_id"]),
        run_id=str(row["run_id"]),
        run_result_id=str(row["run_result_id"]),
    )


def _answer_output_row(row: dict[str, Any]) -> AnswerOutputRow:
    return AnswerOutputRow(
        answer_output_id=str(row["answer_output_id"]),
        run_id=str(row["run_id"]),
        answer_id=str(row["answer_id"]),
        fact_result_id=str(row["fact_result_id"]),
        output_key=str(row["output_key"]),
        value_kind=AnswerValueKind(row["value_kind"]),
        value_json=_dict(row["value_json"]),
        proof_node_refs_json=tuple(
            str(item) for item in _tuple(row["proof_node_refs_json"])
        ),
    )


def _answer_presentation_row(row: dict[str, Any]) -> AnswerPresentationRow:
    return AnswerPresentationRow(
        presentation_id=str(row["presentation_id"]),
        run_id=str(row["run_id"]),
        answer_id=str(row["answer_id"]),
        client_key=PresentationClientKey(row["client_key"]),
        locale=str(row["locale"]),
        presentation_kind=PresentationKind(row["presentation_kind"]),
        render_step_id=str(row["render_step_id"]),
        rendered_value=str(row["rendered_value"] or ""),
        payload_schema=str(row["payload_schema"]),
        payload_schema_rev=row["payload_schema_rev"],
        payload_json=row["payload_json"],
    )


def _catalog_endpoint_row(row: dict[str, Any]) -> CatalogEndpointRow:
    return CatalogEndpointRow(
        catalog_endpoint_id=str(row["catalog_endpoint_id"]),
        run_id=str(row["run_id"]),
        catalog_endpoint_key=str(row["catalog_endpoint_key"]),
        endpoint_name=str(row["endpoint_name"]),
        framework_kind=str(row["framework_kind"]),
        source_namespace_kind=str(row["source_namespace_kind"]),
        source_namespace_path_json=tuple(
            str(item) for item in _tuple(row["source_namespace_path_json"])
        ),
        route_method=str(row["route_method"]),
        route_path_template=str(row["route_path_template"]),
        route_name=str(row["route_name"]),
        api_schema_operation_id=str(row["api_schema_operation_id"]),
        handler_ref=str(row["handler_ref"]),
        domain_resource_names_json=tuple(
            str(item) for item in _tuple(row["domain_resource_names_json"])
        ),
    )


def _source_read_row(row: dict[str, Any]) -> SourceReadRow:
    return SourceReadRow(
        source_read_id=str(row["source_read_id"]),
        run_id=str(row["run_id"]),
        step_id=str(row["step_id"]),
        catalog_endpoint_id=str(row["catalog_endpoint_id"]),
        args_json=_dict(row["args_json"]),
        status=SourceReadStatus(row["status"]),
        row_count=row["row_count"],
        completeness_json=_dict(row["completeness_json"]),
        response_hash=str(row["response_hash"]),
        artifact_id=row["artifact_id"],
        error_json=_dict(row["error_json"]),
    )


def _proof_graph_row(row: dict[str, Any]) -> ProofGraphRow:
    return ProofGraphRow(
        proof_graph_id=str(row["proof_graph_id"]),
        run_id=str(row["run_id"]),
        fact_result_id=str(row["fact_result_id"]),
        compile_step_id=str(row["compile_step_id"]),
        execute_step_id=row["execute_step_id"],
        payload_schema=str(row["payload_schema"]),
        payload_schema_rev=int(row["payload_schema_rev"]),
        payload_json=_dict(row["payload_json"]),
    )
