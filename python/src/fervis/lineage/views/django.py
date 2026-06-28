"""Django-backed lineage view query adapter."""

from __future__ import annotations

from fervis.lineage import models
from fervis.lineage.enums import (
    AnswerValueKind,
    ClarificationBasis,
    FactResultKind,
    MemoryArtifactSourceKind,
    PresentationClientKey,
    PresentationKind,
    RunResultKind,
    RunStepKey,
    RunStepKind,
    RunTriggerKind,
    RuntimeErrorKind,
    SourceReadStatus,
)
from fervis.lineage.views.query import (
    AnswerOutputRow,
    AnswerPresentationRow,
    AnswerRow,
    CatalogEndpointRow,
    ClarificationRequestRow,
    ClarificationResponseRow,
    ConversationRow,
    FactResultRow,
    LineageQueryPort,
    LineageRows,
    MemoryArtifactRow,
    ProofGraphRow,
    QuestionRow,
    RequestedFactRow,
    RunResultRow,
    RunRow,
    RuntimeErrorRow,
    SourceReadRow,
    StepRow,
)


class DjangoLineageQuery(LineageQueryPort):
    def run_id_for_answer(self, answer_id: str) -> str | None:
        answer = (
            models.Answer.objects.filter(answer_id=answer_id).only("run_id").first()
        )
        if answer is None:
            return None
        return answer.run_id

    def run_by_id(self, run_id: str) -> RunRow | None:
        run = models.QuestionRun.objects.filter(run_id=run_id).first()
        if run is None:
            return None
        return _run_row(run)

    def run_ids_for_run(self, run_id: str) -> tuple[str, ...]:
        return tuple(
            models.QuestionRun.objects.filter(run_id=run_id).values_list(
                "run_id", flat=True
            )
        )

    def run_ids_for_question(self, question_id: str) -> tuple[str, ...]:
        return tuple(
            models.QuestionRun.objects.filter(question_id=question_id)
            .order_by("run_number")
            .values_list("run_id", flat=True)
        )

    def run_ids_for_conversation(self, conversation_id: str) -> tuple[str, ...]:
        return tuple(
            models.QuestionRun.objects.filter(question__conversation_id=conversation_id)
            .order_by("question__conversation_sequence", "run_number")
            .values_list("run_id", flat=True)
        )

    def lineage_rows_for_run_ids(self, run_ids: tuple[str, ...]) -> LineageRows:
        if not run_ids:
            return LineageRows()
        run_id_set = set(run_ids)
        runs = tuple(
            models.QuestionRun.objects.filter(run_id__in=run_id_set)
            .select_related("question", "question__conversation")
            .order_by("question__conversation_sequence", "run_number")
        )
        question_ids = {run.question_id for run in runs}
        conversation_ids = {run.question.conversation_id for run in runs}
        return LineageRows(
            conversations=tuple(
                _conversation_row(item)
                for item in models.Conversation.objects.filter(
                    conversation_id__in=conversation_ids
                ).order_by("created_at")
            ),
            questions=tuple(
                _question_row(item)
                for item in models.Question.objects.filter(
                    question_id__in=question_ids
                ).order_by("conversation_sequence")
            ),
            runs=tuple(_run_row(item) for item in runs),
            steps=tuple(
                _step_row(item)
                for item in models.RunStep.objects.filter(
                    run_id__in=run_id_set
                ).order_by("run_id", "sequence")
            ),
            run_results=tuple(
                _run_result_row(item)
                for item in models.RunResult.objects.filter(
                    run_id__in=run_id_set
                ).order_by("run_id")
            ),
            runtime_errors=tuple(
                _runtime_error_row(item)
                for item in models.RuntimeErrorDetail.objects.filter(
                    run_id__in=run_id_set
                ).order_by("run_id", "runtime_error_detail_id")
            ),
            clarification_requests=tuple(
                _clarification_request_row(item)
                for item in models.ClarificationRequest.objects.filter(
                    run_id__in=run_id_set
                ).order_by("run_id", "clarification_id")
            ),
            clarification_responses=tuple(
                _clarification_response_row(item)
                for item in models.ClarificationResponse.objects.filter(
                    run_id__in=run_id_set
                ).order_by("run_id", "response_id")
            ),
            requested_facts=tuple(
                _requested_fact_row(item)
                for item in models.RequestedFact.objects.filter(
                    run_id__in=run_id_set
                ).order_by("run_id", "fact_key")
            ),
            fact_results=tuple(
                _fact_result_row(item)
                for item in models.FactResult.objects.filter(
                    run_id__in=run_id_set
                ).order_by("run_id", "fact_result_id")
            ),
            memory_artifacts=tuple(
                _memory_artifact_row(item)
                for item in models.MemoryArtifact.objects.filter(
                    run_id__in=run_id_set
                ).order_by("run_id", "memory_artifact_id")
            ),
            answers=tuple(
                _answer_row(item)
                for item in models.Answer.objects.filter(
                    run_id__in=run_id_set
                ).order_by("run_id", "answer_id")
            ),
            answer_outputs=tuple(
                _answer_output_row(item)
                for item in models.AnswerOutput.objects.filter(
                    run_id__in=run_id_set
                ).order_by("run_id", "answer_id", "output_key")
            ),
            answer_presentations=tuple(
                _answer_presentation_row(item)
                for item in models.AnswerPresentation.objects.filter(
                    run_id__in=run_id_set
                ).order_by("run_id", "answer_id", "client_key", "locale")
            ),
            catalog_endpoints=tuple(
                _catalog_endpoint_row(item)
                for item in models.CatalogEndpoint.objects.filter(
                    run_id__in=run_id_set
                ).order_by("run_id", "catalog_endpoint_key")
            ),
            source_reads=tuple(
                _source_read_row(item)
                for item in models.SourceRead.objects.select_related("catalog_endpoint")
                .filter(run_id__in=run_id_set)
                .order_by("run_id", "source_read_id")
            ),
            proof_graphs=tuple(
                _proof_graph_row(item)
                for item in models.ExecutionProofGraph.objects.filter(
                    run_id__in=run_id_set
                ).order_by("run_id", "proof_graph_id")
            ),
        )

    def memory_artifact_rows_for_conversation(
        self,
        conversation_id: str,
        *,
        limit: int,
    ) -> tuple[MemoryArtifactRow, ...]:
        if limit <= 0:
            return ()
        run_ids = _recent_artifact_bearing_run_ids(
            conversation_id=conversation_id,
            limit=limit,
        )
        if not run_ids:
            return ()
        run_id_set = frozenset(run_ids)
        rows = tuple(
            models.MemoryArtifact.objects.filter(
                run_id__in=run_id_set,
            )
            .select_related("run", "run__question")
            .order_by(
                "run__question__conversation_sequence",
                "run__run_number",
                "created_at",
                "memory_artifact_id",
            )
        )
        return tuple(_memory_artifact_row(item) for item in rows)


def _recent_artifact_bearing_run_ids(
    *,
    conversation_id: str,
    limit: int,
) -> tuple[str, ...]:
    run_ids: list[str] = []
    seen: set[str] = set()
    rows = (
        models.MemoryArtifact.objects.filter(
            run__question__conversation_id=conversation_id,
        )
        .select_related("run", "run__question")
        .order_by(
            "-run__question__conversation_sequence",
            "-run__run_number",
            "-created_at",
            "-memory_artifact_id",
        )
    )
    for row in rows:
        if row.run_id in seen:
            continue
        seen.add(row.run_id)
        run_ids.append(row.run_id)
        if len(run_ids) >= limit:
            break
    return tuple(run_ids)


def _conversation_row(item: models.Conversation) -> ConversationRow:
    return ConversationRow(
        conversation_id=item.conversation_id, tenant_id=item.tenant_id
    )


def _question_row(item: models.Question) -> QuestionRow:
    return QuestionRow(
        question_id=item.question_id,
        conversation_id=item.conversation_id,
        conversation_sequence=item.conversation_sequence,
        original_question=item.original_question,
    )


def _run_row(item: models.QuestionRun) -> RunRow:
    return RunRow(
        run_id=item.run_id,
        question_id=item.question_id,
        run_number=item.run_number,
        trigger_kind=RunTriggerKind(item.trigger_kind),
        integrated_question=item.integrated_question,
        previous_run_id=item.previous_run_id,
        trigger_clarification_response_run_id=(
            item.trigger_clarification_response_run_id
        ),
        trigger_clarification_response_id=(
            item.trigger_clarification_response_id or None
        ),
    )


def _step_row(item: models.RunStep) -> StepRow:
    return StepRow(
        step_id=item.step_id,
        run_id=item.run_id,
        sequence=item.sequence,
        step_key=RunStepKey(item.step_key),
        kind=RunStepKind(item.kind),
        input_summary_json=item.input_summary_json,
        output_summary_json=item.output_summary_json,
        error_json=item.error_json,
    )


def _run_result_row(item: models.RunResult) -> RunResultRow:
    return RunResultRow(
        run_result_id=item.run_result_id,
        run_id=item.run_id,
        result_kind=RunResultKind(item.result_kind),
    )


def _runtime_error_row(item: models.RuntimeErrorDetail) -> RuntimeErrorRow:
    return RuntimeErrorRow(
        runtime_error_detail_id=item.runtime_error_detail_id,
        run_id=item.run_id,
        run_result_id=item.run_result_id,
        failed_step_id=item.failed_step_id,
        error_kind=RuntimeErrorKind(item.error_kind),
        message=item.message,
    )


def _clarification_request_row(
    item: models.ClarificationRequest,
) -> ClarificationRequestRow:
    return ClarificationRequestRow(
        clarification_id=item.clarification_id,
        run_id=item.run_id,
        basis=ClarificationBasis(item.basis),
        question_text=item.question_text,
        fact_result_id=item.fact_result_id,
        step_id=item.step_id,
        options_json=tuple(item.options_json or ()),
        evidence_refs_json=tuple(item.evidence_refs_json or ()),
    )


def _clarification_response_row(
    item: models.ClarificationResponse,
) -> ClarificationResponseRow:
    return ClarificationResponseRow(
        response_id=item.response_id,
        run_id=item.run_id,
        clarification_id=item.clarification_id,
        evidence_ref=item.evidence_ref,
        source_message_ref=item.source_message_ref,
        selected_option_id=item.selected_option_id,
        response_text=item.response_text,
    )


def _requested_fact_row(item: models.RequestedFact) -> RequestedFactRow:
    return RequestedFactRow(
        requested_fact_id=item.requested_fact_id,
        run_id=item.run_id,
        produced_by_step_id=item.produced_by_step_id,
        fact_key=item.fact_key,
        description=item.description,
        answer_expression_family=item.answer_expression_family,
        requested_fact_json=item.requested_fact_json,
        answer_requests_json=item.answer_requests_json,
    )


def _fact_result_row(item: models.FactResult) -> FactResultRow:
    return FactResultRow(
        fact_result_id=item.fact_result_id,
        run_id=item.run_id,
        requested_fact_id=item.requested_fact_id,
        produced_by_step_id=item.produced_by_step_id,
        result_kind=FactResultKind(item.result_kind),
        evidence_refs_json=tuple(item.evidence_refs_json or ()),
        payload_schema=item.payload_schema,
        payload_schema_rev=item.payload_schema_rev,
        payload_json=item.payload_json,
    )


def _memory_artifact_row(item: models.MemoryArtifact) -> MemoryArtifactRow:
    return MemoryArtifactRow(
        memory_artifact_id=item.memory_artifact_id,
        run_id=item.run_id,
        produced_by_step_id=item.produced_by_step_id,
        source_kind=MemoryArtifactSourceKind(item.source_kind),
        payload_schema=item.payload_schema,
        payload_schema_rev=item.payload_schema_rev,
        payload_json=item.payload_json,
        requested_fact_id=item.requested_fact_id,
        fact_result_id=item.fact_result_id,
    )


def _answer_row(item: models.Answer) -> AnswerRow:
    return AnswerRow(
        answer_id=item.answer_id,
        run_id=item.run_id,
        run_result_id=item.run_result_id,
    )


def _answer_output_row(item: models.AnswerOutput) -> AnswerOutputRow:
    return AnswerOutputRow(
        answer_output_id=item.answer_output_id,
        run_id=item.run_id,
        answer_id=item.answer_id,
        fact_result_id=item.fact_result_id,
        output_key=item.output_key,
        value_kind=AnswerValueKind(item.value_kind),
        value_json=item.value_json,
        proof_node_refs_json=tuple(item.proof_node_refs_json or ()),
    )


def _answer_presentation_row(
    item: models.AnswerPresentation,
) -> AnswerPresentationRow:
    return AnswerPresentationRow(
        presentation_id=item.presentation_id,
        run_id=item.run_id,
        answer_id=item.answer_id,
        client_key=PresentationClientKey(item.client_key),
        locale=item.locale,
        presentation_kind=PresentationKind(item.presentation_kind),
        render_step_id=item.render_step_id,
        rendered_value=item.rendered_value or "",
        payload_schema=item.payload_schema,
        payload_schema_rev=item.payload_schema_rev,
        payload_json=item.payload_json,
    )


def _source_read_row(item: models.SourceRead) -> SourceReadRow:
    return SourceReadRow(
        source_read_id=item.source_read_id,
        run_id=item.run_id,
        step_id=item.step_id,
        catalog_endpoint_id=item.catalog_endpoint_id,
        args_json=item.args_json,
        status=SourceReadStatus(item.status),
        row_count=item.row_count,
        completeness_json=item.completeness_json,
        response_hash=item.response_hash,
        artifact_id=item.artifact_id,
        error_json=item.error_json,
    )


def _catalog_endpoint_row(item: models.CatalogEndpoint) -> CatalogEndpointRow:
    return CatalogEndpointRow(
        catalog_endpoint_id=item.catalog_endpoint_id,
        run_id=item.run_id,
        catalog_endpoint_key=item.catalog_endpoint_key,
        endpoint_name=item.endpoint_name,
        framework_kind=item.framework_kind,
        source_namespace_kind=item.source_namespace_kind,
        source_namespace_path_json=tuple(item.source_namespace_path_json or ()),
        route_method=item.route_method,
        route_path_template=item.route_path_template,
        route_name=item.route_name,
        api_schema_operation_id=item.api_schema_operation_id,
        handler_ref=item.handler_ref,
        domain_resource_names_json=tuple(item.domain_resource_names_json or ()),
    )


def _proof_graph_row(item: models.ExecutionProofGraph) -> ProofGraphRow:
    return ProofGraphRow(
        proof_graph_id=item.proof_graph_id,
        run_id=item.run_id,
        fact_result_id=item.fact_result_id,
        compile_step_id=item.compile_step_id,
        execute_step_id=item.execute_step_id,
        payload_schema=item.payload_schema,
        payload_schema_rev=item.payload_schema_rev,
        payload_json=item.payload_json,
    )
