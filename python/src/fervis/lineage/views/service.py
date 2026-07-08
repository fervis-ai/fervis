"""Framework-neutral lineage view graph services."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from fervis.lineage.proof_summary import (
    proof_applied_inputs,
    proof_computation_link_labels,
    proof_computation_summaries,
    proof_contributions,
    proof_endpoint_args,
    proof_evidence_handle_labels,
    proof_source_read_ids,
)
from fervis.lineage.payloads.execution_proof_graph import (
    read_execution_proof_graph_payload,
)
from fervis.lineage.proof_projection import project_proof_payload
from fervis.lineage.run_chain import run_chain_ids
from fervis.lineage.views.model import (
    AnswerOutputView,
    AnswerPresentationView,
    AnswerView,
    CatalogEndpointView,
    ClarificationRequestView,
    ClarificationResponseView,
    ContributionView,
    ExecutionProofView,
    FactResultView,
    LineageRootKind,
    LineageView,
    MemoryArtifactView,
    ProofComputationLinkView,
    ProofEndpointArgView,
    ProofAppliedInputView,
    QuestionView,
    RequestedFactView,
    RunView,
    RuntimeErrorView,
    SourceReadView,
    StepView,
)
from fervis.lineage.views.query import (
    AnswerOutputRow,
    AnswerPresentationRow,
    AnswerRow,
    CatalogEndpointRow,
    ClarificationRequestRow,
    ClarificationResponseRow,
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
from fervis.lineage.views.step_presenters import (
    step_decision_views,
    step_semantic_view,
)


class LineageRootNotFound(LookupError):
    """Raised when a lineage view root id does not exist."""


class AnswerLineageService:
    def __init__(self, query: LineageQueryPort) -> None:
        self._query = query

    def for_answer(self, answer_id: str) -> LineageView:
        run_id = self._query.run_id_for_answer(answer_id)
        if run_id is None:
            raise LineageRootNotFound(f"answer {answer_id!r} does not exist")
        return _build_view(
            root_kind=LineageRootKind.ANSWER,
            root_id=answer_id,
            rows=self._query.lineage_rows_for_run_ids(
                run_chain_ids(
                    run_id,
                    get_run=self._query.run_by_id,
                    missing=lambda item: LineageRootNotFound(
                        f"run {item!r} does not exist"
                    ),
                )
            ),
        )


class QuestionLineageService:
    def __init__(self, query: LineageQueryPort) -> None:
        self._query = query

    def for_question(self, question_id: str) -> LineageView:
        run_ids = self._query.run_ids_for_question(question_id)
        if not run_ids:
            raise LineageRootNotFound(f"question {question_id!r} does not exist")
        return _build_view(
            root_kind=LineageRootKind.QUESTION,
            root_id=question_id,
            rows=self._query.lineage_rows_for_run_ids(run_ids),
        )

    def for_run(self, run_id: str) -> LineageView:
        run_ids = self._query.run_ids_for_run(run_id)
        if not run_ids:
            raise LineageRootNotFound(f"run {run_id!r} does not exist")
        return _build_view(
            root_kind=LineageRootKind.RUN,
            root_id=run_id,
            rows=self._query.lineage_rows_for_run_ids(run_ids),
        )


class ConversationLineageService:
    def __init__(self, query: LineageQueryPort) -> None:
        self._query = query

    def for_conversation(self, conversation_id: str) -> LineageView:
        run_ids = self._query.run_ids_for_conversation(conversation_id)
        if not run_ids:
            raise LineageRootNotFound(
                f"conversation {conversation_id!r} does not exist"
            )
        return _build_view(
            root_kind=LineageRootKind.CONVERSATION,
            root_id=conversation_id,
            rows=self._query.lineage_rows_for_run_ids(run_ids),
        )


def _build_view(
    *, root_kind: LineageRootKind, root_id: str, rows: LineageRows
) -> LineageView:
    questions_by_id = {item.question_id: item for item in rows.questions}
    runs_by_question: dict[str, list[RunRow]] = defaultdict(list)
    for run in rows.runs:
        runs_by_question[run.question_id].append(run)
    question_views = tuple(
        _question_view(
            question,
            runs=tuple(
                sorted(
                    runs_by_question[question.question_id],
                    key=lambda item: item.run_number,
                )
            ),
            rows=rows,
        )
        for question in sorted(
            questions_by_id.values(), key=lambda item: item.conversation_sequence
        )
    )
    return LineageView(root_kind=root_kind, root_id=root_id, questions=question_views)


def _question_view(
    question: QuestionRow, *, runs: tuple[RunRow, ...], rows: LineageRows
) -> QuestionView:
    return QuestionView(
        question_id=question.question_id,
        conversation_id=question.conversation_id,
        text=question.original_question,
        runs=tuple(_run_view(run, rows=rows) for run in runs),
    )


def _run_view(run: RunRow, *, rows: LineageRows) -> RunView:
    run_result = _one_by_run(rows.run_results).get(run.run_id)
    steps = _run_steps(run, rows=rows)
    return RunView(
        run_id=run.run_id,
        run_number=run.run_number,
        trigger_kind=run.trigger_kind.value,
        result_kind=run_result.result_kind.value if run_result else "unknown",
        activated_memory_ids=_activated_memory_ids(run, rows=rows),
        requested_facts=_run_requested_facts(run, rows=rows, steps=steps),
        answers=_run_answers(run, rows=rows),
        source_reads=_run_source_reads(run, rows=rows),
        steps=steps,
        runtime_errors=_run_runtime_errors(run, rows=rows, steps=steps),
        memory_artifacts=_run_memory_artifacts(run, rows=rows),
        trigger_clarification_response_run_id=(
            run.trigger_clarification_response_run_id
        ),
        trigger_clarification_response_id=run.trigger_clarification_response_id,
        clarification_requests=_run_clarification_requests(run, rows=rows),
        clarification_responses=_run_clarification_responses(run, rows=rows),
    )


def _run_steps(run: RunRow, *, rows: LineageRows) -> tuple[StepView, ...]:
    return tuple(
        _step_view(item)
        for item in sorted(
            (item for item in rows.steps if item.run_id == run.run_id),
            key=lambda item: item.sequence,
        )
    )


def _run_requested_facts(
    run: RunRow, *, rows: LineageRows, steps: tuple[StepView, ...]
) -> tuple[RequestedFactView, ...]:
    fact_results_by_fact = _fact_results_by_fact(run, rows=rows)
    outputs_by_fact_result = _answer_outputs_by_fact_result(run, rows=rows)
    return tuple(
        _requested_fact_view(
            fact,
            fact_results=tuple(fact_results_by_fact.get(fact.requested_fact_id, ())),
            outputs=_outputs_for_fact(
                fact,
                fact_results_by_fact=fact_results_by_fact,
                outputs_by_fact_result=outputs_by_fact_result,
            ),
            rows=rows,
            steps=steps,
        )
        for fact in sorted(
            (fact for fact in rows.requested_facts if fact.run_id == run.run_id),
            key=lambda item: item.fact_key,
        )
    )


def _fact_results_by_fact(
    run: RunRow, *, rows: LineageRows
) -> dict[str, list[FactResultRow]]:
    fact_results_by_fact: dict[str, list[FactResultRow]] = defaultdict(list)
    for fact_result in rows.fact_results:
        if fact_result.run_id == run.run_id:
            fact_results_by_fact[fact_result.requested_fact_id].append(fact_result)
    return fact_results_by_fact


def _outputs_for_fact(
    fact: RequestedFactRow,
    *,
    fact_results_by_fact: dict[str, list[FactResultRow]],
    outputs_by_fact_result: dict[str, tuple[AnswerOutputRow, ...]],
) -> tuple[AnswerOutputRow, ...]:
    return tuple(
        output
        for fact_result in fact_results_by_fact.get(fact.requested_fact_id, ())
        for output in outputs_by_fact_result.get(fact_result.fact_result_id, ())
    )


def _run_answers(run: RunRow, *, rows: LineageRows) -> tuple[AnswerView, ...]:
    return tuple(
        _answer_view(answer, rows=rows)
        for answer in rows.answers
        if answer.run_id == run.run_id
    )


def _run_source_reads(run: RunRow, *, rows: LineageRows) -> tuple[SourceReadView, ...]:
    catalog_endpoints_by_id = _catalog_endpoints_by_id(rows)
    return tuple(
        _source_read_view(item, catalog_endpoints_by_id=catalog_endpoints_by_id)
        for item in sorted(
            (item for item in rows.source_reads if item.run_id == run.run_id),
            key=_source_read_sort_key,
        )
    )


def _run_runtime_errors(
    run: RunRow, *, rows: LineageRows, steps: tuple[StepView, ...]
) -> tuple[RuntimeErrorView, ...]:
    return tuple(
        _runtime_error_view(item, steps=steps)
        for item in rows.runtime_errors
        if item.run_id == run.run_id
    )


def _run_clarification_requests(
    run: RunRow, *, rows: LineageRows
) -> tuple[ClarificationRequestView, ...]:
    return tuple(
        _clarification_request_view(item, rows=rows)
        for item in rows.clarification_requests
        if item.run_id == run.run_id
    )


def _run_clarification_responses(
    run: RunRow, *, rows: LineageRows
) -> tuple[ClarificationResponseView, ...]:
    return tuple(
        _clarification_response_view(item)
        for item in rows.clarification_responses
        if item.run_id == run.run_id
    )


def _requested_fact_view(
    fact: RequestedFactRow,
    *,
    fact_results: tuple[FactResultRow, ...],
    outputs: tuple[AnswerOutputRow, ...],
    rows: LineageRows,
    steps: tuple[StepView, ...],
) -> RequestedFactView:
    proof_rows_by_fact_result = _proof_rows_by_fact_result(rows)
    return RequestedFactView(
        requested_fact_id=fact.requested_fact_id,
        produced_by_step_id=fact.produced_by_step_id,
        fact_key=fact.fact_key,
        description=fact.description or fact.fact_key,
        steps=_steps_for_fact(fact, steps=steps),
        fact_results=tuple(
            _fact_result_view(item, rows=rows, steps=steps) for item in fact_results
        ),
        answer_outputs=tuple(
            _answer_output_view(
                item,
                proof_row=proof_rows_by_fact_result.get(item.fact_result_id),
                rows=rows,
            )
            for item in outputs
        ),
        memory_artifacts=_memory_artifacts_for_requested_fact(fact, rows=rows),
    )


def _fact_result_view(
    fact_result: FactResultRow, *, rows: LineageRows, steps: tuple[StepView, ...]
) -> FactResultView:
    proof_row = next(
        (
            item
            for item in rows.proof_graphs
            if item.run_id == fact_result.run_id
            and item.fact_result_id == fact_result.fact_result_id
        ),
        None,
    )
    return FactResultView(
        fact_result_id=fact_result.fact_result_id,
        produced_by_step_id=fact_result.produced_by_step_id,
        result_kind=fact_result.result_kind.value,
        steps=_steps_for_fact_result(fact_result, proof_row=proof_row, steps=steps),
        proof=_proof_view(
            proof_row,
            source_reads=rows.source_reads,
            catalog_endpoints_by_id=_catalog_endpoints_by_id(rows),
        )
        if proof_row
        else None,
        memory_artifacts=_memory_artifacts_for_fact_result(fact_result, rows=rows),
    )


def _memory_artifacts_for_fact_result(
    fact_result: FactResultRow, *, rows: LineageRows
) -> tuple[MemoryArtifactView, ...]:
    return tuple(
        _memory_artifact_view(item)
        for item in rows.memory_artifacts
        if item.run_id == fact_result.run_id
        and item.fact_result_id == fact_result.fact_result_id
    )


def _memory_artifacts_for_requested_fact(
    fact: RequestedFactRow, *, rows: LineageRows
) -> tuple[MemoryArtifactView, ...]:
    return tuple(
        _memory_artifact_view(item)
        for item in rows.memory_artifacts
        if item.run_id == fact.run_id
        and item.requested_fact_id == fact.requested_fact_id
    )


def _run_memory_artifacts(
    run: RunRow, *, rows: LineageRows
) -> tuple[MemoryArtifactView, ...]:
    return tuple(
        _memory_artifact_view(item)
        for item in rows.memory_artifacts
        if item.run_id == run.run_id
        and item.requested_fact_id is None
        and item.fact_result_id is None
    )


def _activated_memory_ids(run: RunRow, *, rows: LineageRows) -> tuple[str, ...]:
    memory_ids: list[str] = []
    for artifact in rows.memory_artifacts:
        if artifact.run_id != run.run_id:
            continue
        provenance = artifact.payload_json.get("provenance")
        if not isinstance(provenance, dict):
            continue
        activation = provenance.get("conversation_resolution_activation")
        if not isinstance(activation, dict):
            continue
        for memory_id in activation.get("activated_memory_ids") or ():
            text = str(memory_id or "").strip()
            if text:
                memory_ids.append(text)
    return tuple(dict.fromkeys(memory_ids))


def _memory_artifact_view(artifact: MemoryArtifactRow) -> MemoryArtifactView:
    payload = artifact.payload_json
    return MemoryArtifactView(
        memory_artifact_id=artifact.memory_artifact_id,
        source_kind=artifact.source_kind.value,
        payload_schema=artifact.payload_schema,
        payload_schema_rev=artifact.payload_schema_rev,
        outcome=str(payload.get("outcome") or ""),
        source_question=str(payload.get("sourceQuestion") or ""),
        source_answer=str(payload.get("sourceAnswer") or ""),
        address_summaries=_memory_address_summaries(payload),
    )


def _memory_address_summaries(payload: dict[str, object]) -> tuple[str, ...]:
    summaries: list[str] = []
    for address in payload.get("addresses") or ():
        if not isinstance(address, dict):
            continue
        address_id = str(address.get("address") or "").strip()
        kind = str(address.get("kind") or "").strip()
        if address_id and kind:
            summaries.append(f"{kind}:{address_id}")
        elif address_id:
            summaries.append(address_id)
    return tuple(summaries)


def _answer_view(answer: AnswerRow, *, rows: LineageRows) -> AnswerView:
    return AnswerView(
        answer_id=answer.answer_id,
        outputs=tuple(
            _answer_output_view(output)
            for output in rows.answer_outputs
            if output.run_id == answer.run_id and output.answer_id == answer.answer_id
        ),
        presentations=tuple(
            _answer_presentation_view(presentation)
            for presentation in rows.answer_presentations
            if presentation.run_id == answer.run_id
            and presentation.answer_id == answer.answer_id
        ),
    )


def _answer_output_view(
    output: AnswerOutputRow,
    *,
    proof_row: ProofGraphRow | None = None,
    rows: LineageRows | None = None,
) -> AnswerOutputView:
    return AnswerOutputView(
        fact_result_id=output.fact_result_id,
        output_key=output.output_key,
        value_kind=output.value_kind.value,
        value=_format_value(output.value_json),
        value_json=output.value_json,
        proof_node_refs=output.proof_node_refs_json,
        proof=(
            _proof_view(
                proof_row,
                source_reads=rows.source_reads,
                catalog_endpoints_by_id=_catalog_endpoints_by_id(rows),
                target_node_refs=output.proof_node_refs_json,
            )
            if proof_row is not None and rows is not None
            else None
        ),
    )


def _answer_presentation_view(
    presentation: AnswerPresentationRow,
) -> AnswerPresentationView:
    return AnswerPresentationView(
        presentation_id=presentation.presentation_id,
        client_key=presentation.client_key.value,
        locale=presentation.locale,
        presentation_kind=presentation.presentation_kind.value,
        render_step_id=presentation.render_step_id,
        value=_presentation_value(presentation),
    )


def _presentation_value(presentation: AnswerPresentationRow) -> str:
    if presentation.rendered_value:
        return presentation.rendered_value
    if presentation.payload_json is None:
        return ""
    return str(presentation.payload_json.get("summary") or presentation.payload_json)


def _source_read_view(
    source_read: SourceReadRow,
    *,
    catalog_endpoints_by_id: dict[str, CatalogEndpointRow],
) -> SourceReadView:
    return SourceReadView(
        source_read_id=source_read.source_read_id,
        step_id=source_read.step_id,
        catalog_endpoint=_catalog_endpoint_view(
            _required_catalog_endpoint(
                source_read,
                catalog_endpoints_by_id=catalog_endpoints_by_id,
            )
        ),
        args=source_read.args_json,
        row_count=source_read.row_count,
        response_hash=source_read.response_hash,
        status=source_read.status.value,
        completeness=source_read.completeness_json,
        artifact_id=source_read.artifact_id,
        error=source_read.error_json or None,
    )


def _required_catalog_endpoint(
    source_read: SourceReadRow,
    *,
    catalog_endpoints_by_id: dict[tuple[str, str], CatalogEndpointRow],
) -> CatalogEndpointRow:
    try:
        return catalog_endpoints_by_id[
            (source_read.run_id, source_read.catalog_endpoint_id)
        ]
    except KeyError as exc:
        raise ValueError(
            f"source read {source_read.source_read_id!r} references missing catalog endpoint "
            f"{source_read.catalog_endpoint_id!r}"
        ) from exc


def _catalog_endpoint_view(endpoint: CatalogEndpointRow) -> CatalogEndpointView:
    return CatalogEndpointView(
        catalog_endpoint_id=endpoint.catalog_endpoint_id,
        catalog_endpoint_key=endpoint.catalog_endpoint_key,
        endpoint_name=endpoint.endpoint_name,
        framework_kind=endpoint.framework_kind,
        source_namespace_kind=endpoint.source_namespace_kind,
        source_namespace_path=endpoint.source_namespace_path_json,
        route_method=endpoint.route_method,
        route_path_template=endpoint.route_path_template,
        route_name=endpoint.route_name,
        api_schema_operation_id=endpoint.api_schema_operation_id,
        handler_ref=endpoint.handler_ref,
        domain_resource_names=endpoint.domain_resource_names_json,
    )


def _catalog_endpoints_by_id(
    rows: LineageRows,
) -> dict[tuple[str, str], CatalogEndpointRow]:
    return {
        (item.run_id, item.catalog_endpoint_id): item for item in rows.catalog_endpoints
    }


def _runtime_error_view(
    error: RuntimeErrorRow, *, steps: tuple[StepView, ...]
) -> RuntimeErrorView:
    failed_step_key = next(
        (
            step.step_key
            for step in steps
            if error.failed_step_id is not None and step.step_id == error.failed_step_id
        ),
        None,
    )
    return RuntimeErrorView(
        runtime_error_detail_id=error.runtime_error_detail_id,
        error_kind=error.error_kind.value,
        message=error.message,
        failed_step_id=error.failed_step_id,
        failed_step_key=failed_step_key,
    )


def _clarification_request_view(
    clarification: ClarificationRequestRow,
    *,
    rows: LineageRows,
) -> ClarificationRequestView:
    return ClarificationRequestView(
        clarification_id=clarification.clarification_id,
        basis=clarification.basis.value,
        question_text=clarification.question_text,
        requested_fact_id=_clarification_requested_fact_id(clarification, rows=rows),
        known_input_id=_known_input_id_from_refs(clarification.evidence_refs_json),
        fact_result_id=clarification.fact_result_id,
        step_id=clarification.step_id,
        options=clarification.options_json,
        evidence_refs=clarification.evidence_refs_json,
    )


def _clarification_requested_fact_id(
    clarification: ClarificationRequestRow,
    *,
    rows: LineageRows,
) -> str:
    if not clarification.fact_result_id:
        return ""
    fact_result = next(
        (
            item
            for item in rows.fact_results
            if item.run_id == clarification.run_id
            and item.fact_result_id == clarification.fact_result_id
        ),
        None,
    )
    return fact_result.requested_fact_id if fact_result is not None else ""


def _known_input_id_from_refs(refs: tuple[str, ...]) -> str:
    prefix = "known_input:"
    return next(
        (ref.removeprefix(prefix) for ref in refs if ref.startswith(prefix)),
        "",
    )


def _clarification_response_view(
    response: ClarificationResponseRow,
) -> ClarificationResponseView:
    return ClarificationResponseView(
        response_id=response.response_id,
        clarification_id=response.clarification_id,
        evidence_ref=response.evidence_ref,
        source_message_ref=response.source_message_ref,
        selected_option_id=response.selected_option_id,
        response_text=response.response_text,
    )


def _source_read_sort_key(
    source_read: SourceReadRow,
) -> tuple[str, tuple[tuple[str, str], ...]]:
    return (
        source_read.catalog_endpoint_id,
        tuple(
            sorted(
                (str(key), str(value)) for key, value in source_read.args_json.items()
            )
        ),
    )


def _step_view(step: StepRow) -> StepView:
    return StepView(
        step_id=step.step_id,
        step_key=step.step_key.value,
        sequence=step.sequence,
        fact_refs=_fact_references(step.output_summary_json),
        decisions=step_decision_views(step),
        semantic=step_semantic_view(step),
        error=step.error_json or None,
    )


def _steps_for_fact(
    fact: RequestedFactRow, *, steps: tuple[StepView, ...]
) -> tuple[StepView, ...]:
    return _dedupe_steps(
        step
        for step in steps
        if step.step_id == fact.produced_by_step_id or _step_references_fact(step, fact)
    )


def _steps_for_fact_result(
    fact_result: FactResultRow,
    *,
    proof_row: ProofGraphRow | None,
    steps: tuple[StepView, ...],
) -> tuple[StepView, ...]:
    step_ids = {fact_result.produced_by_step_id}
    if proof_row is not None:
        step_ids.add(proof_row.compile_step_id)
        if proof_row.execute_step_id:
            step_ids.add(proof_row.execute_step_id)
    return _dedupe_steps(step for step in steps if step.step_id in step_ids)


def _step_references_fact(step: StepView, fact: RequestedFactRow) -> bool:
    return fact.requested_fact_id in step.fact_refs or fact.fact_key in step.fact_refs


def _dedupe_steps(steps: Iterable[StepView]) -> tuple[StepView, ...]:
    output: list[StepView] = []
    seen: set[str] = set()
    for step in steps:
        if step.step_id in seen:
            continue
        seen.add(step.step_id)
        output.append(step)
    return tuple(sorted(output, key=lambda item: item.sequence))


def _fact_references(output: dict[str, object]) -> tuple[str, ...]:
    references: list[str] = []
    for key in ("requested_fact_id", "fact_key"):
        value = output.get(key)
        if isinstance(value, str) and value:
            references.append(value)
    for key in ("requested_fact_ids", "fact_keys"):
        value = output.get(key)
        if isinstance(value, list):
            references.extend(str(item) for item in value if item)
    return tuple(dict.fromkeys(references))


def _proof_view(
    proof_row: ProofGraphRow,
    *,
    source_reads: tuple[SourceReadRow, ...],
    catalog_endpoints_by_id: dict[str, CatalogEndpointRow],
    target_node_refs: tuple[str, ...] = (),
) -> ExecutionProofView:
    payload = read_execution_proof_graph_payload(
        payload_schema=proof_row.payload_schema,
        payload_schema_rev=proof_row.payload_schema_rev,
        payload_json=proof_row.payload_json,
    )
    payload = project_proof_payload(payload, target_node_ids=target_node_refs)
    proof_source_reads = _proof_source_reads(
        payload,
        proof_row=proof_row,
        source_reads=source_reads,
        catalog_endpoints_by_id=catalog_endpoints_by_id,
    )
    return ExecutionProofView(
        proof_graph_id=proof_row.proof_graph_id,
        evidence_handles=tuple(node.id for node in payload.nodes),
        endpoint_args=_endpoint_args(payload, source_reads=proof_source_reads),
        computation_links=tuple(
            ProofComputationLinkView(
                source=edge.source,
                target=edge.target,
                role=edge.role.value,
            )
            for edge in payload.edges
        ),
        computation_summaries=proof_computation_summaries(payload.edges),
        debug_evidence_handles=proof_evidence_handle_labels(payload),
        debug_computation_links=proof_computation_link_labels(payload.edges),
        contributions=_contributions(payload),
        applied_inputs=_applied_inputs(payload),
        source_reads=proof_source_reads,
    )


def _endpoint_args(
    payload, *, source_reads: tuple[SourceReadView, ...]
) -> tuple[ProofEndpointArgView, ...]:
    return tuple(
        ProofEndpointArgView(
            handle=item.handle,
            arg_name=item.arg_name,
            values=item.values,
        )
        for item in proof_endpoint_args(payload, source_reads=source_reads)
    )


def _contributions(payload) -> tuple[ContributionView, ...]:
    return tuple(
        ContributionView(
            origin=item.origin,
            label=item.label,
            node_refs=item.node_refs,
            proof_refs=item.proof_refs,
        )
        for item in proof_contributions(payload)
    )


def _applied_inputs(payload) -> tuple[ProofAppliedInputView, ...]:
    return tuple(
        ProofAppliedInputView(
            handle=item.handle,
            label=item.label,
            action=item.action,
            proof_refs=item.proof_refs,
        )
        for item in proof_applied_inputs(payload)
    )


def _proof_source_reads(
    payload,
    *,
    proof_row: ProofGraphRow,
    source_reads: tuple[SourceReadRow, ...],
    catalog_endpoints_by_id: dict[str, CatalogEndpointRow],
) -> tuple[SourceReadView, ...]:
    source_read_ids = proof_source_read_ids(payload)
    return tuple(
        _source_read_view(
            source_read,
            catalog_endpoints_by_id=catalog_endpoints_by_id,
        )
        for source_read in source_reads
        if source_read.run_id == proof_row.run_id
        and source_read.source_read_id in source_read_ids
    )


def _answer_outputs_by_fact_result(
    run: RunRow, *, rows: LineageRows
) -> dict[str, list[AnswerOutputRow]]:
    grouped: dict[str, list[AnswerOutputRow]] = defaultdict(list)
    for output in rows.answer_outputs:
        if output.run_id != run.run_id:
            continue
        grouped[output.fact_result_id].append(output)
    return grouped


def _proof_rows_by_fact_result(rows: LineageRows) -> dict[str, ProofGraphRow]:
    return {item.fact_result_id: item for item in rows.proof_graphs}


def _one_by_run(items: tuple[RunResultRow, ...]) -> dict[str, RunResultRow]:
    return {item.run_id: item for item in items}


def _format_value(value: dict[str, object]) -> str:
    entity_type = str(value.get("entity_type") or "")
    entity_id = str(value.get("entity_id") or "")
    if entity_type and entity_id:
        return f"{entity_type}:{entity_id}"
    if "value" in value:
        return str(value["value"])
    return str(value)
