"""Agent-readable lineage explain projection."""

from __future__ import annotations

from fervis.interfaces.agent.actions import provide_clarification_action
from fervis.lineage.views.detail import (
    LineageRenderDetail,
    include_step_decision,
)
from fervis.lineage.enums import ContributionOrigin
from fervis.lineage.views.json_payload import view_json
from fervis.lineage.views.model import (
    AnswerPresentationView,
    CatalogEndpointView,
    ClarificationRequestView,
    ClarificationResponseView,
    ExecutionProofView,
    InputLineageResultView,
    InputLineageView,
    LineageTimelineView,
    MemoryArtifactView,
    ModelCallInspectionView,
    ObservabilityNoticeView,
    RuntimeErrorView,
    SourceReadView,
    TimelineAnswerOutputView,
    TimelineFactResultView,
    TimelineRequestedFactView,
    TimelineRunView,
    TimelineStepView,
)


def agent_lineage_view(
    view: LineageTimelineView, *, detail: LineageRenderDetail
) -> dict[str, object]:
    """Return the structured default view for agents.

    This is not a raw dataclass dump. It follows the same additive detail
    contract as the human renderer: compact shows the answer path, verbose adds
    model/source/proof summaries, and debug adds low-level handles.
    """

    questions = tuple(
        _question_json(question, detail=detail) for question in view.questions
    )
    return {
        "view_kind": "lineage",
        "root": {"kind": view.root_kind.value, "id": view.root_id},
        "summary": _summary(view),
        "index": _index(view, detail=detail),
        "questions": questions,
        "observability_notices": tuple(
            _notice_json(notice) for notice in view.observability_notices
        ),
    }


def agent_input_lineage_view(
    view: InputLineageView, *, detail: LineageRenderDetail
) -> dict[str, object]:
    return {
        "root": {"kind": view.root_kind.value, "id": view.root_id},
        "results": tuple(
            _input_lineage_result_json(result, detail=detail) for result in view.results
        ),
    }


def _input_lineage_result_json(
    result: InputLineageResultView, *, detail: LineageRenderDetail
) -> dict[str, object]:
    data: dict[str, object] = {
        "fact_result_id": result.fact_result_id,
        "requested_fact_id": result.requested_fact_id,
        "fact_description": result.fact_description,
        "explicit": result.explicit,
        "derived": result.derived,
        "contextual": result.contextual,
        "applied": result.applied,
    }
    if detail.includes_verbose():
        data["evidence_refs"] = result.evidence_refs
    if detail.includes_debug():
        data["proof_handles"] = result.proof_handles
    return _clean(data)


def _summary(view: LineageTimelineView) -> dict[str, int]:
    runs = tuple(run for question in view.questions for run in question.runs)
    steps = tuple(step for run in runs for step in run.steps)
    return {
        "question_count": len(view.questions),
        "run_count": len(runs),
        "step_count": len(steps),
        "model_call_count": sum(len(step.model_calls) for step in steps),
        "source_read_count": sum(len(step.source_reads) for step in steps),
        "answer_output_count": sum(len(step.answer_outputs) for step in steps),
        "runtime_error_count": sum(len(step.runtime_errors) for step in steps),
    }


def _index(
    view: LineageTimelineView, *, detail: LineageRenderDetail
) -> dict[str, object]:
    answer_outputs: list[dict[str, object]] = []
    source_reads: list[dict[str, object]] = []
    runtime_errors: list[dict[str, object]] = []
    model_calls: list[dict[str, object]] = []
    for question in view.questions:
        for run in question.runs:
            for step in run.steps:
                path = {
                    "question_id": question.question_id,
                    "run_id": run.run_id,
                    "step_key": step.step_key,
                    "step_id": step.step_id,
                }
                answer_outputs.extend(
                    {**path, **_answer_output_index_json(output)}
                    for output in step.answer_outputs
                )
                source_reads.extend(
                    {**path, **_source_read_index_json(source_read)}
                    for source_read in step.source_reads
                )
                runtime_errors.extend(
                    {**path, **_runtime_error_json(error)}
                    for error in step.runtime_errors
                )
                if detail.includes_verbose():
                    model_calls.extend(
                        {**path, **_model_call_json(call)} for call in step.model_calls
                    )
    return _clean(
        {
            "answer_outputs": tuple(answer_outputs),
            "source_reads": tuple(source_reads),
            "runtime_errors": tuple(runtime_errors),
            "model_calls": tuple(model_calls),
        }
    )


def _answer_output_index_json(output: TimelineAnswerOutputView) -> dict[str, object]:
    return {
        "output_key": output.output_key,
        "fact_result_id": output.fact_result_id,
        "value_kind": output.value_kind,
        "value": output.value,
    }


def _source_read_index_json(source_read: SourceReadView) -> dict[str, object]:
    return {
        "source_read_id": source_read.source_read_id,
        "endpoint": source_read.catalog_endpoint.label,
        "status": source_read.status,
        "row_count": source_read.row_count,
        "args": _display_args(source_read.args),
        "response_hash": source_read.response_hash,
    }


def _question_json(question, *, detail: LineageRenderDetail) -> dict[str, object]:
    return {
        "question_id": question.question_id,
        "conversation_id": question.conversation_id,
        "text": question.text,
        "runs": tuple(
            _run_json(
                run,
                detail=detail,
                conversation_id=question.conversation_id,
                question_id=question.question_id,
            )
            for run in question.runs
        ),
    }


def _run_json(
    run: TimelineRunView,
    *,
    detail: LineageRenderDetail,
    conversation_id: str,
    question_id: str,
) -> dict[str, object]:
    data: dict[str, object] = _clean(
        {
            "run_id": run.run_id,
            "run_number": run.run_number,
            "kind": run.kind,
            "trigger_kind": run.trigger_kind,
            "result_kind": run.result_kind,
            "base_run_id": run.base_run_id,
            "program_derivation": view_json(run.program_derivation),
            "activated_memory_ids": run.activated_memory_ids,
            "memory_artifacts": tuple(
                _memory_artifact_json(artifact) for artifact in run.memory_artifacts
            ),
            "clarification_responses": tuple(
                _clarification_response_json(response)
                for response in run.clarification_responses
            ),
            "steps": tuple(
                _step_json(
                    step,
                    detail=detail,
                    conversation_id=conversation_id,
                    question_id=question_id,
                    run_id=run.run_id,
                )
                for step in run.steps
            ),
        }
    )
    return data


def _step_json(
    step: TimelineStepView,
    *,
    detail: LineageRenderDetail,
    conversation_id: str,
    question_id: str,
    run_id: str,
) -> dict[str, object]:
    data: dict[str, object] = _clean(
        {
            "step_id": step.step_id,
            "step_key": step.step_key,
            "sequence": step.sequence,
            "runtime_errors": tuple(
                _runtime_error_json(error) for error in step.runtime_errors
            ),
            "clarifications": tuple(
                _clarification_json(
                    clarification,
                    conversation_id=conversation_id,
                    question_id=question_id,
                    run_id=run_id,
                )
                for clarification in step.clarifications
            ),
            "decisions": tuple(
                _decision_json(decision)
                for decision in step.decisions
                if include_step_decision(decision.detail, detail)
            ),
            "semantic": view_json(step.semantic),
            "requested_facts": tuple(
                _requested_fact_json(fact) for fact in step.requested_facts
            ),
            "fact_results": tuple(
                _fact_result_json(result) for result in step.fact_results
            ),
            "source_reads": tuple(
                _source_read_json(source_read, detail=detail)
                for source_read in step.source_reads
            ),
            "answer_outputs": tuple(
                _answer_output_json(output, detail=detail)
                for output in step.answer_outputs
            ),
            "answer_presentations": tuple(
                _presentation_json(presentation)
                for presentation in step.answer_presentations
            ),
        }
    )
    if detail.includes_verbose():
        data["model_call_ids"] = tuple(call.model_call_id for call in step.model_calls)
    return data


def _decision_json(decision) -> dict[str, object]:
    return _clean(
        {
            "detail": decision.detail.value,
            "is_explanation": decision.is_explanation,
            "lines": decision.lines,
            "items": tuple(
                _clean(
                    {
                        "text": item.text,
                        "is_explanation": item.is_explanation,
                        "path": item.path,
                        "subject": item.subject,
                        "disposition": item.disposition,
                        "basis": item.basis,
                    }
                )
                for item in decision.items
            ),
        }
    )


def _requested_fact_json(fact: TimelineRequestedFactView) -> dict[str, object]:
    return {
        "requested_fact_id": fact.requested_fact_id,
        "fact_key": fact.fact_key,
        "description": fact.description,
    }


def _fact_result_json(result: TimelineFactResultView) -> dict[str, object]:
    return {
        "fact_result_id": result.fact_result_id,
        "requested_fact_id": result.requested_fact_id,
        "result_kind": result.result_kind,
        "memory_artifacts": tuple(
            _memory_artifact_json(artifact) for artifact in result.memory_artifacts
        ),
    }


def _answer_output_json(
    output: TimelineAnswerOutputView, *, detail: LineageRenderDetail
) -> dict[str, object]:
    data: dict[str, object] = {
        "fact_result_id": output.fact_result_id,
        "output_key": output.output_key,
        "value_kind": output.value_kind,
        "value": output.value,
        "value_json": output.value_json,
    }
    if output.proof is not None:
        data["proof"] = _proof_json(output.proof, detail=detail)
    if detail.includes_debug():
        data["proof_node_refs"] = output.proof_node_refs
    return data


def _proof_json(
    proof: ExecutionProofView, *, detail: LineageRenderDetail
) -> dict[str, object]:
    data: dict[str, object] = {
        "proof_graph_id": proof.proof_graph_id,
        "inputs": _proof_inputs_json(proof),
    }
    if detail.includes_verbose():
        data.update(
            {
                "endpoint_args": tuple(
                    _proof_endpoint_arg_json(arg, detail=detail)
                    for arg in proof.endpoint_args
                ),
                "computation_summaries": proof.computation_summaries,
                "source_read_ids": tuple(
                    source_read.source_read_id for source_read in proof.source_reads
                ),
            }
        )
    if detail.includes_debug():
        data.update(
            {
                "applied_inputs": tuple(
                    _clean(
                        {
                            "handle": item.handle,
                            "label": item.label,
                            "action": item.action,
                            "proof_refs": item.proof_refs,
                        }
                    )
                    for item in proof.applied_inputs
                ),
                "contributions": tuple(
                    _clean(
                        {
                            "origin": contribution.origin.value,
                            "label": contribution.label,
                            "node_refs": contribution.node_refs,
                            "proof_refs": contribution.proof_refs,
                        }
                    )
                    for contribution in proof.contributions
                ),
                "evidence_handles": proof.evidence_handles,
                "debug_evidence_handles": proof.debug_evidence_handles,
                "computation_links": tuple(
                    {
                        "source": link.source,
                        "target": link.target,
                        "role": link.role,
                    }
                    for link in proof.computation_links
                ),
                "debug_computation_links": proof.debug_computation_links,
            }
        )
    return data


def _proof_endpoint_arg_json(arg, *, detail: LineageRenderDetail) -> dict[str, object]:
    data: dict[str, object] = {
        "arg_name": arg.arg_name,
        "values": arg.values,
    }
    if detail.includes_debug():
        data["handle"] = arg.handle
    return data


def _proof_inputs_json(proof: ExecutionProofView) -> dict[str, tuple[str, ...]]:
    inputs = {
        "explicit": _proof_contribution_labels(proof, ContributionOrigin.EXPLICIT),
        "derived": _proof_contribution_labels(proof, ContributionOrigin.DERIVED),
        "contextual": _proof_contribution_labels(proof, ContributionOrigin.CONTEXTUAL),
    }
    return {key: value for key, value in inputs.items() if value}


def _proof_contribution_labels(
    proof: ExecutionProofView, origin: ContributionOrigin
) -> tuple[str, ...]:
    return tuple(
        _dedupe_keep_order(
            contribution.label
            for contribution in proof.contributions
            if contribution.origin is origin
        )
    )


def _presentation_json(presentation: AnswerPresentationView) -> dict[str, object]:
    return {
        "presentation_id": presentation.presentation_id,
        "client_key": presentation.client_key,
        "locale": presentation.locale,
        "presentation_kind": presentation.presentation_kind,
        "render_step_id": presentation.render_step_id,
        "value": presentation.value,
    }


def _source_read_json(
    source_read: SourceReadView, *, detail: LineageRenderDetail
) -> dict[str, object]:
    endpoint = source_read.catalog_endpoint
    data: dict[str, object] = {
        "source_read_id": source_read.source_read_id,
        "step_id": source_read.step_id,
        "endpoint": endpoint.label,
        "endpoint_name": source_read.endpoint_name,
        "status": source_read.status,
        "args": _display_args(source_read.args),
        "row_count": source_read.row_count,
        "response_hash": source_read.response_hash,
    }
    if detail.includes_verbose():
        data.update(
            {
                "catalog_endpoint": _catalog_endpoint_json(endpoint),
                "completeness": source_read.completeness,
                "artifact_id": source_read.artifact_id,
                "error": source_read.error,
            }
        )
    return _clean(data)


def _catalog_endpoint_json(endpoint: CatalogEndpointView) -> dict[str, object]:
    return _clean(
        {
            "catalog_endpoint_id": endpoint.catalog_endpoint_id,
            "catalog_endpoint_key": endpoint.catalog_endpoint_key,
            "endpoint_name": endpoint.endpoint_name,
            "label": endpoint.label,
            "framework_kind": endpoint.framework_kind,
            "source_namespace_kind": endpoint.source_namespace_kind,
            "source_namespace_path": endpoint.source_namespace_path,
            "route_method": endpoint.route_method,
            "route_path_template": endpoint.route_path_template,
            "route_name": endpoint.route_name,
            "api_schema_operation_id": endpoint.api_schema_operation_id,
            "handler_ref": endpoint.handler_ref,
            "domain_resource_names": endpoint.domain_resource_names,
        }
    )


def _model_call_json(call: ModelCallInspectionView) -> dict[str, object]:
    return {
        "model_call_id": call.model_call_id,
        "call_index": call.call_index,
        "provider": call.provider,
        "model_key": call.model_key,
        "status": call.status,
        "prompt_chars": call.prompt_chars,
        "schema_chars": call.schema_chars,
        "tool_spec_chars": call.tool_spec_chars,
        "duration_ms": call.duration_ms,
        "artifacts": tuple(
            {
                "artifact_kind": artifact.artifact_kind,
                "artifact_id": artifact.artifact_id,
                "size_bytes": artifact.size_bytes,
            }
            for artifact in call.artifacts
        ),
    }


def _runtime_error_json(error: RuntimeErrorView) -> dict[str, object]:
    return {
        "runtime_error_detail_id": error.runtime_error_detail_id,
        "error_kind": error.error_kind,
        "message": error.message,
        "failed_step_id": error.failed_step_id,
        "failed_step_key": error.failed_step_key,
    }


def _clarification_json(
    clarification: ClarificationRequestView,
    *,
    conversation_id: str,
    question_id: str,
    run_id: str,
) -> dict[str, object]:
    return {
        **clarification.payload_json,
        "next_actions": (
            provide_clarification_action(
                conversation_id,
                question_id=question_id,
                run_id=run_id,
                clarification_id=clarification.clarification_id,
            ),
        ),
    }


def _clarification_response_json(
    response: ClarificationResponseView,
) -> dict[str, object]:
    return {
        "response_id": response.response_id,
        "clarification_id": response.clarification_id,
        "evidence_ref": response.evidence_ref,
        "source_message_ref": response.source_message_ref,
        "selected_option_id": response.selected_option_id,
        "response_text": response.response_text,
    }


def _memory_artifact_json(artifact: MemoryArtifactView) -> dict[str, object]:
    return {
        "memory_artifact_id": artifact.memory_artifact_id,
        "source_kind": artifact.source_kind,
        "payload_schema": artifact.payload_schema,
        "payload_schema_rev": artifact.payload_schema_rev,
        "outcome": artifact.outcome,
        "source_question": artifact.source_question,
        "source_answer": artifact.source_answer,
        "address_summaries": artifact.address_summaries,
    }


def _notice_json(notice: ObservabilityNoticeView) -> dict[str, object]:
    return _clean(
        {
            "kind": notice.kind,
            "severity": notice.severity,
            "message": notice.message,
            "run_ids": notice.run_ids,
            "details": notice.details,
        }
    )


def _display_args(args: dict[str, object]) -> dict[str, object]:
    return {_arg_name(key): value for key, value in sorted(args.items())}


def _arg_name(key: str) -> str:
    return key.rsplit(".", 1)[-1]


def _dedupe_keep_order(values) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _clean(data: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in data.items() if not _is_empty(value)}


def _is_empty(value: object) -> bool:
    return value is None or value == "" or value == () or value == [] or value == {}
