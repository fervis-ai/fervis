"""Human-readable chronological lineage renderers."""

from __future__ import annotations

from fervis.lineage.enums import ContributionOrigin
from fervis.lineage.views.detail import (
    LineageRenderDetail,
    include_step_decision,
)
from fervis.lineage.views.model import (
    ExecutionProofView,
    LineageTimelineView,
    SourceReadView,
    TimelineAnswerOutputView,
    TimelineStepView,
)


def render_lineage(
    view: LineageTimelineView,
    *,
    answer_output: str | None = None,
    fact_filter: str | None = None,
    step: str | None = None,
    errors_only: bool = False,
    detail: LineageRenderDetail = LineageRenderDetail.COMPACT,
) -> str:
    lines: list[str] = []
    for question in view.questions:
        _append_question(
            lines,
            question,
            answer_output=answer_output,
            fact_filter=fact_filter,
            step=step,
            errors_only=errors_only,
            detail=detail,
        )
    _append_observability_notices(lines, view.observability_notices)
    return "\n".join(lines)


def _append_observability_notices(lines: list[str], notices) -> None:
    if not notices:
        return
    if lines:
        lines.append("")
    lines.append("Observability")
    for notice in notices:
        lines.append(f"- {notice.kind} ({notice.severity}): {notice.message}")
        if notice.run_ids:
            lines.append(f"  runs: {', '.join(notice.run_ids)}")


def _append_question(
    lines: list[str],
    question,
    *,
    answer_output: str | None,
    fact_filter: str | None,
    step: str | None,
    errors_only: bool,
    detail: LineageRenderDetail,
) -> None:
    question_start = len(lines)
    lines.append(f"Question {question.question_id}: {question.text}")
    for run in question.runs:
        if errors_only and run.result_kind != "runtime_error":
            continue
        _append_run(
            lines,
            run,
            answer_output=answer_output,
            fact_filter=fact_filter,
            step=step,
            detail=detail,
        )
    if errors_only and len(lines) == question_start + 1:
        lines.append("  No runtime errors.")


def _append_run(
    lines: list[str],
    run,
    *,
    answer_output: str | None,
    fact_filter: str | None,
    step: str | None,
    detail: LineageRenderDetail,
) -> None:
    lines.append(f"  Run {run.run_id} (#{run.run_number}): {run.result_kind}")
    _append_run_trigger(lines, run, indent=4)
    _append_clarification_responses(lines, run.clarification_responses, indent=4)
    _append_activated_memory(lines, run.activated_memory_ids, indent=4)
    _append_memory_artifacts(lines, run.memory_artifacts, indent=4)
    for timeline_step in run.steps:
        if step is not None and timeline_step.step_key != step:
            continue
        _append_step(
            lines,
            timeline_step,
            answer_output=answer_output,
            fact_filter=fact_filter,
            detail=detail,
        )


def _append_step(
    lines: list[str],
    step: TimelineStepView,
    *,
    answer_output: str | None,
    fact_filter: str | None,
    detail: LineageRenderDetail,
) -> None:
    lines.append(f"    Step {step.sequence}: {step.step_key}")
    _append_runtime_errors(lines, step.runtime_errors, indent=6)
    _append_clarifications(lines, step.clarifications, indent=6)
    _append_step_decisions(lines, step, indent=6, detail=detail)
    _append_requested_facts(lines, step, fact_filter=fact_filter, indent=6)
    _append_fact_results(lines, step, fact_filter=fact_filter, indent=6)
    _append_source_reads(
        lines,
        _visible_source_reads(step, answer_output=answer_output),
        indent=6,
        include_audit=detail.includes_verbose(),
    )
    _append_outputs(lines, step, answer_output=answer_output, indent=6, detail=detail)
    if answer_output is None and fact_filter is None:
        _append_presentations(lines, step, indent=6)
    if detail.includes_verbose():
        _append_model_calls(lines, step, indent=6)
    if detail.includes_debug():
        _append_debug_proofs(lines, step, answer_output=answer_output, indent=6)


def _append_step_decisions(
    lines: list[str],
    step: TimelineStepView,
    *,
    indent: int,
    detail: LineageRenderDetail,
) -> None:
    prefix = " " * indent
    for decision in step.decisions:
        if not include_step_decision(decision.detail, detail):
            continue
        for line in decision.lines:
            lines.append(f"{prefix}{line}")


def _append_requested_facts(
    lines: list[str], step: TimelineStepView, *, fact_filter: str | None, indent: int
) -> None:
    prefix = " " * indent
    for fact in step.requested_facts:
        if _fact_matches(fact, fact_filter=fact_filter):
            lines.append(
                f"{prefix}Requested fact {fact.requested_fact_id}: {fact.description}"
            )


def _append_fact_results(
    lines: list[str], step: TimelineStepView, *, fact_filter: str | None, indent: int
) -> None:
    prefix = " " * indent
    visible_fact_ids = {
        fact.requested_fact_id
        for fact in step.requested_facts
        if _fact_matches(fact, fact_filter=fact_filter)
    }
    for result in step.fact_results:
        if visible_fact_ids and result.requested_fact_id not in visible_fact_ids:
            continue
        if result.result_kind != "answered":
            lines.append(
                f"{prefix}Fact result {result.fact_result_id}: {result.result_kind}"
            )
        _append_memory_artifacts(lines, result.memory_artifacts, indent=indent)


def _append_outputs(
    lines: list[str],
    step: TimelineStepView,
    *,
    answer_output: str | None,
    indent: int,
    detail: LineageRenderDetail,
) -> None:
    prefix = " " * indent
    for output in step.answer_outputs:
        if not _output_matches(output, answer_output=answer_output):
            continue
        lines.append(
            f"{prefix}Answer output {output.output_key}: {output.value_kind} {output.value}"
        )
        if output.proof is not None:
            _append_contributions(lines, output.proof, indent=indent)
            if detail.includes_verbose():
                lines.append(f"{prefix}Proof record: {output.proof.proof_graph_id}")
                _append_proof_summary(lines, output.proof, indent=indent + 2)


def _append_presentations(
    lines: list[str], step: TimelineStepView, *, indent: int
) -> None:
    prefix = " " * indent
    for presentation in step.answer_presentations:
        if presentation.value:
            lines.append(
                f"{prefix}Answer presentation "
                f"({presentation.client_key}/{presentation.presentation_kind}): "
                f"{presentation.value}"
            )


def _append_model_calls(
    lines: list[str], step: TimelineStepView, *, indent: int
) -> None:
    if not step.model_calls:
        return
    prefix = " " * indent
    lines.append(f"{prefix}Model calls:")
    for call in step.model_calls:
        lines.append(
            f"{prefix}  - {call.step_key}#{call.call_index}: "
            f"{call.provider}/{call.model_key} {call.status}"
        )
        lines.append(
            f"{prefix}    chars: prompt={call.prompt_chars}, "
            f"schema={call.schema_chars}, tool_spec={call.tool_spec_chars}"
        )
        for artifact in call.artifacts:
            lines.append(
                f"{prefix}    {artifact.artifact_kind}: "
                f"{artifact.artifact_id} size={artifact.size_bytes}"
            )


def _append_runtime_errors(lines: list[str], runtime_errors, *, indent: int) -> None:
    prefix = " " * indent
    for error in runtime_errors:
        lines.append(f"{prefix}{error.error_kind}: {error.message}")
        if error.failed_step_key:
            lines.append(f"{prefix}failed step: {error.failed_step_key}")
        elif error.failed_step_id:
            lines.append(f"{prefix}failed step id: {error.failed_step_id}")


def _append_clarifications(lines: list[str], clarifications, *, indent: int) -> None:
    prefix = " " * indent
    for clarification in clarifications:
        lines.append(
            f"{prefix}Clarification request {clarification.clarification_id}: "
            f"{clarification.question_text}"
        )
        for option in clarification.options:
            lines.append(f"{prefix}  option: {_clarification_option(option)}")


def _append_run_trigger(lines: list[str], run, *, indent: int) -> None:
    if not run.trigger_clarification_response_id:
        return
    source_run = run.trigger_clarification_response_run_id or "unknown run"
    lines.append(
        f"{' ' * indent}Triggered by clarification response "
        f"{run.trigger_clarification_response_id} from run {source_run}"
    )


def _append_clarification_responses(
    lines: list[str], responses, *, indent: int
) -> None:
    prefix = " " * indent
    for response in responses:
        selected = response.selected_option_id or response.response_text
        if selected:
            lines.append(
                f"{prefix}Clarification response {response.response_id}: "
                f"selected {selected}"
            )
        else:
            lines.append(f"{prefix}Clarification response {response.response_id}")


def _append_activated_memory(
    lines: list[str], activated_memory_ids: tuple[str, ...], *, indent: int
) -> None:
    if activated_memory_ids:
        lines.append(
            f"{' ' * indent}Activated memory: {', '.join(activated_memory_ids)}"
        )


def _append_memory_artifacts(
    lines: list[str], memory_artifacts, *, indent: int
) -> None:
    if not memory_artifacts:
        return
    prefix = " " * indent
    lines.append(f"{prefix}Memory artifacts:")
    for artifact in memory_artifacts:
        lines.append(f"{prefix}  - {_memory_artifact_summary(artifact)}")


def _append_contributions(
    lines: list[str], proof: ExecutionProofView, *, indent: int
) -> None:
    groups = (
        ("Explicit inputs", ContributionOrigin.EXPLICIT),
        ("Derived inputs", ContributionOrigin.DERIVED),
        ("Applied constraints", ContributionOrigin.CONTEXTUAL),
    )
    prefix = " " * indent
    for label, origin in groups:
        values = _dedupe_keep_order(
            item.label for item in proof.contributions if item.origin is origin
        )
        if values:
            lines.append(f"{prefix}{label}: {', '.join(values)}")


def _append_proof_summary(
    lines: list[str], proof: ExecutionProofView, *, indent: int
) -> None:
    prefix = " " * indent
    evidence = tuple(item for item in proof.endpoint_args if item.values)
    if evidence:
        lines.append(f"{prefix}Evidence used:")
        for item in evidence:
            lines.append(f"{prefix}  - applied {item.arg_name}")
    if proof.computation_summaries:
        lines.append(f"{prefix}Computation:")
        for item in proof.computation_summaries:
            lines.append(f"{prefix}  - {item}")


def _append_debug_proofs(
    lines: list[str],
    step: TimelineStepView,
    *,
    answer_output: str | None,
    indent: int,
) -> None:
    prefix = " " * indent
    for output in step.answer_outputs:
        if output.proof is None or not _output_matches(
            output, answer_output=answer_output
        ):
            continue
        lines.append(
            f"{prefix}Proof details for {output.output_key}: {output.proof.proof_graph_id}"
        )
        if output.proof.debug_evidence_handles:
            lines.append(f"{prefix}  Evidence handles:")
            for item in output.proof.debug_evidence_handles:
                lines.append(f"{prefix}    - {item}")
        if output.proof.debug_computation_links:
            lines.append(f"{prefix}  Computation links:")
            for item in output.proof.debug_computation_links:
                lines.append(f"{prefix}    - {item}")


def _append_source_reads(
    lines: list[str],
    source_reads: tuple[SourceReadView, ...],
    *,
    indent: int,
    include_audit: bool,
) -> None:
    prefix = " " * indent
    for source_read in source_reads:
        row_count = (
            "unknown" if source_read.row_count is None else source_read.row_count
        )
        hash_segment = (
            f" hash={source_read.response_hash}" if source_read.response_hash else ""
        )
        lines.append(
            f"{prefix}Source read: {source_read.catalog_endpoint.label} "
            f"rows={row_count}{hash_segment}"
        )
        if source_read.args:
            lines.append(f"{prefix}  Args: {_format_args(source_read.args)}")
        if include_audit:
            _append_source_read_audit(lines, source_read, indent=indent + 2)


def _append_source_read_audit(
    lines: list[str], source_read: SourceReadView, *, indent: int
) -> None:
    prefix = " " * indent
    endpoint = source_read.catalog_endpoint
    lines.append(f"{prefix}Status: {source_read.status}")
    lines.append(f"{prefix}Catalog endpoint: {endpoint.label}")
    lines.append(f"{prefix}Catalog endpoint key: {endpoint.catalog_endpoint_key}")
    lines.append(f"{prefix}Catalog endpoint row id: {endpoint.catalog_endpoint_id}")
    lines.append(f"{prefix}Framework: {endpoint.framework_kind}")
    if endpoint.source_namespace_path:
        lines.append(
            f"{prefix}Namespace: {endpoint.source_namespace_kind} "
            f"{'/'.join(endpoint.source_namespace_path)}"
        )
    if endpoint.route_method or endpoint.route_path_template:
        lines.append(
            f"{prefix}Route: {endpoint.route_method} {endpoint.route_path_template}"
        )
    if endpoint.handler_ref:
        lines.append(f"{prefix}Handler: {endpoint.handler_ref}")
    if endpoint.api_schema_operation_id:
        lines.append(
            f"{prefix}API schema operation: {endpoint.api_schema_operation_id}"
        )
    if source_read.completeness:
        lines.append(f"{prefix}Completeness: {_format_args(source_read.completeness)}")
    if source_read.artifact_id:
        lines.append(f"{prefix}Artifact: {source_read.artifact_id}")
    if source_read.error:
        lines.append(f"{prefix}Error: {_format_args(source_read.error)}")


def _visible_source_reads(
    step: TimelineStepView, *, answer_output: str | None
) -> tuple[SourceReadView, ...]:
    if answer_output is None:
        return step.source_reads
    source_read_ids = {
        source_read.source_read_id
        for output in step.answer_outputs
        if _output_matches(output, answer_output=answer_output)
        and output.proof is not None
        for source_read in output.proof.source_reads
    }
    return tuple(
        source_read
        for source_read in step.source_reads
        if source_read.source_read_id in source_read_ids
    )


def _fact_matches(fact, *, fact_filter: str | None) -> bool:
    if fact_filter is None:
        return True
    return fact.requested_fact_id == fact_filter or fact.fact_key == fact_filter


def _output_matches(
    output: TimelineAnswerOutputView, *, answer_output: str | None
) -> bool:
    return answer_output is None or output.output_key == answer_output


def _memory_artifact_summary(artifact) -> str:
    parts: list[str] = []
    if artifact.source_question:
        parts.append(f"source question: {artifact.source_question}")
    if artifact.source_answer:
        parts.append(f"source answer: {artifact.source_answer}")
    if artifact.outcome:
        parts.append(f"outcome={artifact.outcome}")
    if artifact.address_summaries:
        parts.append(f"addresses={', '.join(artifact.address_summaries)}")
    return "; ".join(parts) if parts else artifact.memory_artifact_id


def _clarification_option(option: dict[str, object]) -> str:
    label = str(option.get("label") or option.get("id") or "")
    description = str(option.get("description") or "")
    if description and description != label:
        return f"{label} [{description}]"
    return label


def _format_args(args: dict[str, object]) -> str:
    return ", ".join(f"{_arg_name(key)}={value}" for key, value in sorted(args.items()))


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
