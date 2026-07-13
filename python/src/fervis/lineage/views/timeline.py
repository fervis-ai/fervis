"""Chronological lineage projection."""

from __future__ import annotations

from dataclasses import replace

from fervis.lineage.views.model import (
    AnswerPresentationView,
    ClarificationRequestView,
    LineageTimelineView,
    LineageView,
    ModelCallInspectionView,
    ObservabilityNoticeView,
    RequestedFactView,
    RuntimeErrorView,
    SourceReadView,
    TimelineQuestionView,
    TimelineRunView,
    TimelineAnswerOutputView,
    TimelineFactResultView,
    TimelineRequestedFactView,
    TimelineStepView,
)


def lineage_timeline_view(
    view: LineageView,
    *,
    model_calls: tuple[ModelCallInspectionView, ...] = (),
    observability_notices: tuple[ObservabilityNoticeView, ...] = (),
) -> LineageTimelineView:
    return LineageTimelineView(
        root_kind=view.root_kind,
        root_id=view.root_id,
        observability_notices=observability_notices,
        questions=tuple(
            TimelineQuestionView(
                question_id=question.question_id,
                conversation_id=question.conversation_id,
                text=question.text,
                runs=tuple(
                    _timeline_run(run, model_calls=model_calls) for run in question.runs
                ),
            )
            for question in view.questions
        ),
    )


def _timeline_run(run, *, model_calls: tuple[ModelCallInspectionView, ...]):
    return TimelineRunView(
        run_id=run.run_id,
        run_number=run.run_number,
        kind=run.kind,
        trigger_kind=run.trigger_kind,
        result_kind=run.result_kind,
        activated_memory_ids=run.activated_memory_ids,
        memory_artifacts=run.memory_artifacts,
        program_derivation=run.program_derivation,
        steps=tuple(
            _timeline_step(step, run, model_calls=model_calls) for step in run.steps
        ),
        base_run_id=run.base_run_id,
        clarification_responses=run.clarification_responses,
    )


def _timeline_step(step, run, *, model_calls: tuple[ModelCallInspectionView, ...]):
    return TimelineStepView(
        step_id=step.step_id,
        step_key=step.step_key,
        sequence=step.sequence,
        decisions=step.decisions,
        semantic=step.semantic,
        model_calls=_step_model_calls(step, model_calls=model_calls),
        source_reads=_step_source_reads(step, run.source_reads),
        requested_facts=_step_requested_facts(step, run.requested_facts),
        fact_results=_step_fact_results(step, run.requested_facts),
        answer_outputs=_step_answer_outputs(step, run.requested_facts),
        answer_presentations=_step_answer_presentations(step, run.answers),
        clarifications=_step_clarifications(step, run.clarification_requests),
        runtime_errors=_step_runtime_errors(step, run.runtime_errors),
    )


def _step_model_calls(
    step, *, model_calls: tuple[ModelCallInspectionView, ...]
) -> tuple[ModelCallInspectionView, ...]:
    return tuple(call for call in model_calls if call.step_id == step.step_id)


def _step_source_reads(
    step, source_reads: tuple[SourceReadView, ...]
) -> tuple[SourceReadView, ...]:
    return tuple(
        source_read
        for source_read in source_reads
        if source_read.step_id == step.step_id
    )


def _step_requested_facts(
    step, requested_facts: tuple[RequestedFactView, ...]
) -> tuple[TimelineRequestedFactView, ...]:
    return tuple(
        TimelineRequestedFactView(
            requested_fact_id=fact.requested_fact_id,
            fact_key=fact.fact_key,
            description=fact.description,
        )
        for fact in requested_facts
        if fact.produced_by_step_id == step.step_id
    )


def _step_fact_results(
    step, requested_facts: tuple[RequestedFactView, ...]
) -> tuple[TimelineFactResultView, ...]:
    return tuple(
        TimelineFactResultView(
            fact_result_id=result.fact_result_id,
            requested_fact_id=fact.requested_fact_id,
            result_kind=result.result_kind,
            proof=result.proof,
            memory_artifacts=result.memory_artifacts,
        )
        for fact in requested_facts
        for result in fact.fact_results
        if result.produced_by_step_id == step.step_id
    )


def _step_answer_outputs(
    step, requested_facts: tuple[RequestedFactView, ...]
) -> tuple[TimelineAnswerOutputView, ...]:
    result_ids = {
        result.fact_result_id for result in _step_fact_results(step, requested_facts)
    }
    return tuple(
        TimelineAnswerOutputView(
            fact_result_id=output.fact_result_id,
            output_key=output.output_key,
            value_kind=output.value_kind,
            value=output.value,
            value_json=output.value_json,
            proof_node_refs=output.proof_node_refs,
            proof=output.proof,
        )
        for fact in requested_facts
        for output in fact.answer_outputs
        if output.fact_result_id in result_ids
    )


def _step_answer_presentations(step, answers) -> tuple[AnswerPresentationView, ...]:
    return tuple(
        presentation
        for answer in answers
        for presentation in answer.presentations
        if presentation.render_step_id == step.step_id
    )


def _step_clarifications(
    step, clarifications: tuple[ClarificationRequestView, ...]
) -> tuple[ClarificationRequestView, ...]:
    return tuple(
        clarification
        for clarification in clarifications
        if clarification.step_id == step.step_id
    )


def _step_runtime_errors(
    step, runtime_errors: tuple[RuntimeErrorView, ...]
) -> tuple[RuntimeErrorView, ...]:
    return tuple(
        error for error in runtime_errors if error.failed_step_id == step.step_id
    )


def compact_lineage_timeline_view(view: LineageTimelineView) -> LineageTimelineView:
    return replace(
        view,
        questions=tuple(
            replace(
                question,
                runs=tuple(
                    replace(
                        run,
                        steps=tuple(_compact_step(step) for step in run.steps),
                    )
                    for run in question.runs
                ),
            )
            for question in view.questions
        ),
    )


def _compact_step(step: TimelineStepView) -> TimelineStepView:
    return replace(
        step,
        model_calls=(),
        fact_results=tuple(replace(result, proof=None) for result in step.fact_results),
        answer_outputs=tuple(
            replace(output, proof=None) for output in step.answer_outputs
        ),
    )
