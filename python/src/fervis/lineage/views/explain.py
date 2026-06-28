"""Framework-neutral explain view composition."""

from __future__ import annotations

from dataclasses import dataclass, replace

from fervis.lineage.enums import RunStepKey
from fervis.lineage.views.model import (
    ArtifactInspectionView,
    LineageTimelineView,
    LineageView,
    ModelCallInspectionView,
    ObservabilityNoticeView,
    RunView,
)
from fervis.lineage.views.query import LineageQueryPort
from fervis.lineage.views.service import (
    AnswerLineageService,
    ConversationLineageService,
    LineageRootNotFound,
    QuestionLineageService,
)
from fervis.observability.query import (
    ObservabilityModelCall,
    ObservabilityQueryPort,
)
from fervis.lineage.views.timeline import lineage_timeline_view


@dataclass(frozen=True)
class ExplainView:
    lineage: LineageView
    timeline: LineageTimelineView
    model_calls: tuple[ModelCallInspectionView, ...] = ()


@dataclass(frozen=True)
class LineageSlice:
    answer_output: str | None = None
    fact_filter: str | None = None
    step_key: RunStepKey | None = None
    errors_only: bool = False


class ExplainViewService:
    def __init__(
        self,
        *,
        lineage_query: LineageQueryPort,
        observability_query: ObservabilityQueryPort,
    ) -> None:
        self._answer_lineage = AnswerLineageService(lineage_query)
        self._question_lineage = QuestionLineageService(lineage_query)
        self._conversation_lineage = ConversationLineageService(lineage_query)
        self._observability_query = observability_query

    def for_answer(
        self,
        answer_id: str,
        *,
        step_key: RunStepKey | None = None,
        lineage_slice: LineageSlice | None = None,
    ) -> ExplainView:
        return self._explain(
            self._answer_lineage.for_answer(answer_id),
            step_key=step_key,
            lineage_slice=lineage_slice,
        )

    def for_question(
        self,
        question_id: str,
        *,
        step_key: RunStepKey | None = None,
        lineage_slice: LineageSlice | None = None,
    ) -> ExplainView:
        return self._explain(
            self._question_lineage.for_question(question_id),
            step_key=step_key,
            lineage_slice=lineage_slice,
        )

    def for_question_run(
        self,
        question_id: str,
        run_id: str,
        *,
        step_key: RunStepKey | None = None,
        lineage_slice: LineageSlice | None = None,
    ) -> ExplainView:
        view = self._question_lineage.for_question(question_id)
        if not _view_has_run(view, run_id):
            raise LineageRootNotFound(
                f"run {run_id!r} is not in question {question_id!r}"
            )
        return self._explain(
            _slice_to_run(view, run_id),
            step_key=step_key,
            lineage_slice=lineage_slice,
        )

    def for_run(
        self,
        run_id: str,
        *,
        step_key: RunStepKey | None = None,
        lineage_slice: LineageSlice | None = None,
    ) -> ExplainView:
        return self._explain(
            self._question_lineage.for_run(run_id),
            step_key=step_key,
            lineage_slice=lineage_slice,
        )

    def for_conversation(
        self,
        conversation_id: str,
        *,
        step_key: RunStepKey | None = None,
        lineage_slice: LineageSlice | None = None,
    ) -> ExplainView:
        return self._explain(
            self._conversation_lineage.for_conversation(conversation_id),
            step_key=step_key,
            lineage_slice=lineage_slice,
        )

    def for_conversation_question(
        self,
        conversation_id: str,
        question_id: str,
        *,
        step_key: RunStepKey | None = None,
        lineage_slice: LineageSlice | None = None,
    ) -> ExplainView:
        view = self._conversation_lineage.for_conversation(conversation_id)
        if not _view_has_question(view, question_id):
            raise LineageRootNotFound(
                f"question {question_id!r} is not in conversation {conversation_id!r}"
            )
        return self._explain(
            _slice_to_question(view, question_id),
            step_key=step_key,
            lineage_slice=lineage_slice,
        )

    def for_conversation_run(
        self,
        conversation_id: str,
        run_id: str,
        *,
        step_key: RunStepKey | None = None,
        lineage_slice: LineageSlice | None = None,
    ) -> ExplainView:
        view = self._conversation_lineage.for_conversation(conversation_id)
        if not _view_has_run(view, run_id):
            raise LineageRootNotFound(
                f"run {run_id!r} is not in conversation {conversation_id!r}"
            )
        return self._explain(
            _slice_to_run(view, run_id),
            step_key=step_key,
            lineage_slice=lineage_slice,
        )

    def _explain(
        self,
        lineage: LineageView,
        *,
        step_key: RunStepKey | None,
        lineage_slice: LineageSlice | None,
    ) -> ExplainView:
        lineage_slice = lineage_slice or LineageSlice()
        original_lineage = lineage
        lineage = _apply_slice(lineage, lineage_slice)
        run_ids = _view_run_ids(lineage)
        model_calls = tuple(
            _model_call_inspection_view(call)
            for call in self._observability_query.model_calls_for_run_ids(
                run_ids, detail="inspection"
            )
            if step_key is None or call.step_key is step_key
        )
        observability_notices = _observability_notices(
            original_lineage=original_lineage,
            sliced_lineage=lineage,
            lineage_slice=lineage_slice,
            model_calls=model_calls,
        )
        return ExplainView(
            lineage=lineage,
            timeline=lineage_timeline_view(
                lineage,
                model_calls=model_calls,
                observability_notices=observability_notices,
            ),
            model_calls=model_calls,
        )


def _model_call_inspection_view(
    call: ObservabilityModelCall,
) -> ModelCallInspectionView:
    return ModelCallInspectionView(
        model_call_id=call.model_call_id,
        run_id=call.run_id,
        step_id=call.step_id,
        step_key=call.step_key.value,
        step_sequence=call.step_sequence,
        call_index=call.call_index,
        provider=call.provider,
        model_key=call.model_key,
        status=call.status.value,
        prompt_chars=call.prompt_chars,
        schema_chars=call.schema_chars,
        tool_spec_chars=call.tool_spec_chars,
        artifacts=tuple(
            ArtifactInspectionView(
                artifact_kind=artifact.artifact_kind.value,
                artifact_id=artifact.artifact_id,
                size_bytes=artifact.size_bytes,
            )
            for artifact in call.artifacts
        ),
    )


def _view_run_ids(view: LineageView) -> tuple[str, ...]:
    return tuple(run.run_id for question in view.questions for run in question.runs)


def _observability_notices(
    *,
    original_lineage: LineageView,
    sliced_lineage: LineageView,
    lineage_slice: LineageSlice,
    model_calls: tuple[ModelCallInspectionView, ...],
) -> tuple[ObservabilityNoticeView, ...]:
    notices: list[ObservabilityNoticeView] = []
    original_runs = _view_runs(original_lineage)
    sliced_runs = _view_runs(sliced_lineage)
    if lineage_slice.errors_only and original_runs and not sliced_runs:
        notices.append(_errors_filter_notice(original_runs))
    if sliced_runs and lineage_slice.step_key is None and not model_calls:
        notices.append(_missing_model_call_notice(sliced_runs))
    return tuple(notices)


def _errors_filter_notice(runs: tuple[RunView, ...]) -> ObservabilityNoticeView:
    run_kinds = {run.run_id: run.result_kind for run in runs}
    return ObservabilityNoticeView(
        kind="errors_filter_no_runtime_error_runs",
        severity="info",
        message=(
            "--errors shows only runs whose result_kind is runtime_error. "
            "The selected lineage has no runtime_error runs; run without --errors "
            "to inspect terminal facts, clarifications, source reads, and model calls."
        ),
        run_ids=tuple(run_kinds),
        details={"result_kinds_by_run_id": run_kinds},
    )


def _missing_model_call_notice(runs: tuple[RunView, ...]) -> ObservabilityNoticeView:
    return ObservabilityNoticeView(
        kind="missing_model_call_audits",
        severity="warning",
        message=(
            "No model-call audit rows were found for the selected runs. "
            "Lineage may still show deterministic steps, source reads, and runtime "
            "errors, but prompts, schemas, raw outputs, and parsed payloads are not "
            "available through explain."
        ),
        run_ids=tuple(run.run_id for run in runs),
    )


def _view_runs(view: LineageView) -> tuple[RunView, ...]:
    return tuple(run for question in view.questions for run in question.runs)


def _view_has_question(view: LineageView, question_id: str) -> bool:
    return any(question.question_id == question_id for question in view.questions)


def _view_has_run(view: LineageView, run_id: str) -> bool:
    return any(
        run.run_id == run_id for question in view.questions for run in question.runs
    )


def _slice_to_question(view: LineageView, question_id: str) -> LineageView:
    return replace(
        view,
        questions=tuple(
            question
            for question in view.questions
            if question.question_id == question_id
        ),
    )


def _slice_to_run(view: LineageView, run_id: str) -> LineageView:
    return replace(
        view,
        questions=tuple(
            replace(
                question,
                runs=tuple(run for run in question.runs if run.run_id == run_id),
            )
            for question in view.questions
            if any(run.run_id == run_id for run in question.runs)
        ),
    )


def _apply_slice(view: LineageView, lineage_slice: LineageSlice) -> LineageView:
    _validate_slice(view, lineage_slice)
    questions = tuple(
        _slice_question(question, lineage_slice) for question in view.questions
    )
    return replace(
        view,
        questions=tuple(
            question
            for question in questions
            if question.runs or lineage_slice.errors_only
        ),
    )


def _validate_slice(view: LineageView, lineage_slice: LineageSlice) -> None:
    if lineage_slice.answer_output is not None and not any(
        output.output_key == lineage_slice.answer_output
        for question in view.questions
        for run in question.runs
        for fact in run.requested_facts
        for output in fact.answer_outputs
    ):
        raise LineageRootNotFound(
            f"answer output {lineage_slice.answer_output!r} is not in lineage view"
        )
    if lineage_slice.fact_filter is not None and not any(
        fact.requested_fact_id == lineage_slice.fact_filter
        or fact.fact_key == lineage_slice.fact_filter
        for question in view.questions
        for run in question.runs
        for fact in run.requested_facts
    ):
        raise LineageRootNotFound(
            f"fact {lineage_slice.fact_filter!r} is not in lineage view"
        )
    if lineage_slice.step_key is not None and not any(
        step.step_key == lineage_slice.step_key.value
        for question in view.questions
        for run in question.runs
        for step in run.steps
    ):
        raise LineageRootNotFound(
            f"step {lineage_slice.step_key.value!r} is not in lineage view"
        )


def _slice_question(question, lineage_slice: LineageSlice):
    return replace(
        question,
        runs=tuple(
            _slice_run(run, lineage_slice)
            for run in question.runs
            if not lineage_slice.errors_only or run.result_kind == "runtime_error"
        ),
    )


def _slice_run(run, lineage_slice: LineageSlice):
    return replace(
        run,
        requested_facts=tuple(
            _slice_fact(fact, lineage_slice)
            for fact in run.requested_facts
            if _fact_in_slice(fact, lineage_slice)
        ),
        answers=tuple(_slice_answer(answer, lineage_slice) for answer in run.answers),
        steps=tuple(
            step
            for step in run.steps
            if lineage_slice.step_key is None
            or step.step_key == lineage_slice.step_key.value
        ),
    )


def _fact_in_slice(fact, lineage_slice: LineageSlice) -> bool:
    if lineage_slice.fact_filter is not None and not (
        fact.requested_fact_id == lineage_slice.fact_filter
        or fact.fact_key == lineage_slice.fact_filter
    ):
        return False
    if lineage_slice.answer_output is not None:
        return any(
            output.output_key == lineage_slice.answer_output
            for output in fact.answer_outputs
        )
    return True


def _slice_fact(fact, lineage_slice: LineageSlice):
    return replace(
        fact,
        steps=_slice_steps(fact.steps, lineage_slice),
        fact_results=tuple(
            _slice_fact_result(result, lineage_slice) for result in fact.fact_results
        ),
        answer_outputs=tuple(
            output
            for output in fact.answer_outputs
            if lineage_slice.answer_output is None
            or output.output_key == lineage_slice.answer_output
        ),
    )


def _slice_fact_result(result, lineage_slice: LineageSlice):
    return replace(result, steps=_slice_steps(result.steps, lineage_slice))


def _slice_answer(answer, lineage_slice: LineageSlice):
    if lineage_slice.answer_output is None:
        return answer
    return replace(
        answer,
        outputs=tuple(
            output
            for output in answer.outputs
            if output.output_key == lineage_slice.answer_output
        ),
    )


def _slice_steps(steps, lineage_slice: LineageSlice):
    return tuple(
        step
        for step in steps
        if lineage_slice.step_key is None
        or step.step_key == lineage_slice.step_key.value
    )
