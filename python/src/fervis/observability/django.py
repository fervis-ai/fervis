"""Django-backed observability query adapter."""

from __future__ import annotations

from fervis.lineage import models
from fervis.lineage.enums import (
    ArtifactKind,
    ModelCallStatus,
    ModelUsageKind,
    ModelUsageUnit,
    RunStepKey,
)
from fervis.observability.query import (
    ObservabilityArtifactContent,
    ModelCallDetailLevel,
    ObservabilityArtifact,
    ObservabilityModelCall,
    ObservabilityQueryPort,
    ObservabilityRun,
    ObservabilityUsage,
)


class DjangoObservabilityQuery(ObservabilityQueryPort):
    def run_id_for_answer(self, answer_id: str) -> str | None:
        answer = (
            models.Answer.objects.filter(answer_id=answer_id)
            .select_related("run")
            .first()
        )
        if answer is None:
            return None
        return answer.run_id

    def run_by_id(self, run_id: str) -> ObservabilityRun | None:
        run = (
            models.QuestionRun.objects.filter(run_id=run_id)
            .only(
                "run_id",
                "previous_run_id",
                "trigger_clarification_response_run_id",
            )
            .first()
        )
        if run is None:
            return None
        return ObservabilityRun(
            run_id=run.run_id,
            previous_run_id=run.previous_run_id,
            trigger_clarification_response_run_id=(
                run.trigger_clarification_response_run_id
            ),
        )

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

    def model_calls_for_run_ids(
        self, run_ids: tuple[str, ...], *, detail: ModelCallDetailLevel = "inspection"
    ) -> tuple[ObservabilityModelCall, ...]:
        if not run_ids:
            return ()
        calls = _model_call_queryset(
            models.ModelCall.objects.filter(run_id__in=run_ids),
            detail=detail,
        ).order_by("run_id", "step__sequence", "call_index")
        return tuple(_model_call_row(call, detail=detail) for call in calls)

    def model_calls_for_run(
        self,
        run_id: str,
        step_key: RunStepKey | None = None,
        *,
        detail: ModelCallDetailLevel = "inspection",
    ) -> tuple[ObservabilityModelCall, ...]:
        calls = _model_call_queryset(
            models.ModelCall.objects.filter(run_id=run_id),
            detail=detail,
        )
        if step_key is not None:
            calls = calls.filter(step__step_key=step_key.value)
        calls = calls.order_by("step__sequence", "call_index")
        return tuple(_model_call_row(call, detail=detail) for call in calls)

    def artifact_content(self, artifact_id: str) -> ObservabilityArtifactContent | None:
        artifact = models.RunArtifact.objects.filter(artifact_id=artifact_id).first()
        if artifact is None:
            return None
        return ObservabilityArtifactContent(
            artifact_id=artifact.artifact_id,
            artifact_kind=ArtifactKind(artifact.artifact_kind),
            content_hash=artifact.content_hash,
            content_type=artifact.content_type,
            size_bytes=artifact.size_bytes,
            content=artifact.content,
            storage_ref=artifact.storage_ref,
        )


def _model_call_queryset(queryset, *, detail: ModelCallDetailLevel):
    queryset = queryset.select_related("step").prefetch_related("usage_rows")
    if detail == "inspection":
        queryset = queryset.prefetch_related("artifacts")
    return queryset


def _model_call_row(
    call: models.ModelCall, *, detail: ModelCallDetailLevel
) -> ObservabilityModelCall:
    return ObservabilityModelCall(
        model_call_id=call.model_call_id,
        run_id=call.run_id,
        step_id=call.step_id,
        step_key=RunStepKey(call.step.step_key),
        step_sequence=call.step.sequence,
        call_index=call.call_index,
        provider=call.provider,
        model_key=call.model_key,
        status=ModelCallStatus(call.status),
        provider_request_id=call.provider_request_id,
        finish_reason=call.finish_reason,
        duration_ms=call.duration_ms,
        reasoning_effort=call.reasoning_effort,
        reasoning_budget_tokens=call.reasoning_budget_tokens,
        max_output_tokens=call.max_output_tokens,
        prompt_chars=call.prompt_chars,
        schema_chars=call.schema_chars,
        tool_spec_chars=call.tool_spec_chars,
        submitted_payload_chars=call.submitted_payload_chars,
        raw_output_chars=call.raw_output_chars,
        model_subcalls=tuple(
            dict(item)
            for item in (call.model_subcalls_json or ())
            if isinstance(item, dict)
        ),
        usage_rows=tuple(_usage_row(usage) for usage in call.usage_rows.all()),
        artifacts=(
            tuple(_artifact_row(artifact) for artifact in call.artifacts.all())
            if detail == "inspection"
            else ()
        ),
    )


def _usage_row(usage: models.ModelCallUsage) -> ObservabilityUsage:
    return ObservabilityUsage(
        usage_kind=ModelUsageKind(usage.usage_kind),
        quantity=usage.quantity,
        unit=ModelUsageUnit(usage.unit),
        provider_usage_key=usage.provider_usage_key,
        cost_micros=usage.cost_micros,
        currency=usage.currency,
        price_basis_json=usage.price_basis_json,
    )


def _artifact_row(artifact: models.RunArtifact) -> ObservabilityArtifact:
    return ObservabilityArtifact(
        artifact_id=artifact.artifact_id,
        artifact_kind=ArtifactKind(artifact.artifact_kind),
        content_hash=artifact.content_hash,
        content_type=artifact.content_type,
        size_bytes=artifact.size_bytes,
        has_content=artifact.content is not None,
        storage_ref=artifact.storage_ref,
    )
