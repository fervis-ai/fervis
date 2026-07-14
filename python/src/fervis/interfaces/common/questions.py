"""Transport-neutral question interface over the Fervis lifecycle."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from fervis.host_api.contracts.authority import ReadContextRef
from fervis.host_api.contracts.credentials import DelegatedReadCredential
from fervis.interfaces.common.admission import (
    ConfiguredModelPolicy,
    ModelPolicyValidationError,
)
from fervis.questions import (
    AskRequest,
    AskRequestLimits,
    AskResult,
    ClarificationResponseRequest,
    ExecutionMode,
    QuestionPrincipal,
    RerunQuestionRequest,
)
from fervis.interfaces.common.binding_patches import (
    binding_patch_from_payload,
    capability_application_from_payload,
)
from fervis.questions.result_data import result_data_clarifications
from fervis.run_work.events import QuestionRunEventSink

RERUN_TRIGGER = "rerun"

_CREATE_QUESTION_KEYS = frozenset(
    {
        "question",
        "conversationId",
        "contextRunId",
        "provider",
        "modelKey",
        "maxBudgetUsd",
        "maxThinkingTokens",
    }
)
_CONTINUE_QUESTION_KEYS = frozenset(
    {
        "responseText",
        "runId",
        "clarificationId",
        "selectedOptionId",
    }
)
_RERUN_QUESTION_KEYS = frozenset(
    {"triggerKind", "baseRunId", "patch", "capabilityApplication"}
)


@dataclass(frozen=True)
class InterfacePrincipal:
    principal_id: str
    tenant_id: str
    raw: Any = None
    read_context_ref: ReadContextRef = field(
        default_factory=lambda: ReadContextRef(scheme="anonymous")
    )
    delegated_credential: DelegatedReadCredential | None = None


@dataclass(frozen=True)
class QuestionInterfaceResponse:
    status_code: int
    payload: Any


@dataclass
class QuestionInterfaceValidationError(ValueError):
    field: str
    message: str
    code: str = "invalid"

    def __str__(self) -> str:
        return self.message


class QuestionLifecycle(Protocol):
    def ask(
        self,
        request: AskRequest,
        *,
        event_sink: QuestionRunEventSink | None = None,
    ) -> AskResult: ...

    def respond_to_clarification(
        self,
        request: ClarificationResponseRequest,
        *,
        event_sink: QuestionRunEventSink | None = None,
    ) -> AskResult: ...

    def rerun_question(
        self,
        request: RerunQuestionRequest,
        *,
        event_sink: QuestionRunEventSink | None = None,
    ) -> AskResult: ...

    def list_conversations(
        self,
        *,
        principal: QuestionPrincipal,
    ) -> list[dict[str, Any]]: ...

    def get_question_state(
        self,
        question_id: str,
        *,
        principal: QuestionPrincipal,
    ) -> dict[str, Any] | None: ...

    def list_question_runs(
        self,
        question_id: str,
        *,
        principal: QuestionPrincipal,
    ) -> list[dict[str, Any]]: ...

    def get_question_run(
        self,
        question_id: str,
        run_id: str,
        *,
        principal: QuestionPrincipal,
    ) -> dict[str, Any] | None: ...


@dataclass(frozen=True)
class QuestionInterface:
    questions: QuestionLifecycle
    limits: AskRequestLimits = field(default_factory=AskRequestLimits)
    model_policy: ConfiguredModelPolicy = field(default_factory=ConfiguredModelPolicy)
    close_callback: Callable[[], None] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def close(self) -> None:
        if self.close_callback is not None:
            self.close_callback()

    def list_conversations(
        self,
        *,
        principal: InterfacePrincipal,
    ) -> QuestionInterfaceResponse:
        return QuestionInterfaceResponse(
            status_code=200,
            payload={
                "conversations": self.questions.list_conversations(
                    principal=_question_principal(principal),
                )
            },
        )

    def create_question(
        self,
        payload: dict[str, Any],
        *,
        principal: InterfacePrincipal,
        idempotency_key: str | None = None,
        event_sink: QuestionRunEventSink | None = None,
    ) -> QuestionInterfaceResponse:
        try:
            request = self._create_request(
                payload,
                principal=principal,
                idempotency_key=idempotency_key,
            )
            result = self.questions.ask(request, event_sink=event_sink)
        except QuestionInterfaceValidationError:
            raise
        except ModelPolicyValidationError as exc:
            raise QuestionInterfaceValidationError(
                field=exc.field,
                message=exc.message,
            ) from exc
        except PermissionError as exc:
            raise QuestionInterfaceValidationError(
                field="contextRunId",
                message=str(exc),
            ) from exc
        except ValueError as exc:
            raise QuestionInterfaceValidationError(
                field=_ask_request_error_field(str(exc)),
                message=str(exc),
            ) from exc
        if result.status == "ACTIVE_RUN_CONFLICT":
            return QuestionInterfaceResponse(
                status_code=409,
                payload=_active_conflict_payload(result),
            )
        question = self.questions.get_question_state(
            result.question_id,
            principal=_question_principal(principal),
        )
        return QuestionInterfaceResponse(
            status_code=202,
            payload=_with_next_actions(
                _require_admitted_question_state(question),
                principal,
            ),
        )

    def get_question(
        self,
        question_id: str,
        *,
        principal: InterfacePrincipal,
    ) -> QuestionInterfaceResponse:
        question = self.questions.get_question_state(
            question_id,
            principal=_question_principal(principal),
        )
        if question is None:
            return QuestionInterfaceResponse(
                status_code=404,
                payload=_not_found_payload("fervis_question", question_id),
            )
        return QuestionInterfaceResponse(
            status_code=200,
            payload=_with_next_actions(question, principal),
        )

    def list_question_runs(
        self,
        question_id: str,
        *,
        principal: InterfacePrincipal,
    ) -> QuestionInterfaceResponse:
        question = self.questions.get_question_state(
            question_id,
            principal=_question_principal(principal),
        )
        if question is None:
            return QuestionInterfaceResponse(
                status_code=404,
                payload=_not_found_payload("fervis_question", question_id),
            )
        return QuestionInterfaceResponse(
            status_code=200,
            payload={
                "questionId": question_id,
                "runs": self.questions.list_question_runs(
                    question_id,
                    principal=_question_principal(principal),
                ),
            },
        )

    def create_question_run(
        self,
        question_id: str,
        payload: dict[str, Any],
        *,
        principal: InterfacePrincipal,
        idempotency_key: str | None = None,
        event_sink: QuestionRunEventSink | None = None,
    ) -> QuestionInterfaceResponse:
        try:
            if payload.get("triggerKind") == RERUN_TRIGGER:
                rerun = self._rerun_request(
                    question_id,
                    payload,
                    principal=principal,
                    idempotency_key=idempotency_key,
                )
                result = self.questions.rerun_question(rerun, event_sink=event_sink)
            else:
                continuation = self._continue_request(
                    question_id,
                    payload,
                    principal=principal,
                )
                result = self.questions.respond_to_clarification(
                    continuation,
                    event_sink=event_sink,
                )
        except QuestionInterfaceValidationError:
            raise
        except ModelPolicyValidationError as exc:
            raise QuestionInterfaceValidationError(
                field=exc.field,
                message=exc.message,
            ) from exc
        except ValueError as exc:
            raise QuestionInterfaceValidationError(
                field=_ask_request_error_field(str(exc)),
                message=str(exc),
            ) from exc
        if result.status == "ACTIVE_RUN_CONFLICT":
            return QuestionInterfaceResponse(
                status_code=409,
                payload=_active_conflict_payload(result),
            )
        question = self.questions.get_question_state(
            result.question_id,
            principal=_question_principal(principal),
        )
        return QuestionInterfaceResponse(
            status_code=202,
            payload=_with_next_actions(
                _require_admitted_question_state(question),
                principal,
            ),
        )

    def get_question_run(
        self,
        question_id: str,
        run_id: str,
        *,
        principal: InterfacePrincipal,
    ) -> QuestionInterfaceResponse:
        run = self.questions.get_question_run(
            question_id,
            run_id,
            principal=_question_principal(principal),
        )
        if run is None:
            return QuestionInterfaceResponse(
                status_code=404,
                payload=_not_found_payload("fervis_run", run_id),
            )
        return QuestionInterfaceResponse(
            status_code=200,
            payload=_with_next_actions(run, principal),
        )

    def _create_request(
        self,
        payload: dict[str, Any],
        *,
        principal: InterfacePrincipal,
        idempotency_key: str | None,
    ) -> AskRequest:
        if not isinstance(payload, dict):
            raise QuestionInterfaceValidationError(
                field="__all__",
                message="Payload must be an object.",
            )
        _reject_unknown_fields(payload)
        question = str(payload.get("question") or "").strip()
        if not question:
            raise QuestionInterfaceValidationError(
                field="question",
                message="question is required.",
            )
        model = self.model_policy.admit(
            requested_provider=payload.get("provider"),
            requested_model_key=_optional_string(payload, "modelKey"),
        )
        return AskRequest(
            conversation_id=str(payload.get("conversationId") or ""),
            context_run_id=_optional_identifier(payload, "contextRunId"),
            question=question,
            principal=_question_principal(principal),
            execution_mode=ExecutionMode.QUEUED,
            provider=model.provider,
            model_key=model.model_key,
            idempotency_key=idempotency_key,
            max_budget_usd=_optional_float(payload, "maxBudgetUsd"),
            max_thinking_tokens=_optional_int(payload, "maxThinkingTokens"),
            limits=self.limits,
        )

    def _continue_request(
        self,
        question_id: str,
        payload: dict[str, Any],
        *,
        principal: InterfacePrincipal,
    ) -> ClarificationResponseRequest:
        if not isinstance(payload, dict):
            raise QuestionInterfaceValidationError(
                field="__all__",
                message="Payload must be an object.",
        )
        _reject_unknown_fields(payload, allowed=_CONTINUE_QUESTION_KEYS)
        response_text = str(payload.get("responseText") or "").strip()
        selected_option_id = str(payload.get("selectedOptionId") or "").strip()
        if not response_text and not selected_option_id:
            raise QuestionInterfaceValidationError(
                field="__all__",
                message="responseText or selectedOptionId is required.",
            )
        run_id = str(payload.get("runId") or "").strip()
        if not run_id:
            raise QuestionInterfaceValidationError(
                field="runId",
                message="runId is required.",
            )
        clarification_id = str(payload.get("clarificationId") or "").strip()
        if not clarification_id:
            raise QuestionInterfaceValidationError(
                field="clarificationId",
                message="clarificationId is required.",
            )
        return ClarificationResponseRequest(
            question_id=str(question_id),
            run_id=run_id,
            clarification_id=clarification_id,
            response_text=response_text,
            principal=_question_principal(principal),
            execution_mode=ExecutionMode.QUEUED,
            selected_option_id=selected_option_id,
        )

    def _rerun_request(
        self,
        question_id: str,
        payload: dict[str, Any],
        *,
        principal: InterfacePrincipal,
        idempotency_key: str | None,
    ) -> RerunQuestionRequest:
        if not isinstance(payload, dict):
            raise QuestionInterfaceValidationError(
                field="__all__",
                message="Payload must be an object.",
            )
        _reject_unknown_fields(payload, allowed=_RERUN_QUESTION_KEYS)
        if payload.get("triggerKind") != RERUN_TRIGGER:
            raise QuestionInterfaceValidationError(
                field="triggerKind",
                message="triggerKind must be rerun.",
            )
        base_run_id = payload.get("baseRunId")
        if not isinstance(base_run_id, str) or not base_run_id.strip():
            raise QuestionInterfaceValidationError(
                field="baseRunId",
                message="baseRunId is required.",
            )
        patch = None
        if "patch" in payload:
            try:
                patch = binding_patch_from_payload(payload["patch"])
            except ValueError as exc:
                raise QuestionInterfaceValidationError(
                    field="patch",
                    message=str(exc),
                ) from exc
        capability_application = None
        if "capabilityApplication" in payload:
            try:
                capability_application = capability_application_from_payload(
                    payload["capabilityApplication"]
                )
            except ValueError as exc:
                raise QuestionInterfaceValidationError(
                    field="capabilityApplication",
                    message=str(exc),
                ) from exc
        if patch is not None and capability_application is not None:
            raise QuestionInterfaceValidationError(
                field="__all__",
                message="Use patch or capabilityApplication, not both.",
            )
        return RerunQuestionRequest(
            question_id=question_id,
            base_run_id=base_run_id,
            patch=patch,
            capability_application=capability_application,
            principal=_question_principal(principal),
            execution_mode=ExecutionMode.QUEUED,
            idempotency_key=idempotency_key,
        )


def _reject_unknown_fields(
    payload: dict[str, Any],
    *,
    allowed: frozenset[str] = _CREATE_QUESTION_KEYS,
) -> None:
    unknown = sorted(str(key) for key in payload if str(key) not in allowed)
    if unknown:
        field = unknown[0]
        raise QuestionInterfaceValidationError(
            field=field,
            message=f"{field} is not a supported field.",
            code="unknown",
        )


def _question_principal(principal: InterfacePrincipal) -> QuestionPrincipal:
    return QuestionPrincipal(
        principal_id=principal.principal_id,
        tenant_id=principal.tenant_id,
        raw=principal.raw,
        read_context_ref=principal.read_context_ref,
        delegated_credential=principal.delegated_credential,
    )


def _optional_string(payload: dict[str, Any], key: str) -> str:
    if key not in payload:
        return ""
    value = str(payload.get(key) or "").strip()
    if not value:
        raise QuestionInterfaceValidationError(
            field=key,
            message=f"{key} must not be empty.",
        )
    return value


def _optional_identifier(payload: dict[str, Any], key: str) -> str | None:
    if key not in payload:
        return None
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise QuestionInterfaceValidationError(
            field=key,
            message=f"{key} must be a non-empty string.",
        )
    return value.strip()


def _optional_float(payload: dict[str, Any], key: str) -> float | None:
    if key not in payload or payload.get(key) in {None, ""}:
        return None
    try:
        return float(payload[key])
    except (TypeError, ValueError) as exc:
        raise QuestionInterfaceValidationError(
            field=key,
            message=f"{key} must be a number.",
        ) from exc


def _optional_int(payload: dict[str, Any], key: str) -> int | None:
    if key not in payload or payload.get(key) in {None, ""}:
        return None
    try:
        return int(payload[key])
    except (TypeError, ValueError) as exc:
        raise QuestionInterfaceValidationError(
            field=key,
            message=f"{key} must be an integer.",
        ) from exc


def _ask_request_error_field(message: str) -> str:
    if message.startswith("max_budget_usd"):
        return "maxBudgetUsd"
    if message.startswith("max_thinking_tokens"):
        return "maxThinkingTokens"
    if message.startswith("modelKey") or "model policy" in message:
        return "modelKey"
    if message.startswith("provider"):
        return "provider"
    return "__all__"


def _require_admitted_question_state(
    question: dict[str, Any] | None,
) -> dict[str, Any]:
    if question is None:
        raise RuntimeError("admitted question is missing its persisted projection")
    return question


def _active_conflict_payload(result: AskResult) -> dict[str, Any]:
    return {
        "error": {
            "type": "conflict",
            "code": "fervis_question_already_active",
            "message": "This conversation already has an active Fervis question run.",
            "retryable": True,
            "details": [],
            "context": {
                "questionId": result.question_id,
                "activeRunId": result.active_run_id or result.run_id,
            },
        }
    }


def _not_found_payload(resource: str, identifier: str) -> dict[str, Any]:
    return {
        "error": {
            "type": "not_found",
            "code": f"{resource}_not_found",
            "message": f"{resource} was not found.",
            "retryable": False,
            "details": [],
            "context": {"id": identifier},
        }
    }


def _with_next_actions(
    payload: dict[str, Any],
    principal: InterfacePrincipal,
) -> dict[str, Any]:
    del principal
    if str(payload.get("status") or "") != "WAITING_FOR_CLARIFICATION":
        return payload
    conversation_id = str(payload.get("conversationId") or "")
    if not conversation_id:
        return payload
    question_id = str(payload.get("questionId") or "")
    run_id = str(
        payload.get("latestRunId")
        or payload.get("primaryRunId")
        or payload.get("runId")
        or ""
    )
    clarification_id = _first_clarification_id(payload)
    return {
        **payload,
        "nextActions": [
            _clarification_next_action(
                conversation_id=conversation_id,
                question_id=question_id,
                run_id=run_id,
                clarification_id=clarification_id,
            )
        ],
    }


def _clarification_next_action(
    *,
    conversation_id: str,
    question_id: str,
    run_id: str,
    clarification_id: str,
) -> dict[str, object]:
    return {
        "kind": "provide_clarification",
        "questionId": question_id,
        "conversationId": conversation_id,
        "runId": run_id,
        "clarificationId": clarification_id,
        "request": {
            "method": "POST",
            "path": f"/questions/{question_id}/runs/",
            "body": {
                "responseText": "<clarification-answer>",
                "runId": run_id,
                "clarificationId": clarification_id,
                "selectedOptionId": "<selected-option-id>",
            },
        },
    }


def _first_clarification_id(payload: dict[str, Any]) -> str:
    result_data = payload.get("resultData")
    if not isinstance(result_data, dict):
        return ""
    clarifications = result_data_clarifications(result_data)
    if not clarifications:
        return ""
    first = clarifications[0]
    if not isinstance(first, dict):
        return ""
    return str(first.get("id") or "")
