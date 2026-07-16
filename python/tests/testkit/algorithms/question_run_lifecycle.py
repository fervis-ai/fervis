from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any
from fervis.lineage.enums import ProgramInvocationKind

from fervis.questions import (
    AskRequest,
    ClarificationResponseRequest,
    ExecutionMode,
    QuestionPrincipal,
    RerunQuestionRequest,
)
from fervis.lookup.clarification.payload import clarification_from_payload
from fervis.questions.contracts import QuestionLifecycleError
from fervis.lookup.answer_program.codec import answer_program_id, decode_answer_program
from fervis.lookup.answer_program.persistence import (
    StoredProgramInvocation,
    parse_stored_program_invocation,
    program_invocation,
)
from fervis.questions.service import QuestionService, clarification_successor_run
from fervis.lookup.clarification.model import ConversationResolutionResponse
from fervis.questions.execution_specs import execution_spec_kind
from fervis.run_work import FailQueuedRunRequest, QueuedRunRequest
from fervis.questions.ports import (
    AuthorizedQuestionAccess,
    ClarificationRunResponse,
    LookupExecutionRequest,
    LookupExecutionResult,
    ParsedQuestionRunSubmission,
    QuestionRunRecord,
    QuestionRunSubmissionKind,
    QuestionRunSubmissionResult,
    QueuedRun,
    ResolveQuestionRunSpec,
    RunSubmission,
)
from fervis.lookup.clarification.response import (
    clarification_response_payload,
    parse_clarification_response,
)
from tests.testkit.answer_program_contracts import (
    binding_patch_from_payload,
    binding_set_from_payload,
    capability_application_from_payload,
)
from tests.testkit.assertions import subset_mismatches


def run_question_run_lifecycle_case(payload: dict[str, Any]) -> list[str]:
    input_payload = payload["input"]
    if input_payload.get("schema_revision") == 1:
        operation = str(input_payload["operation"])
        if operation == "rerun":
            return _run_portable_rerun_case(payload)
        if operation == "rerun_eligibility":
            return _run_portable_rerun_eligibility_case(payload)
        return [f"unsupported portable question lifecycle operation: {operation}"]
    scenario = str(input_payload["scenario"])
    if scenario == "ask":
        return _run_ask_case(payload)
    if scenario == "clarification_resume":
        return _run_clarification_resume_case(payload)
    return [f"unsupported question-run lifecycle scenario: {scenario}"]


def _run_clarification_resume_case(payload: dict[str, Any]) -> list[str]:
    input_payload = payload["input"]
    lineage = _InMemoryLineage()
    runs = _InMemoryRuns(lineage=lineage)
    clarification_payload = dict(input_payload["clarification"])
    lookup = _FakeLookup(
        {
            "status": "WAITING_FOR_CLARIFICATION",
            "result_data": {
                "kind": "needs_clarification",
                "details": {"clarifications": [clarification_payload]},
            },
        }
    )
    service = QuestionService(
        lineage=lineage,
        runs=runs,
        lookup=lookup,
        program=lookup,
        ids=_DeterministicIds(),
        adapter_ref="conformance",
        runtime_version="test-runtime",
    )
    first = service.ask(
        AskRequest(
            question=str(input_payload["question"]),
            principal=_principal(input_payload),
            execution_mode=ExecutionMode.INLINE,
        )
    )
    runs.pending_clarifications[str(clarification_payload["id"])] = (
        clarification_payload
    )
    lookup.result_payload = dict(input_payload["resumed_lookup_result"])
    resumed = service.respond_to_clarification(
        ClarificationResponseRequest(
            question_id=first.question_id,
            run_id=first.run_id,
            clarification_id=str(clarification_payload["id"]),
            response_text=str(input_payload["response_text"]),
            principal=_principal(input_payload),
            execution_mode=ExecutionMode.INLINE,
            selected_option_id=str(input_payload.get("selected_option_id") or ""),
        )
    )
    actual = {
        "first_result": _result_payload(first),
        "resumed_result": _result_payload(resumed),
        "suspended_run_status": runs.runs[first.run_id].status,
        "run_count": lineage.run_count,
        "question_count": lineage.question_count,
        "clarification_response_count": runs.clarification_response_count,
        "lookup_call_count": len(lookup.calls),
        "resumed_original_question": (
            lookup.calls[-1].question if lookup.calls else None
        ),
        "resumed_has_clarification_response": bool(
            lookup.calls and lookup.calls[-1].clarification_responses
        ),
        "resumed_clarification": _resumed_clarification_summary(lookup),
    }
    return subset_mismatches(
        actual=actual,
        expected_subset=payload["expect"]["result_contains"],
    )


def _resumed_clarification_summary(lookup: "_FakeLookup") -> dict[str, object] | None:
    if not lookup.calls or not lookup.calls[-1].clarification_responses:
        return None
    response = lookup.calls[-1].clarification_responses[-1]
    payload = clarification_response_payload(response)
    kind = str(payload["kind"])
    summary: dict[str, object] = {
        "owner": kind.removesuffix("_catalog_input")
        .removesuffix("_identity")
        .removesuffix("_text"),
        "continuation_kind": kind,
        "dispatch_count": 1,
    }
    if kind in {"conversation_resolution", "question_contract"}:
        summary["attributed_source"] = payload["source"]
    if kind == "conversation_resolution" and payload.get("annotation") is not None:
        summary["annotation"] = payload["annotation"]
    if kind == "grounding_identity":
        summary["closed_target"] = {
            "knownInputId": payload["knownInputId"],
            "optionId": dict(payload["option"])["id"],
        }
    if kind.endswith("_catalog_input"):
        target = dict(payload["target"])
        summary["closed_target"] = {
            "rowSourceId": target["row_source_id"],
            "paramId": target["param_id"],
            "valueType": target["value_type"],
            "value": payload["value"],
        }
    return summary


def _run_portable_rerun_eligibility_case(
    payload: dict[str, Any],
) -> list[str]:
    outcomes: list[dict[str, Any]] = []
    for attempt in payload["input"]["attempts"]:
        service, runs, lineage, execution = _portable_rerun_state(attempt["state"])
        initial_run_count = lineage.run_count
        initial_invocation_count = len(runs.stored_invocations)
        initial_revision_count = len(runs.revision_ids)
        initial_queue_count = len(runs.runs)
        try:
            service.rerun_question(_portable_rerun_request(attempt["command"]))
        except QuestionLifecycleError as exc:
            outcome = {
                "id": str(attempt["id"]),
                "status": "rejected",
                "code": exc.code,
            }
        else:
            outcome = {"id": str(attempt["id"]), "status": "accepted"}
        outcome.update(
            {
                "new_runs": lineage.run_count - initial_run_count,
                "new_invocations": (
                    len(runs.stored_invocations) - initial_invocation_count
                ),
                "new_revisions": len(runs.revision_ids) - initial_revision_count,
                "queued_work": len(runs.runs) - initial_queue_count,
                "model_calls": len(execution.calls),
                "program_calls": execution.program_call_count,
                "memory_calls": lineage.memory_call_count,
                "read_calls": execution.read_call_count,
            }
        )
        outcomes.append(outcome)
    return subset_mismatches(
        actual={"outcomes": outcomes},
        expected_subset=payload["expect"]["result_contains"],
    )


def _run_portable_rerun_case(payload: dict[str, Any]) -> list[str]:
    input_payload = payload["input"]
    service, runs, lineage, execution = _portable_rerun_state(input_payload["state"])
    result = service.rerun_question(_portable_rerun_request(input_payload["command"]))
    created = runs.runs[result.run_id]
    invocation = runs.stored_invocations[result.run_id]
    base_run_id = str(input_payload["command"]["base_run_id"])
    base_before = runs.stored_invocations.get(base_run_id)
    after_enqueue = input_payload.get("after_enqueue") or {}
    if after_enqueue.get("revoke_question"):
        runs.questions.pop(str(input_payload["command"]["question_id"]), None)
    executed = None
    if after_enqueue.get("process"):
        executed = service.run_work.process_queued_run(
            QueuedRunRequest(
                run_id=result.run_id,
                worker_id=str(after_enqueue.get("worker_id") or "worker_1"),
                active_attempt=int(after_enqueue.get("active_attempt") or 1),
            )
        )
    actual = {
        "result": _result_payload(result),
        "execution_result": _result_payload(executed),
        "question_count": lineage.question_count,
        "run_count": lineage.run_count,
        "run": {
            "kind": execution_spec_kind(created.submission.spec).value,
            "lineage_kind": lineage.last_run_kind,
            "trigger_kind": lineage.trigger_kind,
            "base_run_id": lineage.base_run_id,
            "program_id": invocation.invocation.program_id,
            "patch_id_present": invocation.invocation.patch_id is not None,
            "revision_id_present": invocation.invocation.revision_id is not None,
            "binding_ids": list(invocation.bindings.parameter_ids),
        },
        "memory_call_count": lineage.memory_call_count,
        "model_call_count": len(execution.calls),
        "program_call_count": execution.program_call_count,
        "base_run_unchanged": runs.stored_invocations.get(base_run_id) is base_before,
    }
    return subset_mismatches(
        actual=actual,
        expected_subset=payload["expect"]["result_contains"],
    )


def _portable_rerun_state(
    payload: dict[str, Any],
) -> tuple[QuestionService, "_InMemoryRuns", "_InMemoryLineage", "_FakeLookup"]:
    lineage = _InMemoryLineage(
        question_count=int(payload.get("question_count") or 0),
        run_count=int(payload.get("run_count") or 0),
    )
    runs = _InMemoryRuns(lineage=lineage)
    for question in payload.get("questions") or ():
        principal = _principal(question["principal"])
        runs.questions[str(question["question_id"])] = AuthorizedQuestionAccess._issue(
            question_id=str(question["question_id"]),
            conversation_id=str(question["conversation_id"]),
            tenant_id=principal.tenant_id,
            original_question=str(question["original_question"]),
            read_context_ref=principal.read_context_ref,
        )
    for base in payload.get("program_invocations") or ():
        program = decode_answer_program(base["program"])
        bindings = binding_set_from_payload(base)
        invocation = program_invocation(
            run_id=str(base["run_id"]),
            program_id=answer_program_id(program),
            bindings=bindings,
            kind=ProgramInvocationKind(str(base["kind"])),
            base_invocation_id=(
                str(base["base_invocation_id"])
                if base.get("base_invocation_id") is not None
                else None
            ),
        )
        runs.stored_invocations[str(base["run_id"])] = StoredProgramInvocation(
            invocation=invocation,
            program=program,
        )
    execution = _FakeLookup(
        {},
        program_result_payload=dict(payload.get("program_result") or {}),
    )
    service = QuestionService(
        lineage=lineage,
        runs=runs,
        lookup=execution,
        program=execution,
        ids=_DeterministicIds(),
        adapter_ref="conformance",
        runtime_version="test-runtime",
    )
    return service, runs, lineage, execution


def _portable_rerun_request(payload: dict[str, Any]) -> RerunQuestionRequest:
    return RerunQuestionRequest(
        question_id=str(payload["question_id"]),
        base_run_id=str(payload["base_run_id"]),
        patch=(
            binding_patch_from_payload(payload["patch"]) if "patch" in payload else None
        ),
        capability_application=(
            capability_application_from_payload(payload["capability_application"])
            if "capability_application" in payload
            else None
        ),
        principal=_principal(payload["principal"]),
        execution_mode=ExecutionMode.QUEUED,
        idempotency_key=payload.get("idempotency_key"),
    )


def _principal(payload: dict[str, Any]) -> QuestionPrincipal:
    return QuestionPrincipal(
        principal_id=str(payload["principal_id"]),
        tenant_id=str(payload["tenant_id"]),
    )


def _run_ask_case(payload: dict[str, Any]) -> list[str]:
    input_payload = payload["input"]
    lineage = _InMemoryLineage(
        memory_by_conversation=_memory_by_conversation(input_payload),
        fail_memory_after_calls=input_payload.get("fail_memory_after_calls"),
    )
    runs = _InMemoryRuns(lineage=lineage)
    lookup = _FakeLookup(input_payload.get("lookup_result") or {})
    service = QuestionService(
        lineage=lineage,
        runs=runs,
        lookup=lookup,
        program=lookup,
        ids=_DeterministicIds(),
        adapter_ref="conformance",
        runtime_version="test-runtime",
    )
    model = dict(input_payload.get("model") or {})
    result = _ask(service, input_payload, model=model)
    repeated_result = None
    if int(input_payload.get("repeat_count") or 1) > 1:
        repeated_result = _ask(service, input_payload, model=model)
    queued_result = None
    second_queued_result = None
    failed_queued_result = None
    if input_payload.get("process_queued_run"):
        execute_count = int(input_payload.get("process_queued_run_count") or 1)
        queued_result = service.run_work.process_queued_run(
            QueuedRunRequest(
                run_id=result.run_id,
                worker_id=str(input_payload.get("worker_id") or "worker_1"),
                active_attempt=int(input_payload.get("active_attempt") or 1),
            )
        )
        if execute_count > 1:
            second_queued_result = service.run_work.process_queued_run(
                QueuedRunRequest(
                    run_id=result.run_id,
                    worker_id=str(input_payload.get("worker_id") or "worker_1"),
                    active_attempt=int(input_payload.get("active_attempt") or 1),
                )
            )
    if input_payload.get("fail_queued_run"):
        failed_queued_result = service.run_work.fail_queued_run(
            FailQueuedRunRequest(
                run_id=result.run_id,
                worker_id=str(input_payload.get("worker_id") or "worker_1"),
                active_attempt=int(input_payload.get("active_attempt") or 1),
                error=str(input_payload.get("worker_error") or "worker_failed"),
            )
        )
    actual = {
        "ask_result": _result_payload(result),
        "repeated_ask_result": _result_payload(repeated_result),
        "queued_result": _result_payload(queued_result),
        "second_queued_result": _result_payload(second_queued_result),
        "failed_queued_result": _result_payload(failed_queued_result),
        "lineage": lineage.summary(),
        "queue": runs.summary(),
        "lookup": lookup.summary(),
    }
    return subset_mismatches(
        actual=actual,
        expected_subset=payload["expect"]["result_contains"],
    )


def _ask(
    service: QuestionService,
    input_payload: dict[str, Any],
    *,
    model: dict[str, Any],
):
    return service.ask(
        AskRequest(
            question=str(input_payload["question"]),
            principal=QuestionPrincipal(
                principal_id=str(input_payload["principal_id"]),
                tenant_id=str(input_payload["tenant_id"]),
                raw=input_payload.get("principal_raw"),
            ),
            execution_mode=input_payload["execution_mode"],
            conversation_id=str(input_payload.get("conversation_id") or ""),
            provider=model.get("provider"),
            model_key=str(model.get("model_key") or ""),
            idempotency_key=input_payload.get("idempotency_key"),
            max_budget_usd=input_payload.get("max_budget_usd"),
            max_thinking_tokens=input_payload.get("max_thinking_tokens"),
            runtime_context=dict(input_payload.get("runtime_context") or {}),
        )
    )


def _result_payload(result: Any | None) -> dict[str, Any] | None:
    return asdict(result) if result is not None else None


def _memory_by_conversation(
    input_payload: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    memory_context = input_payload.get("conversation_memory_context")
    if not memory_context:
        return {}
    tenant_id = str(input_payload["tenant_id"])
    conversation_id = str(input_payload.get("conversation_id") or "conversation_1")
    return {(tenant_id, conversation_id): dict(memory_context)}


@dataclass
class _DeterministicIds:
    conversation_count: int = 0
    question_count: int = 0
    run_count: int = 0
    clarification_response_count: int = 0

    def new_conversation_id(self) -> str:
        self.conversation_count += 1
        return f"conversation_{self.conversation_count}"

    def new_question_id(self) -> str:
        self.question_count += 1
        return f"question_{self.question_count}"

    def new_run_id(self) -> str:
        self.run_count += 1
        return f"run_{self.run_count}"

    def new_clarification_response_id(self) -> str:
        self.clarification_response_count += 1
        return f"clarification_response_{self.clarification_response_count}"


@dataclass
class _InMemoryLineage:
    conversations: set[tuple[str, str]] = field(default_factory=set)
    question_counts_by_conversation: dict[str, int] = field(default_factory=dict)
    question_count: int = 0
    run_count: int = 0
    failed_runtime_fallback_count: int = 0
    first_question_sequence: int | None = None
    trigger_kind: str | None = None
    last_run_kind: str | None = None
    base_run_id: str | None = None
    memory_by_conversation: dict[tuple[str, str], dict[str, Any]] = field(
        default_factory=dict
    )
    fail_memory_after_calls: int | None = None
    memory_call_count: int = 0

    def ensure_conversation(self, *, conversation_id: str, tenant_id: str) -> None:
        self.conversations.add((tenant_id, conversation_id))

    def conversation_memory_context(
        self,
        *,
        conversation_id: str,
        authority,
        context_run_id: str | None = None,
        continuation_run_id: str | None = None,
    ) -> dict[str, Any]:
        del context_run_id, continuation_run_id
        self.memory_call_count += 1
        if (
            self.fail_memory_after_calls is not None
            and self.memory_call_count > self.fail_memory_after_calls
        ):
            raise RuntimeError("conversation memory should not be read")
        return dict(
            self.memory_by_conversation.get((authority.tenant_id, conversation_id))
            or {}
        )

    def record_question_run(self, record: QuestionRunRecord) -> None:
        if record.question is not None:
            sequence = (
                self.question_counts_by_conversation.get(
                    record.question.conversation_id,
                    0,
                )
                + 1
            )
            self.question_counts_by_conversation[record.question.conversation_id] = (
                sequence
            )
            self.question_count += 1
            self.first_question_sequence = self.first_question_sequence or sequence
        self.run_count += 1
        self.trigger_kind = record.run.trigger_kind.value
        self.last_run_kind = record.run.kind.value
        self.base_run_id = record.run.base_run_id

    def record_failed_runtime_fallback(
        self,
        *,
        run_id: str,
        status: str,
        answer: str | None,
        result_data: dict[str, Any] | None,
        error: str | None,
    ) -> None:
        del run_id, status, answer, result_data, error
        self.failed_runtime_fallback_count += 1

    def summary(self) -> dict[str, Any]:
        return {
            "conversation_count": len(self.conversations),
            "question_count": self.question_count,
            "run_count": self.run_count,
            "failed_runtime_fallback_count": self.failed_runtime_fallback_count,
            "memory_call_count": self.memory_call_count,
            "first_question_sequence": self.first_question_sequence,
            "trigger_kind": self.trigger_kind,
        }


@dataclass
class _InMemoryRuns:
    lineage: _InMemoryLineage
    runs: dict[str, QueuedRun] = field(default_factory=dict)
    questions: dict[str, AuthorizedQuestionAccess] = field(default_factory=dict)
    stored_invocations: dict[str, StoredProgramInvocation] = field(default_factory=dict)
    revision_ids: set[str] = field(default_factory=set)
    pending_clarifications: dict[str, dict[str, Any]] = field(default_factory=dict)
    clarification_response_count: int = 0

    def get_question(
        self,
        *,
        question_id: str,
        authority,
    ) -> AuthorizedQuestionAccess | None:
        question = self.questions.get(question_id)
        if question is None or question.tenant_id != authority.tenant_id:
            return None
        return question

    def authorize_conversation(self, *, conversation_id: str, authority) -> None:
        del conversation_id, authority

    def load_answered_program_invocation(
        self,
        *,
        run_id: str,
        access: AuthorizedQuestionAccess,
    ) -> StoredProgramInvocation | None:
        if access.question_id not in self.questions:
            return None
        return self.stored_invocations.get(run_id)

    def load_program_invocation_for_execution(
        self,
        *,
        invocation_id: str,
        run_id: str,
        question_id: str,
        tenant_id: str,
    ) -> StoredProgramInvocation | None:
        invocation = self.stored_invocations.get(run_id)
        question = self.questions.get(question_id)
        if (
            invocation is None
            or invocation.invocation.invocation_id != invocation_id
            or question is None
            or question.tenant_id != tenant_id
        ):
            return None
        return invocation

    def find_idempotent_run(
        self,
        *,
        principal,
        conversation_id: str | None,
        idempotency_key: str | None,
    ) -> QueuedRun | None:
        return self._idempotent_run(
            tenant_id=principal.tenant_id,
            principal_id=principal.principal_id,
            read_context_ref=principal.read_context_ref,
            conversation_id=conversation_id,
            idempotency_key=idempotency_key,
            idempotency_scope=(
                f"conversation:{conversation_id}"
                if conversation_id is not None
                else "new_conversation"
            ),
        )

    def submit_question_run_atomically(
        self,
        *,
        submission: RunSubmission,
        record: QuestionRunRecord,
    ) -> QuestionRunSubmissionResult:
        parsed = ParsedQuestionRunSubmission(submission=submission, record=record)
        submission = parsed.submission
        record = parsed.record
        existing = self._idempotent_run(
            tenant_id=submission.tenant_id,
            principal_id=submission.principal.principal_id,
            read_context_ref=submission.principal.read_context_ref,
            conversation_id=submission.conversation_id,
            idempotency_key=submission.idempotency_key,
            idempotency_scope=submission.idempotency_scope,
        )
        if existing is not None:
            return QuestionRunSubmissionResult(
                kind=QuestionRunSubmissionKind.EXISTING,
                run=existing,
            )
        active = self._active_run(
            tenant_id=submission.tenant_id,
            conversation_id=submission.conversation_id,
            read_context_ref=submission.principal.read_context_ref,
        )
        if active is not None:
            return QuestionRunSubmissionResult(
                kind=QuestionRunSubmissionKind.ACTIVE_CONFLICT,
                run=active,
            )
        if record.question is not None:
            self.lineage.ensure_conversation(
                conversation_id=record.question.conversation_id,
                tenant_id=record.question.tenant_id,
            )
            self.questions[record.question.question_id] = (
                AuthorizedQuestionAccess._issue(
                    question_id=record.question.question_id,
                    conversation_id=record.question.conversation_id,
                    tenant_id=record.question.tenant_id,
                    original_question=record.question.question,
                    read_context_ref=submission.principal.read_context_ref,
                )
            )
        self.lineage.record_question_run(record)
        if record.program_revision is not None:
            self.revision_ids.add(record.program_revision.revision.revision_id)
        if record.program_invocation is not None:
            bundle = record.program_invocation
            self.stored_invocations[submission.run_id] = (
                parse_stored_program_invocation(
                    invocation_id=bundle.invocation.invocation_id,
                    run_id=bundle.invocation.run_id,
                    program_id=bundle.invocation.program_id,
                    canonical_json=bundle.program.canonical_json,
                    bindings_json=bundle.invocation.bindings_json,
                    kind=bundle.invocation.kind.value,
                    base_invocation_id=bundle.invocation.base_invocation_id,
                    patch_id=bundle.invocation.patch_id,
                    binding_patch_json=bundle.invocation.binding_patch_json,
                    revision_id=bundle.invocation.revision_id,
                )
            )
        run = QueuedRun(
            submission=submission,
            status="RUNNING"
            if submission.execution_mode is ExecutionMode.INLINE
            else "QUEUED",
        )
        self.runs[submission.run_id] = run
        return QuestionRunSubmissionResult(
            kind=QuestionRunSubmissionKind.CREATED,
            run=run,
        )

    def respond_to_clarification_atomically(
        self,
        resume: ClarificationRunResponse,
    ) -> QuestionRunSubmissionResult:
        current = self.runs[resume.run_id]
        payload = self.pending_clarifications[resume.clarification_id]
        spec = current.submission.spec
        if not isinstance(spec, ResolveQuestionRunSpec):
            raise ValueError("clarification can resume only a question lookup")
        response = parse_clarification_response(
            clarification_from_payload(payload),
            response_id=resume.response_id,
            response_text=resume.response_text,
            selected_option_id=resume.selected_option_id,
            suspended_question_text=spec.question,
        )
        if (
            isinstance(response, ConversationResolutionResponse)
            and response.annotation is not None
        ):
            submission, record = clarification_successor_run(
                current.submission,
                response=resume,
                annotation=response,
            )
            self.runs[resume.run_id] = replace(current, status="SUPERSEDED")
            self.lineage.record_question_run(record)
            active_attempt = 1
        else:
            submission = replace(
                current.submission,
                spec=replace(
                    spec,
                    clarification_responses=(
                        *spec.clarification_responses,
                        response,
                    ),
                ),
                execution_mode=resume.execution_mode,
            )
            active_attempt = 2
        queued = QueuedRun(
            submission=submission,
            status=(
                "RUNNING" if resume.execution_mode is ExecutionMode.INLINE else "QUEUED"
            ),
            active_attempt=active_attempt,
        )
        self.runs[submission.run_id] = queued
        self.clarification_response_count += 1
        return QuestionRunSubmissionResult(
            kind=QuestionRunSubmissionKind.CREATED,
            run=queued,
        )

    def wait_for_clarification(
        self,
        *,
        run_id: str,
        worker_id: str,
        active_attempt: int,
    ) -> QueuedRun:
        del worker_id, active_attempt
        current = self.runs[run_id]
        clarifications = list(self.pending_clarifications.values())
        result_data = {
            "kind": "needs_clarification",
            "details": {"clarifications": clarifications},
        }
        waiting = replace(
            current,
            status="WAITING_FOR_CLARIFICATION",
            result_data=result_data,
        )
        self.runs[run_id] = waiting
        return waiting

    def _idempotent_run(
        self,
        *,
        tenant_id: str,
        principal_id: str,
        read_context_ref,
        conversation_id: str | None,
        idempotency_key: str | None,
        idempotency_scope: str,
    ) -> QueuedRun | None:
        if not idempotency_key:
            return None
        for run in self.runs.values():
            submission = run.submission
            if (
                submission.tenant_id == tenant_id
                and submission.principal.principal_id == principal_id
                and submission.principal.read_context_ref == read_context_ref
                and (
                    conversation_id is None
                    or submission.conversation_id == conversation_id
                )
                and submission.idempotency_key == idempotency_key
                and submission.idempotency_scope == idempotency_scope
            ):
                return run
        return None

    def _active_run(
        self,
        *,
        tenant_id: str,
        conversation_id: str,
        read_context_ref,
    ) -> QueuedRun | None:
        for run in self.runs.values():
            if run.status not in {"QUEUED", "RUNNING"}:
                continue
            submission = run.submission
            if (
                submission.tenant_id == tenant_id
                and submission.conversation_id == conversation_id
                and submission.principal.read_context_ref == read_context_ref
            ):
                return run
        return None

    def load_executable_run(
        self,
        *,
        run_id: str,
        worker_id: str,
        active_attempt: int | None,
    ) -> QueuedRun:
        del worker_id, active_attempt
        run = self.runs[run_id]
        if run.status == "QUEUED":
            running = QueuedRun(submission=run.submission, status="RUNNING")
            self.runs[run_id] = running
            return running
        return run

    def load_failable_run(
        self,
        *,
        run_id: str,
        worker_id: str,
        active_attempt: int | None,
    ) -> QueuedRun:
        del worker_id, active_attempt
        return self.runs[run_id]

    def terminalize(
        self,
        *,
        run_id: str,
        status: str,
        answer: str | None,
        result_data: dict[str, Any] | None,
        error: str | None,
        worker_id: str = "",
        active_attempt: int | None = None,
    ) -> QueuedRun:
        del worker_id, active_attempt
        current = self.runs[run_id]
        terminal = QueuedRun(
            submission=current.submission,
            status=status,
            answer=answer,
            result_data=result_data,
            error=error,
        )
        self.runs[run_id] = terminal
        return terminal

    def summary(self) -> dict[str, Any]:
        return {
            "enqueued_count": len(self.runs),
            "terminal_count": sum(
                1
                for run in self.runs.values()
                if run.status not in {"QUEUED", "RUNNING"}
            ),
        }


@dataclass
class _FakeLookup:
    result_payload: dict[str, Any]
    program_result_payload: dict[str, Any] = field(default_factory=dict)
    calls: list[LookupExecutionRequest] = field(default_factory=list)
    program_call_count: int = 0
    read_call_count: int = 0

    def run_lookup(
        self,
        request: LookupExecutionRequest,
        *,
        progress_sink=None,
    ) -> LookupExecutionResult:
        del progress_sink
        self.calls.append(request)
        if self.result_payload.get("raise_error"):
            raise RuntimeError(str(self.result_payload.get("error") or "lookup_failed"))
        terminal_lineage_recorded = self.result_payload.get(
            "terminal_lineage_recorded",
            True,
        )
        return LookupExecutionResult(
            status=str(self.result_payload.get("status") or "COMPLETED"),
            answer=self.result_payload.get("answer"),
            result_data=self.result_payload.get("result_data"),
            error=self.result_payload.get("error"),
            terminal_lineage_recorded=bool(terminal_lineage_recorded),
        )

    def run_program(self, request, *, progress_sink=None) -> LookupExecutionResult:
        del request, progress_sink
        self.program_call_count += 1
        self.read_call_count += 1
        if not self.program_result_payload:
            raise RuntimeError(
                "question lifecycle fixture did not configure a program run"
            )
        return LookupExecutionResult(
            status=str(self.program_result_payload.get("status") or "COMPLETED"),
            answer=self.program_result_payload.get("answer"),
            result_data=self.program_result_payload.get("result_data"),
            error=self.program_result_payload.get("error"),
            terminal_lineage_recorded=bool(
                self.program_result_payload.get("terminal_lineage_recorded", True)
            ),
        )

    def summary(self) -> dict[str, Any]:
        last = self.calls[-1] if self.calls else None
        return {
            "call_count": len(self.calls),
            "last_question": last.question if last is not None else None,
            "last_principal": last.principal if last is not None else None,
            "last_conversation_id": last.conversation_id if last is not None else None,
            "last_conversation_context": (
                last.conversation_context if last is not None else None
            ),
            "last_provider": last.provider if last is not None else None,
            "last_model_key": last.model_key if last is not None else None,
            "last_runtime_context": last.runtime_context if last is not None else None,
            "last_max_budget_usd": (
                str(last.max_budget_usd) if last is not None else None
            ),
            "last_max_thinking_tokens": (
                last.max_thinking_tokens if last is not None else None
            ),
            "last_active_attempt": last.active_attempt if last is not None else None,
        }
