from __future__ import annotations

from dataclasses import dataclass, field

from fervis.host_api.contracts.authority import ReadContextRef
from fervis.lookup.answer_program import AnswerProgram, BindingSet, answer_program_id
from fervis.lookup.answer_program.persistence import (
    StoredProgramInvocation,
    program_invocation,
)
from fervis.questions import ExecutionMode, QuestionPrincipal
from fervis.questions.ports import (
    AuthorizedQuestionAccess,
    DeterministicRunSpec,
    LookupExecutionResult,
    QueuedRun,
    RunSubmission,
)
from fervis.run_work import QueuedRunRequest
from fervis.run_work.service import RunWorkService


def test_queued_deterministic_run_uses_program_invoker() -> None:
    program = AnswerProgram()
    bindings = BindingSet()
    invocation = program_invocation(
        run_id="run_rerun",
        program_id=answer_program_id(program),
        bindings=bindings,
    )
    stored = StoredProgramInvocation(
        invocation=invocation,
        program=program,
    )
    current_read_context = ReadContextRef(
        scheme="fastapi_principal",
        key="current-user",
    )
    queued = QueuedRun(
        submission=RunSubmission(
            conversation_id="conversation_1",
            tenant_id="tenant_1",
            question_id="question_1",
            run_id="run_rerun",
            principal=QuestionPrincipal(
                principal_id="user_1",
                tenant_id="tenant_1",
                read_context_ref=current_read_context,
            ),
            spec=DeterministicRunSpec(invocation_id=invocation.invocation_id),
            execution_mode=ExecutionMode.QUEUED,
        ),
        status="QUEUED",
    )
    runs = _Runs(queued=queued, stored=stored)
    program_port = _ProgramPort()
    service = RunWorkService(
        lineage=_Lineage(),
        runs=runs,
        lookup=_ForbiddenLookup(),
        program=program_port,
    )

    result = service.process_queued_run(
        QueuedRunRequest(
            run_id="run_rerun",
            worker_id="worker_1",
            active_attempt=1,
        )
    )

    assert (result.status, result.answer) == ("COMPLETED", "3")
    assert len(program_port.requests) == 1
    assert program_port.requests[0].invocation is stored
    assert program_port.requests[0].read_context_ref == current_read_context
    assert runs.terminal.status == "COMPLETED"


def test_queued_deterministic_run_rechecks_current_question_authority() -> None:
    program = AnswerProgram()
    bindings = BindingSet()
    invocation = program_invocation(
        run_id="run_rerun",
        program_id=answer_program_id(program),
        bindings=bindings,
    )
    queued = QueuedRun(
        submission=RunSubmission(
            conversation_id="conversation_1",
            tenant_id="tenant_1",
            question_id="question_1",
            run_id="run_rerun",
            principal=QuestionPrincipal(
                principal_id="user_1",
                tenant_id="tenant_1",
                read_context_ref=ReadContextRef(
                    scheme="fastapi_principal",
                    key="current-user",
                ),
            ),
            spec=DeterministicRunSpec(invocation_id=invocation.invocation_id),
            execution_mode=ExecutionMode.QUEUED,
        ),
        status="QUEUED",
    )
    runs = _Runs(
        queued=queued,
        stored=StoredProgramInvocation(
            invocation=invocation,
            program=program,
        ),
        authorized=False,
    )
    lineage = _RecordingLineage()
    program_port = _ProgramPort()

    result = RunWorkService(
        lineage=lineage,
        runs=runs,
        lookup=_ForbiddenLookup(),
        program=program_port,
    ).process_queued_run(
        QueuedRunRequest(
            run_id="run_rerun",
            worker_id="worker_1",
            active_attempt=1,
        )
    )

    assert result.status == "FAILED"
    assert result.error == "deterministic run question is not authorized"
    assert program_port.requests == []
    assert lineage.failures == [
        {
            "run_id": "run_rerun",
            "status": "FAILED",
            "error": "deterministic run question is not authorized",
        }
    ]


@dataclass
class _Runs:
    queued: QueuedRun
    stored: StoredProgramInvocation
    authorized: bool = True
    terminal: QueuedRun | None = None

    def load_executable_run(self, **_kwargs) -> QueuedRun:
        return self.queued

    def get_question(self, *, question_id, authority):
        if (
            not self.authorized
            or question_id != "question_1"
            or authority.tenant_id != "tenant_1"
        ):
            return None
        return AuthorizedQuestionAccess._issue(
            question_id="question_1",
            conversation_id="conversation_1",
            tenant_id="tenant_1",
            original_question="How many sales?",
            read_context_ref=self.queued.submission.principal.read_context_ref,
        )

    def load_program_invocation_for_execution(self, **kwargs):
        if (
            kwargs["invocation_id"] != self.stored.invocation.invocation_id
            or kwargs["run_id"] != "run_rerun"
        ):
            return None
        return self.stored

    def terminalize(self, *, status, answer, result_data, error, **_kwargs):
        self.terminal = QueuedRun(
            submission=self.queued.submission,
            status=status,
            answer=answer,
            result_data=result_data,
            error=error,
        )
        return self.terminal


class _ForbiddenLookup:
    def run_lookup(self, *_args, **_kwargs):
        raise AssertionError("deterministic run must not call model lookup")


@dataclass
class _ProgramPort:
    requests: list = field(default_factory=list)

    def run_program(self, request, *, progress_sink=None):
        del progress_sink
        self.requests.append(request)
        return LookupExecutionResult(
            status="COMPLETED",
            answer="3",
            result_data={"value": 3},
            terminal_lineage_recorded=True,
        )


class _Lineage:
    def record_failed_runtime_fallback(self, **_kwargs) -> None:
        raise AssertionError("successful deterministic run must not record fallback")


@dataclass
class _RecordingLineage:
    failures: list[dict] = field(default_factory=list)

    def record_failed_runtime_fallback(
        self,
        *,
        run_id,
        status,
        answer,
        result_data,
        error,
    ) -> None:
        del answer, result_data
        self.failures.append(
            {"run_id": run_id, "status": status, "error": error}
        )
