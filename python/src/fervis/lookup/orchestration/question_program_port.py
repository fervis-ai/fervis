"""Question-runtime adapter for model-free answer-program execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from fervis.lookup.answer_rendering import rendered_fact_payload
from fervis.lookup.orchestration.program_service import (
    AnswerProgramService,
    StoredProgramRunRequest,
)
from fervis.lookup.orchestration.result import delivery_result_data
from fervis.questions.ports import (
    LookupExecutionResult,
    ProgramExecutionRequest,
    QuestionProgramPort,
)


ProgramTerminalLineageCheck = Callable[[ProgramExecutionRequest], bool]


@dataclass(frozen=True)
class AnswerProgramQuestionPort(QuestionProgramPort):
    program_service: AnswerProgramService
    terminal_lineage_recorded: ProgramTerminalLineageCheck

    def run_program(
        self,
        request: ProgramExecutionRequest,
        *,
        progress_sink=None,
    ) -> LookupExecutionResult:
        del progress_sink
        result = self.program_service.run_program(
            StoredProgramRunRequest(
                run_id=request.run_id,
                conversation_id=request.conversation_id,
                tenant_id=request.tenant_id,
                question=request.question,
                read_context_ref=request.read_context_ref,
                delegated_credential=request.delegated_credential,
                principal=request.principal,
                invocation=request.invocation,
                runtime_context=dict(request.runtime_context),
                active_attempt=request.active_attempt,
            )
        )
        result_data = (
            delivery_result_data(rendered_fact_payload(result.rendered_fact))
            if result.rendered_fact is not None
            else delivery_result_data(result.result_data)
        )
        return LookupExecutionResult(
            status=result.status,
            answer=result.answer,
            result_data=result_data,
            error=result.error,
            usage=dict(result.usage or {}),
            terminal_lineage_recorded=self.terminal_lineage_recorded(request),
        )
