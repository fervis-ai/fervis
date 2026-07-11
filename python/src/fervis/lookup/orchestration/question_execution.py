"""Closed model-assisted execution paths selected by conversation resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeAlias, TypeVar, assert_never

from fervis.lookup.answer_program.persistence import PriorProgramInvocationReader
from fervis.lookup.conversation_resolution.callable_frames import (
    CallableFrameProgram,
    load_callable_frame_program,
)
from fervis.lookup.conversation_resolution.compilation import (
    CompiledConversationResolution,
)
from fervis.memory.conversation_context import ConversationMemoryCardProjection


@dataclass(frozen=True)
class CompileQuestionExecution:
    resolution: CompiledConversationResolution | None


@dataclass(frozen=True)
class ContinuePriorRequestExecution:
    frame: CallableFrameProgram


QuestionExecution: TypeAlias = (
    CompileQuestionExecution | ContinuePriorRequestExecution
)
_Result = TypeVar("_Result")


def parse_question_execution(
    *,
    resolution: CompiledConversationResolution | None,
    memory_projection: ConversationMemoryCardProjection,
    prior_program_invocations: PriorProgramInvocationReader | None,
    conversation_id: str,
    tenant_id: str,
) -> QuestionExecution:
    if resolution is None or resolution.frame_call is None:
        return CompileQuestionExecution(resolution=resolution)
    if prior_program_invocations is None or not conversation_id or not tenant_id:
        raise ValueError("callable prior frame execution is unavailable")
    return ContinuePriorRequestExecution(
        frame=load_callable_frame_program(
            resolution=resolution,
            memory_projection=memory_projection,
            reader=prior_program_invocations,
            conversation_id=conversation_id,
            tenant_id=tenant_id,
        )
    )


def fold_question_execution(
    execution: QuestionExecution,
    *,
    compile_question: Callable[[CompileQuestionExecution], _Result],
    continue_prior_request: Callable[[ContinuePriorRequestExecution], _Result],
) -> _Result:
    match execution:
        case CompileQuestionExecution():
            return compile_question(execution)
        case ContinuePriorRequestExecution():
            return continue_prior_request(execution)
        case _:
            assert_never(execution)
