"""Closed admission contract for deterministic answer-program reruns."""

from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import assert_never

from fervis.lookup.answer_program.codec import answer_program_id
from fervis.lookup.answer_program.inputs import program_value_expressions
from fervis.lookup.answer_program.model import AnswerProgram
from fervis.lookup.answer_program.persistence import (
    ProgramInvocation,
    StoredProgramInvocation,
)
from fervis.lookup.answer_program.relations import SourceKind
from fervis.lookup.answer_program.values import BindingSet
from fervis.lookup.answer_program.values import (
    FactValue,
    ValueDependency,
    ValueDependencyKind,
)
from fervis.lookup.answer_program.expressions import expression_constant


class ProgramNotRerunnableError(ValueError):
    """The stored invocation cannot execute in the deterministic environment."""


@dataclass(frozen=True)
class RerunnableProgramInvocation:
    """A stored invocation parsed against the deterministic-worker contract."""

    invocation: ProgramInvocation
    program: AnswerProgram

    @classmethod
    def parse(
        cls,
        stored: StoredProgramInvocation,
    ) -> "RerunnableProgramInvocation":
        if answer_program_id(stored.program) != stored.invocation.program_id:
            raise ProgramNotRerunnableError(
                "stored invocation does not reference its canonical program"
            )
        for relation in stored.program.relations:
            _require_rerunnable_source(relation.source.kind)
        for value in (
            *(binding.value for binding in stored.invocation.bindings.bindings),
            *_program_constant_values(stored.program),
        ):
            _require_rerunnable_value(value.dependencies)
        return cls(
            invocation=stored.invocation,
            program=stored.program,
        )

    @property
    def bindings(self) -> BindingSet:
        return self.invocation.bindings


def _require_rerunnable_source(kind: SourceKind) -> None:
    match kind:
        case SourceKind.API_READ | SourceKind.GENERATED_CALENDAR:
            return
        case SourceKind.MEMORY_READ:
            raise ProgramNotRerunnableError(
                "deterministic reruns cannot depend on conversation memory"
            )
        case _:
            assert_never(kind)


def _program_constant_values(program: AnswerProgram) -> tuple[FactValue, ...]:
    return tuple(
        constant.value
        for named in program_value_expressions(program)
        if (constant := expression_constant(named.expression)) is not None
    )


def _require_rerunnable_value(
    dependencies: tuple[ValueDependency, ...],
) -> None:
    for dependency in dependencies:
        match dependency.kind:
            case ValueDependencyKind.CONVERSATION_MEMORY:
                raise ProgramNotRerunnableError(
                    "deterministic reruns cannot reuse conversation-memory values"
                )
            case _:
                assert_never(dependency.kind)
