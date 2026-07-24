"""Atomic handoff from verified semantics to executable relational code."""

from dataclasses import dataclass

from fervis.lookup.answer_program.model import AnswerProgram
from fervis.lookup.answer_program.values import BindingSet


@dataclass(frozen=True)
class FactCompilationResult:
    answer_program: AnswerProgram
    initial_bindings: BindingSet


__all__ = ["FactCompilationResult"]
