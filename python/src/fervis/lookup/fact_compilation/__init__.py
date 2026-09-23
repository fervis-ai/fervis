"""Deterministic semantic-to-relational fact compilation."""

from .compiler import (
    compile_verified_source_strategies,
    compile_verified_source_strategy,
)
from .inputs import semantic_compiler_inputs
from .model import FactCompilationResult

__all__ = [
    "FactCompilationResult",
    "compile_verified_source_strategies",
    "compile_verified_source_strategy",
    "semantic_compiler_inputs",
]
