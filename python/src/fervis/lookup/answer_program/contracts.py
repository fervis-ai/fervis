"""Answer-program contract errors and canonical value projection."""

from __future__ import annotations

from typing import Any

from fervis.lookup.answer_program.errors import AnswerProgramContractError
from fervis.lookup.answer_program.values import (
    BindingPatch,
    BindingPatchOperation,
    BindingPatchOperationKind,
    BindingProvenance,
    BindingProvenanceKind,
    BindingSet,
    ConstantRef,
    EnvironmentRef,
    FactValue,
    NamedValueExpression,
    NodeOutputRef,
    ParameterBinding,
    ParameterDeclaration,
    ParameterRef,
    ParameterRole,
    ParameterValueType,
    ProgramInputs,
    SetParameter,
    UnsetParameter,
    ValueExpression,
    ValueExpressionOrigin,
    ValueDependency,
    ValueDependencyKind,
)


def parameter_value_type(value: FactValue) -> ParameterValueType:
    return ParameterValueType(value.payload.parameter_value_type)


def canonical_fact_value(value: FactValue) -> Any:
    return value.payload.canonical_value()


__all__ = [
    "AnswerProgramContractError",
    "BindingPatch",
    "BindingPatchOperation",
    "BindingPatchOperationKind",
    "BindingProvenance",
    "BindingProvenanceKind",
    "BindingSet",
    "ConstantRef",
    "EnvironmentRef",
    "NamedValueExpression",
    "NodeOutputRef",
    "ParameterBinding",
    "ParameterDeclaration",
    "ParameterRef",
    "ParameterRole",
    "ParameterValueType",
    "ProgramInputs",
    "SetParameter",
    "UnsetParameter",
    "ValueExpression",
    "ValueExpressionOrigin",
    "ValueDependency",
    "ValueDependencyKind",
    "canonical_fact_value",
    "parameter_value_type",
]
