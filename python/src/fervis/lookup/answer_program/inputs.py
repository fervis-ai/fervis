"""Compilation and immutable patching for answer-program inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing_extensions import assert_never

from fervis.lookup.answer_program.contracts import (
    AnswerProgramContractError,
    BindingPatch,
    BindingProvenance,
    BindingProvenanceKind,
    BindingSet,
    NamedValueExpression,
    ParameterBinding,
    ParameterDeclaration,
    ParameterRole,
    ParameterValueType,
    ProgramInputs,
    SetParameter,
    UnsetParameter,
    canonical_fact_value,
    parameter_value_type,
)
from fervis.lookup.answer_program.values import (
    ConstantRef,
    EnvironmentRef,
    FactValue,
    NodeOutputRef,
    ParameterRef,
    TimeComponent,
    ValueComponent,
)
from fervis.lookup.answer_program.expressions import Expression, expression_references
from fervis.lookup.answer_program.model import AnswerProgram
from fervis.lookup.answer_program.operations import (
    AggregateSpec,
    AntiJoinSpec,
    CrossJoinSpec,
    ComputeSpec,
    FilterSpec,
    JoinSpec,
    Operation,
    Predicate,
    ProjectSpec,
    ProjectToKeySpec,
    OrderSpec,
    Take,
    RoleExpandSpec,
    UnionSpec,
    UniversalConditionSpec,
)
from fervis.lookup.fact_planning.value_components import value_component


@dataclass(frozen=True)
class ResolvedValueExpression:
    value: Any
    fact_value: Any
    proof_refs: tuple[str, ...] = ()
    source_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompiledProgramInputs:
    parameters: tuple[ParameterDeclaration, ...]
    bindings: BindingSet
    expressions: tuple[NamedValueExpression, ...] = ()


def compile_program_inputs(inputs: ProgramInputs) -> CompiledProgramInputs:
    """Validate and canonicalize the closed input surface of one program."""

    parameters = _parameter_index(inputs.parameters)
    _validate_binding_set(parameters, inputs.bindings)
    for named in inputs.expressions:
        try:
            references = expression_references(named.expression)
        except (AssertionError, TypeError):
            raise AnswerProgramContractError(
                "unclassified_value_origin",
                f"expression {named.sink} has no declared value origin",
            ) from None
        for reference in references.parameters:
            if reference.parameter_id not in parameters:
                raise AnswerProgramContractError(
                    "unknown_parameter",
                    f"expression {named.sink} references an unknown parameter",
                )
    return CompiledProgramInputs(
        parameters=tuple(sorted(inputs.parameters, key=lambda item: item.id)),
        bindings=BindingSet.from_bindings(inputs.bindings.bindings),
        expressions=inputs.expressions,
    )


def compile_answer_program_inputs(
    program: AnswerProgram,
    *,
    bindings: BindingSet,
) -> CompiledProgramInputs:
    compiled = compile_program_inputs(
        ProgramInputs(
            parameters=program.parameters,
            bindings=bindings,
            expressions=program_value_expressions(program),
        )
    )
    _validate_population_choice_parameters(program, compiled.parameters)
    _validate_compute_parameters(program, compiled.parameters)
    return compiled


def _validate_population_choice_parameters(
    program: AnswerProgram,
    parameters: tuple[ParameterDeclaration, ...],
) -> None:
    by_id = {parameter.id: parameter for parameter in parameters}
    for relation in program.relations:
        for choice in relation.source.population_choices:
            declaration = by_id[choice.selection_expr.parameter_id]
            if (
                declaration.role is not ParameterRole.SEMANTIC_CONTROL
                or declaration.value_type is not ParameterValueType.STRING_SET
                or choice.selection_expr.component != ValueComponent.VALUE.value
                or choice.selection_expr.item_index is not None
            ):
                raise AnswerProgramContractError(
                    "invalid_population_choice_parameter",
                    "population choice requires a whole semantic-control string set",
                )


def _validate_compute_parameters(
    program: AnswerProgram,
    parameters: tuple[ParameterDeclaration, ...],
) -> None:
    by_id = {parameter.id: parameter for parameter in parameters}
    for operation in program.operations:
        spec = operation.spec
        if not isinstance(spec, ComputeSpec):
            continue
        references = expression_references(spec.expression)
        if any(
            by_id[reference.parameter_id].value_type is not ParameterValueType.NUMBER
            or reference.component != ValueComponent.VALUE.value
            or reference.item_index is not None
            for reference in references.parameters
        ) or any(
            parameter_value_type(reference.value) is not ParameterValueType.NUMBER
            or reference.component != ValueComponent.VALUE.value
            or reference.item_index is not None
            for reference in references.constants
        ):
            raise AnswerProgramContractError(
                "invalid_compute_parameter",
                "compute expressions require whole numeric values",
            )


def program_value_expressions(
    program: AnswerProgram,
) -> tuple[NamedValueExpression, ...]:
    """Return every answer-affecting value expression with its program sink."""

    expressions: list[NamedValueExpression] = []
    for relation in program.relations:
        for binding in relation.source.param_bindings:
            expressions.append(
                NamedValueExpression(
                    f"relation.{relation.id}.param.{binding.param_id}",
                    binding.value_expr,
                )
            )
        for choice in relation.source.population_choices:
            expressions.append(
                NamedValueExpression(
                    f"relation.{relation.id}.population.{choice.controller_id}",
                    choice.selection_expr,
                )
            )
    for operation in program.operations:
        expressions.extend(_operation_value_expressions(operation))
    return tuple(expressions)


def _operation_value_expressions(
    operation: Operation,
) -> tuple[NamedValueExpression, ...]:
    spec = operation.spec
    if isinstance(spec, OrderSpec) and isinstance(spec.selection, Take):
        return (
            NamedValueExpression(
                sink=f"operation.{operation.id}.order.limit",
                expression=spec.selection.limit,
            ),
        )
    if isinstance(spec, ComputeSpec):
        return tuple(
            NamedValueExpression(
                sink=f"operation.{operation.id}.compute.{index}",
                expression=expression,
            )
            for index, expression in enumerate(
                expression_references(spec.expression).leaves
            )
        )
    if isinstance(spec, FilterSpec):
        return _predicate_value_expressions(operation.id, spec.predicate)
    if isinstance(spec, ProjectSpec):
        return tuple(
            NamedValueExpression(
                sink=f"operation.{operation.id}.project.{output.output_field}",
                expression=output.expression,
            )
            for output in spec.outputs
        )
    if isinstance(spec, UniversalConditionSpec):
        return _predicate_value_expressions(operation.id, spec.predicate)
    if isinstance(
        spec,
        (
            ProjectToKeySpec,
            JoinSpec,
            UnionSpec,
            RoleExpandSpec,
            CrossJoinSpec,
            AntiJoinSpec,
            AggregateSpec,
            OrderSpec,
        ),
    ):
        return ()
    assert_never(spec)


def _predicate_value_expressions(
    operation_id: str,
    predicate: Predicate,
) -> tuple[NamedValueExpression, ...]:
    return tuple(
        NamedValueExpression(
            sink=f"operation.{operation_id}.predicate.{index}",
            expression=expression,
        )
        for index, expression in enumerate(
            (
                predicate.left,
                *((predicate.right,) if predicate.right is not None else ()),
            )
        )
    )


def apply_binding_patch(
    *,
    program: AnswerProgram,
    bindings: BindingSet,
    patch: BindingPatch,
) -> BindingSet:
    """Return a new validated binding set without mutating the base set."""

    parameter_index = _parameter_index(program.parameters)
    _validate_binding_set(parameter_index, bindings)
    revised = {binding.parameter_id: binding for binding in bindings.bindings}
    for operation in patch.operations:
        parameter = parameter_index.get(operation.parameter_id)
        if parameter is None:
            raise AnswerProgramContractError(
                "unknown_parameter",
                f"patch references unknown parameter {operation.parameter_id}",
            )
        if isinstance(operation, SetParameter):
            if operation.value.kind.value in {"identity", "identity_set"}:
                raise AnswerProgramContractError(
                    "identity_patch_requires_current_grounding",
                    "identity bindings must be certified under current authority",
                )
            binding = ParameterBinding(
                parameter_id=operation.parameter_id,
                value=operation.value,
                provenance=BindingProvenance(
                    kind=BindingProvenanceKind.RERUN_PATCH,
                    refs=patch.provenance_refs,
                ),
            )
            _validate_binding(parameter, binding)
            revised[operation.parameter_id] = binding
            continue
        if isinstance(operation, UnsetParameter):
            if parameter.required:
                raise AnswerProgramContractError(
                    "required_parameter_unset",
                    f"required parameter {parameter.id} cannot be unset",
                )
            revised.pop(operation.parameter_id, None)
            continue
        raise AnswerProgramContractError(
            "unsupported_binding_patch",
            "binding patch contains an unsupported operation",
        )
    result = BindingSet.from_bindings(tuple(revised.values()))
    _validate_binding_set(parameter_index, result)
    return result


def resolve_value_expression(
    expression: Expression,
    *,
    bindings: BindingSet,
) -> ResolvedValueExpression:
    """Materialize a non-derived expression from one immutable binding set."""

    if isinstance(expression, ParameterRef):
        binding = bindings.get(expression.parameter_id)
        if binding is None:
            raise AnswerProgramContractError(
                "missing_parameter_binding",
                f"parameter {expression.parameter_id} is unbound",
            )
        value = _fact_value_component(binding.value, expression.component)
        return ResolvedValueExpression(
            value=_indexed_value(value, expression.item_index),
            fact_value=binding.value,
            proof_refs=tuple(
                dict.fromkeys((*binding.value.proof_refs, *binding.provenance.refs))
            ),
            source_refs=binding.value.source_refs,
        )
    if isinstance(expression, ConstantRef):
        value = _fact_value_component(expression.value, expression.component)
        return ResolvedValueExpression(
            value=_indexed_value(value, expression.item_index),
            fact_value=expression.value,
            proof_refs=expression.value.proof_refs,
            source_refs=expression.value.source_refs,
        )
    if isinstance(expression, NodeOutputRef):
        raise AnswerProgramContractError(
            "unresolved_node_output",
            f"node output {expression.node_id}.{expression.output_id} is unavailable",
        )
    if isinstance(expression, EnvironmentRef):
        raise AnswerProgramContractError(
            "unresolved_environment_value",
            f"environment value {expression.key} is unavailable",
        )
    raise AnswerProgramContractError(
        "unclassified_value_origin",
        "value expression has no declared origin",
    )


def resolved_value_expression_type(
    expression: ParameterRef | ConstantRef,
    resolved: ResolvedValueExpression,
) -> str:
    """Return the declared comparison type of one resolved expression component."""

    if expression.component != ValueComponent.VALUE.value:
        return ""
    return {
        ParameterValueType.IDENTITY: "",
        ParameterValueType.IDENTITY_SET: "list",
        ParameterValueType.NAMED: "string",
        ParameterValueType.TIME: "",
        ParameterValueType.NUMBER: "number",
        ParameterValueType.STRING: "string",
        ParameterValueType.BOOLEAN: "boolean",
        ParameterValueType.STRING_SET: "list",
    }[parameter_value_type(resolved.fact_value)]


def _fact_value_component(value: Any, component: str) -> Any:
    key_component_prefix = "key_component:"
    if component.startswith(key_component_prefix):
        if not isinstance(value, FactValue):
            raise AnswerProgramContractError(
                "unsupported_value_component",
                f"value does not carry {component}",
            )
        try:
            return value.identity_key_component(
                component.removeprefix(key_component_prefix)
            )
        except ValueError as exc:
            raise AnswerProgramContractError(
                "unsupported_value_component",
                f"value does not carry {component}",
            ) from exc
    try:
        typed_component = (
            TimeComponent(component)
            if component in {item.value for item in TimeComponent}
            else ValueComponent(component)
        )
    except ValueError as exc:
        raise AnswerProgramContractError(
            "unsupported_value_component",
            f"unsupported value component {component}",
        ) from exc
    return value_component(value, typed_component)


def _indexed_value(value: Any, item_index: int | None) -> Any:
    if item_index is None:
        return value
    if not isinstance(value, tuple) or item_index >= len(value):
        raise AnswerProgramContractError(
            "binding_item_out_of_range",
            "parameter item reference is outside its bound value",
        )
    return value[item_index]


def _parameter_index(
    parameters: tuple[ParameterDeclaration, ...],
) -> dict[str, ParameterDeclaration]:
    output: dict[str, ParameterDeclaration] = {}
    for parameter in parameters:
        if parameter.id in output:
            raise AnswerProgramContractError(
                "duplicate_parameter",
                f"duplicate parameter declaration {parameter.id}",
            )
        output[parameter.id] = parameter
    return output


def _validate_binding_set(
    parameters: dict[str, ParameterDeclaration],
    bindings: BindingSet,
) -> None:
    bindings_by_id = {binding.parameter_id: binding for binding in bindings.bindings}
    unknown = set(bindings_by_id) - set(parameters)
    if unknown:
        raise AnswerProgramContractError(
            "unknown_parameter",
            f"bindings contain unknown parameters: {', '.join(sorted(unknown))}",
        )
    missing = {
        parameter.id
        for parameter in parameters.values()
        if parameter.required and parameter.id not in bindings_by_id
    }
    if missing:
        raise AnswerProgramContractError(
            "missing_parameter_binding",
            f"required parameters are unbound: {', '.join(sorted(missing))}",
        )
    for binding in bindings.bindings:
        _validate_binding(parameters[binding.parameter_id], binding)


def _validate_binding(
    parameter: ParameterDeclaration,
    binding: ParameterBinding,
) -> None:
    if parameter_value_type(binding.value) != parameter.value_type:
        raise AnswerProgramContractError(
            "binding_type_mismatch",
            f"binding for {parameter.id} has the wrong type",
        )
    if not parameter.allowed_values:
        return
    value = canonical_fact_value(binding.value)
    values = value if isinstance(value, list) else [value]
    if any(str(item) not in parameter.allowed_values for item in values):
        raise AnswerProgramContractError(
            "disallowed_parameter_value",
            f"binding for {parameter.id} contains a disallowed value",
        )
