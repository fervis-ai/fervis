"""One owner for endpoint-parameter collection cardinality."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

from fervis.lookup.canonical_data import RuntimeValue
from fervis.lookup.answer_program.expressions import Expression
from fervis.lookup.relation_catalog.parameter_values import (
    CatalogParameterValue,
    CatalogScalarParameterValue,
)
from fervis.types.enums import StrEnum


class RelationInputOrigin(StrEnum):
    QUESTION_INPUT = "question_input"
    PLAN_CONTROL = "plan_control"


@dataclass(frozen=True)
class ParameterBindingCandidate:
    param_id: str
    value: object | None = None
    value_expr: Expression | None = None
    origin_kind: RelationInputOrigin = RelationInputOrigin.PLAN_CONTROL
    value_id: str = ""
    value_component: str = "value"
    value_item_index: int | None = None
    parameter_id: str = ""
    proof_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.param_id:
            raise ValueError("endpoint parameter binding requires a parameter")
        if (self.value is None) == (self.value_expr is None):
            raise ValueError("parameter binding requires one value origin")

    @property
    def compiler_value(self) -> object | None:
        if self.value_item_index is None:
            return self.value
        if not isinstance(self.value, tuple) or self.value_item_index >= len(
            self.value
        ):
            raise ValueError("parameter binding item is outside its value")
        return self.value[self.value_item_index]


ParamBindingSet = tuple[ParameterBindingCandidate, ...]
ParamBindingSetAlternatives = tuple[ParamBindingSet, ...]


def finite_choice_parameter_is_omittable(
    *,
    required: bool,
    default: CatalogParameterValue,
    choices: tuple[CatalogScalarParameterValue, ...],
    included_values: tuple[CatalogScalarParameterValue, ...],
) -> bool:
    """Whether omission has exactly the selected finite-choice population."""

    if required:
        return False
    if isinstance(default, (tuple, dict)):
        raise ValueError("finite-choice parameter default must be scalar")
    omission_values = (default,) if default is not None else choices
    return set(included_values) == set(omission_values)


def parameter_binding_sets(
    *,
    param_id: str,
    value: RuntimeValue,
    parameter_type: str,
    proof_refs: tuple[str, ...] = (),
    origin_kind: RelationInputOrigin,
    value_id: str = "",
    value_component: str = "value",
    parameter_id: str = "",
) -> ParamBindingSetAlternatives:
    """Bind a collection once or expand it into scalar invocations."""

    if isinstance(value, tuple) and parameter_type not in {"array", "list"}:
        return tuple(
            (
                ParameterBindingCandidate(
                    param_id=param_id,
                    value=value,
                    origin_kind=origin_kind,
                    value_id=value_id,
                    value_component=value_component,
                    value_item_index=index,
                    parameter_id=parameter_id,
                    proof_refs=proof_refs,
                ),
            )
            for index, _item in enumerate(value)
        )
    return (
        (
            ParameterBindingCandidate(
                param_id=param_id,
                value=value,
                origin_kind=origin_kind,
                value_id=value_id,
                value_component=value_component,
                parameter_id=parameter_id,
                proof_refs=proof_refs,
            ),
        ),
    )


def combine_param_binding_sets(
    groups: Iterable[ParamBindingSetAlternatives],
) -> ParamBindingSetAlternatives:
    """Combine independent values while preserving each collection item's identity."""

    alternatives = tuple(groups)
    if not alternatives:
        return ((),)
    combined: ParamBindingSetAlternatives = ((),)
    for group in alternatives:
        combined = tuple(
            (*existing, *option)
            for existing in combined
            for option in group
            if _binding_sets_are_compatible(existing, option)
        )
    return combined


def alternate_param_binding_sets(
    groups: Iterable[ParamBindingSetAlternatives],
) -> ParamBindingSetAlternatives:
    """Combine values that are alternatives for one parameter target."""

    return tuple(binding_set for group in groups for binding_set in group)


def intersect_param_binding_sets(
    groups: Iterable[ParamBindingSetAlternatives],
) -> ParamBindingSetAlternatives:
    """Intersect conjunctive constraints on one parameter target."""

    constraints = tuple(groups)
    if not constraints:
        return ((),)
    intersection = constraints[0]
    for constraint in constraints[1:]:
        intersection = tuple(
            merge_equivalent_param_binding_sets((left,), (right,))[0]
            for left in intersection
            for right in constraint
            if equivalent_param_binding_sets((left,), (right,))
        )
    return intersection


def equivalent_param_binding_sets(
    left: ParamBindingSetAlternatives,
    right: ParamBindingSetAlternatives,
) -> bool:
    """Compare executable parameter projections without their proof paths."""

    return _executable_projection(left) == _executable_projection(right)


def merge_equivalent_param_binding_sets(
    left: ParamBindingSetAlternatives,
    right: ParamBindingSetAlternatives,
) -> ParamBindingSetAlternatives:
    """Keep one executable projection while retaining every supporting proof."""

    if not equivalent_param_binding_sets(left, right):
        raise ValueError("parameter binding projections are not equivalent")
    return tuple(
        tuple(
            replace(
                left_binding,
                proof_refs=tuple(
                    dict.fromkeys((*left_binding.proof_refs, *right_binding.proof_refs))
                ),
            )
            for left_binding, right_binding in zip(
                left_set,
                right_set,
                strict=True,
            )
        )
        for left_set, right_set in zip(left, right, strict=True)
    )


def coalesce_equivalent_param_binding_sets(
    groups: Iterable[ParamBindingSetAlternatives],
) -> tuple[ParamBindingSetAlternatives, ...]:
    """Canonicalize repeated executable projections without losing proof refs."""

    distinct: list[ParamBindingSetAlternatives] = []
    for group in groups:
        matching_index = next(
            (
                index
                for index, existing in enumerate(distinct)
                if equivalent_param_binding_sets(existing, group)
            ),
            None,
        )
        if matching_index is None:
            distinct.append(group)
            continue
        distinct[matching_index] = merge_equivalent_param_binding_sets(
            distinct[matching_index],
            group,
        )
    return tuple(distinct)


def _executable_projection(
    alternatives: ParamBindingSetAlternatives,
) -> tuple[tuple[tuple[object, ...], ...], ...]:
    return tuple(
        tuple(
            (
                binding.param_id,
                binding.compiler_value,
                binding.value_expr,
                binding.parameter_id,
            )
            for binding in binding_set
        )
        for binding_set in alternatives
    )


def _binding_sets_are_compatible(
    existing: ParamBindingSet,
    option: ParamBindingSet,
) -> bool:
    existing_item_indexes = {
        binding.value_id: binding.value_item_index
        for binding in existing
        if binding.value_id and binding.value_item_index is not None
    }
    return all(
        existing_item_indexes.get(binding.value_id, binding.value_item_index)
        == binding.value_item_index
        for binding in option
        if binding.value_id and binding.value_item_index is not None
    )


__all__ = [
    "ParamBindingSetAlternatives",
    "ParameterBindingCandidate",
    "RelationInputOrigin",
    "alternate_param_binding_sets",
    "coalesce_equivalent_param_binding_sets",
    "combine_param_binding_sets",
    "equivalent_param_binding_sets",
    "finite_choice_parameter_is_omittable",
    "intersect_param_binding_sets",
    "merge_equivalent_param_binding_sets",
    "parameter_binding_sets",
]
