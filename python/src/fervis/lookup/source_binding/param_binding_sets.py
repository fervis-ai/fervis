"""One owner for endpoint-parameter collection cardinality."""

from __future__ import annotations

from collections.abc import Iterable

from fervis.lookup.canonical_data import RuntimeValue
from fervis.lookup.source_binding.candidates.model import CandidateParameter
from fervis.lookup.source_binding.compiler_ir import (
    DraftEndpointParamBinding,
    RelationInputOrigin,
)


ParamBindingSet = tuple[DraftEndpointParamBinding, ...]
ParamBindingSetAlternatives = tuple[ParamBindingSet, ...]


def parameter_binding_sets(
    *,
    param_id: str,
    value: RuntimeValue,
    param: CandidateParameter,
    proof_refs: tuple[str, ...] = (),
    origin_kind: RelationInputOrigin,
    value_id: str = "",
    value_component: str = "value",
    parameter_id: str = "",
) -> ParamBindingSetAlternatives:
    """Bind a collection once or expand it into scalar invocations."""

    if isinstance(value, tuple) and param.type not in {"array", "list"}:
        return tuple(
            (
                DraftEndpointParamBinding(
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
            DraftEndpointParamBinding(
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
    "combine_param_binding_sets",
    "parameter_binding_sets",
]
