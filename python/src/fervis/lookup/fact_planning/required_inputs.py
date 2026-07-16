"""Required input handles exposed by row-source contracts."""

from __future__ import annotations

from dataclasses import dataclass
from fervis.lookup.fact_planning.grounded_params import (
    unique_grounded_param_ids_by_row_source,
)
from fervis.lookup.answer_program.values import FactValue
from fervis.lookup.answer_program.values import (
    IdentitySetValuePayload,
    IdentityValuePayload,
    LiteralValuePayload,
    ValueKind,
)
from fervis.lookup.fact_plan.row_sources.model import RowSourceCatalog, RowSourceParam
from fervis.lookup.grounding.model import GroundedInputUse
from fervis.lookup.relation_catalog import EntityKeyComponentTarget
from fervis.lookup.canonical_data import EntityKeyValue


@dataclass(frozen=True)
class RequiredInput:
    id: str
    row_source_id: str
    row_source_kind: str
    param_id: str
    param_ref: str
    param_label: str = ""
    param_type: str = ""
    choices: tuple[str, ...] = ()
    choice_labels: dict[str, str] | None = None


def required_input_id(*, row_source_id: str, param_id: str) -> str:
    return f"{row_source_id}.{param_id}"


def required_inputs(row_sources: RowSourceCatalog) -> tuple[RequiredInput, ...]:
    return tuple(
        RequiredInput(
            id=required_input_id(row_source_id=source.id, param_id=param.id),
            row_source_id=source.id,
            row_source_kind=source.kind.value,
            param_id=param.id,
            param_ref=param.param_ref,
            param_label=param.name,
            param_type=param.type,
            choices=tuple(param.choices),
            choice_labels=dict(param.choice_labels or {}),
        )
        for source in row_sources.sources
        for param in source.params
        if param.required and param.default is None
    )


def clarifiable_required_inputs(
    row_sources: RowSourceCatalog,
) -> tuple[RequiredInput, ...]:
    return tuple(
        item
        for item in required_inputs(row_sources)
        if item.row_source_kind == "api_read"
    )


def param_has_bindable_value(
    param: RowSourceParam,
    *,
    available_values: tuple[FactValue, ...],
) -> bool:
    """Return whether Source Binding can offer a value for this parameter."""

    if param.choices:
        return True
    target = param.entity_target
    if target is not None:
        return any(
            _identity_value_matches_target(value, target=target)
            for value in available_values
        )
    if param.type in {"date", "datetime"}:
        return any(value.kind == ValueKind.TIME for value in available_values)
    if param.type in {"integer", "number", "decimal", "float", "string"}:
        return any(
            value.kind == ValueKind.LITERAL
            and isinstance(value.payload, LiteralValuePayload)
            for value in available_values
        )
    return False


def _identity_value_matches_target(
    value: FactValue,
    *,
    target: EntityKeyComponentTarget,
) -> bool:
    payload = value.payload
    keys: tuple[EntityKeyValue, ...]
    if isinstance(payload, IdentityValuePayload):
        keys = (payload.key,)
    elif isinstance(payload, IdentitySetValuePayload):
        keys = payload.keys
    else:
        return False
    return (
        payload.entity_kind == target.entity_kind
        and payload.key_id == target.key_id
        and all(
            target.component_id
            in {component.component_id for component in key.components}
            for key in keys
        )
    )


def grounded_required_input_ids(
    row_sources: RowSourceCatalog,
    *,
    values: tuple[FactValue, ...],
    grounded_input_uses: tuple[GroundedInputUse, ...],
) -> frozenset[str]:
    grounded_params = unique_grounded_param_ids_by_row_source(
        values=values,
        grounded_input_uses=grounded_input_uses,
    )
    return frozenset(
        required_input_id(row_source_id=source.id, param_id=param.id)
        for source in row_sources.sources
        for param in source.params
        if param.required
        and param.default is None
        and param.id in grounded_params.get(source.id, frozenset())
    )
