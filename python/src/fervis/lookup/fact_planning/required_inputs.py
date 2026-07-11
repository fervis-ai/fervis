"""Required input handles exposed by row-source contracts."""

from __future__ import annotations

from dataclasses import dataclass
from fervis.lookup.fact_planning.grounded_params import (
    unique_grounded_param_ids_by_row_source,
)
from fervis.lookup.answer_program.values import FactValue
from fervis.lookup.fact_plan.row_sources.model import RowSourceCatalog
from fervis.lookup.grounding.model import GroundedInputUse


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
