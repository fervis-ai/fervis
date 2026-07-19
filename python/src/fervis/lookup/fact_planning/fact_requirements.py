"""Fact-scoped endpoint input requirements."""

from __future__ import annotations

from dataclasses import dataclass

from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.relation_catalog.selection import CatalogSelectionResult
from fervis.lookup.fact_planning.grounded_params import (
    GroundedParamValue,
    unique_grounded_param_values,
)
from fervis.lookup.fact_planning.required_inputs import (
    RequiredInput,
    param_has_bindable_value,
    required_inputs,
)
from fervis.lookup.fact_planning.row_set_filters import (
    value_applies_to_requested_fact,
)
from fervis.lookup.fact_plan.row_sources import (
    RowSource,
    RowSourceCatalog,
    RowSourceKind,
    build_row_source_catalog,
    row_source_ids_for_read_ids,
)
from fervis.lookup.answer_program.values import FactValue
from fervis.lookup.grounding.model import GroundedInputUse


@dataclass(frozen=True)
class RequestedFactEndpointRequirement:
    requested_fact_id: str
    selected_row_source_ids: tuple[str, ...]
    executable_row_source_ids: tuple[str, ...]
    missing_inputs: tuple[RequiredInput, ...]

    @property
    def needs_clarification(self) -> bool:
        return not self.executable_row_source_ids and bool(self.missing_inputs)


@dataclass(frozen=True)
class FactEndpointRequirements:
    requested_facts: tuple[RequestedFactEndpointRequirement, ...]

    @property
    def executable_row_source_ids(self) -> frozenset[str]:
        return frozenset(
            row_source_id
            for item in self.requested_facts
            for row_source_id in item.executable_row_source_ids
        )

    @property
    def clarifiable_missing_inputs(self) -> tuple[RequiredInput, ...]:
        return tuple(
            input_item
            for item in self.requested_facts
            if item.needs_clarification
            for input_item in item.missing_inputs
        )


def fact_endpoint_requirements(
    *,
    catalog: RelationCatalog,
    catalog_selection: CatalogSelectionResult | None,
    available_values: tuple[FactValue, ...],
    available_value_uses: tuple[GroundedInputUse, ...],
    row_sources: RowSourceCatalog | None = None,
) -> FactEndpointRequirements:
    row_source_catalog = row_sources or build_row_source_catalog(catalog)
    if catalog_selection is None:
        selected_ids = tuple(source.id for source in row_source_catalog.sources)
        return FactEndpointRequirements(
            requested_facts=(
                _requested_fact_endpoint_requirement(
                    requested_fact_id="",
                    selected_row_source_ids=selected_ids,
                    row_sources=row_source_catalog,
                    available_values=available_values,
                    grounded_params=unique_grounded_param_values(
                        values=available_values,
                        grounded_input_uses=available_value_uses,
                    ),
                ),
            )
        )
    return FactEndpointRequirements(
        requested_facts=tuple(
            _requested_fact_endpoint_requirement(
                requested_fact_id=selection.requested_fact_id,
                selected_row_source_ids=row_source_ids_for_read_ids(
                    selection.selected_read_ids,
                    row_sources=row_source_catalog,
                ),
                row_sources=row_source_catalog,
                available_values=tuple(
                    value
                    for value in available_values
                    if value_applies_to_requested_fact(
                        value,
                        selection.requested_fact_id,
                    )
                ),
                grounded_params=unique_grounded_param_values(
                    values=available_values,
                    grounded_input_uses=available_value_uses,
                    requested_fact_id=selection.requested_fact_id,
                ),
            )
            for selection in catalog_selection.requested_fact_selections
        )
    )


def _requested_fact_endpoint_requirement(
    *,
    requested_fact_id: str,
    selected_row_source_ids: tuple[str, ...],
    row_sources: RowSourceCatalog,
    available_values: tuple[FactValue, ...],
    grounded_params: dict[tuple[str, str], GroundedParamValue],
) -> RequestedFactEndpointRequirement:
    executable_row_source_ids = frozenset(
        source.id
        for source in row_sources.sources
        if _source_is_executable(
            source,
            grounded_params=grounded_params,
            available_values=available_values,
        )
    )
    missing_inputs_by_source = _missing_inputs_by_source(
        row_sources,
        grounded_params=grounded_params,
        available_values=available_values,
    )
    executable = tuple(
        row_source_id
        for row_source_id in selected_row_source_ids
        if row_source_id in executable_row_source_ids
    )
    if executable:
        missing_inputs: tuple[RequiredInput, ...] = ()
    else:
        missing_inputs = tuple(
            input_item
            for row_source_id in selected_row_source_ids
            for input_item in missing_inputs_by_source.get(row_source_id, ())
        )
    return RequestedFactEndpointRequirement(
        requested_fact_id=requested_fact_id,
        selected_row_source_ids=selected_row_source_ids,
        executable_row_source_ids=executable,
        missing_inputs=missing_inputs,
    )


def _source_is_executable(
    source: RowSource,
    *,
    grounded_params: dict[tuple[str, str], GroundedParamValue],
    available_values: tuple[FactValue, ...],
) -> bool:
    return not _missing_required_inputs(
        source,
        grounded_params=grounded_params,
        available_values=available_values,
    )


def _missing_inputs_by_source(
    row_sources: RowSourceCatalog,
    *,
    grounded_params: dict[tuple[str, str], GroundedParamValue],
    available_values: tuple[FactValue, ...],
) -> dict[str, tuple[RequiredInput, ...]]:
    all_required_inputs = required_inputs(row_sources)
    return {
        source.id: tuple(
            item
            for item in all_required_inputs
            if item.row_source_id == source.id
            and item.row_source_kind == RowSourceKind.API_READ.value
            and (item.row_source_id, item.param_id) not in grounded_params
            and not param_has_bindable_value(
                source.param(item.param_id),
                available_values=available_values,
            )
        )
        for source in row_sources.sources
    }


def _missing_required_inputs(
    source: RowSource,
    *,
    grounded_params: dict[tuple[str, str], GroundedParamValue],
    available_values: tuple[FactValue, ...],
) -> tuple[str, ...]:
    return tuple(
        param.id
        for param in source.params
        if param.required
        and param.default is None
        and not param.choices
        and (source.id, param.id) not in grounded_params
        and not param_has_bindable_value(
            param,
            available_values=available_values,
        )
    )
