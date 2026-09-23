"""Catalog clarification values remain typed, scoped and attributable."""

from dataclasses import replace

import pytest

from fervis.lookup.available_sources import snapshot_source_catalog
from fervis.lookup.clarification.model import (
    CatalogInputTarget,
    SourceBindingCatalogInputResponse,
)
from fervis.lookup.relation_catalog import RelationCatalog, CatalogParam, ParamSource
from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
from fervis.lookup.source_binding.catalog_responses import catalog_response_values
from fervis.lookup.answer_program.values import ValueComponent
from tests.lookup.relational_engine.test_dependent_reads import _read


def case():
    read = _read(
        "entries",
        params=(
            CatalogParam(
                "scope",
                "entries.scope",
                ParamSource.PATH,
                "integer",
                required=True,
                choices=("1", "2"),
            ),
        ),
    )
    (source,) = build_api_row_source_catalog(RelationCatalog(reads=(read,))).sources
    (param,) = source.params
    target = CatalogInputTarget(
        source.id, param.id, param.param_ref, param.type.value, param.choices
    )
    response = SourceBindingCatalogInputResponse("r1", "c1", "fact_1", target, "2")
    return snapshot_source_catalog((source,)), response


def test_catalog_response_certifies_the_requested_api_argument_only():
    catalog, response = case()
    (value,) = catalog_response_values(
        requested_fact_id="fact_1", source_catalog=catalog, responses=(response,)
    )
    assert value.target_ref == response.target.param_ref
    assert value.typed_value.payload.component_value(ValueComponent.VALUE) == 2
    assert value.certification_refs == ("clarification_response:r1",)
    assert value.typed_value.known_input_id == ""
    assert (
        catalog_response_values(
            requested_fact_id="fact_2", source_catalog=catalog, responses=(response,)
        )
        == ()
    )


@pytest.mark.parametrize(
    "change", ["type", "choices", "source", "invalid_value", "repeated"]
)
def test_catalog_response_rejects_stale_or_invalid_authority(change):
    catalog, response = case()
    if change == "type":
        response = replace(
            response, target=replace(response.target, value_type="string")
        )
    elif change == "choices":
        response = replace(response, target=replace(response.target, choices=("2",)))
    elif change == "source":
        response = replace(
            response, target=replace(response.target, row_source_id="foreign")
        )
    elif change == "invalid_value":
        response = replace(response, value="3")
    responses = (response, response) if change == "repeated" else (response,)
    with pytest.raises(ValueError):
        catalog_response_values(
            requested_fact_id="fact_1", source_catalog=catalog, responses=responses
        )
