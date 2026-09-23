"""Pagination authoring binds structural references without parameter-name rules."""

from copy import deepcopy
from dataclasses import replace

import pytest
from jsonschema import Draft202012Validator, validate

from fervis.lookup.relation_catalog.from_host_api import (
    relation_catalog_from_endpoint_contracts,
)
from fervis.lookup.source_reads.pagination_discovery import (
    PaginationDiscoveryRequest,
    PaginationDiscoveryPrompt,
    parse_pagination_discovery,
)
from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
from tests.host_api.test_bound_pagination import contract


def case():
    catalog = relation_catalog_from_endpoint_contracts(
        (replace(contract(), resource_names=("records",)),)
    )
    request = PaginationDiscoveryRequest(catalog, ("records",))
    (read,) = request.targets
    params = {param.name: param.ref for param in read.params}
    rows = {path.path: path.id for path in read.row_paths}
    fields = {field.path: field.ref for field in read.fields}
    payload = {
        "reads": {
            "records": {
                "mapping_basis": "The segment selects a numbered slice, width its size, and matched gives the collection total.",
                "mode": "page_number",
                "row_path_ref": rows["records"],
                "position_parameter_ref": params["segment"],
                "size_parameter_ref": params["width"],
                "total_field_ref": fields["matched"],
                "continuation_field_ref": None,
            }
        }
    }
    return request, payload


def test_discovery_uses_the_exact_schema_and_removes_only_transport_inputs():
    request, payload = case()
    prompt = PaginationDiscoveryPrompt(request)
    validate(payload, prompt.tool_contract().tool_specs[0].input_schema)
    assert prompt.response_contract().provider_schema
    result = parse_pagination_discovery(payload, request=request)
    (read,) = result.reads
    assert read.pagination_binding.results_path == "records"
    assert read.pagination_binding.page_size == 2
    source = next(
        source
        for source in build_api_row_source_catalog(result).sources
        if source.row_path == "records"
    )
    assert source.params == ()
    assert source.pagination_binding == read.pagination_binding
    assert request.catalog.reads[0].pagination_binding is None


@pytest.mark.parametrize(
    "change",
    [
        "foreign_param",
        "foreign_field",
        "duplicate_param",
        "row_total",
        "empty_evidence",
        "omitted_read",
    ],
)
def test_discovery_rejects_invalid_cross_boundary_mappings(change):
    request, payload = case()
    payload = deepcopy(payload)
    decision = payload["reads"]["records"]
    if change == "foreign_param":
        decision["position_parameter_ref"] = "foreign"
    elif change == "foreign_field":
        decision["total_field_ref"] = "foreign"
    elif change == "duplicate_param":
        decision["size_parameter_ref"] = decision["position_parameter_ref"]
    elif change == "row_total":
        decision["total_field_ref"] = next(
            field.ref
            for field in request.targets[0].fields
            if field.path == "records.id"
        )
    elif change == "empty_evidence":
        decision["total_field_ref"] = None
    else:
        payload["reads"] = {}
    with pytest.raises(ValueError):
        parse_pagination_discovery(payload, request=request)


@pytest.mark.parametrize("kind", ["unknown", "single_response"])
def test_unsupported_or_unpaginated_decisions_create_no_traversal(kind):
    request, _ = case()
    payload = {
        "reads": {
            "records": {
                "mode": kind,
                "row_path_ref": None,
                "position_parameter_ref": None,
                "size_parameter_ref": None,
                "total_field_ref": None,
                "continuation_field_ref": None,
                "mapping_basis": "This contract does not establish supported traversal mechanics.",
            }
        }
    }
    result = parse_pagination_discovery(payload, request=request)
    if kind == "single_response":
        assert result == request.catalog
    else:
        assert result.reads == ()


def test_nonpaged_schema_cannot_author_page_only_references():
    request, page = case()
    schema = PaginationDiscoveryPrompt(request).tool_contract().tool_specs[0].input_schema
    nonpaged = deepcopy(page)
    decision = nonpaged["reads"]["records"]
    decision.update({
        "mode": "single_response",
        "row_path_ref": None,
        "position_parameter_ref": None,
        "size_parameter_ref": None,
        "total_field_ref": None,
        "continuation_field_ref": None,
    })
    validate(nonpaged, schema)
    decision["row_path_ref"] = page["reads"]["records"]["row_path_ref"]
    assert list(Draft202012Validator(schema).iter_errors(nonpaged))


def test_unknown_traversal_cannot_supply_a_population_count():
    from fervis.lookup.available_sources import snapshot_source_catalog
    from fervis.lookup.source_binding.candidates import candidate_source_strategy
    from tests.lookup.fact_compilation.test_compiler import _compile_memory_count

    request, _ = case()
    result = parse_pagination_discovery(
        {
            "reads": {
                "records": {
                    "mode": "unknown",
                    "row_path_ref": None,
                    "position_parameter_ref": None,
                    "size_parameter_ref": None,
                    "total_field_ref": None,
                    "continuation_field_ref": None,
                    "mapping_basis": "The declared protocol cannot establish complete enumeration.",
                }
            }
        },
        request=request,
    )
    _, _, _, _, _, verified = _compile_memory_count(())
    sources = snapshot_source_catalog(build_api_row_source_catalog(result).sources)
    assert candidate_source_strategy(verified.request.index, sources, ()).branches == ()


@pytest.mark.parametrize("shape", ["cursor", "root_array"])
def test_unsupported_collection_shapes_require_an_explicit_completeness_decision(shape):
    request, _ = case()
    read = request.catalog.reads[0]
    if shape == "cursor":
        read = replace(
            read,
            params=(replace(read.params[0], name="cursor", type="string"),),
        )
    elif shape == "root_array":
        read = replace(
            read,
            row_paths=tuple(
                replace(path, path="") if path.path == "records" else path
                for path in read.row_paths
            ),
        )
    request = replace(request, catalog=replace(request.catalog, reads=(read,)))
    assert request.targets == (read,)
    schema = PaginationDiscoveryPrompt(request).tool_contract().tool_specs[0].input_schema
    modes = schema["properties"]["reads"]["properties"][read.id]["properties"]["mode"]["enum"]
    assert modes == ["single_response", "unknown"]
    decision = {
        "reads": {read.id: {
            "mode": "unknown",
            "mapping_basis": "Complete traversal is not established for this collection.",
            "row_path_ref": None,
            "position_parameter_ref": None,
            "size_parameter_ref": None,
            "total_field_ref": None,
            "continuation_field_ref": None,
        }}
    }
    validate(decision, schema)
    assert parse_pagination_discovery(decision, request=request).reads == ()


def test_collection_without_advertised_traversal_uses_its_one_response_contract():
    request, _ = case()
    read = replace(request.catalog.reads[0], params=())
    request = replace(request, catalog=replace(request.catalog, reads=(read,)))
    assert request.targets == ()


def test_root_collection_with_only_a_text_filter_has_no_page_mechanism():
    request, _ = case()
    read = request.catalog.reads[0]
    read = replace(
        read,
        params=(replace(read.params[0], name="label", type="string"),),
        row_paths=tuple(
            replace(path, path="") if path.path == "records" else path
            for path in read.row_paths
        ),
    )
    request = replace(request, catalog=replace(request.catalog, reads=(read,)))
    assert request.targets == ()
