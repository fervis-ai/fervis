"""A schema-free addressed read can only be probed with a supplied fact input."""

from types import SimpleNamespace

import pytest
from jsonschema import validate

from fervis.lookup.relation_catalog import CatalogParam, EndpointRead, ParamSource, RelationCatalog
from fervis.lookup.source_reads.inspection_inputs import (
    InspectionInputTurnPrompt, inspection_input_request, inspection_input_schema,
    parse_inspection_inputs,
)
from fervis.lookup.turn_prompts import TurnPromptContext


def _request():
    read = EndpointRead(
        "readings", "readings", path="/facilities/{facility_id}/readings",
        resource_names=("readings",),
        params=(CatalogParam("facility_id", "facility_id", ParamSource.PATH,
                             "uuid", required=True),),
    )
    supplied = SimpleNamespace(
        id="facility", operand="00000000-0000-0000-0000-000000000001"
    )
    unrelated = SimpleNamespace(
        id="order", operand="00000000-0000-0000-0000-000000000002"
    )
    return inspection_input_request(
        catalog=RelationCatalog(reads=(read,)), read_ids=(read.id,),
        contract=SimpleNamespace(
            inputs=(supplied, unrelated),
            input_denotations=(SimpleNamespace(
                input_ref="facility", operand_meaning="the supplied facility"
            ),),
        ),
        indexes=(SimpleNamespace(
            requested_fact_id="fact", input_use_sites=(SimpleNamespace(input_ref="facility"),),
        ),),
        fact_selections=(SimpleNamespace(
            requested_fact_id="fact", selected_read_ids=(read.id,),
        ),),
        certified_values=(
            SimpleNamespace(input_ref="facility", certification_refs=("question_input:facility",)),
            SimpleNamespace(input_ref="order", certification_refs=("question_input:order",)),
        ),
    )


def test_inspection_input_contract_excludes_unrelated_supplied_values():
    request = _request()
    assert [item.id for item in request.targets[0].options["facility_id"]] == ["facility"]
    prompt = InspectionInputTurnPrompt(request).to_model_payload(
        TurnPromptContext(current_question="How many readings at the supplied facility?")
    )
    assert "the supplied facility" in prompt.prompt_text
    assert "00000000-0000-0000-0000-000000000002" not in prompt.prompt_text
    payload = {"reads": {"readings": {
        "kind": "bound_arguments", "mapping_basis": "The path names the supplied facility.",
        "parameter_inputs": {"facility_id": "facility"},
    }}}
    validate(payload, inspection_input_schema(request))
    assert parse_inspection_inputs(payload, request=request) == {
        "readings": {"facility_id": "00000000-0000-0000-0000-000000000001"}
    }
    payload["reads"]["readings"]["parameter_inputs"]["facility_id"] = "order"
    with pytest.raises(ValueError, match="eligible original input"):
        parse_inspection_inputs(payload, request=request)


def test_inspection_input_contract_allows_explicit_unsupported_decision():
    request = _request()
    payload = {"reads": {"readings": {
        "kind": "unsupported", "reason": "The question supplies no location address.",
    }}}
    validate(payload, inspection_input_schema(request))
    assert parse_inspection_inputs(payload, request=request) == {}


def test_optional_schema_free_argument_can_bind_or_be_explicitly_omitted():
    read = EndpointRead(
        "reports", "reports", path="/reports", resource_names=("reports",),
        params=(CatalogParam("shape", "shape", ParamSource.QUERY, "string"),),
    )
    supplied = SimpleNamespace(id="shape", operand="compact")
    request = inspection_input_request(
        catalog=RelationCatalog(reads=(read,)), read_ids=(read.id,),
        contract=SimpleNamespace(
            inputs=(supplied,), input_denotations=(SimpleNamespace(
                input_ref="shape", operand_meaning="compact response shape"
            ),),
        ),
        indexes=(SimpleNamespace(requested_fact_id="fact", input_use_sites=(
            SimpleNamespace(input_ref="shape"),
        )),),
        fact_selections=(SimpleNamespace(requested_fact_id="fact",
                                         selected_read_ids=(read.id,)),),
        certified_values=(SimpleNamespace(input_ref="shape",
            certification_refs=("question_input:shape",)),),
    )
    assert len(request.targets) == 1
    payload = {"reads": {"reports": {
        "kind": "bound_arguments", "mapping_basis": "Compact is the requested representation.",
        "parameter_inputs": {"shape": "shape"},
    }}}
    validate(payload, inspection_input_schema(request))
    assert parse_inspection_inputs(payload, request=request) == {
        "reports": {"shape": "compact"}
    }
    payload["reads"]["reports"]["parameter_inputs"]["shape"] = "omit"
    validate(payload, inspection_input_schema(request))
    assert parse_inspection_inputs(payload, request=request) == {"reports": {}}


def test_schema_free_inspection_can_select_a_declared_finite_shape_choice():
    read = EndpointRead(
        "reports", "reports", path="/reports", resource_names=("reports",),
        params=(CatalogParam("shape", "shape", ParamSource.QUERY, "choice",
                             choices=("summary", "detail")),),
    )
    request = inspection_input_request(
        catalog=RelationCatalog(reads=(read,)), read_ids=(read.id,),
        contract=SimpleNamespace(inputs=(), input_denotations=()),
        indexes=(SimpleNamespace(requested_fact_id="fact", input_use_sites=()),),
        fact_selections=(SimpleNamespace(requested_fact_id="fact",
                                         selected_read_ids=(read.id,)),),
        certified_values=(),
    )
    assert len(request.targets) == 1
    payload = {"reads": {"reports": {
        "kind": "bound_arguments", "mapping_basis": "The summary shape carries the requested total.",
        "parameter_inputs": {"shape": "choice:summary"},
    }}}
    validate(payload, inspection_input_schema(request))
    assert parse_inspection_inputs(payload, request=request) == {
        "reports": {"shape": "summary"}
    }
    payload["reads"]["reports"]["parameter_inputs"]["shape"] = "choice:invented"
    with pytest.raises(ValueError, match="declared inspection choice"):
        parse_inspection_inputs(payload, request=request)


def test_boolean_catalog_choice_is_typed_before_schema_free_inspection():
    read = EndpointRead(
        "flags", "flags", path="/flags", resource_names=("flags",),
        params=(CatalogParam("active", "active", ParamSource.QUERY,
                             "boolean", required=True),),
    )
    request = inspection_input_request(
        catalog=RelationCatalog(reads=(read,)), read_ids=(read.id,),
        contract=SimpleNamespace(inputs=(), input_denotations=()),
        indexes=(SimpleNamespace(requested_fact_id="fact", input_use_sites=()),),
        fact_selections=(SimpleNamespace(requested_fact_id="fact",
                                         selected_read_ids=(read.id,)),),
        certified_values=(),
    )
    payload = {"reads": {"flags": {
        "kind": "bound_arguments", "mapping_basis": "The requested flags are active.",
        "parameter_inputs": {"active": "choice:true"},
    }}}
    validate(payload, inspection_input_schema(request))
    assert parse_inspection_inputs(payload, request=request) == {
        "flags": {"active": True}
    }


def test_numeric_catalog_choice_is_parsed_under_its_declared_parameter_type():
    read = EndpointRead(
        "tiers", "tiers", path="/tiers", resource_names=("tiers",),
        params=(CatalogParam("tier", "tier", ParamSource.QUERY, "integer",
                             required=True, choices=("1", "2")),),
    )
    request = inspection_input_request(
        catalog=RelationCatalog(reads=(read,)), read_ids=(read.id,),
        contract=SimpleNamespace(inputs=(), input_denotations=()),
        indexes=(SimpleNamespace(requested_fact_id="fact", input_use_sites=()),),
        fact_selections=(SimpleNamespace(requested_fact_id="fact",
                                         selected_read_ids=(read.id,)),),
        certified_values=(),
    )
    assert request.targets[0].choice_values["tier"] == ("1", "2")
    payload = {"reads": {"tiers": {
        "kind": "bound_arguments", "mapping_basis": "The second tier is requested.",
        "parameter_inputs": {"tier": "choice:2"},
    }}}
    validate(payload, inspection_input_schema(request))
    assert parse_inspection_inputs(payload, request=request) == {
        "tiers": {"tier": 2}
    }


def test_inconsistent_catalog_choices_cannot_supply_required_inspection_input():
    read = EndpointRead(
        "tiers", "tiers", path="/tiers", resource_names=("tiers",),
        params=(CatalogParam("tier", "tier", ParamSource.QUERY, "integer",
                             required=True, choices=("invalid", "2")),),
    )
    request = inspection_input_request(
        catalog=RelationCatalog(reads=(read,)), read_ids=(read.id,),
        contract=SimpleNamespace(inputs=(), input_denotations=()),
        indexes=(SimpleNamespace(requested_fact_id="fact", input_use_sites=()),),
        fact_selections=(SimpleNamespace(requested_fact_id="fact",
                                         selected_read_ids=(read.id,)),),
        certified_values=(),
    )
    assert request.targets == ()
