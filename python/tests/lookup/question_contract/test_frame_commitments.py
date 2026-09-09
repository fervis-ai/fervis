from copy import deepcopy
from pathlib import Path
import pytest
import yaml
from jsonschema import Draft7Validator

from fervis.lookup.question_contract.parser import (
    parse_semantic_question_frame,
    parse_semantic_question_contract,
)
from fervis.lookup.question_contract.schema import (
    build_semantic_question_contract_schema_for_meaning,
)
from fervis.lookup.semantic_types import BooleanType


def case(name):
    path = (
        Path(__file__).parents[2]
        / "conformance/cases/algorithms/semantic_kernel"
        / f"{name}.yaml"
    )
    return yaml.safe_load(path.read_text())["input"]


def parse(request):
    meaning = parse_semantic_question_frame(
        request["frame_payload"],
        question_context_texts=tuple(request["question_context_texts"]),
    )
    return parse_semantic_question_contract(
        request["payload"],
        meaning=meaning,
        question_context_texts=tuple(request["question_context_texts"]),
    )


@pytest.mark.parametrize(
    "kinds", [("warehouse", "supplier"), ("warehouse", "warehouse")]
)
def test_equal_names_preserve_independently_declared_references(kinds):
    request = case("distinct_related_identity_roles_share_candidate_domain")
    operands = request["frame_payload"]["outcome"]["supplied_values"]["operands"]
    for operand, kind in zip(operands, kinds):
        operand["entity_reference"]["value"]["identity_value"] = {"kind": "literal", "value": "Central"}
        operand["entity_reference"]["instance_kind"] = kind
    meaning = parse_semantic_question_frame(
        request["frame_payload"],
        question_context_texts=tuple(request["question_context_texts"]),
    )
    assert [term.id for term in meaning.inputs] == ["i1", "i2"]
    assert [item.denoted_instance_kind for item in meaning.input_denotations] == list(
        kinds
    )
    assert [item.operand_meaning for item in meaning.input_denotations] == [
        item["meaning"] for item in operands
    ]


def test_candidate_output_cannot_substitute_a_related_identity():
    request = case("same_related_row_uses_one_graph_node")
    fact = request["payload"]["outcome"]["answer_requests"][0]
    fact["outputs"]["result_key_outputs"][0]["expression"] = {
        "kind": "fact",
        "identity_path": {"kind": "related_instance", "association_ref": "a1"},
        "origin": fact["origin"],
    }
    with pytest.raises(ValueError, match="candidate identity"):
        parse(request)


def test_root_qualification_supplies_boolean_context_to_a_recorded_fact():
    request = case("same_related_row_uses_one_graph_node")
    request["frame_payload"]["outcome"]["answer_requests"][0]["request"][
        "relational_shape"
    ] = "ordinary"
    fact = request["payload"]["outcome"]["answer_requests"][0]
    fact["set_graph"]["other_related_sets"] = []
    fact["qualification"] = {
        "kind": "fact",
        "observed_for_ref": "s1",
        "origin": fact["origin"],
    }
    parsed = parse(request)
    assert isinstance(
        parsed.contract.requested_facts[0].facts[0].value_type, BooleanType
    )


def test_same_row_schema_can_express_three_relationships():
    request = case("same_related_row_uses_one_graph_node")
    graph = request["payload"]["outcome"]["answer_requests"][0]["set_graph"]
    third = deepcopy(graph["other_related_sets"][0]["associations"][0])
    third["id"] = "a3"
    third["origin"]["meaning"] = "employee reviews project"
    graph["other_related_sets"][0]["associations"].append(third)
    meaning = parse_semantic_question_frame(
        request["frame_payload"],
        question_context_texts=tuple(request["question_context_texts"]),
    )
    Draft7Validator(
        build_semantic_question_contract_schema_for_meaning(meaning)
    ).validate(request["payload"])
    parsed = parse(request)
    assert parsed.contract.requested_facts[0].expressions[0].association_refs == (
        "a1",
        "a2",
        "a3",
    )


@pytest.mark.parametrize(
    ("form", "accepted"),
    [
        ("forall", True),
        ("exists", False),
        ("not_forall", False),
        ("and", True),
        ("or", False),
    ],
)
def test_universal_frame_requires_universal_qualification(form, accepted):
    request = case("same_related_row_uses_one_graph_node")
    request["frame_payload"]["outcome"]["answer_requests"][0]["request"][
        "relational_shape"
    ] = "every_related_row"
    fact = request["payload"]["outcome"]["answer_requests"][0]
    related = fact["set_graph"]["other_related_sets"][0]
    related["association"] = related.pop("associations")[0]
    condition = {
        "kind": "null_check",
        "operator": "not_null",
        "argument": {
            "kind": "fact",
            "observed_for_ref": "s2",
            "origin": fact["origin"],
        },
    }
    universal = {
        "kind": "quantify",
        "quantifier": "exists" if form == "exists" else "forall",
        "over_set_ref": "s2",
        "association_refs": ["a1"],
        "condition": condition,
    }
    if form == "not_forall":
        universal = {"kind": "not", "argument": universal}
    elif form in ("and", "or"):
        universal = {
            "kind": form,
            "arguments": [
                universal,
                {"kind": "fact", "observed_for_ref": "s1", "origin": fact["origin"]},
            ],
        }
    fact["qualification"] = universal
    if accepted:
        parse(request)
    else:
        with pytest.raises(ValueError, match="universal qualification"):
            parse(request)


def test_identity_usage_cannot_retype_an_already_returned_property():
    import json
    request = json.loads((Path(__file__).parent / 'fixtures/identity_and_property_share_meaning.json').read_text())
    with pytest.raises(ValueError, match="distinct_by must equal the complete output tuple"):
        parse(request)


def test_identity_and_property_with_equal_origins_remain_separate_terms():
    import json
    from fervis.lookup.semantic_types import IdentifierType
    request = json.loads((Path(__file__).parent / 'fixtures/identity_and_property_share_meaning.json').read_text())
    payload = request['payload']['outcome']['answer_requests'][0]
    payload['distinct_by'] = []
    payload['qualification']['fact']['origin']['meaning'] = 'precision'
    fact = parse(request).contract.requested_facts[0]
    output_ref = fact.outputs[0].expression_ref
    terms = {term.id: term for term in fact.facts}
    assert not isinstance(terms[output_ref].value_type, IdentifierType)
    identity_terms = [term for term in fact.facts if isinstance(term.value_type, IdentifierType)]
    assert len(identity_terms) == 1
    assert identity_terms[0].origin == terms[output_ref].origin
    assert identity_terms[0].id != output_ref
