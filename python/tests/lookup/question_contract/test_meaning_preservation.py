"""Cross-boundary commitments must survive schema generation and direct parsing."""

from copy import deepcopy
from pathlib import Path

from jsonschema import Draft7Validator
import pytest
import yaml

from fervis.lookup.question_contract import (
    parse_semantic_question_contract,
    parse_semantic_question_frame,
)
from fervis.lookup.question_contract.schema import (
    build_semantic_question_contract_schema_for_meaning,
)


CASES = Path(__file__).resolve().parents[2] / "conformance/cases/algorithms/semantic_kernel"


def _daily():
    return yaml.safe_load(
        (CASES / "non_identity_group_value_is_a_result_key.yaml").read_text()
    )["input"]


def _meaning(case):
    return parse_semantic_question_frame(
        case["frame_payload"],
        question_context_texts=tuple(case["question_context_texts"]),
    )


def _parse(case):
    return parse_semantic_question_contract(
        case["payload"],
        meaning=_meaning(case),
        question_context_texts=tuple(case["question_context_texts"]),
    )


def _schema_errors(case):
    return list(Draft7Validator(
        build_semantic_question_contract_schema_for_meaning(_meaning(case))
    ).iter_errors(case["payload"]))


def _fact(meaning):
    return {
        "kind": "fact", "observed_for_ref": "s1",
        "origin": {"source": "question_context", "meaning": meaning,
                   "resolved_input_ref": None},
    }


def _sum():
    return {"kind": "aggregate", "function": "sum",
            "argument": _fact("event amount"), "distinct_argument": False}


def _ranked_amount():
    case = _daily()
    result = case["frame_payload"]["outcome"]["answer_requests"][0]["request"]["result"]
    result["projection"]["explicitly_requested_values"][0]["meaning"] = "total event amount"
    result["result_order"]["ordering"]["values"] = [{
        "kind": "requested_value_ref", "value_ref": "v1",
        "ownership_basis": "Rank by the explicitly returned total.",
    }]
    request = case["payload"]["outcome"]["answer_requests"][0]
    request["outputs"]["requested_value_outputs"][0]["expression"] = _sum()
    request["ordering"][0]["expression"] = {
        "kind": "requested_value_ref", "value_ref": "v1",
    }
    request["ordering"][0]["direction"] = "descending"
    return case


def test_returned_value_ordering_reuses_the_exact_output_expression():
    case = _ranked_amount()
    assert _schema_errors(case) == []
    [fact] = _parse(case).contract.requested_facts
    assert fact.ordering[0].expression_ref == fact.outputs[1].expression_ref


def test_returned_value_ordering_cannot_reauthor_a_different_aggregate():
    case = _ranked_amount()
    expression = _sum()
    expression["function"] = "maximum"
    case["payload"]["outcome"]["answer_requests"][0]["ordering"][0]["expression"] = expression
    assert _schema_errors(case)
    with pytest.raises(ValueError, match="ordering.*requested value"):
        _parse(case)


@pytest.mark.parametrize("grain", ["week", "month", "invalid"])
def test_direct_parser_preserves_the_declared_calendar_grain(grain):
    case = _daily()
    case["payload"]["outcome"]["answer_requests"][0]["grouping"][0]["expression"]["grain"] = grain
    with pytest.raises(ValueError):
        _parse(case)


def test_each_grouping_shape_is_bound_to_its_own_reference():
    case = _daily()
    result = case["frame_payload"]["outcome"]["answer_requests"][0]["request"]["result"]
    second = deepcopy(result["grouping_meanings"][0])
    second.update(group_ref="g2", meaning="event week")
    second["grouping_value"]["grain"] = "week"
    result["grouping_meanings"].append(second)
    request = case["payload"]["outcome"]["answer_requests"][0]
    second = deepcopy(request["grouping"][0])
    second["id"] = "g2"
    second["expression"]["grain"] = "week"
    request["grouping"].append(second)
    request["outputs"]["result_key_outputs"].append({"expression": {"kind": "group_ref", "ref": "g2"}})
    assert _schema_errors(case) == []
    assert len(_parse(case).contract.requested_facts[0].grouping_refs) == 2
    request["grouping"][0]["expression"]["grain"] = "week"
    request["grouping"][1]["expression"]["grain"] = "day"
    assert _schema_errors(case)
    with pytest.raises(ValueError):
        _parse(case)


def test_multiple_independent_ordering_values_have_a_satisfiable_schema():
    case = _daily()
    result = case["frame_payload"]["outcome"]["answer_requests"][0]["request"]["result"]
    result["result_order"]["ordering"]["values"] = [{
        "kind": "unreturned_ordering_meaning", "meaning": meaning,
        "origin": {"kind": "question"}, "ownership_basis": "Used only for ordering.",
    } for meaning in ("total event amount", "largest event amount")]
    maximum = _sum()
    maximum["function"] = "maximum"
    case["payload"]["outcome"]["answer_requests"][0]["ordering"] = [{
        "ordering_basis": "Compare the shown ordering value.",
        "expression": expression, "direction": "descending",
    } for expression in (_sum(), maximum)]
    assert _schema_errors(case) == []
    assert len(_parse(case).contract.requested_facts[0].ordering) == 2


def test_direct_parser_cannot_change_all_results_to_one_rank():
    case = _daily()
    case["payload"]["outcome"]["answer_requests"][0]["selection"] = {"kind": "first_rank_with_ties"}
    with pytest.raises(ValueError, match="selection"):
        _parse(case)


@pytest.mark.parametrize("boolean_key,comparison_form", [(False,""),(True,"input"),(True,"value"),(True,"reversed_value")])
def test_computed_row_values_can_define_grouping_keys(boolean_key,comparison_form):
    case = _daily()
    result = case["frame_payload"]["outcome"]["answer_requests"][0]["request"]["result"]
    result["grouping_meanings"][0].update(
        meaning="amount band" if boolean_key else "gross amount minus fee amount",
        grouping_value={"kind": "condition" if boolean_key else "computed_value"},
    )
    expression = {"kind": "subtract", "left": _fact("gross amount"), "right": _fact("fee amount")}
    if boolean_key:
        result["result_order"] = {"ordering_request_basis": "No order is requested.",
                                  "ordering": {"kind": "no_ordering_requested"},
                                  "selection": {"kind": "all_results"}}
        case["payload"]["outcome"]["answer_requests"][0]["ordering"] = []
        case["frame_payload"]["outcome"]["supplied_values"]["operands"].append({
            "meaning": "amount cutoff", "denotation_basis": "100 separates the requested groups.",
            "answer_request_numbers": [1],
            "non_entity_value": {"kind": "number", "value": {
                "operands": ["100"], "origin": {"kind": "question"},
            }},
        })
        expression = {"kind": "input_comparison", "fact": _fact("amount"),
                      "operator": "gt", "input": {"kind": "input_ref", "input_ref": "i2"}}
    if boolean_key and comparison_form in {"value", "reversed_value"}:
        left, right = _fact("amount"), {"kind":"input_ref","input_ref":"i2"}
        expression = {"kind":"value_comparison","operator":"gt", "left":left,"right":right}
        if comparison_form == "reversed_value":
            expression.update(left=right,right=left,operator="lt")
    case["payload"]["outcome"]["answer_requests"][0]["grouping"][0]["expression"] = expression
    assert _schema_errors(case) == []
    fact = _parse(case).contract.requested_facts[0]
    assert fact.grouping_refs[0] == fact.outputs[0].expression_ref


def test_frame_parser_rejects_unknown_calendar_grains():
    case = _daily()
    result = case["frame_payload"]["outcome"]["answer_requests"][0]["request"]["result"]
    result["grouping_meanings"][0]["grouping_value"]["grain"] = "fortnight"
    with pytest.raises(ValueError):
        _meaning(case)


def test_recorded_grouping_cannot_be_replaced_by_a_tautological_condition():
    case = _daily()
    result = case["frame_payload"]["outcome"]["answer_requests"][0]["request"]["result"]
    result["grouping_meanings"][0].update(
        meaning="recorded status", grouping_value={"kind": "observed_value"},
    )
    grouping = case["payload"]["outcome"]["answer_requests"][0]["grouping"][0]
    grouping["expression"] = {
        "kind": "value_comparison", "operator": "equals",
        "left": _fact("recorded status"), "right": _fact("recorded status"),
    }
    assert _schema_errors(case)
    with pytest.raises(ValueError, match="declared recorded value"):
        _parse(case)


def test_grouping_tuple_can_combine_two_related_identity_domains():
    import json
    from fervis.lookup.semantic_types import IdentifierType
    case = json.loads((Path(__file__).parent / 'fixtures/two_related_grouping_keys.json').read_text())
    assert _schema_errors(case) == []
    parsed = _parse(case)
    fact = parsed.contract.requested_facts[0]
    keys = {term.id: term for term in fact.facts}
    assert [keys[ref].value_type for ref in fact.grouping_refs] == [
        IdentifierType('s2'), IdentifierType('s3'),
    ]


def test_grouping_tuple_rejects_an_aggregate_key():
    import json
    from dataclasses import replace
    from fervis.lookup.question_contract.analysis import analyze_requested_fact
    from fervis.lookup.question_contract.model import Aggregate
    case = json.loads((Path(__file__).parent / 'fixtures/two_related_grouping_keys.json').read_text())
    fact = _parse(case).contract.requested_facts[0]
    aggregate = next(item for item in fact.expressions if isinstance(item, Aggregate))
    with pytest.raises(ValueError, match='grouping requires row-level values'):
        analyze_requested_fact(replace(fact, grouping_refs=(aggregate.id,)), inputs={item.id: item for item in _meaning(case).inputs},
            input_denotations={item.input_ref: item for item in _meaning(case).input_denotations})


def test_grouping_tuple_rejects_a_disconnected_identity_domain():
    import json
    from dataclasses import replace
    from fervis.lookup.question_contract.analysis import analyze_requested_fact
    from fervis.lookup.question_contract.model import SetTerm
    case = json.loads((Path(__file__).parent / 'fixtures/two_related_grouping_keys.json').read_text())
    fact = _parse(case).contract.requested_facts[0]
    disconnected = replace(fact,
        sets=(*fact.sets, SetTerm('s4', fact.origin)),
        associations=tuple(replace(item, from_set_ref='s4') if item.to_set_ref=='s3' else item for item in fact.associations))
    with pytest.raises(ValueError, match='grouping has an unrelated row domain'):
        analyze_requested_fact(disconnected,
            inputs={item.id: item for item in _meaning(case).inputs},
            input_denotations={item.input_ref: item for item in _meaning(case).input_denotations})


def test_frame_can_group_by_its_own_observation_identity():
    import json
    from fervis.lookup.question_contract.schema import build_semantic_question_frame_schema
    case = json.loads((Path(__file__).parent / 'fixtures/two_related_grouping_keys.json').read_text())
    frame = case['frame_payload']
    result = frame['outcome']['answer_requests'][0]['request']['result']
    group = result['grouping_meanings'][0]
    group.update(grouping_kind='qualifying_row_identity', meaning='sales transaction',
                 grouping_basis='Each sale identifies its own result group.')
    result['grouping_meanings'] = [group]
    assert list(Draft7Validator(build_semantic_question_frame_schema()).iter_errors(frame)) == []
    parsed = parse_semantic_question_frame(frame, question_context_texts=tuple(case['question_context_texts']))
    assert parsed.answer_requests[0].grouping_kinds == ('qualifying_row_identity',)
