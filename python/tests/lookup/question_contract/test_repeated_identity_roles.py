from pathlib import Path
import yaml

from fervis.lookup.question_contract.parser import (
    parse_semantic_question_contract,
    parse_semantic_question_frame,
)


def test_one_supplied_identity_can_qualify_two_distinct_relationships():
    fixture = (
        Path(__file__).parents[2]
        / "conformance/cases/algorithms/semantic_kernel/distinct_related_identity_roles_share_candidate_domain.yaml"
    )
    request = yaml.safe_load(fixture.read_text())["input"]
    frame = request["frame_payload"]
    frame["outcome"]["supplied_values"]["operands"].pop()
    question = ("How many transfers went from Account A back to Account A?",)
    meaning = parse_semantic_question_frame(frame, question_context_texts=question)
    payload = request["payload"]
    fact = payload["outcome"]["answer_requests"][0]
    graph = fact["set_graph"]
    graph["other_related_sets"].append(graph["identity_input_relations"].pop("i2"))
    first, second = fact["qualification"]["arguments"]
    second["input"] = dict(first["input"])
    first["fact"]["identity_path"]["association_ref"] = "a1"
    second["fact"]["identity_path"]["association_ref"] = "a2"
    parsed = parse_semantic_question_contract(
        payload, meaning=meaning, question_context_texts=question
    )
    assert {term.owner_ref for term in parsed.contract.requested_facts[0].facts} == {
        "a1",
        "a2",
    }
    assert [use.input_ref for use in parsed.semantic_indexes[0].input_use_sites] == [
        "i1",
        "i1",
    ]
