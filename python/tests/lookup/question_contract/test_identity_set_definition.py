"""Input-owned set domains cannot absorb the reference's selection predicate."""
from copy import deepcopy
from pathlib import Path
import yaml
import pytest

from fervis.lookup.question_contract import parse_semantic_question_frame, parse_semantic_question_contract


@pytest.mark.parametrize("population_kind", ["district", "operating districts"])
def test_set_population_uses_its_declared_kind_instead_of_reference_provenance(population_kind):
    path=Path(__file__).resolve().parents[2]/'conformance/cases/algorithms/semantic_kernel/identity_input_owns_target_set_and_direct_association.yaml'
    fixture=yaml.safe_load(path.read_text())['input']
    texts=tuple(fixture['question_context_texts'])
    meaning=parse_semantic_question_frame(fixture['frame_payload'],question_context_texts=texts)
    payload=deepcopy(fixture['payload'])
    graph=payload['outcome']['answer_requests'][0]['set_graph']
    target=graph['identity_input_relations']['i1']['set']
    target['origin']['meaning']='the district named River District'
    extra=deepcopy(graph['identity_input_relations']['i1'])
    extra['association']['id']='a2'
    extra['set']['id']='s3'
    extra['set']['instance_kind'] = population_kind
    extra['set']['origin']['meaning']='operating districts named by the current input'
    graph['other_related_sets'].append(extra)
    request = payload['outcome']['answer_requests'][0]
    request['qualification'] = {
        'kind': 'and',
        'arguments': [request['qualification'], {
            'kind': 'quantify', 'quantifier': 'exists', 'over_set_ref': 's3',
            'association_refs': ['a2'],
            'condition': {'kind': 'null_check', 'operator': 'not_null',
                          'argument': {'kind': 'fact', 'observed_for_ref': 's3',
                                       'origin': extra['set']['origin']}},
        }],
    }
    parsed=parse_semantic_question_contract(payload,meaning=meaning,question_context_texts=texts)
    sets={item.id:item for item in parsed.contract.requested_facts[0].sets}
    assert sets['s2'].origin.meaning == 'district'
    assert sets['s3'].origin.meaning == population_kind
    use,=parsed.semantic_indexes[0].input_use_sites
    assert use.input_ref=='i1'
    assert use.identity_set_ref.local_id=='s2'


def test_declared_relationship_must_participate_in_the_requested_computation():
    import pytest
    path = Path(__file__).resolve().parents[2] / 'conformance/cases/algorithms/semantic_kernel/identity_input_owns_target_set_and_direct_association.yaml'
    fixture = yaml.safe_load(path.read_text())['input']
    texts = tuple(fixture['question_context_texts'])
    meaning = parse_semantic_question_frame(fixture['frame_payload'], question_context_texts=texts)
    payload = deepcopy(fixture['payload'])
    graph = payload['outcome']['answer_requests'][0]['set_graph']
    unused = deepcopy(graph['identity_input_relations']['i1'])
    unused['association']['id'] = 'a2'
    unused['set']['id'] = 's3'
    graph['other_related_sets'].append(unused)
    with pytest.raises(ValueError, match='unused relationship declarations.*a2'):
        parse_semantic_question_contract(payload, meaning=meaning, question_context_texts=texts)
