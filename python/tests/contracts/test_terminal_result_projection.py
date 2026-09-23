"""Persisted factual limitations must survive the public run projection."""
from types import SimpleNamespace
import pytest
from fervis.lineage.enums import FactResultKind
from fervis.lineage.views.query import FactResultRow, LineageRows
from fervis.lineage.views.service import _fact_result_view
from fervis.questions.run_views import _result_data, _answer_text


def run_with(kind, payload, *, answers=()):
    result = _fact_result_view(FactResultRow(
        fact_result_id='result', run_id='run', requested_fact_id='requested',
        produced_by_step_id='step', result_kind=FactResultKind(kind), payload_json=payload,
    ), rows=LineageRows(), steps=())
    fact = SimpleNamespace(requested_fact_id='requested', fact_key='fact_1', fact_results=(result,))
    return SimpleNamespace(clarification_requests=(), clarification_responses=(),
                           answers=answers, requested_facts=(fact,))


@pytest.mark.parametrize(('kind', 'payload'), [
    ('impossible', {'blockedRequirements': [{'requiredFor': 'a requested measure'}]}),
    ('no_data', {'emptyRelation': {'relationId': 'observations'}}),
    ('undefined', {'operation': {'reasonCode': 'division_by_zero'}}),
])
def test_terminal_details_are_not_lost_when_there_are_no_answer_outputs(kind, payload):
    from fervis.questions.result_data import terminal_result_message
    assert _result_data(run_with(kind, payload)) == {'kind': kind, **payload, 'message': terminal_result_message(kind, payload)}


def test_terminal_presentation_is_preserved_independently_of_answer_records():
    run = run_with('impossible', {'message': 'The requested observation is unavailable.', 'blockedRequirements': []})
    assert _answer_text(run) == 'The requested observation is unavailable.'


def test_partial_answers_do_not_hide_unavailable_requested_facts():
    answer = SimpleNamespace(presentations=(), outputs=(SimpleNamespace(
        output_key='known', value_kind='number', value_json={'kind':'number','value':'3'}, value='3'),))
    run = run_with('impossible', {'message':'Another requested observation is unavailable.', 'blockedRequirements': []}, answers=(answer,))
    projected = _result_data(run)
    assert projected['kind'] == 'partial'
    assert projected['outputs'][0]['displayValue'] == '3'
    assert projected['facts'][0]['kind'] == 'impossible'
    assert _answer_text(run) == '3\nAnother requested observation is unavailable.'


def test_terminal_kind_comes_from_the_record_and_audit_refs_stay_out_of_delivery():
    result = _result_data(run_with('impossible', {'kind': 'answer', 'blockedRequirements': [
        {'requiredFor': 'measure', 'proofRefs': ['internal-proof']},
    ]}))
    assert result == {'kind': 'impossible', 'blockedRequirements': [{'requiredFor': 'measure'}], 'message': 'I cannot answer measure from the available API evidence.'}


def test_terminal_message_is_persisted_using_the_same_renderer():
    from fervis.lookup.outcomes.model import BlockedRequirement, BlockedRequirementKind, FactResult, Impossible
    from fervis.lookup.answer_rendering import render_fact_result
    from fervis.lookup.lineage.results import _terminal_fact_result_payload
    result = FactResult(outcome=Impossible(blocked_requirements=(BlockedRequirement(
        id='missing', kind=BlockedRequirementKind.COMPLETE_EVIDENCE_PATH,
        requested_fact_id='fact_1', fact_ref='fact_1:f1', required_for='the requested quantity'),)))
    assert _terminal_fact_result_payload(result)['message'] == render_fact_result(result).message


def test_public_wire_fixtures_are_generated_by_the_backend_projection():
    import json
    from pathlib import Path
    fixtures = json.loads((Path(__file__).parent / 'fixtures/public_terminal_results.json').read_text())
    for kind in ('impossible', 'no_data', 'undefined'):
        payload = {key:value for key,value in fixtures[kind].items() if key != 'kind'}
        assert _result_data(run_with(kind, payload)) == fixtures[kind]
    value = fixtures['partial']['outputs'][0]
    answer = SimpleNamespace(presentations=(), outputs=(SimpleNamespace(
        output_key=value['key'], value_kind=value['valueKind'], value_json=value['value'], value=value['displayValue']),))
    payload = {key:value for key,value in fixtures['impossible'].items() if key != 'kind'}
    assert _result_data(run_with('impossible', payload, answers=(answer,))) == fixtures['partial']


def test_message_absent_partial_records_still_produce_complete_prose():
    answer = SimpleNamespace(presentations=(), outputs=(SimpleNamespace(
        output_key='known', value_kind='number', value_json={'kind':'number','value':'3'}, value='3'),))
    run = run_with('impossible', {'blockedRequirements': [{'requiredFor':'unknown quantity'}]}, answers=(answer,))
    assert _answer_text(run) == '3\nI cannot answer unknown quantity from the available API evidence.'
    assert _result_data(run)['facts'][0]['message'] == 'I cannot answer unknown quantity from the available API evidence.'


@pytest.mark.parametrize('with_terminal', [False, True])
def test_equal_display_values_do_not_merge_distinct_requested_outputs(with_terminal):
    answer = SimpleNamespace(presentations=(), outputs=tuple(SimpleNamespace(
        output_key=key, value_kind='number', value_json={'kind':'number','value':'3'}, value='3')
        for key in ('orders', 'customers')))
    run = run_with('impossible' if with_terminal else 'answered',
                   {'blockedRequirements':[{'requiredFor':'another quantity'}]}, answers=(answer,))
    expected = '3\n3'
    if with_terminal:
        expected += '\nI cannot answer another quantity from the available API evidence.'
    assert _answer_text(run) == expected
    assert len(_result_data(run)['outputs']) == 2
