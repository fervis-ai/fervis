"""Structural compatibility must not force a semantic source substitution."""
import pytest
from jsonschema import Draft7Validator
from fervis.lookup.source_binding import schema as schemas
from fervis.lookup.source_binding.parser import compile_source_realization
from tests.lookup.source_binding._candidate_fixture import daily_observation_request


def rejection(request):
    ref = next(item.token for item in request.index.source_requirement_refs if item.kind.value == 'set')
    return {'kind':'unavailable_source_realization','unmet_requirement_refs':[ref],
            'explanation':'The candidate rows describe departments, not the requested observations.'}


def test_structurally_available_candidates_can_be_declared_unrealizable():
    request = daily_observation_request()
    assert request.strategy.branches
    payload = rejection(request)
    schema = schemas.build_unavailable_source_realization_schema(tuple(ref.token for ref in request.index.source_requirement_refs))
    assert list(Draft7Validator(schema).iter_errors(payload)) == []
    result = compile_source_realization(payload, request=request)
    assert type(result).__name__ == 'SourceRealizationUnavailable'
    assert result.unmet_requirement_refs == tuple(payload['unmet_requirement_refs'])


@pytest.mark.parametrize('refs', [[], ['fact_1:set:unknown'], ['other:set:s1'], ['fact_1:set:s1','fact_1:set:s1']])
def test_unrealizable_outcome_requires_unique_owned_requirements(refs):
    request = daily_observation_request()
    payload = rejection(request)
    payload['unmet_requirement_refs'] = refs
    with pytest.raises(ValueError):
        compile_source_realization(payload, request=request)




def test_source_realization_exposes_success_and_unavailable_tools():
    from fervis.lookup.source_binding.prompt import SemanticSourceRealizationTurnPrompt
    prompt = SemanticSourceRealizationTurnPrompt(daily_observation_request())
    assert {item.name for item in prompt.tool_contract().tool_specs} == {
        'submit_source_realization', 'report_unavailable_source_realization'}
