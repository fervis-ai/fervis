"""An explicit no-request decision must never become a request filter."""
from jsonschema import Draft7Validator
import pytest
from fervis.lookup.provider_contract import ProviderObject
from fervis.lookup.source_binding.parser import _resolved_input_applications
from fervis.lookup.source_binding.schema import _branch_resolved_input_applications_schema
from tests.lookup.relational_engine.test_scoped_compilation import employee_query


def fixture():
    request = employee_query(manager_minimum=True).request
    owner = next(owner for owner in request.invocation_application_owner_refs
                 if request.unapplied_input_value_refs_for_owner(owner,branch_id='branch'))
    value = request.unapplied_input_value_refs_for_owner(owner,branch_id='branch')[0]
    return request, {'kind':'no_request_application','mapping_basis':'Evaluate the predicate from returned rows.','owner_ref':owner,'value_ref':value}


def test_explicit_nonapplication_is_a_valid_distinct_schema_choice():
    request,item = fixture()
    payload = dict(item)
    schema = _branch_resolved_input_applications_schema(request,branch_id='branch')
    assert list(Draft7Validator(schema).iter_errors([payload])) == []
    assert _resolved_input_applications({'branch':(ProviderObject(item),)},request=request) == ()


@pytest.mark.parametrize('changes', [
    {'owner_ref':'unknown'}, {'value_ref':'unknown'}, {'value_component':'WHOLE_VALUE'},
])
def test_nonapplication_cannot_claim_an_unknown_owner_value_or_projection(changes):
    request,item = fixture()
    with pytest.raises(ValueError):
        _resolved_input_applications({'branch':(ProviderObject({**item,**changes}),)},request=request)


def test_one_owner_cannot_both_apply_and_decline_the_same_value():
    request,item = fixture()
    option = request.direct_value_options_for_owner(item['owner_ref'],branch_id='branch')[0]
    applied = {**item, 'kind':'request_application', 'target_ref':option.target_ref,
        'value_component':option.component_ref or option.projection.value}
    with pytest.raises(ValueError,match='both applied and unapplied'):
        _resolved_input_applications({'branch':(ProviderObject(item),ProviderObject(applied))},request=request)


def test_production_prompt_explains_explicit_nonapplication():
    from fervis.lookup.source_binding.prompt import (
        SOURCE_INPUT_APPLICATION_INSTRUCTION, SemanticSourceBindingTurnPrompt,
    )
    from fervis.lookup.turn_prompts.builder import TurnPromptBuilder
    from fervis.lookup.turn_prompts.rendering import PromptRenderer
    query = employee_query(manager_minimum=True)
    prompt = SemanticSourceBindingTurnPrompt.__new__(SemanticSourceBindingTurnPrompt)
    prompt.request = query.request
    sections = prompt.instruction_sections(TurnPromptBuilder(None))
    rendered = '\n'.join(section.render(PromptRenderer()) for section in sections)
    assert SOURCE_INPUT_APPLICATION_INSTRUCTION in rendered
    assert 'Only request_application includes value_component and target_ref' in rendered
    assert 'must apply every fact-local resolved input' not in rendered
