from dataclasses import replace
from types import SimpleNamespace

import pytest
from jsonschema import validate
from fervis.lookup.orchestration.reference_authoring import FactualQueryPrompt, ReferenceDemand, parse_factual_query, reference_contracts
from fervis.lookup.orchestration.reference_slots import reference_input_menu
from fervis.lookup.relational_sql.parameters import query_parameter_menu
from fervis.lookup.relational_sql.catalog import build_query_view_catalog
from fervis.lookup.relational_sql.execution import QueryValidationError
from fervis.lookup.relation_catalog import RelationCatalog, CatalogParam, ParamSource, EntityKeyComponentTarget, CandidateKey, CandidateKeyComponent
from fervis.lookup.answer_program.values import FactValue, LiteralType
from fervis.lookup.grounding.semantic import CanonicalInputValue
from fervis.lookup.question_contract import InputTerm, InputDenotation
from fervis.lookup.question_contract.model import InputDenotationKind
from fervis.lookup.semantic_types import SourceOrigin, SourceOriginKind, TextType
from tests.lookup.relational_engine.test_dependent_reads import _read
from tests.lookup.relational_sql.test_authoring import payload


def context(*, extra_authorities=0, composite=False, opaque=False):
    origin = SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, '42')
    term = InputTerm('i1', origin, '42', TextType())
    denotation = InputDenotation('d1', 'i1', 'specified record', 'The input identifies the record.', 'record', InputDenotationKind.IDENTITY_REFERENCE)
    value = FactValue.literal(id='input', known_input_id='i1', literal_type=LiteralType.STRING, value='42', proof_refs=('question_input:i1',))
    menu = reference_input_menu(query_parameter_menu((CanonicalInputValue(value.id, 'i1', ('fact_1:sql_input:i1',), value, value.proof_refs),)), {'i1':denotation})
    records = _read('records', paired=composite)
    if composite:
        records = replace(records, candidate_keys=(CandidateKey('primary', 'records', (
            CandidateKeyComponent('country', 'records.zone'), CandidateKeyComponent('id', 'records.id')), primary=True),))
    params = (CatalogParam('record_id', 'record_id', ParamSource.PATH, 'integer', required=True,
        entity_target=None if opaque else EntityKeyComponentTarget('records', 'primary', 'id')),)
    if composite:
        params += (CatalogParam('country', 'country', ParamSource.PATH, 'string', required=True,
            entity_target=EntityKeyComponentTarget('records', 'primary', 'country')),)
    child = _read('observations', params=params)
    views = build_query_view_catalog(RelationCatalog(reads=(child,)))
    reference_views = build_query_view_catalog(RelationCatalog(reads=(records, *( _read('unrelated_'+str(i)) for i in range(extra_authorities)))))
    fact = SimpleNamespace(requested_fact_id='fact_1', input_refs=('i1',), output_kinds=('value',),
        output_origins=(origin,), ordering_origins=(), result_kind='scalar', selection_kind='all_results', selection_limit_input_ref=None)
    prompt = FactualQueryPrompt(question='How many observations belong to the specified record?', meaning=fact,
        tables=views.tables, parameters=menu.descriptions, inputs={'i1':term}, denotations={'i1':denotation}, reference_tables=reference_views.tables)
    return prompt, menu, next(iter(views.tables))


def test_unrelated_authorities_do_not_multiply_reference_relations_or_argument_symbols():
    selections = []
    for count in (0, 30):
        prompt, menu, _ = context(extra_authorities=count)
        assert len(prompt.tables) == 1 and len(prompt.reference_inputs) == 1
        assert not any(description.get('kind') == 'reference_argument' for description in prompt.parameters.values())
        slots, _, lowered = reference_contracts(prompt, (ReferenceDemand('i1', 'records/primary(id)'),), menu)
        assert len(slots) == 1
        assert sum(description.get('kind') == 'reference_argument' for description in lowered.descriptions.values()) == 1
        selections.append(slots[0])
    assert selections[0] == selections[1]
    assert selections[0].view.name == 'i1'


def test_composite_rest_binding_projects_each_declared_parameter_component():
    prompt, menu, view = context(composite=True)
    body = payload(query=f'SELECT COUNT(*) AS total FROM "{view}"',
        reference_demands=[{'input_ref':'i1', 'authority':'records/primary(country,id)'}],
        api_bindings=[{'view':view, 'parameter_ref':parameter, 'binding':{'reference_input':'i1'}}
            for parameter in ('record_id', 'country')])
    validate(body, prompt._schema())
    declared = parse_factual_query(body, prompt=prompt, menu=menu, selection_limit=None)
    _, _, lowered = reference_contracts(prompt, declared.demands, menu)
    components = {argument.parameter_ref: lowered.descriptions[argument.binding]['projection']
        for argument in declared.authored.request_arguments}
    assert components == {'record_id':'key_component:id', 'country':'key_component:country'}


@pytest.mark.parametrize('demands', [[], [{'input_ref':'i1','authority':'invented'}],
    [{'input_ref':'i1','authority':'records/primary(id)'}]*2])
def test_invalid_demands_are_rejected_inside_the_factual_parser(demands):
    prompt, menu, view = context()
    with pytest.raises(QueryValidationError):
        parse_factual_query(payload(query=f'SELECT COUNT(*) AS total FROM "{view}"', reference_demands=demands),
            prompt=prompt, menu=menu, selection_limit=None)


def test_mixed_mode_error_is_owned_by_parser_and_corrected_declaration_succeeds():
    prompt, menu, view = context(opaque=True)
    symbol = next(iter(menu.expressions))
    body = payload(query=f'SELECT COUNT(*) AS total FROM "{view}"',
        reference_demands=[{'input_ref':'i1','authority':'records/primary(id)'}],
        api_bindings=[{'view':view,'parameter_ref':'record_id','binding':symbol}])
    with pytest.raises(QueryValidationError, match='declared resolution mode'):
        parse_factual_query(body, prompt=prompt, menu=menu, selection_limit=None)
    corrected = {**body, 'reference_demands':[{'input_ref':'i1','authority':None}]}
    validate(corrected, prompt._schema())
    assert parse_factual_query(corrected, prompt=prompt, menu=menu, selection_limit=None).demands == (ReferenceDemand('i1', None),)


def test_schema_does_not_offer_reference_keys_to_incompatible_scalar_parameters():
    from jsonschema import ValidationError
    prompt, _, view = context(opaque=True)
    prompt.tables[view]['request_parameters'].append({'param_ref':'name', 'name':'name', 'type':'string', 'source':'query'})
    bad = payload(query=f'SELECT COUNT(*) AS total FROM "{view}"',
        reference_demands=[{'input_ref':'i1','authority':'records/primary(id)'}],
        api_bindings=[{'view':view,'parameter_ref':'name','binding':{'reference_input':'i1','component_id':'id'}}])
    with pytest.raises(ValidationError):
        validate(bad, prompt._schema())


def test_opaque_parameter_schema_offers_only_type_compatible_key_components():
    from jsonschema import ValidationError
    prompt, _, view = context(composite=True, opaque=True)
    body = payload(query=f'SELECT COUNT(*) AS total FROM "{view}"',
        reference_demands=[{'input_ref':'i1','authority':'records/primary(country,id)'}],
        api_bindings=[{'view':view,'parameter_ref':'record_id','binding':{'reference_input':'i1','component_id':'id'}}])
    validate(body, prompt._schema())
    body['api_invocations'][0]['arguments'][0]['binding']['component_id'] = 'country'
    with pytest.raises(ValidationError):
        validate(body, prompt._schema())


@pytest.mark.parametrize(('value','source','kind','allowed'), [
    ('42','path','integer',True), ('Alpha','path','integer',False),
    ('Alpha','path','string',True), ('Alpha','query','string',False),
    ('Alpha','query','uuid',False), ('00000000-0000-0000-0000-000000000001','query','uuid',True),
])
def test_literal_address_mode_requires_an_available_compatible_api_argument(value,source,kind,allowed):
    original, menu, view = context(opaque=True)
    parameter = {**original.tables[view]['request_parameters'][0], 'type':kind, 'source':source}
    tables = {view:{**original.tables[view],'request_parameters':[parameter]}}
    descriptions = {name:{**item,'value':value,'label':value} for name,item in menu.descriptions.items()}
    prompt = FactualQueryPrompt(question='Read the supplied record.',meaning=original.meaning,tables=tables,
        parameters=descriptions,inputs={'i1':replace(original.inputs['i1'],operand=value)},
        denotations=original.denotations,reference_tables=original.reference_tables)
    assert prompt.reference_inputs['i1']['allows_literal_address'] is allowed
    variant = prompt._schema()['properties']['reference_demands']['items']['anyOf'][0]
    assert (None in variant['properties']['authority']['enum']) is allowed


def test_reference_sql_names_are_local_while_program_identifiers_remain_disjoint():
    prompt, menu, _ = context()
    first, _, _ = reference_contracts(prompt,(ReferenceDemand('i1','records/primary(id)'),),menu)
    prompt.meaning = SimpleNamespace(**{**vars(prompt.meaning),'requested_fact_id':'fact_2'})
    second, _, _ = reference_contracts(prompt,(ReferenceDemand('i1','records/primary(id)'),),menu)
    assert first[0].view.name == second[0].view.name == 'i1'
    assert first[0].view.relation_id != second[0].view.relation_id
    assert first[0].reference_id != second[0].reference_id
