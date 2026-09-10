from fervis.lookup.relational_sql.outputs import identity_authorities


def test_output_identity_choices_are_logical_and_independent_of_carriers():
    tables = {
        'staff': {'candidate_keys': [{'entity_kind': 'staff', 'key_id': 'pk', 'components': {'id': 'staff_id'}}]},
        'sales': {'entity_references': [{'target_entity_kind': 'staff', 'target_key_id': 'pk', 'components': {'id': 'seller_id'}}]},
        'slot': {'candidate_keys': [{'entity_kind': 'staff', 'key_id': 'pk', 'components': {'id': 'id'}}]},
    }
    assert identity_authorities(tables) == {
        'staff/pk(id)': {'entity_kind': 'staff', 'key_id': 'pk', 'components': ('id',)}}


def test_distinct_identity_domains_and_composite_shapes_remain_separate():
    tables = {'records': {'candidate_keys': [
        {'entity_kind': kind, 'key_id': key, 'components': components}
        for kind, key, components in [
            ('staff', 'pk', {'id': 'id'}), ('sale', 'pk', {'id': 'id'}),
            ('staff', 'external', {'id': 'external_id'}),
            ('district', 'pk', {'country': 'country', 'id': 'id'})]]}}
    assert set(identity_authorities(tables)) == {'staff/pk(id)', 'sale/pk(id)', 'staff/external(id)', 'district/pk(country,id)'}


def test_prompt_columns_expose_declared_identity_roles_without_promoting_pointers():
    from fervis.lookup.relational_sql.outputs import annotate_identity_columns
    tables = {
        'sites': {'columns':{'id':{'type':'integer'},'name':{'type':'string'}},
                  'candidate_keys':[{'entity_kind':'site','key_id':'pk','components':{'id':'id'}}]},
        'settings': {'columns':{'reporting_site_id':{'type':'integer'}}},
    }
    shown = annotate_identity_columns(tables)
    assert shown['sites']['columns']['id']['identity_roles'] == [{'authority':'site/pk(id)','component':'id'}]
    assert 'identity_roles' not in shown['settings']['columns']['reporting_site_id']
    assert 'identity_roles' not in tables['sites']['columns']['id']
