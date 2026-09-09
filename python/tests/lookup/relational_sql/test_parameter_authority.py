import pytest
from fervis.lookup.relational_sql.parameter_usage import compatible_argument, validate_identity_key_uses
from fervis.lookup.relational_sql.execution import QueryValidationError

TARGET = {'entity_kind': 'shop.customer', 'key_id': 'primary', 'component_id': 'id'}


def test_catalog_enum_cannot_bind_an_identity_parameter():
    assert not compatible_argument({'type': 'uuid', 'entity_target': TARGET},
        {'kind': 'catalog_choice', 'type': 'choice', 'value': 'DRAFT'})


def test_untyped_uuid_cannot_replace_a_grounded_identity():
    assert not compatible_argument({'type': 'uuid', 'entity_target': TARGET},
        {'kind': 'input', 'value_type': 'uuid', 'value': '00000000-0000-0000-0000-000000000001'})


@pytest.mark.parametrize('domain,expected', [('shop.customer', True), ('accounts.customer', False)])
def test_identity_parameter_requires_its_declared_domain(domain, expected):
    assert compatible_argument({'type': 'uuid', 'entity_target': TARGET},
        {'kind': 'reference_argument', 'identity': {'entity_kind': domain, 'key_id': 'primary'},
         'projection': 'key_component:id', 'value_type': 'uuid'}) is expected


@pytest.mark.parametrize('wrapped', [False, True, 'in', 'cast'])
def test_reference_relation_cannot_compare_keys_from_another_domain(wrapped):
    def view(domain):
        return {'columns': {'id': {'type': 'uuid'}}, 'candidate_keys': [
            {'entity_kind': domain, 'key_id': 'primary', 'components': {'id': 'id'}}]}
    tables = {'resolved': {**view('accounts.customer'), 'kind': 'resolved_reference'},
              'carts': view('shop.customer')}
    query = ('WITH selected AS (SELECT id FROM resolved) SELECT c.id FROM carts c JOIN selected r ON c.id=r.id'
             if wrapped else 'SELECT c.id FROM carts c JOIN resolved r ON c.id=r.id')
    if wrapped == 'in':
        query = 'SELECT c.id FROM carts c WHERE c.id IN (SELECT id FROM resolved)'
    if wrapped == 'cast':
        query = 'SELECT c.id FROM carts c JOIN resolved r ON CAST(c.id AS VARCHAR)=CAST(r.id AS VARCHAR)'
    with pytest.raises(QueryValidationError, match='identity'):
        validate_identity_key_uses(query, tables, {})


def test_nested_predicate_parameter_is_checked_in_its_own_scope():
    tables = {
        'customers': {'columns': {'id': {'type': 'integer'}}, 'candidate_keys': [
            {'entity_kind': 'customer', 'key_id': 'primary', 'components': {'id': 'id'}}]},
        'orders': {'columns': {'id': {'type': 'integer'}, 'customer_id': {'type': 'integer'}}, 'candidate_keys': [
            {'entity_kind': 'order', 'key_id': 'primary', 'components': {'id': 'id'}}], 'entity_references': [
            {'target_entity_kind': 'customer', 'target_key_id': 'primary', 'components': {'id': 'customer_id'}}]},
    }
    validate_identity_key_uses('SELECT id FROM customers WHERE id IN (SELECT customer_id FROM orders WHERE id=$order)',
        tables, {'order': {'identity': {'entity_kind': 'order', 'key_id': 'primary'}, 'projection': 'key_component:id'}})


def test_guarded_identity_cannot_be_used_as_an_untyped_uuid_argument():
    assert not compatible_argument({'source':'query','type':'uuid','entity_target':None},
        {'kind':'reference_argument','identity':{'entity_kind':'customer','key_id':'primary'},
         'projection':'key_component:id','value_type':'uuid'})


@pytest.mark.parametrize('source,kind,allowed', [('path','uuid',True), ('query','uuid',True), ('query','string',False)])
def test_original_reference_literal_can_address_an_opaque_resource(source, kind, allowed):
    description = {'kind':'reference_literal','value_type':'string','value':'00000000-0000-0000-0000-000000000001'}
    assert compatible_argument({'source':source,'type':kind,'entity_target':None},description) is allowed
    assert not compatible_argument({'source':source,'type':kind,'entity_target':TARGET},description)


@pytest.mark.parametrize(('kind', 'valid', 'invalid'), [
    ('integer', '42', '42.5'), ('number', '12.5', 'twelve'),
    ('uuid', '00000000-0000-0000-0000-000000000001', '42'),
    ('boolean', 'true', 'sometimes'), ('date', '2026-03-01', '2026-02-30')])
def test_literal_address_uses_declared_scalar_parser_without_identity_promotion(kind, valid, invalid):
    parameter = {'source': 'path', 'type': kind, 'entity_target': None}
    description = {'kind': 'reference_literal', 'value_type': 'string', 'value': valid}
    assert compatible_argument(parameter, description)
    assert not compatible_argument(parameter, {**description, 'value': invalid})
    assert not compatible_argument({**parameter, 'entity_target': TARGET}, description)
