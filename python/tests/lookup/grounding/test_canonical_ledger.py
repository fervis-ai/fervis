"""Repeated uses share one certified value without interning distinct inputs."""
from dataclasses import replace
import pytest
from fervis.lookup.answer_program.values import FactValue, LiteralType
from fervis.lookup.grounding.semantic import CanonicalInputValue, build_canonical_input_ledger


def _value(use, proof, *, input_ref='i1', value_id='v1', value='4'):
    return CanonicalInputValue(value_id, input_ref, (use,),
        FactValue.literal(id=value_id, known_input_id=input_ref, literal_type=LiteralType.NUMBER,
                          value=value, proof_refs=(proof,)), (proof,))


def test_repeated_input_uses_merge_certification_without_duplicate_value_ids():
    ledger = build_canonical_input_ledger((_value('use1','proof1'), _value('use2','proof2')),
                                         required_use_refs=('use1','use2'))
    assert len(ledger) == 1
    assert ledger[0].use_refs == ('use1','use2')
    assert ledger[0].certification_refs == ('proof1','proof2')
    assert ledger[0].typed_value.proof_refs == ('proof1','proof2')


def test_distinct_equal_inputs_keep_separate_value_ownership():
    ledger = build_canonical_input_ledger((_value('use1','proof1'), _value('use2','proof2',input_ref='i2',value_id='v2')),
                                         required_use_refs=('use1','use2'))
    assert [value.input_ref for value in ledger] == ['i1','i2']


@pytest.mark.parametrize('second', (_value('use2','proof2',input_ref='i2'), _value('use2','proof2',value='5')))
def test_one_value_id_cannot_merge_conflicting_inputs(second):
    with pytest.raises(ValueError, match='conflicting'):
        build_canonical_input_ledger((_value('use1','proof1'),second),required_use_refs=('use1','use2'))


def test_duplicate_use_is_not_hidden_by_value_merging():
    with pytest.raises(ValueError, match='repeats an input use'):
        build_canonical_input_ledger((_value('use1','proof1'),_value('use1','proof2')),required_use_refs=('use1',))


def test_equal_identity_keys_keep_each_routes_match_evidence():
    from fervis.lookup.canonical_data import EntityKeyValue, EntityKeyComponentValue
    key = EntityKeyValue('store', 'primary', (EntityKeyComponentValue('id','s1'),))
    def resolved(use, route, entity_key=key):
        value = FactValue.identity(id='i1:identity:1', known_input_id='i1', key=entity_key,
            display_value='Nairobi', matched_field_ref=route+':name', matched_field_path='data.name',
            matched_value='Nairobi', proof_refs=('read:'+route,), source_refs=(route,))
        return CanonicalInputValue(value.id,'i1',(use,),value,value.proof_refs)
    merged, = build_canonical_input_ledger((resolved('use1','list'), resolved('use2','search')),
                                           required_use_refs=('use1','use2'))
    assert merged.typed_value.payload.key == key
    assert {e.matched_field_ref for e in merged.typed_value.identity_evidence} == {'list:name','search:name'}
    assert {e.proof_refs for e in merged.typed_value.identity_evidence} == {('read:list',),('read:search',)}
    from fervis.lookup.contract_codec import decode_canonical_contract, canonical_contract_payload
    assert decode_canonical_contract(canonical_contract_payload(merged.typed_value), FactValue) == merged.typed_value
    different = EntityKeyValue('store', 'primary', (EntityKeyComponentValue('id','s2'),))
    with pytest.raises(ValueError, match='conflicting'):
        build_canonical_input_ledger((resolved('use1','list'), resolved('use2','search',different)),
                                     required_use_refs=('use1','use2'))
