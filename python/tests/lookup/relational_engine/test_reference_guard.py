"""Reference cardinality is a typed operation independent of query syntax."""
import pytest
from fervis.lookup.answer_program.operations import ReferenceGuardSpec
from fervis.lookup.answer_program.result_projection import EntityKeyProjection,EntityKeyProjectionComponent
from fervis.lookup.plan_execution.operation_engine import execute_operations
from fervis.lookup.plan_execution.operation_runtime import ExecutableOperation,RelationEngineInput
from fervis.lookup.plan_execution.relations import RelationRows,CompletenessProof,CompletenessStatus


def run_guard(rows, *, identity=False, complete=True):
    key=EntityKeyProjection('record','primary',(EntityKeyProjectionComponent('id','id'),)) if identity else None
    fields=('id',) if identity else ('id','label')
    spec=ReferenceGuardSpec('candidates',fields,'i1',entity_key=key)
    from fervis.lookup.contract_codec import canonical_contract_json,decode_canonical_contract
    spec=decode_canonical_contract(canonical_contract_json(spec),ReferenceGuardSpec)
    return execute_operations(RelationEngineInput(relations=(RelationRows('candidates',tuple(rows),
        field_types={'id':'integer','label':'string'},completeness=CompletenessProof(
            status=CompletenessStatus.COMPLETE if complete else CompletenessStatus.INCOMPLETE,
            proof_refs=('read:candidates',))),),
        operations=(ExecutableOperation('guard',spec,'selected'),)))


@pytest.mark.parametrize('identity', [False, True])
def test_guard_preserves_selected_observed_values(identity):
    result=run_guard([{'id':7,'label':'Alpha','unrelated':'unused'}],identity=identity)
    assert result.issue is None
    assert result.relation('selected').grain_keys==(('id',) if identity else ())
    assert result.relation('selected').rows==(({'id':7} if identity else {'id':7,'label':'Alpha'}),)


@pytest.mark.parametrize('rows,reason', [([], 'NOT_FOUND'),
    ([{'id':1,'label':'Alpha'},{'id':1,'label':'Alpha'}], 'AMBIGUOUS_RESULT'),
    ([{'id':1,'label':'Alpha'},{'id':2,'label':'Alpha'}], 'AMBIGUOUS_RESULT')])
def test_uncertified_occurrences_are_not_deduplicated(rows,reason):
    result=run_guard(rows)
    assert result.issue.reference.reason.value==reason
    assert result.issue.reference.candidates==()


def test_nominal_key_uniqueness_deduplicates_only_certified_keys():
    result=run_guard([{'id':1,'label':'A'},{'id':1,'label':'B'}],identity=True)
    assert result.issue is None
    assert result.relation('selected').rows==({'id':1},)


def test_incomplete_candidates_cannot_certify_uniqueness():
    result=run_guard([{'id':1,'label':'Alpha'}],complete=False)
    assert result.issue is not None
