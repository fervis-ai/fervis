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


@pytest.mark.parametrize("regions,expected", [
    (("West", "East"), 2),
    (("West", "West"), 0),
    ((None, "West"), 0),
])
def test_anonymous_ambiguity_offers_only_distinguishable_observed_properties(
    regions, expected,
):
    from fervis.lookup.answer_program.operations import ObservedReferenceProperty

    spec = ReferenceGuardSpec(
        "candidates", ("occurrence", "name", "region"), "district_name",
        occurrence_fields=("occurrence",),
        observed_source_ref="districts",
        observed_properties=(
            ObservedReferenceProperty("name", "district.name", "string", "name"),
            ObservedReferenceProperty("region", "district.region", "string", "region"),
        ),
    )
    rows = tuple({"occurrence": index, "name": "River District", "region": region}
                 for index, region in enumerate(regions, start=1))
    result = execute_operations(RelationEngineInput(
        relations=(RelationRows(
            "candidates", rows,
            field_types={"occurrence": "integer", "name": "string", "region": "string"},
            completeness=CompletenessProof(status=CompletenessStatus.COMPLETE),
        ),),
        operations=(ExecutableOperation("guard", spec, "selected"),),
    ))
    assert result.issue.reference.reason.value == "AMBIGUOUS_RESULT"
    candidates = result.issue.reference.observed_candidates
    assert len(candidates) == expected
    if candidates:
        assert {candidate.display_properties[0].field_ref for candidate in candidates} == {
            "district.region"
        }
        assert {candidate.display_properties[0].value for candidate in candidates} == {
            "West", "East"
        }
        assert all({item.field_ref for item in candidate.properties} == {
            "district.name", "district.region"
        } for candidate in candidates)


def test_anonymous_choice_can_require_a_combination_of_observed_properties():
    from fervis.lookup.answer_program.operations import ObservedReferenceProperty

    spec = ReferenceGuardSpec(
        "candidates", ("occurrence", "name", "region", "zone"), "district_name",
        occurrence_fields=("occurrence",), observed_source_ref="districts",
        observed_properties=(
            ObservedReferenceProperty("name", "district.name", "string", "name"),
            ObservedReferenceProperty("region", "district.region", "string", "region"),
            ObservedReferenceProperty("zone", "district.zone", "integer", "zone"),
        ),
    )
    rows = tuple({"occurrence": index, "name": "River District",
                  "region": region, "zone": zone}
                 for index, (region, zone) in enumerate((
                     ("West", 1), ("West", 2), ("East", 1), ("East", 2),
                 ), start=1))
    result = execute_operations(RelationEngineInput(
        relations=(RelationRows(
            "candidates", rows,
            field_types={"occurrence": "integer", "name": "string",
                         "region": "string", "zone": "integer"},
            completeness=CompletenessProof(status=CompletenessStatus.COMPLETE),
        ),),
        operations=(ExecutableOperation("guard", spec, "selected"),),
    ))
    candidates = result.issue.reference.observed_candidates
    assert len(candidates) == 4
    assert all({item.field_ref for item in candidate.display_properties} == {
        "district.region", "district.zone"
    } for candidate in candidates)
    assert all(len(candidate.properties) == 3 for candidate in candidates)
