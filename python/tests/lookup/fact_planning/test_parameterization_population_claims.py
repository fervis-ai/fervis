from fervis.lookup.answer_program.relations import (
    PopulationCoverageClaim,
    PopulationCoverageRole,
)
from fervis.lookup.fact_planning.pattern_plan.parameterization import (
    _partition_population_claims,
)
from fervis.lookup.question_contract import MembershipTestRef


def test_combined_source_and_returned_filter_claim_runs_after_the_filter() -> None:
    claim = PopulationCoverageClaim(
        test_ref=MembershipTestRef("fact_1", "requested_state"),
        role=PopulationCoverageRole.ROW_POPULATION,
        proof_refs=("source_param:state", "returned_field:data.state"),
    )

    returned, source = _partition_population_claims(
        (claim,),
        returned_filter_proof_refs=("returned_field:data.state",),
    )

    assert returned == (claim,)
    assert source == ()
