from fervis.lookup.canonical_data import EntityKeyComponentValue, EntityKeyValue
from fervis.lookup.grounding import (
    IdentityExecutionCandidate,
    IdentityExecutionClarification,
    IdentityExecutionFailureReason,
)
from fervis.lookup.orchestration.terminal_results import (
    semantic_clarification_fact_result,
)
from fervis.lookup.outcomes.model import NeedsClarification
from tests.lookup.read_eligibility.test_semantic_read_eligibility import (
    _semantic_contract,
)


def test_ambiguous_identity_keys_produce_stable_clarification_options() -> None:
    parsed = _semantic_contract()
    [index] = parsed.semantic_indexes
    [use] = index.input_use_sites
    candidates = tuple(
        IdentityExecutionCandidate(
            key=EntityKeyValue(
                entity_kind="staff",
                key_id="primary_key",
                components=(EntityKeyComponentValue("staff_id", value),),
            ),
            display_value="Ada",
            matched_field_ref="staff.name",
            matched_field_path="data.name",
            resolver_read_id="list_staff_list",
        )
        for value in ("staff_1", "staff_2")
    )
    cause = IdentityExecutionClarification(
        task_ref=f"{use.use_ref}:reference_grounding",
        input_ref=use.input_ref,
        use_refs=(use.use_ref,),
        reason=IdentityExecutionFailureReason.AMBIGUOUS_RESULT,
        evidence_refs=("list_staff_list",),
        candidates=candidates,
    )

    result = semantic_clarification_fact_result(
        cause,
        contract=parsed.contract,
    )

    assert isinstance(result.outcome, NeedsClarification)
    [clarification] = result.outcome.clarifications
    [subject] = clarification.subjects
    assert len(subject.options) == 2
    assert len({option.id for option in subject.options}) == 2
