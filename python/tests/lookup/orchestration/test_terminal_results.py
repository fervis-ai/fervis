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


def test_executed_reference_uses_real_keys_without_inventing_field_match_evidence():
    from fervis.lookup.identity_types import ReferenceResolutionFailure
    from fervis.lookup.outcomes.errors import ExecutionIssue,ExecutionIssueKind
    from fervis.lookup.orchestration.terminal_results import reference_clarification_fact_result
    parsed=_semantic_contract();use=parsed.semantic_indexes[0].input_use_sites[0]
    keys=tuple(EntityKeyValue('staff','primary_key',(EntityKeyComponentValue('staff_id',value),)) for value in ('one','two'))
    issue=ExecutionIssue(ExecutionIssueKind.REFERENCE_RESOLUTION,'Ambiguous reference',proof_refs=('read:staff',),
        reference=ReferenceResolutionFailure(use.input_ref,IdentityExecutionFailureReason.AMBIGUOUS_RESULT,keys))
    result=reference_clarification_fact_result(issue,contract=parsed.contract)
    options=result.outcome.clarifications[0].subjects[0].options
    assert tuple(option.key for option in options)==keys
    assert len({option.id for option in options})==2
    assert all(not option.matched_field and not option.resolver_read_id for option in options)


def test_reference_choice_resumes_with_user_authority_and_no_fabricated_read():
    from fervis.lookup.identity_types import ReferenceResolutionFailure
    from fervis.lookup.outcomes.errors import ExecutionIssue,ExecutionIssueKind
    from fervis.lookup.orchestration.terminal_results import reference_clarification_fact_result
    from fervis.lookup.clarification.response import parse_clarification_response
    from fervis.lookup.orchestration.semantic_compilation import _grounding_response_values
    from fervis.lookup.question_contract.grounding_context import grounding_context_from_index
    from fervis.lookup.relational_sql.parameters import query_parameter_menu
    parsed=_semantic_contract();index=parsed.semantic_indexes[0];use=index.input_use_sites[0]
    keys=tuple(EntityKeyValue('staff','primary_key',(EntityKeyComponentValue('staff_id',value),)) for value in ('one','two'))
    issue=ExecutionIssue(ExecutionIssueKind.REFERENCE_RESOLUTION,'Choose a staff member',proof_refs=('read:staff',),
        reference=ReferenceResolutionFailure(use.input_ref,IdentityExecutionFailureReason.AMBIGUOUS_RESULT,keys))
    clarification=reference_clarification_fact_result(issue,contract=parsed.contract).outcome.clarifications[0]
    option=clarification.subjects[0].options[1]
    response=parse_clarification_response(clarification,response_id='user_selection',response_text=option.label,selected_option_id=option.id)
    (value,)=_grounding_response_values(indexes=(grounding_context_from_index(index),),responses=(response,))
    assert value.typed_value.payload.key==keys[1]
    assert value.typed_value.identity_evidence==()
    assert value.typed_value.source_refs==()
    assert len(value.typed_value.proof_refs)==1 and 'user_selection' in value.typed_value.proof_refs[0]
    menu=query_parameter_menu((value,))
    assert len(menu.program_inputs.bindings.bindings)==1


def test_collection_reference_clarification_preserves_the_unresolved_member():
    from fervis.lookup.identity_types import ReferenceResolutionFailure
    from fervis.lookup.outcomes.errors import ExecutionIssue,ExecutionIssueKind
    from fervis.lookup.orchestration.terminal_results import reference_clarification_fact_result
    from fervis.lookup.clarification.response import parse_clarification_response,clarification_response_payload,clarification_response_from_payload
    from fervis.lookup.question_contract.model import IntentQuestionContract,IntentFact,IntentOutput,InputTerm,InputDenotation,InputDenotationKind
    from fervis.lookup.semantic_types import SourceOrigin,SourceOriginKind,TextType,CollectionType
    from fervis.lookup.question_contract.grounding_context import grounding_fact_context
    from fervis.lookup.orchestration.semantic_compilation import _grounding_response_values
    origin=SourceOrigin(SourceOriginKind.QUESTION_CONTEXT,'Alpha and Beta')
    term=InputTerm('i1',origin,('Alpha','Beta'),CollectionType(TextType()))
    denotation=InputDenotation('d1','i1','sites','The two names identify sites.','site',InputDenotationKind.IDENTITY_REFERENCE)
    fact=IntentFact('fact_1',origin,(IntentOutput('r1',origin),),('i1',),'scalar')
    contract=IntentQuestionContract((term,),(fact,),(denotation,))
    key=EntityKeyValue('site','primary_key',(EntityKeyComponentValue('id',2),))
    issue=ExecutionIssue(ExecutionIssueKind.REFERENCE_RESOLUTION,'Reference unresolved',
        reference=ReferenceResolutionFailure('i1',IdentityExecutionFailureReason.AMBIGUOUS_RESULT,(key,EntityKeyValue('site','primary_key',(EntityKeyComponentValue('id',3),))),operand='Beta'))
    clarification=reference_clarification_fact_result(issue,contract=contract).outcome.clarifications[0]
    assert clarification.subjects[0].source_text=='Beta'
    assert clarification.continuation.reference_operand=='Beta'
    option=clarification.subjects[0].options[0]
    response=parse_clarification_response(clarification,response_id='choose_beta',response_text=option.label,selected_option_id=option.id)
    restored=clarification_response_from_payload(clarification_response_payload(response))
    assert restored.reference_operand=='Beta' and restored.option.key==key
    context=grounding_fact_context(fact,inputs={'i1':term},input_denotations={'i1':denotation})
    assert _grounding_response_values(indexes=(context,),responses=(restored,))==()
