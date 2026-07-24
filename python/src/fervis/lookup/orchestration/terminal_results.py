"""Terminal-result assembly owned by the semantic lookup runtime."""

from hashlib import sha256

from fervis.lookup.clarification import (
    ClarificationCause,
    ClarificationOption,
    CatalogInputTarget,
    MissingCatalogChoice,
    MissingCatalogRequiredValue,
    MissingAnswerMetric,
    SourceBindingCatalogInputContinuation,
    TargetReferenceAmbiguous,
    TargetReferenceNotFound,
    TargetReferenceUnsupported,
    clarify,
)
from fervis.lookup.canonical_data import (
    EntityKeyValue,
    canonical_runtime_json,
    entity_key_to_payload,
)
from fervis.lookup.grounding import (
    IdentityExecutionClarification,
    IdentityExecutionFailureReason,
)
from fervis.lookup.outcomes.model import FactResult, NeedsClarification
from fervis.lookup.question_contract import QuestionContract, analyze_requested_fact
from fervis.lookup.question_contract.clarification import (
    IncompleteFactualRequestKind,
    QuestionContractNeedsClarification,
)
from fervis.lookup.read_eligibility import NoCanonicalInterpretation, NoResolverRoute
from fervis.lookup.source_binding import SourceBindingClarification


def _question_contract_clarification_fact_result(
    outcome: QuestionContractNeedsClarification,
) -> FactResult:
    clarifications = []
    for index, item in enumerate(outcome.missing, start=1):
        if (
            item.missing_kind
            is IncompleteFactualRequestKind.UNRESOLVED_PRIOR_TURN_REFERENCE
        ):
            known_input_id = f"question_contract:{item.source_text}"
            cause: ClarificationCause = TargetReferenceNotFound(
                clarification_id=f"clarify_question_contract_{index}",
                requested_fact_id="question_contract",
                known_input_id=known_input_id,
                source_text=item.source_text,
                target_label=item.target_label,
                proof_refs=(
                    f"known_input:{known_input_id}",
                    "question_contract:needs_clarification",
                ),
            )
        else:
            cause = MissingAnswerMetric(
                clarification_id=f"clarify_question_contract_{index}",
                requested_fact_id="question_contract",
                source_text=item.source_text,
                metric_needed=item.why_question_is_incomplete,
                proof_refs=(
                    "requested_fact:question_contract",
                    "question_contract:needs_clarification",
                ),
            )
        clarifications.append(clarify(cause))
    return FactResult(outcome=NeedsClarification(clarifications=tuple(clarifications)))


def semantic_clarification_fact_result(
    cause: object,
    *,
    contract: QuestionContract | None,
) -> FactResult:
    if isinstance(cause, QuestionContractNeedsClarification):
        return _question_contract_clarification_fact_result(cause)
    if isinstance(cause, SourceBindingClarification):
        causes = _source_binding_causes(cause)
    elif isinstance(
        cause,
        (IdentityExecutionClarification, NoCanonicalInterpretation, NoResolverRoute),
    ):
        if contract is None:
            raise ValueError("identity clarification requires a question contract")
        causes = (_identity_cause(cause, contract=contract),)
    else:
        raise TypeError(f"unsupported semantic clarification: {type(cause).__name__}")
    return FactResult(
        outcome=NeedsClarification(
            clarifications=tuple(clarify(item) for item in causes)
        )
    )


def _identity_cause(
    cause: IdentityExecutionClarification
    | NoCanonicalInterpretation
    | NoResolverRoute,
    *,
    contract: QuestionContract,
) -> ClarificationCause:
    task_ref = cause.task_ref
    input_ref, use_refs = _identity_subject(cause, contract=contract)
    input_term = next(item for item in contract.inputs if item.id == input_ref)
    denotation = next(
        item for item in contract.input_denotations if item.input_ref == input_ref
    )
    requested_fact_id = _requested_fact_id(contract, use_refs=use_refs)
    source_text = (
        input_term.operand
        if isinstance(input_term.operand, str)
        else ", ".join(input_term.operand)
    )
    clarification_id = f"clarify_identity:{task_ref}"
    proof_refs = tuple(cause.evidence_refs)
    if not isinstance(cause, IdentityExecutionClarification):
        return TargetReferenceUnsupported(
            clarification_id=clarification_id,
            requested_fact_id=requested_fact_id,
            known_input_id=input_ref,
            source_text=source_text,
            target_label=denotation.operand_meaning,
            proof_refs=proof_refs,
        )
    if cause.reason is IdentityExecutionFailureReason.NOT_FOUND:
        return TargetReferenceNotFound(
            clarification_id=clarification_id,
            requested_fact_id=requested_fact_id,
            known_input_id=input_ref,
            source_text=source_text,
            target_label=denotation.operand_meaning,
            proof_refs=proof_refs,
        )
    if cause.reason is IdentityExecutionFailureReason.AMBIGUOUS_RESULT:
        return TargetReferenceAmbiguous(
            clarification_id=clarification_id,
            requested_fact_id=requested_fact_id,
            known_input_id=input_ref,
            source_text=source_text,
            target_label=denotation.operand_meaning,
            proof_refs=proof_refs,
            options=tuple(
                ClarificationOption(
                    id=_identity_option_id(candidate.key),
                    label=candidate.display_value,
                    key=candidate.key,
                    matched_field=candidate.matched_field_path,
                    matched_value=candidate.display_value,
                    resolver_read_id=candidate.resolver_read_id,
                )
                for candidate in cause.candidates
            ),
        )
    return TargetReferenceUnsupported(
        clarification_id=clarification_id,
        requested_fact_id=requested_fact_id,
        known_input_id=input_ref,
        source_text=source_text,
        target_label=denotation.operand_meaning,
        proof_refs=proof_refs,
    )


def _identity_subject(
    cause: IdentityExecutionClarification
    | NoCanonicalInterpretation
    | NoResolverRoute,
    *,
    contract: QuestionContract,
) -> tuple[str, tuple[str, ...]]:
    if isinstance(cause, IdentityExecutionClarification):
        return cause.input_ref, cause.use_refs
    for fact in contract.requested_facts:
        index = analyze_requested_fact(
            fact,
            inputs={item.id: item for item in contract.inputs},
            input_denotations={
                item.input_ref: item for item in contract.input_denotations
            },
        )
        for use in index.input_use_sites:
            if cause.task_ref.startswith(f"{use.use_ref}:"):
                return use.input_ref, (use.use_ref,)
    raise ValueError("identity clarification task has no question input")


def _requested_fact_id(
    contract: QuestionContract,
    *,
    use_refs: tuple[str, ...],
) -> str:
    use_ref_set = set(use_refs)
    for fact in contract.requested_facts:
        index = analyze_requested_fact(
            fact,
            inputs={item.id: item for item in contract.inputs},
            input_denotations={
                item.input_ref: item for item in contract.input_denotations
            },
        )
        if use_ref_set.intersection(item.use_ref for item in index.input_use_sites):
            return fact.id
    raise ValueError("identity clarification has no requested fact")


def _identity_option_id(key: EntityKeyValue) -> str:
    fingerprint = sha256(
        canonical_runtime_json(entity_key_to_payload(key)).encode("utf-8")
    ).hexdigest()
    return f"identity_option:{fingerprint}"


def _source_binding_causes(
    cause: SourceBindingClarification,
) -> tuple[ClarificationCause, ...]:
    output: list[ClarificationCause] = []
    for missing in cause.missing_catalog_values:
        target = CatalogInputTarget(
            row_source_id=missing.source_ref,
            param_id=missing.parameter_id,
            param_ref=missing.target_ref,
            value_type=missing.value_type,
            choices=missing.allowed_values,
        )
        continuation = SourceBindingCatalogInputContinuation(
            requested_fact_id=cause.requested_fact_id,
            target=target,
        )
        if missing.allowed_values:
            output.append(
                MissingCatalogChoice(
                    clarification_id=missing.catalog_input_ref,
                    requested_fact_id=cause.requested_fact_id,
                    label=missing.label,
                    continuation=continuation,
                    proof_refs=missing.evidence_refs,
                    required_choice_input_id=(
                        f"{missing.source_ref}.{missing.parameter_id}"
                    ),
                    options=tuple(
                        ClarificationOption(id=value, label=value, value=value)
                        for value in missing.allowed_values
                    ),
                )
            )
        else:
            output.append(
                MissingCatalogRequiredValue(
                    clarification_id=missing.catalog_input_ref,
                    requested_fact_id=cause.requested_fact_id,
                    label=missing.label,
                    continuation=continuation,
                    proof_refs=missing.evidence_refs,
                    required_input_id=f"{missing.source_ref}.{missing.parameter_id}",
                )
            )
    return tuple(output)
