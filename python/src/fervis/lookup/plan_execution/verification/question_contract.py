"""Question-contract checks for fact-plan verification."""

from ._shared import QuestionContract, RequestedFact, VerificationError


def _verify_question_contract(question_contract: QuestionContract) -> None:
    question_input_ids = _verify_question_inputs(question_contract)
    _verify_requested_facts(
        question_contract.requested_facts,
        question_input_ids=question_input_ids,
    )


def _verify_question_inputs(question_contract: QuestionContract) -> set[str]:
    output: set[str] = set()
    for known in question_contract.question_inputs:
        if known.id in output:
            raise VerificationError("duplicate question input")
        output.add(known.id)
        if not known.text:
            raise VerificationError("known input requires text")
    return output


def _verify_requested_facts(
    requested_facts: tuple[RequestedFact, ...],
    *,
    question_input_ids: set[str],
) -> None:
    seen: set[str] = set()
    known_input_ids: set[str] = set()
    for fact in requested_facts:
        if not fact.id or not fact.description:
            raise VerificationError("requested fact requires id and description")
        if not fact.answer_outputs:
            raise VerificationError("requested fact requires answer outputs")
        if fact.id in seen:
            raise VerificationError("duplicate requested fact")
        seen.add(fact.id)
        _verify_unique_ids(
            (output.id for output in fact.support_answer_outputs),
            message="duplicate requested fact answer output",
        )
        _verify_unique_ids(
            (known.id for known in fact.known_inputs),
            message="duplicate requested fact known input",
        )
        output_ids = {output.id for output in fact.support_answer_outputs}
        known_ids = {known.id for known in fact.known_inputs}
        if output_ids & known_ids:
            raise VerificationError(
                "answer output and known input ids must be disjoint"
            )
        for input_ref in fact.input_refs:
            if input_ref not in question_input_ids:
                raise VerificationError(
                    "answer request references unknown question input"
                )
        for known in fact.known_inputs:
            if not question_input_ids and known.id in known_input_ids:
                raise VerificationError("duplicate requested fact known input")
            known_input_ids.add(known.id)
            if not known.text:
                raise VerificationError("known input requires text")


def _verify_unique_ids(ids: object, *, message: str) -> None:
    seen: set[str] = set()
    for item in ids:
        value = str(item or "")
        if not value:
            raise VerificationError(message)
        if value in seen:
            raise VerificationError(message)
        seen.add(value)
