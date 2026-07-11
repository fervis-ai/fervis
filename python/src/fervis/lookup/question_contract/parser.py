"""Parse provider-authored question-contract decisions."""

from __future__ import annotations

import re
from typing import Any

from fervis.lookup.conversation_resolution.compilation import (
    CompiledConversationResolution,
)
from fervis.lookup.question_contract._text_spans import copied_span
from fervis.lookup.question_contract import provider_contract as provider_output
from fervis.lookup.question_inputs import KnownInputKind, LiteralInputRole
from fervis.lookup.question_contract.model import (
    AnswerPopulationMembershipTestKind,
    AnswerPopulationMembershipTestPolarity,
    AnswerSubjectInstanceInterpretationKind,
    KnownInputSource,
    MissingQuestionInput,
    MissingQuestionInputType,
    QuestionContract,
    QuestionContractNeedsClarification,
    QuestionContractResult,
    GroupKeyDomainKind,
    RequestedFact,
    RequestedFactAnswerExpression,
    RequestedFactAnswerExpressionFamily,
    RequestedFactGroupKey,
    RequestedFactAnswerPopulation,
    RequestedFactAnswerPopulationMembershipTest,
    RequestedFactAnswerOutput,
    RequestedFactAnswerSubject,
    RequestedFactAnswerSubjectInstanceInterpretation,
    RequestedFactKnownInput,
    RequestedFactLiteralInput,
    RequestedFactRowSetReferenceInput,
)
from fervis.lookup.question_contract.tools import (
    ANSWER_REQUEST_CONTRACT_TOOL_NAME,
    MISSING_INPUT_CLARIFICATION_TOOL_NAME,
)


def parse_question_contract(
    *,
    tool_name: str,
    payload: dict[str, Any],
    question_context: str,
    question_context_texts: tuple[str, ...] = (),
    current_question_context_texts: tuple[str, ...] = (),
    conversation_resolution: CompiledConversationResolution | None = None,
) -> QuestionContractResult:
    question_text = _text(question_context)
    if not question_text:
        raise ValueError("question context is required")
    current_question_texts = (question_text, *current_question_context_texts)
    context_texts = (question_text, *question_context_texts)

    if tool_name == MISSING_INPUT_CLARIFICATION_TOOL_NAME:
        parsed = provider_output.MissingInputClarificationOutput.parse(payload)
        if _text(parsed.kind) != "needs_clarification":
            raise ValueError("invalid question contract clarification kind")
        return QuestionContractResult(
            outcome=QuestionContractNeedsClarification(
                missing=_missing_question_inputs(
                    parsed.missing,
                    question_context_texts=context_texts,
                ),
            )
        )

    if tool_name != ANSWER_REQUEST_CONTRACT_TOOL_NAME:
        raise ValueError("unknown question contract tool")

    parsed = provider_output.QuestionContractOutput.parse(payload)
    kind = _text(parsed.kind)
    if kind != "question_contract":
        raise ValueError("invalid question contract kind")
    question_inputs = _question_inputs(
        parsed.question_inputs,
        current_question_texts=current_question_texts,
        question_context_texts=context_texts,
    )
    _validate_conversation_resolution_question_inputs(
        question_inputs,
        conversation_resolution=conversation_resolution,
    )
    requested_facts = _requested_facts(
        parsed.answer_requests,
        question_inputs=question_inputs,
        question_context_texts=context_texts,
    )
    _reject_unowned_question_inputs(
        question_inputs,
        requested_facts=requested_facts,
    )
    _reject_answer_subject_question_inputs(
        question_inputs,
        requested_facts=requested_facts,
    )
    question_inputs = _referenced_question_inputs(
        question_inputs,
        requested_facts=requested_facts,
    )
    _validate_answer_requests_count(
        parsed.answer_requests_count,
        requested_facts=requested_facts,
    )
    _question_input_inventory_check(parsed.question_input_inventory_check)
    return QuestionContractResult(
        outcome=QuestionContract(
            question_inputs=question_inputs,
            requested_facts=requested_facts,
        )
    )


def _missing_question_inputs(
    raw: Any,
    *,
    question_context_texts: tuple[str, ...],
) -> tuple[MissingQuestionInput, ...]:
    return tuple(
        _missing_question_input(
            item,
            question_context_texts=question_context_texts,
            path=f"missing[{index}]",
        )
        for index, item in enumerate(_required_items(raw, "missing"))
    )


def _missing_question_input(
    raw: object,
    *,
    question_context_texts: tuple[str, ...],
    path: str,
) -> MissingQuestionInput:
    parsed = provider_output.MissingQuestionInputOutput.parse(raw)
    return MissingQuestionInput(
        type=MissingQuestionInputType(
            _required_text(parsed.type, path=f"{path}.type")
        ),
        source_text=_copied_text(
            parsed.source_text,
            question_context_texts=question_context_texts,
            path=f"{path}.source_text",
        ),
        entity_type=_text(parsed.entity_type),
        why_context_is_insufficient=_required_text(
            parsed.why_context_is_insufficient,
            path=f"{path}.why_context_is_insufficient",
        ),
    )


def _validate_answer_requests_count(
    raw: Any,
    *,
    requested_facts: tuple[RequestedFact, ...],
) -> None:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise ValueError("answer_requests_count must be a positive integer")
    if raw != len(requested_facts):
        raise ValueError("answer_requests_count must equal answer_requests length")


def _requested_facts(
    raw: Any,
    *,
    question_inputs: tuple[RequestedFactKnownInput, ...],
    question_context_texts: tuple[str, ...],
) -> tuple[RequestedFact, ...]:
    output: list[RequestedFact] = []
    inputs_by_id = {item.id: item for item in question_inputs}
    for fact_index, item in enumerate(_required_items(raw, "answer_requests"), start=1):
        path = f"answer_requests[{fact_index - 1}]"
        parsed = provider_output.AnswerRequestOutput.parse(item)
        fact_id = f"fact_{fact_index}"
        answer_outputs = _answer_outputs(
            parsed.answer_outputs,
            path=f"{path}.answer_outputs",
        )
        input_refs = _used_question_inputs(
            parsed.used_question_inputs,
            inputs_by_id=inputs_by_id,
            path=f"{path}.used_question_inputs",
        )
        answer_subject = _answer_subject(
            parsed.answer_subject,
            question_context_texts=question_context_texts,
            path=f"{path}.answer_subject",
        )
        answer_population = _answer_population(
            parsed.answer_population,
            used_question_input_refs=input_refs,
            path=f"{path}.answer_population",
        )
        output.append(
            RequestedFact(
                id=fact_id,
                description=_required_text(
                    parsed.answer_fact,
                    path=f"{path}.answer_fact",
                ),
                answer_expression=_answer_expression(
                    parsed.answer_expression,
                    path=f"{path}.answer_expression",
                ),
                answer_subject=answer_subject,
                answer_population=answer_population,
                answer_outputs=answer_outputs,
                known_inputs=tuple(inputs_by_id[input_ref] for input_ref in input_refs),
                input_refs=input_refs,
            )
        )
    return tuple(output)


def _answer_expression(
    raw: Any,
    *,
    path: str,
) -> RequestedFactAnswerExpression:
    item = provider_output.AnswerExpressionOutput.parse(raw)
    return RequestedFactAnswerExpression(
        family=RequestedFactAnswerExpressionFamily(
            _required_text(item.family, path=f"{path}.family")
        ),
        group_key=_answer_expression_group_key(
            item.group_key,
            path=f"{path}.group_key",
        ),
    )


def _answer_expression_group_key(
    raw: Any,
    *,
    path: str,
) -> RequestedFactGroupKey | None:
    if raw is None:
        return None
    item = provider_output.GroupKeyOutput.parse(raw)
    try:
        domain = GroupKeyDomainKind(_required_text(item.domain, path=f"{path}.domain"))
    except ValueError as exc:
        raise ValueError(f"{path}.domain is invalid") from exc
    return RequestedFactGroupKey(
        description=_required_text(item.description, path=f"{path}.description"),
        domain=domain,
        question_input_refs=_answer_expression_group_key_refs(
            item.question_input_refs,
            path=f"{path}.question_input_refs",
        ),
    )


def _answer_subject(
    raw: Any,
    *,
    question_context_texts: tuple[str, ...],
    path: str,
) -> RequestedFactAnswerSubject:
    item = provider_output.AnswerSubjectOutput.parse(raw)
    return RequestedFactAnswerSubject(
        subject_text=_required_text(item.subject_text, path=f"{path}.subject_text"),
        instance_interpretation=_instance_interpretation(
            item.instance_interpretation,
            path=f"{path}.instance_interpretation",
        ),
    )


def _instance_interpretation(
    raw: Any,
    *,
    path: str,
) -> RequestedFactAnswerSubjectInstanceInterpretation:
    item = provider_output.AnswerSubjectInstanceInterpretationOutput.parse(raw)
    return RequestedFactAnswerSubjectInstanceInterpretation(
        kind=AnswerSubjectInstanceInterpretationKind(_text(item.kind))
    )


def _answer_population(
    raw: Any,
    *,
    used_question_input_refs: tuple[str, ...],
    path: str,
) -> RequestedFactAnswerPopulation:
    item = provider_output.AnswerPopulationOutput.parse(raw)
    return RequestedFactAnswerPopulation(
        population_label=_required_text(
            item.population_label,
            path=f"{path}.population_label",
        ),
        counted_unit=_required_text(
            item.counted_unit,
            path=f"{path}.counted_unit",
        ),
        membership_tests=_answer_population_membership_tests(
            item.membership_tests,
            used_question_input_refs=used_question_input_refs,
            path=f"{path}.membership_tests",
        ),
    )


def _answer_population_membership_tests(
    raw: Any,
    *,
    used_question_input_refs: tuple[str, ...],
    path: str,
) -> tuple[RequestedFactAnswerPopulationMembershipTest, ...]:
    output: list[RequestedFactAnswerPopulationMembershipTest] = []
    for index, raw_item in enumerate(_required_items(raw, path)):
        item_path = f"{path}[{index}]"
        item = provider_output.AnswerPopulationMembershipTestOutput.parse(raw_item)
        output.append(
            RequestedFactAnswerPopulationMembershipTest(
                id=_required_text(item.test_id, path=f"{item_path}.test_id"),
                kind=AnswerPopulationMembershipTestKind(
                    _text(item.kind),
                ),
                polarity=AnswerPopulationMembershipTestPolarity(
                    _text(item.polarity),
                ),
                test_question=_required_text(
                    item.test_question,
                    path=f"{item_path}.test_question",
                ),
                owned_question_input_refs=_owned_question_input_refs(
                    item.owned_question_input_refs,
                    used_question_input_refs=used_question_input_refs,
                    path=f"{item_path}.owned_question_input_refs",
                ),
            )
        )
    if not output:
        raise ValueError(f"{path} must not be empty")
    return tuple(output)


def _owned_question_input_refs(
    raw: Any,
    *,
    used_question_input_refs: tuple[str, ...],
    path: str,
) -> tuple[str, ...]:
    if not isinstance(raw, list):
        raise ValueError(f"{path} must be a list")
    allowed_refs = set(used_question_input_refs)
    output: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        input_ref = _required_text(item, path=f"{path}[{index}]")
        if input_ref not in allowed_refs:
            raise ValueError(
                f"{path}[{index}] references question input not used by this fact"
            )
        if input_ref in seen:
            raise ValueError(f"{path}[{index}] duplicates question input")
        seen.add(input_ref)
        output.append(input_ref)
    return tuple(output)


def _referenced_question_inputs(
    question_inputs: tuple[RequestedFactKnownInput, ...],
    *,
    requested_facts: tuple[RequestedFact, ...],
) -> tuple[RequestedFactKnownInput, ...]:
    referenced = {
        input_ref for fact in requested_facts for input_ref in fact.input_refs
    }
    return tuple(known for known in question_inputs if known.id in referenced)


def _reject_unowned_question_inputs(
    question_inputs: tuple[RequestedFactKnownInput, ...],
    *,
    requested_facts: tuple[RequestedFact, ...],
) -> None:
    referenced = {
        input_ref for fact in requested_facts for input_ref in fact.input_refs
    }
    unowned = [
        known.id
        for known in question_inputs
        if known.id not in referenced
    ]
    if unowned:
        raise ValueError(
            "question inputs must be owned by a requested fact: "
            + ", ".join(unowned)
        )


def _validate_conversation_resolution_question_inputs(
    question_inputs: tuple[RequestedFactKnownInput, ...],
    *,
    conversation_resolution: CompiledConversationResolution | None,
) -> None:
    for known in question_inputs:
        if known.source != KnownInputSource.CONVERSATION_RESOLUTION:
            continue
        if (
            conversation_resolution is not None
            and conversation_resolution.accepts_question_input(known)
        ):
            continue
        raise ValueError(
            "conversation_resolution question input must match a declared resolved input"
        )


def _reject_answer_subject_question_inputs(
    question_inputs: tuple[RequestedFactKnownInput, ...],
    *,
    requested_facts: tuple[RequestedFact, ...],
) -> None:
    duplicate_input_ids = {
        known.id
        for known in question_inputs
        if known.is_reference_value
        and any(
            _same_question_text(known.text, text)
            and _known_input_describes_same_text(known, text)
            for fact in requested_facts
            for text in _answer_subject_input_exclusion_texts(fact)
        )
    }
    if duplicate_input_ids:
        raise ValueError(
            "answer subject must not be declared as a question input: "
            + ", ".join(sorted(duplicate_input_ids))
        )


def _answer_subject_input_exclusion_texts(fact: RequestedFact) -> tuple[str, ...]:
    texts: list[str] = []
    if fact.answer_subject is not None:
        texts.append(fact.answer_subject.subject_text)
    if fact.answer_population is not None:
        texts.extend(
            (
                fact.answer_population.counted_unit,
                fact.answer_population.population_label,
            )
        )
    return tuple(text for text in texts if text.strip())


def _same_question_text(left: str, right: str) -> bool:
    return _normalized_question_text(left) == _normalized_question_text(right)


def _known_input_describes_same_text(
    known: RequestedFactKnownInput,
    text: str,
) -> bool:
    return _same_question_text(
        known.value_meaning_hint,
        text,
    ) or _same_question_text(known.value_meaning_hint, known.text)


def _normalized_question_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).casefold()


def _answer_outputs(
    raw: Any,
    *,
    path: str,
) -> tuple[RequestedFactAnswerOutput, ...]:
    output: list[RequestedFactAnswerOutput] = []
    for output_index, raw_item in enumerate(_required_items(raw, path), start=1):
        item_path = f"{path}[{output_index - 1}]"
        item = provider_output.AnswerOutputOutput.parse(raw_item)
        output.append(
            RequestedFactAnswerOutput(
                id=f"answer_{output_index}",
                description=_required_text(
                    item.description,
                    path=f"{item_path}.description",
                ),
                role=_text(item.role),
            )
        )
    if not output:
        raise ValueError(f"{path} must not be empty")
    return tuple(output)


def _answer_expression_group_key_refs(raw: Any, *, path: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"{path} must be a list")
    refs: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        input_ref = _required_text(item, path=f"{path}[{index}]")
        if input_ref in seen:
            raise ValueError(f"{path}[{index}] duplicates question input")
        seen.add(input_ref)
        refs.append(input_ref)
    return tuple(refs)


def _question_inputs(
    raw: Any,
    *,
    current_question_texts: tuple[str, ...],
    question_context_texts: tuple[str, ...],
) -> tuple[RequestedFactKnownInput, ...]:
    output: list[RequestedFactKnownInput] = []
    seen_ids: set[str] = set()
    for index, raw_item in enumerate(_optional_items(raw, "question_inputs")):
        path = f"question_inputs[{index}]"
        parsed = _question_input_output(raw_item, path=path)
        _question_input_item_inventory_check(
            parsed.inventory_check,
            path=f"{path}.inventory_check",
        )
        input_ref = _generated_unique_id(
            _required_text(parsed.input_ref, path=f"{path}.input_ref"),
            seen_ids=seen_ids,
        )
        kind = _question_input_kind(parsed.kind, path=f"{path}.kind")
        source = _question_input_source(
            parsed.source,
            kind=kind,
            path=f"{path}.source",
        )
        reference_text = (
            parsed.value_source_text
            if isinstance(parsed, provider_output.LiteralTextInputOutput)
            else parsed.reference_text
        )
        input_text_key = (
            "value_source_text"
            if kind == KnownInputKind.LITERAL
            else "reference_text"
        )
        span_contexts = (
            current_question_texts
            if source == KnownInputSource.QUESTION_CONTEXT
            else question_context_texts
        )
        copied_reference_text = _copied_text(
            reference_text,
            question_context_texts=span_contexts,
            path=f"{path}.{input_text_key}",
        )
        output.append(
            _question_input(
                parsed,
                input_ref=input_ref,
                kind=kind,
                source=source,
                reference_text=copied_reference_text,
                question_context_texts=question_context_texts,
                path=path,
            )
        )
    return tuple(output)


def _question_input_output(
    raw: object,
    *,
    path: str,
) -> provider_output.LiteralTextInputOutput | provider_output.RowSetReferenceInputOutput:
    item = _required_dict(raw, path)
    kind = _question_input_kind(item.get("kind"), path=f"{path}.kind")
    if kind == KnownInputKind.LITERAL:
        return provider_output.LiteralTextInputOutput.parse(item)
    if kind == KnownInputKind.ROW_SET_REFERENCE:
        return provider_output.RowSetReferenceInputOutput.parse(item)
    raise ValueError(f"{path}.kind is invalid")


def _question_input_inventory_check(raw: Any) -> None:
    item = provider_output.QuestionInputInventoryCheckOutput.parse(raw)
    if item.all_input_like_phrases_declared is not True:
        raise ValueError(
            "question_input_inventory_check.all_input_like_phrases_declared must be true"
        )


def _question_input_item_inventory_check(raw: Any, *, path: str) -> None:
    item = provider_output.QuestionInputItemInventoryCheckOutput.parse(raw)
    _required_text(
        item.why_this_is_an_input,
        path=f"{path}.why_this_is_an_input",
    )


def _question_input(
    item: provider_output.LiteralTextInputOutput | provider_output.RowSetReferenceInputOutput,
    *,
    input_ref: str,
    kind: KnownInputKind,
    source: KnownInputSource,
    reference_text: str,
    question_context_texts: tuple[str, ...],
    path: str,
) -> RequestedFactKnownInput:
    if isinstance(item, provider_output.LiteralTextInputOutput):
        role = LiteralInputRole(_required_text(item.role, path=f"{path}.role"))
        field_label_text = _text(item.field_label_text)
        resolved_input_ref = _text(item.resolved_input_ref)
        resolved_value_text = _required_text(
            item.resolved_value_text,
            path=f"{path}.resolved_value_text",
        )
        raw_occurrence = item.occurrence
        return RequestedFactLiteralInput(
            id=input_ref,
            source=source,
            text=reference_text,
            resolved_value_text=resolved_value_text,
            field_label_text=field_label_text,
            value_meaning_hint=_text(item.value_meaning_hint),
            role=role,
            resolved_input_ref=resolved_input_ref,
            occurrence=(
                1
                if raw_occurrence is None
                else _positive_int(raw_occurrence, path=f"{path}.occurrence")
            ),
        )
    if isinstance(item, provider_output.RowSetReferenceInputOutput):
        return RequestedFactRowSetReferenceInput(
            id=input_ref,
            text=reference_text,
            occurrence=_positive_int(
                item.occurrence,
                path=f"{path}.occurrence",
            ),
            resolved_input_ref=_required_text(
                item.resolved_input_ref,
                path=f"{path}.resolved_input_ref",
            ),
        )
    raise ValueError("unsupported question input kind")


def _used_question_inputs(
    raw: Any,
    *,
    inputs_by_id: dict[str, RequestedFactKnownInput],
    path: str,
) -> tuple[str, ...]:
    if not isinstance(raw, list):
        raise ValueError(f"{path} must be a list")
    used_input_refs: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        input_ref = _required_text(item, path=f"{path}[{index}]")
        if input_ref not in inputs_by_id:
            raise ValueError(
                f"{path}[{index}] references unknown question input"
            )
        if input_ref in seen:
            raise ValueError(f"{path}[{index}] duplicates question input")
        seen.add(input_ref)
        used_input_refs.append(input_ref)
    return tuple(used_input_refs)


def _question_input_kind(value: Any, *, path: str) -> KnownInputKind:
    kind = _required_text(value, path=path)
    if kind == KnownInputKind.LITERAL.value:
        return KnownInputKind.LITERAL
    if kind == KnownInputKind.ROW_SET_REFERENCE.value:
        return KnownInputKind.ROW_SET_REFERENCE
    raise ValueError(f"{path} is invalid")


def _question_input_source(
    value: Any,
    *,
    kind: KnownInputKind,
    path: str,
) -> KnownInputSource:
    source = _required_text(value, path=path)
    if kind == KnownInputKind.LITERAL:
        if source == KnownInputSource.CONVERSATION_RESOLUTION.value:
            return KnownInputSource.CONVERSATION_RESOLUTION
        if source == KnownInputSource.QUESTION_CONTEXT.value:
            return KnownInputSource.QUESTION_CONTEXT
        raise ValueError(f"{path} must be question_context or conversation_resolution")
    if kind == KnownInputKind.ROW_SET_REFERENCE:
        if source != KnownInputSource.CONVERSATION_RESOLUTION.value:
            raise ValueError(f"{path} must be conversation_resolution")
        return KnownInputSource.CONVERSATION_RESOLUTION
    if source != KnownInputSource.QUESTION_CONTEXT.value:
        raise ValueError(f"{path} must be question_context")
    return KnownInputSource.QUESTION_CONTEXT


def _positive_int(value: Any, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _copied_text(
    value: Any,
    *,
    question_context_texts: tuple[str, ...],
    path: str,
) -> str:
    text = _required_text(value, path=path)
    try:
        copied_span(text, question_context_texts=question_context_texts)
    except ValueError as exc:
        raise ValueError(f"{path} must come from question context") from exc
    return text


def _generated_unique_id(value: str, *, seen_ids: set[str]) -> str:
    if value in seen_ids:
        raise ValueError("duplicate question input")
    seen_ids.add(value)
    return value


def _required_text(value: Any, *, path: str) -> str:
    text = _text(value)
    if not text:
        raise ValueError(f"{path} must not be empty")
    return text


def _optional_items(value: Any, path: str) -> tuple[object, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list")
    return tuple(value)


def _required_dict(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _required_items(value: Any, path: str) -> tuple[object, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list")
    if not value:
        raise ValueError(f"{path} must contain at least one value")
    return tuple(value)


def _text(value: Any) -> str:
    return str(value or "").strip()
