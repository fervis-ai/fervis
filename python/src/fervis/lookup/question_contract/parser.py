"""Parse provider-authored question-contract decisions."""

from __future__ import annotations

from dataclasses import replace
import re
from typing import Any

from fervis.lookup.conversation_resolution import ConversationResolutionOverlay
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
    RequestedFact,
    RequestedFactAnswerExpression,
    RequestedFactAnswerExpressionFamily,
    RequestedFactAnswerPopulation,
    RequestedFactAnswerPopulationMembershipTest,
    RequestedFactAnswerOutput,
    RequestedFactAnswerSubject,
    RequestedFactAnswerSubjectInstanceInterpretation,
    RequestedFactTimeRequirement,
    RequestedFactInputRequirements,
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
    conversation_resolution_overlay: ConversationResolutionOverlay | None = None,
) -> QuestionContractResult:
    question_text = _text(question_context)
    if not question_text:
        raise ValueError("question context is required")
    current_question_texts = (question_text, *current_question_context_texts)
    context_texts = (question_text, *question_context_texts)

    if tool_name == MISSING_INPUT_CLARIFICATION_TOOL_NAME:
        parsed = _ParsedObject(payload, "question_contract")
        if _text(parsed.take("kind")) != "needs_clarification":
            raise ValueError("invalid question contract clarification kind")
        missing = parsed.take("missing")
        clarification_question = parsed.take("clarification_question")
        parsed.finish()
        return QuestionContractResult(
            outcome=QuestionContractNeedsClarification(
                missing=_missing_question_inputs(
                    missing,
                    question_context_texts=context_texts,
                ),
                clarification_question=_required_text(
                    clarification_question,
                    path="clarification_question",
                ),
            )
        )

    if tool_name != ANSWER_REQUEST_CONTRACT_TOOL_NAME:
        raise ValueError("unknown question contract tool")

    parsed = _ParsedObject(payload, "question_contract")
    kind = _text(parsed.take("kind"))
    if kind != "question_contract":
        raise ValueError("invalid question contract kind")
    question_inputs = _question_inputs(
        parsed.take("question_inputs"),
        current_question_texts=current_question_texts,
        question_context_texts=context_texts,
    )
    _validate_conversation_resolution_question_inputs(
        question_inputs,
        conversation_resolution_overlay=conversation_resolution_overlay,
    )
    requested_facts = _requested_facts(
        parsed.take("answer_requests"),
        question_inputs=question_inputs,
        question_context_texts=context_texts,
    )
    _reject_unowned_literal_inputs(
        question_inputs,
        requested_facts=requested_facts,
    )
    question_inputs, requested_facts = _drop_answer_subject_question_inputs(
        question_inputs,
        requested_facts=requested_facts,
    )
    question_inputs = _referenced_question_inputs(
        question_inputs,
        requested_facts=requested_facts,
    )
    _validate_input_requirements(
        requested_facts=requested_facts,
        question_inputs=question_inputs,
    )
    _validate_answer_requests_count(
        parsed.take("answer_requests_count"),
        requested_facts=requested_facts,
    )
    _question_input_inventory_check(parsed.take("question_input_inventory_check"))
    parsed.finish()
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
        for index, item in enumerate(_required_dicts(raw, "missing"))
    )


def _missing_question_input(
    raw: dict[str, Any],
    *,
    question_context_texts: tuple[str, ...],
    path: str,
) -> MissingQuestionInput:
    parsed = _ParsedObject(raw, path)
    input_type = parsed.take("type")
    source_text = parsed.take("source_text")
    entity_type = parsed.take("entity_type")
    why_context_is_insufficient = parsed.take("why_context_is_insufficient")
    parsed.finish()
    return MissingQuestionInput(
        type=MissingQuestionInputType(_required_text(input_type, path=f"{path}.type")),
        source_text=_copied_text(
            source_text,
            question_context_texts=question_context_texts,
            path=f"{path}.source_text",
        ),
        entity_type=_text(entity_type),
        why_context_is_insufficient=_required_text(
            why_context_is_insufficient,
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
    input_ids = tuple(inputs_by_id)
    for fact_index, item in enumerate(_required_dicts(raw, "answer_requests"), start=1):
        path = f"answer_requests[{fact_index - 1}]"
        parsed = _ParsedObject(item, path)
        fact_id = f"fact_{fact_index}"
        answer_outputs = _answer_outputs(
            parsed.take("answer_outputs"),
            path=f"{path}.answer_outputs",
        )
        input_refs = _input_decisions(
            parsed.take("input_decisions"),
            inputs_by_id=inputs_by_id,
            input_ids=input_ids,
            path=f"{path}.input_decisions",
        )
        answer_subject = _answer_subject(
            parsed.take("answer_subject"),
            question_context_texts=question_context_texts,
            path=f"{path}.answer_subject",
        )
        input_requirements = _input_requirements(
            parsed.take("input_requirements"),
            question_context_texts=question_context_texts,
            path=f"{path}.input_requirements",
        )
        answer_population = _answer_population(
            parsed.take("answer_population"),
            path=f"{path}.answer_population",
        )
        output.append(
            RequestedFact(
                id=fact_id,
                description=_required_text(
                    parsed.take("answer_fact"),
                    path=f"{path}.answer_fact",
                ),
                answer_expression=_answer_expression(
                    parsed.take("answer_expression"),
                    path=f"{path}.answer_expression",
                ),
                answer_subject=answer_subject,
                input_requirements=input_requirements,
                answer_population=answer_population,
                answer_outputs=answer_outputs,
                known_inputs=tuple(inputs_by_id[input_ref] for input_ref in input_refs),
                input_refs=input_refs,
            )
        )
        parsed.finish()
    return tuple(output)


def _answer_expression(
    raw: Any,
    *,
    path: str,
) -> RequestedFactAnswerExpression:
    item = _ParsedObject(raw, path)
    family = item.take("family")
    item.finish()
    return RequestedFactAnswerExpression(
        family=RequestedFactAnswerExpressionFamily(
            _required_text(family, path=f"{path}.family")
        )
    )


def _answer_subject(
    raw: Any,
    *,
    question_context_texts: tuple[str, ...],
    path: str,
) -> RequestedFactAnswerSubject:
    item = _ParsedObject(raw, path)
    subject_text = item.take("subject_text")
    instance_interpretation = item.take("instance_interpretation")
    item.finish()
    return RequestedFactAnswerSubject(
        subject_text=_required_text(subject_text, path=f"{path}.subject_text"),
        instance_interpretation=_instance_interpretation(
            instance_interpretation,
            path=f"{path}.instance_interpretation",
        ),
    )


def _instance_interpretation(
    raw: Any,
    *,
    path: str,
) -> RequestedFactAnswerSubjectInstanceInterpretation:
    item = _ParsedObject(raw, path)
    kind = item.take("kind")
    item.finish()
    return RequestedFactAnswerSubjectInstanceInterpretation(
        kind=AnswerSubjectInstanceInterpretationKind(_text(kind))
    )


def _input_requirements(
    raw: Any,
    *,
    question_context_texts: tuple[str, ...],
    path: str,
) -> RequestedFactInputRequirements:
    item = _ParsedObject(raw, path)
    time_requirements = item.take("time_requirements")
    item.finish()
    requirements: list[RequestedFactTimeRequirement] = []
    seen_ids: set[str] = set()
    for index, requirement in enumerate(
        _optional_dicts(time_requirements, f"{path}.time_requirements")
    ):
        requirement_path = f"{path}.time_requirements[{index}]"
        parsed_requirement = _ParsedObject(requirement, requirement_path)
        requirement_id = _required_text(
            parsed_requirement.take("requirement_id"),
            path=f"{requirement_path}.requirement_id",
        )
        if requirement_id in seen_ids:
            raise ValueError("duplicate time requirement")
        seen_ids.add(requirement_id)
        requirements.append(
            RequestedFactTimeRequirement(
                id=requirement_id,
                source_text=_copied_text(
                    parsed_requirement.take("source_text"),
                    question_context_texts=question_context_texts,
                    path=f"{requirement_path}.source_text",
                ),
                why_required=_required_text(
                    parsed_requirement.take("why_required"),
                    path=f"{requirement_path}.why_required",
                ),
            )
        )
        parsed_requirement.finish()
    return RequestedFactInputRequirements(time_requirements=tuple(requirements))


def _answer_population(
    raw: Any,
    *,
    path: str,
) -> RequestedFactAnswerPopulation:
    item = _ParsedObject(raw, path)
    population_label = item.take("population_label")
    counted_unit = item.take("counted_unit")
    membership_tests = item.take("membership_tests")
    item.finish()
    return RequestedFactAnswerPopulation(
        population_label=_required_text(
            population_label,
            path=f"{path}.population_label",
        ),
        counted_unit=_required_text(
            counted_unit,
            path=f"{path}.counted_unit",
        ),
        membership_tests=_answer_population_membership_tests(
            membership_tests,
            path=f"{path}.membership_tests",
        ),
    )


def _answer_population_membership_tests(
    raw: Any,
    *,
    path: str,
) -> tuple[RequestedFactAnswerPopulationMembershipTest, ...]:
    output: list[RequestedFactAnswerPopulationMembershipTest] = []
    for index, item in enumerate(_required_dicts(raw, path)):
        item_path = f"{path}[{index}]"
        parsed = _ParsedObject(item, item_path)
        output.append(
            RequestedFactAnswerPopulationMembershipTest(
                id=_required_text(parsed.take("test_id"), path=f"{item_path}.test_id"),
                kind=AnswerPopulationMembershipTestKind(
                    _text(parsed.take("kind")),
                ),
                polarity=AnswerPopulationMembershipTestPolarity(
                    _text(parsed.take("polarity")),
                ),
                test_question=_required_text(
                    parsed.take("test_question"),
                    path=f"{item_path}.test_question",
                ),
            )
        )
        parsed.finish()
    if not output:
        raise ValueError(f"{path} must not be empty")
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


def _reject_unowned_literal_inputs(
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
        if known.kind == KnownInputKind.LITERAL and known.id not in referenced
    ]
    if unowned:
        raise ValueError(
            "literal_text question inputs must be owned by a requested fact: "
            + ", ".join(unowned)
        )


def _validate_conversation_resolution_question_inputs(
    question_inputs: tuple[RequestedFactKnownInput, ...],
    *,
    conversation_resolution_overlay: ConversationResolutionOverlay | None,
) -> None:
    resolved_inputs = (
        conversation_resolution_overlay.resolved_question_inputs
        if conversation_resolution_overlay is not None
        else ()
    )
    resolved_by_ref = {
        item.resolved_input_ref: item
        for item in resolved_inputs
        if item.resolved_input_ref
    }
    for known in question_inputs:
        if known.source != KnownInputSource.CONVERSATION_RESOLUTION:
            continue
        resolved = resolved_by_ref.get(known.resolved_input_ref)
        if resolved is None or not _conversation_resolution_input_matches(
            known,
            resolved,
        ):
            raise ValueError(
                "conversation_resolution question input must match "
                "resolved_question_inputs"
            )


def _conversation_resolution_input_matches(
    known: RequestedFactKnownInput,
    resolved: Any,
) -> bool:
    resolved_text = (
        resolved.source_text
        if resolved.kind == KnownInputKind.LITERAL
        else resolved.reference_text
    )
    if known.text != resolved_text:
        return False
    if known.kind == KnownInputKind.ROW_SET_REFERENCE:
        return (
            resolved.kind == KnownInputKind.ROW_SET_REFERENCE
            and known.occurrence == resolved.occurrence
        )
    if known.kind == KnownInputKind.LITERAL:
        return (
            resolved.kind == KnownInputKind.LITERAL
            and known.resolved_value_text == resolved.resolved_value_text
            and known.role is not None
            and known.role == resolved.role
        )
    return False


def _drop_answer_subject_question_inputs(
    question_inputs: tuple[RequestedFactKnownInput, ...],
    *,
    requested_facts: tuple[RequestedFact, ...],
) -> tuple[tuple[RequestedFactKnownInput, ...], tuple[RequestedFact, ...]]:
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
    if not duplicate_input_ids:
        return question_inputs, requested_facts
    kept_inputs = tuple(
        known for known in question_inputs if known.id not in duplicate_input_ids
    )
    kept_facts = tuple(
        _without_question_input_refs(fact, duplicate_input_ids)
        for fact in requested_facts
    )
    return kept_inputs, kept_facts


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


def _without_question_input_refs(
    fact: RequestedFact,
    input_ids: set[str],
) -> RequestedFact:
    input_refs = tuple(
        input_ref for input_ref in fact.input_refs if input_ref not in input_ids
    )
    known_inputs = tuple(
        known for known in fact.known_inputs if known.id not in input_ids
    )
    return replace(fact, input_refs=input_refs, known_inputs=known_inputs)


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
    for output_index, item in enumerate(_required_dicts(raw, path), start=1):
        item_path = f"{path}[{output_index - 1}]"
        parsed = _ParsedObject(item, item_path)
        output.append(
            RequestedFactAnswerOutput(
                id=f"answer_{output_index}",
                description=_required_text(
                    parsed.take("description"),
                    path=f"{item_path}.description",
                ),
            )
        )
        parsed.finish()
    if not output:
        raise ValueError(f"{path} must not be empty")
    return tuple(output)


def _question_inputs(
    raw: Any,
    *,
    current_question_texts: tuple[str, ...],
    question_context_texts: tuple[str, ...],
) -> tuple[RequestedFactKnownInput, ...]:
    output: list[RequestedFactKnownInput] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(_optional_dicts(raw, "question_inputs")):
        path = f"question_inputs[{index}]"
        parsed = _ParsedObject(item, path)
        _question_input_item_inventory_check(
            parsed.take("inventory_check"),
            path=f"{path}.inventory_check",
        )
        input_ref = _generated_unique_id(
            _required_text(parsed.take("input_ref"), path=f"{path}.input_ref"),
            seen_ids=seen_ids,
        )
        kind = _question_input_kind(parsed.take("kind"), path=f"{path}.kind")
        source = _question_input_source(
            parsed.take("source"),
            kind=kind,
            path=f"{path}.source",
        )
        input_text_key = (
            "source_text" if kind == KnownInputKind.LITERAL else "reference_text"
        )
        span_contexts = (
            current_question_texts
            if source == KnownInputSource.QUESTION_CONTEXT
            else question_context_texts
        )
        reference_text = _copied_text(
            parsed.take(input_text_key),
            question_context_texts=span_contexts,
            path=f"{path}.{input_text_key}",
        )
        output.append(
            _question_input(
                parsed,
                input_ref=input_ref,
                kind=kind,
                source=source,
                reference_text=reference_text,
                question_context_texts=question_context_texts,
                path=path,
            )
        )
        parsed.finish()
    return tuple(output)


def _question_input_inventory_check(raw: Any) -> None:
    item = _ParsedObject(raw, "question_input_inventory_check")
    all_declared = item.take("all_input_like_phrases_declared")
    item.finish()
    if all_declared is not True:
        raise ValueError(
            "question_input_inventory_check.all_input_like_phrases_declared must be true"
        )


def _question_input_item_inventory_check(raw: Any, *, path: str) -> None:
    item = _ParsedObject(raw, path)
    _required_text(
        item.take("why_this_is_an_input"),
        path=f"{path}.why_this_is_an_input",
    )
    item.finish()


def _question_input(
    item: "_ParsedObject",
    *,
    input_ref: str,
    kind: KnownInputKind,
    source: KnownInputSource,
    reference_text: str,
    question_context_texts: tuple[str, ...],
    path: str,
) -> RequestedFactKnownInput:
    if kind == KnownInputKind.LITERAL:
        role = LiteralInputRole(_required_text(item.take("role"), path=f"{path}.role"))
        field_label_text = _text(item.take("field_label_text"))
        if field_label_text and not any(
            _contains_text_span(context, field_label_text)
            for context in question_context_texts
        ):
            raise ValueError("field label text must come from question context")
        resolved_input_ref = _text(item.take("resolved_input_ref"))
        resolved_value_text = _required_text(
            item.take("resolved_value_text"),
            path=f"{path}.resolved_value_text",
        )
        if role == LiteralInputRole.RESULT_LIMIT:
            resolved_value_text = _result_limit_value_text(
                resolved_value_text,
                path=f"{path}.resolved_value_text",
            )
        return RequestedFactLiteralInput(
            id=input_ref,
            source=source,
            text=reference_text,
            resolved_value_text=resolved_value_text,
            field_label_text=field_label_text,
            value_meaning_hint=_text(item.take("value_meaning_hint")),
            role=role,
            satisfies_requirement_id=_text(item.take("satisfies_requirement_id")),
            resolved_input_ref=resolved_input_ref,
        )
    if kind == KnownInputKind.ROW_SET_REFERENCE:
        return RequestedFactRowSetReferenceInput(
            id=input_ref,
            text=reference_text,
            occurrence=_positive_int(
                item.take("occurrence"),
                path=f"{path}.occurrence",
            ),
            resolved_input_ref=_required_text(
                item.take("resolved_input_ref"),
                path=f"{path}.resolved_input_ref",
            ),
        )
    raise ValueError("unsupported question input kind")


def _validate_input_requirements(
    *,
    requested_facts: tuple[RequestedFact, ...],
    question_inputs: tuple[RequestedFactKnownInput, ...],
) -> None:
    inputs_by_id = {known.id: known for known in question_inputs}
    for fact in requested_facts:
        used_inputs = tuple(
            inputs_by_id[input_ref]
            for input_ref in fact.input_refs
            if input_ref in inputs_by_id
        )
        for requirement in fact.input_requirements.time_requirements:
            matching = [
                known
                for known in used_inputs
                if (known.is_time_value)
                and known.satisfies_requirement_id == requirement.id
                and known.text == requirement.source_text
            ]
            if not matching:
                raise ValueError(
                    "time requirement requires matching used question input"
                )


def _input_decisions(
    raw: Any,
    *,
    inputs_by_id: dict[str, RequestedFactKnownInput],
    input_ids: tuple[str, ...],
    path: str,
) -> tuple[str, ...]:
    if not isinstance(raw, list):
        raise ValueError(f"{path} must be a list")
    decisions: dict[str, bool] = {}
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"{path}[{index}] must be an object")
        parsed = _ParsedObject(item, f"{path}[{index}]")
        input_ref = _required_text(
            parsed.take("input_ref"),
            path=f"{path}[{index}].input_ref",
        )
        if input_ref not in inputs_by_id:
            raise ValueError(
                f"{path}[{index}].input_ref references unknown question input"
            )
        if input_ref in decisions:
            raise ValueError(f"{path}[{index}].input_ref duplicates question input")
        use_input = parsed.take("use_input")
        if not isinstance(use_input, bool):
            raise ValueError(f"{path}[{index}].use_input must be boolean")
        decisions[input_ref] = use_input
        parsed.finish()
    missing = [input_id for input_id in input_ids if input_id not in decisions]
    if missing:
        raise ValueError(f"{path} must decide every question input")
    return tuple(input_id for input_id in input_ids if decisions[input_id])


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


def _result_limit_value_text(value: str, *, path: str) -> str:
    text = value.strip()
    if not text.isdigit() or int(text) < 1:
        raise ValueError(f"{path} must be canonical positive integer digits")
    return str(int(text))


def _copied_text(
    value: Any,
    *,
    question_context_texts: tuple[str, ...],
    path: str,
) -> str:
    text = _required_text(value, path=path)
    if not any(
        _contains_text_span(context, text) for context in question_context_texts
    ):
        raise ValueError(f"{path} must come from question context")
    return text


def _contains_text_span(container: str, text: str) -> bool:
    for match in re.finditer(re.escape(text), container):
        start = match.start()
        end = match.end()
        if _has_alnum_edge(text, at_start=True) and start > 0:
            if container[start - 1].isalnum():
                continue
        if _has_alnum_edge(text, at_start=False) and end < len(container):
            if container[end].isalnum():
                continue
        return True
    return False


def _has_alnum_edge(text: str, *, at_start: bool) -> bool:
    value = text.strip()
    if not value:
        return False
    char = value[0] if at_start else value[-1]
    return char.isalnum()


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


def _optional_dicts(value: Any, path: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list")
    output: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{path}[{index}] must be an object")
        output.append(item)
    return tuple(output)


def _required_dict(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


class _ParsedObject:
    def __init__(self, value: Any, path: str) -> None:
        self._payload = _required_dict(value, path)
        self._path = path
        self._unparsed = set(self._payload)

    def take(self, key: str) -> Any:
        self._unparsed.discard(key)
        return self._payload.get(key)

    def finish(self) -> None:
        if self._unparsed:
            fields = ", ".join(sorted(self._unparsed))
            raise ValueError(f"{self._path} has unparsed fields: {fields}")


def _required_dicts(value: Any, path: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list")
    output: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{path}[{index}] must be an object")
        output.append(item)
    if not output:
        raise ValueError(f"{path} must contain at least one value")
    return tuple(output)


def _text(value: Any) -> str:
    return str(value or "").strip()
