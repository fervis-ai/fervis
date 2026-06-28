"""Parse provider-authored question-contract decisions."""

from __future__ import annotations

from dataclasses import replace
import re
from typing import Any

from fervis.lookup.question_contract._normalization import (
    number_text,
)
from fervis.lookup.question_contract.model import (
    AnswerPopulationMembershipTestKind,
    AnswerPopulationMembershipTestPolarity,
    AnswerSubjectInstanceInterpretationKind,
    KnownInputKind,
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
) -> QuestionContractResult:
    question_text = _text(question_context)
    if not question_text:
        raise ValueError("question context is required")

    if tool_name == MISSING_INPUT_CLARIFICATION_TOOL_NAME:
        _reject_unexpected_keys(
            payload,
            {"kind", "missing", "clarification_question"},
            "question_contract",
        )
        if _text(payload.get("kind")) != "needs_clarification":
            raise ValueError("invalid question contract clarification kind")
        return QuestionContractResult(
            outcome=QuestionContractNeedsClarification(
                missing=_missing_question_inputs(
                    payload.get("missing"),
                    question_context_texts=(question_text, *question_context_texts),
                ),
                clarification_question=_required_text(
                    payload.get("clarification_question"),
                    path="clarification_question",
                ),
            )
        )

    if tool_name != ANSWER_REQUEST_CONTRACT_TOOL_NAME:
        raise ValueError("unknown question contract tool")

    _reject_unexpected_keys(
        payload,
        {
            "kind",
            "answer_requests_count",
            "answer_requests",
            "question_inputs",
            "question_input_inventory_check",
        },
        "question_contract",
    )
    kind = _text(payload.get("kind"))
    if kind != "question_contract":
        raise ValueError("invalid question contract kind")
    question_inputs = _question_inputs(
        payload.get("question_inputs"),
        question_context_texts=(question_text, *question_context_texts),
    )
    requested_facts = _requested_facts(
        payload.get("answer_requests"),
        question_inputs=question_inputs,
        question_context_texts=(question_text, *question_context_texts),
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
        payload.get("answer_requests_count"),
        requested_facts=requested_facts,
    )
    _question_input_inventory_check(payload.get("question_input_inventory_check"))
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
    _reject_unexpected_keys(
        raw,
        {"type", "source_text", "entity_type", "why_context_is_insufficient"},
        path,
    )
    return MissingQuestionInput(
        type=MissingQuestionInputType(
            _required_text(raw.get("type"), path=f"{path}.type")
        ),
        source_text=_copied_text(
            raw.get("source_text"),
            question_context_texts=question_context_texts,
            path=f"{path}.source_text",
        ),
        entity_type=_text(raw.get("entity_type")),
        why_context_is_insufficient=_required_text(
            raw.get("why_context_is_insufficient"),
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
        allowed_keys = {
            "answer_fact",
            "answer_expression",
            "answer_subject",
            "input_requirements",
            "answer_population",
            "answer_outputs",
            "input_decisions",
        }
        _reject_unexpected_keys(item, allowed_keys, path)
        fact_id = f"fact_{fact_index}"
        answer_outputs = _answer_outputs(
            item.get("answer_outputs"),
            path=f"{path}.answer_outputs",
        )
        input_refs = _input_decisions(
            item.get("input_decisions"),
            inputs_by_id=inputs_by_id,
            input_ids=input_ids,
            path=f"{path}.input_decisions",
        )
        answer_subject = _answer_subject(
            item.get("answer_subject"),
            question_context_texts=question_context_texts,
            path=f"{path}.answer_subject",
        )
        input_requirements = _input_requirements(
            item.get("input_requirements"),
            question_context_texts=question_context_texts,
            path=f"{path}.input_requirements",
        )
        answer_population = _answer_population(
            item.get("answer_population"),
            path=f"{path}.answer_population",
        )
        output.append(
            RequestedFact(
                id=fact_id,
                description=_required_text(
                    item.get("answer_fact"),
                    path=f"{path}.answer_fact",
                ),
                answer_expression=_answer_expression(
                    item.get("answer_expression"),
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
    return tuple(output)


def _answer_expression(
    raw: Any,
    *,
    path: str,
) -> RequestedFactAnswerExpression:
    item = _required_dict(raw, path)
    _reject_unexpected_keys(item, {"family"}, path)
    return RequestedFactAnswerExpression(
        family=RequestedFactAnswerExpressionFamily(
            _required_text(item.get("family"), path=f"{path}.family")
        )
    )


def _answer_subject(
    raw: Any,
    *,
    question_context_texts: tuple[str, ...],
    path: str,
) -> RequestedFactAnswerSubject:
    item = _required_dict(raw, path)
    _reject_unexpected_keys(item, {"subject_text", "instance_interpretation"}, path)
    return RequestedFactAnswerSubject(
        subject_text=_required_text(
            item.get("subject_text"), path=f"{path}.subject_text"
        ),
        instance_interpretation=_instance_interpretation(
            item.get("instance_interpretation"),
            path=f"{path}.instance_interpretation",
        ),
    )


def _instance_interpretation(
    raw: Any,
    *,
    path: str,
) -> RequestedFactAnswerSubjectInstanceInterpretation:
    item = _required_dict(raw, path)
    _reject_unexpected_keys(item, {"kind"}, path)
    return RequestedFactAnswerSubjectInstanceInterpretation(
        kind=AnswerSubjectInstanceInterpretationKind(_text(item.get("kind")))
    )


def _input_requirements(
    raw: Any,
    *,
    question_context_texts: tuple[str, ...],
    path: str,
) -> RequestedFactInputRequirements:
    item = _required_dict(raw, path)
    _reject_unexpected_keys(item, {"time_requirements"}, path)
    requirements: list[RequestedFactTimeRequirement] = []
    seen_ids: set[str] = set()
    for index, requirement in enumerate(
        _optional_dicts(item.get("time_requirements"), f"{path}.time_requirements")
    ):
        requirement_path = f"{path}.time_requirements[{index}]"
        _reject_unexpected_keys(
            requirement,
            {"requirement_id", "source_text", "why_required"},
            requirement_path,
        )
        requirement_id = _required_text(
            requirement.get("requirement_id"),
            path=f"{requirement_path}.requirement_id",
        )
        if requirement_id in seen_ids:
            raise ValueError("duplicate time requirement")
        seen_ids.add(requirement_id)
        requirements.append(
            RequestedFactTimeRequirement(
                id=requirement_id,
                source_text=_copied_text(
                    requirement.get("source_text"),
                    question_context_texts=question_context_texts,
                    path=f"{requirement_path}.source_text",
                ),
                why_required=_required_text(
                    requirement.get("why_required"),
                    path=f"{requirement_path}.why_required",
                ),
            )
        )
    return RequestedFactInputRequirements(time_requirements=tuple(requirements))


def _answer_population(
    raw: Any,
    *,
    path: str,
) -> RequestedFactAnswerPopulation:
    item = _required_dict(raw, path)
    _reject_unexpected_keys(
        item,
        {"population_label", "counted_unit", "membership_tests"},
        path,
    )
    return RequestedFactAnswerPopulation(
        population_label=_required_text(
            item.get("population_label"),
            path=f"{path}.population_label",
        ),
        counted_unit=_required_text(
            item.get("counted_unit"),
            path=f"{path}.counted_unit",
        ),
        membership_tests=_answer_population_membership_tests(
            item.get("membership_tests"),
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
        _reject_unexpected_keys(
            item,
            {"test_id", "kind", "polarity", "test_question"},
            item_path,
        )
        output.append(
            RequestedFactAnswerPopulationMembershipTest(
                id=_required_text(item.get("test_id"), path=f"{item_path}.test_id"),
                kind=AnswerPopulationMembershipTestKind(
                    _text(item.get("kind")),
                ),
                polarity=AnswerPopulationMembershipTestPolarity(
                    _text(item.get("polarity")),
                ),
                test_question=_required_text(
                    item.get("test_question"),
                    path=f"{item_path}.test_question",
                ),
            )
        )
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


def _drop_answer_subject_question_inputs(
    question_inputs: tuple[RequestedFactKnownInput, ...],
    *,
    requested_facts: tuple[RequestedFact, ...],
) -> tuple[tuple[RequestedFactKnownInput, ...], tuple[RequestedFact, ...]]:
    duplicate_input_ids = {
        known.id
        for known in question_inputs
        if known.kind == KnownInputKind.REFERENCE
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


def _known_input_describes_same_text(
    known: RequestedFactKnownInput,
    text: str,
) -> bool:
    return _same_question_text(known.description, text) or _same_question_text(
        known.description,
        known.text,
    )


def _same_question_text(left: str, right: str) -> bool:
    return _normalized_question_text(left) == _normalized_question_text(right)


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
        _reject_unexpected_keys(
            item,
            {"description"},
            item_path,
        )
        output.append(
            RequestedFactAnswerOutput(
                id=f"answer_{output_index}",
                description=_required_text(
                    item.get("description"),
                    path=f"{item_path}.description",
                ),
            )
        )
    if not output:
        raise ValueError(f"{path} must not be empty")
    return tuple(output)


def _question_inputs(
    raw: Any,
    *,
    question_context_texts: tuple[str, ...],
) -> tuple[RequestedFactKnownInput, ...]:
    output: list[RequestedFactKnownInput] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(_optional_dicts(raw, "question_inputs")):
        path = f"question_inputs[{index}]"
        _reject_unexpected_keys(
            item,
            {
                "input_ref",
                "kind",
                "source",
                "reference_text",
                "target_meaning",
                "lookup_text",
                "numeric_value",
                "value_source_text",
                "occurrence",
                "resolved_input_ref",
                "satisfies_requirement_id",
                "inventory_check",
            },
            path,
        )
        _question_input_item_inventory_check(
            item.get("inventory_check"),
            path=f"{path}.inventory_check",
        )
        input_ref = _generated_unique_id(
            _required_text(item.get("input_ref"), path=f"{path}.input_ref"),
            seen_ids=seen_ids,
        )
        kind = _question_input_kind(item.get("kind"), path=f"{path}.kind")
        source = _question_input_source(
            item.get("source"),
            kind=kind,
            path=f"{path}.source",
        )
        reference_text = _copied_text(
            item.get("reference_text"),
            question_context_texts=question_context_texts,
            path=f"{path}.reference_text",
        )
        output.append(
            _question_input(
                item,
                input_ref=input_ref,
                kind=kind,
                source=source,
                reference_text=reference_text,
                question_context_texts=question_context_texts,
                path=path,
            )
        )
    return tuple(output)


def _question_input_inventory_check(raw: Any) -> None:
    item = _required_dict(raw, "question_input_inventory_check")
    _reject_unexpected_keys(
        item,
        {"all_input_like_phrases_declared"},
        "question_input_inventory_check",
    )
    if item.get("all_input_like_phrases_declared") is not True:
        raise ValueError(
            "question_input_inventory_check.all_input_like_phrases_declared must be true"
        )


def _question_input_item_inventory_check(raw: Any, *, path: str) -> None:
    item = _required_dict(raw, path)
    _reject_unexpected_keys(item, {"why_this_is_an_input"}, path)
    _required_text(
        item.get("why_this_is_an_input"),
        path=f"{path}.why_this_is_an_input",
    )


def _question_input(
    item: dict[str, Any],
    *,
    input_ref: str,
    kind: KnownInputKind,
    source: KnownInputSource,
    reference_text: str,
    question_context_texts: tuple[str, ...],
    path: str,
) -> RequestedFactKnownInput:
    if kind == KnownInputKind.REFERENCE:
        _reject_kind_specific_fields(
            item,
            forbidden=(
                "numeric_value",
                "occurrence",
                "resolved_input_ref",
                "satisfies_requirement_id",
            ),
            path=path,
        )
        lookup_text = _lookup_text(
            item.get("lookup_text"),
            question_context_texts=question_context_texts,
            path=f"{path}.lookup_text",
        )
        return RequestedFactKnownInput(
            id=input_ref,
            kind=KnownInputKind.REFERENCE,
            source=source,
            description=_required_text(
                item.get("target_meaning"),
                path=f"{path}.target_meaning",
            ),
            text=reference_text,
            lookup_text=lookup_text,
        )
    if kind == KnownInputKind.ROW_SET_REFERENCE:
        _reject_kind_specific_fields(
            item,
            forbidden=(
                "lookup_text",
                "target_meaning",
                "numeric_value",
                "satisfies_requirement_id",
            ),
            path=path,
        )
        return RequestedFactKnownInput(
            id=input_ref,
            kind=KnownInputKind.ROW_SET_REFERENCE,
            source=source,
            text=reference_text,
            occurrence=_positive_int(
                item.get("occurrence"),
                path=f"{path}.occurrence",
            ),
            resolved_input_ref=_required_text(
                item.get("resolved_input_ref"),
                path=f"{path}.resolved_input_ref",
            ),
        )
    if kind == KnownInputKind.TIME:
        _reject_kind_specific_fields(
            item,
            forbidden=(
                "target_meaning",
                "lookup_text",
                "numeric_value",
                "resolved_input_ref",
            ),
            path=path,
        )
        return RequestedFactKnownInput(
            id=input_ref,
            kind=KnownInputKind.TIME,
            source=source,
            text=reference_text,
            satisfies_requirement_id=_text(item.get("satisfies_requirement_id")),
        )
    if kind in {KnownInputKind.LIMIT, KnownInputKind.NUMBER}:
        forbidden = [
            "target_meaning",
            "lookup_text",
            "resolved_input_ref",
            "satisfies_requirement_id",
        ]
        if kind == KnownInputKind.NUMBER:
            forbidden.append("value_source_text")
        _reject_kind_specific_fields(
            item,
            forbidden=tuple(forbidden),
            path=path,
        )
        value_source_text = ""
        if kind == KnownInputKind.LIMIT:
            value_source_text = _limit_value_source_text(
                item.get("value_source_text"),
                reference_text=reference_text,
                path=f"{path}.value_source_text",
            )
        return RequestedFactKnownInput(
            id=input_ref,
            kind=kind,
            source=source,
            text=reference_text,
            value_source_text=value_source_text,
            numeric_value=_literal_value(
                item.get("numeric_value"),
                kind=kind,
                text=reference_text,
                path=f"{path}.numeric_value",
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
                if known.kind == KnownInputKind.TIME
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
        _reject_unexpected_keys(item, {"input_ref", "use_input"}, f"{path}[{index}]")
        input_ref = _required_text(
            item.get("input_ref"),
            path=f"{path}[{index}].input_ref",
        )
        if input_ref not in inputs_by_id:
            raise ValueError(
                f"{path}[{index}].input_ref references unknown question input"
            )
        if input_ref in decisions:
            raise ValueError(f"{path}[{index}].input_ref duplicates question input")
        use_input = item.get("use_input")
        if not isinstance(use_input, bool):
            raise ValueError(f"{path}[{index}].use_input must be boolean")
        decisions[input_ref] = use_input
    missing = [input_id for input_id in input_ids if input_id not in decisions]
    if missing:
        raise ValueError(f"{path} must decide every question input")
    return tuple(input_id for input_id in input_ids if decisions[input_id])


def _question_input_kind(value: Any, *, path: str) -> KnownInputKind:
    kind = _required_text(value, path=path)
    if kind == KnownInputKind.REFERENCE.value:
        return KnownInputKind.REFERENCE
    if kind == KnownInputKind.ROW_SET_REFERENCE.value:
        return KnownInputKind.ROW_SET_REFERENCE
    if kind == KnownInputKind.TIME.value:
        return KnownInputKind.TIME
    if kind == KnownInputKind.LIMIT.value:
        return KnownInputKind.LIMIT
    if kind == KnownInputKind.NUMBER.value:
        return KnownInputKind.NUMBER
    raise ValueError(f"{path} is invalid")


def _reject_kind_specific_fields(
    item: dict[str, Any],
    *,
    forbidden: tuple[str, ...],
    path: str,
) -> None:
    present = sorted(field for field in forbidden if field in item)
    if present:
        raise ValueError(f"{path} includes unsupported fields: {', '.join(present)}")


def _literal_value(
    value: Any,
    *,
    kind: KnownInputKind,
    text: str,
    path: str,
) -> int | float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{path} must be numeric")
    if kind == KnownInputKind.LIMIT:
        if not isinstance(value, int) or value < 1:
            raise ValueError("limit value must be a positive integer")
        return value
    if number_text(text) != number_text(value):
        raise ValueError("number input value must match reference_text")
    return value


def _limit_value_source_text(
    value: Any,
    *,
    reference_text: str,
    path: str,
) -> str:
    text = _required_text(value, path=path)
    if not _contains_text_span(reference_text, text):
        raise ValueError("limit value source text must come from reference_text")
    return text


def _lookup_text(
    raw: Any,
    *,
    question_context_texts: tuple[str, ...],
    path: str,
) -> str:
    lookup_text = _required_text(raw, path=path)
    if not any(
        _contains_text_span(context, lookup_text) for context in question_context_texts
    ):
        raise ValueError("reference lookup text must come from question context")
    return lookup_text


def _question_input_source(
    value: Any,
    *,
    kind: KnownInputKind,
    path: str,
) -> KnownInputSource:
    source = _required_text(value, path=path)
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
    if not any(
        _contains_text_span(context, text) for context in question_context_texts
    ):
        raise ValueError("input text must come from question context")
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


def _reject_unexpected_keys(
    payload: dict[str, Any],
    allowed: set[str],
    path: str,
) -> None:
    unexpected = sorted(set(payload) - allowed)
    if unexpected:
        raise ValueError(f"{path} includes unsupported fields: {', '.join(unexpected)}")


def _text(value: Any) -> str:
    return str(value or "").strip()
