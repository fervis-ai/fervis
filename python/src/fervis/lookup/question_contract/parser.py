"""Parse provider-authored question-contract decisions."""

from __future__ import annotations

from dataclasses import dataclass
import re

from fervis.lookup.conversation_resolution.compilation import (
    CompiledConversationResolution,
)
from fervis.lookup.question_contract._text_spans import copied_span
from fervis.lookup.question_contract import provider_contract as provider_output
from fervis.lookup.question_inputs import (
    KnownInputKind,
    LiteralInputRole,
)
from fervis.lookup.question_contract.model import (
    AnswerPopulationMembershipTestKind,
    AnswerPopulationMembershipTestPolarity,
    AnswerSubjectInstanceInterpretationKind,
    KnownInputSource,
    IncompleteFactualRequestItem,
    IncompleteFactualRequestKind,
    QuestionContract,
    QuestionContractNeedsClarification,
    QuestionContractResult,
    GroupKeyDomainKind,
    RequestedFact,
    RequestedFactAnswerExpression,
    RequestedFactAnswerExpressionFamily,
    RequestedFactOrderingDirection,
    ResultSelectionKind,
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
    QUESTION_CONTRACT_TOOL_NAME,
)
from fervis.lookup.provider_contract import ProviderObject


@dataclass(frozen=True, slots=True)
class _ParsedMembershipTest:
    id: str
    kind: AnswerPopulationMembershipTestKind
    polarity: AnswerPopulationMembershipTestPolarity
    test_question: str
    owned_question_input_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ParsedQuestionInputUses:
    input_refs: tuple[str, ...]
    group_key_input_refs: tuple[str, ...]
    group_key_derivation_input_refs: tuple[str, ...]
    population_input_refs_by_use_id: dict[str, str]
    compute_input_refs: tuple[str, ...]
    result_limit_input_ref: str


def parse_question_contract(
    *,
    tool_name: str,
    payload: dict[str, object],
    question_context: str,
    question_context_texts: tuple[str, ...] = (),
    current_question_context_texts: tuple[str, ...] = (),
    conversation_resolution: CompiledConversationResolution | None = None,
) -> QuestionContractResult:
    question_text = question_context.strip()
    if not question_text:
        raise ValueError("question context is required")
    current_question_texts = (question_text, *current_question_context_texts)
    context_texts = (question_text, *question_context_texts)

    if tool_name != QUESTION_CONTRACT_TOOL_NAME:
        raise ValueError("unknown question contract tool")

    decision = provider_output.QuestionContractDecisionOutput.parse(payload)
    decision_basis = _required_text(
        decision.decision_basis,
        path="decision_basis",
    )
    outcome = decision.outcome
    outcome_kind = outcome.discriminator("kind")
    if outcome_kind == "missing_requested_fact":
        missing_fact = outcome.parse_as(provider_output.MissingRequestedFactOutput)
        return QuestionContractResult(
            decision_basis=decision_basis,
            outcome=QuestionContractNeedsClarification(
                missing=(
                    _missing_requested_fact(
                        missing_fact,
                        question_context_texts=context_texts,
                    ),
                ),
            ),
        )
    if outcome_kind == "unresolved_prior_turn_references":
        unresolved_references = outcome.parse_as(
            provider_output.UnresolvedPriorTurnReferencesOutput
        )
        return QuestionContractResult(
            decision_basis=decision_basis,
            outcome=QuestionContractNeedsClarification(
                missing=_unresolved_prior_turn_references(
                    unresolved_references.references,
                    question_context_texts=context_texts,
                ),
            ),
        )

    parsed = outcome.parse_as(provider_output.QuestionContractOutput)
    kind = parsed.kind
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
        decision_basis=decision_basis,
        outcome=QuestionContract(
            question_inputs=question_inputs,
            requested_facts=requested_facts,
            clarification_lineage_refs=(
                conversation_resolution.clarification_lineage_refs
                if conversation_resolution is not None
                else ()
            ),
        ),
    )


def _unresolved_prior_turn_references(
    items: tuple[provider_output.UnresolvedPriorTurnReferenceOutput, ...],
    *,
    question_context_texts: tuple[str, ...],
) -> tuple[IncompleteFactualRequestItem, ...]:
    return tuple(
        _unresolved_prior_turn_reference(
            item,
            question_context_texts=question_context_texts,
            path=f"missing[{index}]",
        )
        for index, item in enumerate(items)
    )


def _unresolved_prior_turn_reference(
    parsed: provider_output.UnresolvedPriorTurnReferenceOutput,
    *,
    question_context_texts: tuple[str, ...],
    path: str,
) -> IncompleteFactualRequestItem:
    return IncompleteFactualRequestItem(
        missing_kind=(IncompleteFactualRequestKind.UNRESOLVED_PRIOR_TURN_REFERENCE),
        source_text=_copied_text(
            parsed.source_text,
            question_context_texts=question_context_texts,
            path=f"{path}.source_text",
        ),
        why_question_is_incomplete=_required_text(
            parsed.why_question_is_incomplete,
            path=f"{path}.why_question_is_incomplete",
        ),
        target_label=str(parsed.target_label or "").strip(),
    )


def _missing_requested_fact(
    parsed: provider_output.MissingRequestedFactOutput,
    *,
    question_context_texts: tuple[str, ...],
) -> IncompleteFactualRequestItem:
    return IncompleteFactualRequestItem(
        missing_kind=IncompleteFactualRequestKind.MISSING_REQUESTED_FACT,
        source_text=_copied_text(
            parsed.source_text,
            question_context_texts=question_context_texts,
            path="source_text",
        ),
        why_question_is_incomplete=_required_text(
            parsed.why_question_is_incomplete,
            path="why_question_is_incomplete",
        ),
    )


def _validate_answer_requests_count(
    count: int,
    *,
    requested_facts: tuple[RequestedFact, ...],
) -> None:
    if count < 1:
        raise ValueError("answer_requests_count must be a positive integer")
    if count != len(requested_facts):
        raise ValueError("answer_requests_count must equal answer_requests length")


def _requested_facts(
    items: tuple[provider_output.AnswerRequestOutput, ...],
    *,
    question_inputs: tuple[RequestedFactKnownInput, ...],
    question_context_texts: tuple[str, ...],
) -> tuple[RequestedFact, ...]:
    output: list[RequestedFact] = []
    inputs_by_id = {item.id: item for item in question_inputs}
    if not items:
        raise ValueError("answer_requests must contain at least one value")
    for fact_index, parsed in enumerate(items, start=1):
        path = f"answer_requests[{fact_index - 1}]"
        fact_id = f"fact_{fact_index}"
        answer_outputs = _answer_outputs(
            parsed.answer_outputs,
            path=f"{path}.answer_outputs",
        )
        input_uses = _question_input_uses(
            parsed.question_input_uses,
            inputs_by_id=inputs_by_id,
            answer_expression=parsed.answer_expression,
            expression_path=f"{path}.answer_expression",
            path=f"{path}.question_input_uses",
        )
        membership_tests = _parsed_membership_tests(
            parsed.answer_population.membership_tests,
            population_input_refs_by_use_id=(
                input_uses.population_input_refs_by_use_id
            ),
            path=f"{path}.answer_population.membership_tests",
        )
        answer_subject = _answer_subject(
            parsed.answer_subject,
            question_context_texts=question_context_texts,
            path=f"{path}.answer_subject",
        )
        known_inputs = tuple(
            inputs_by_id[input_ref] for input_ref in input_uses.input_refs
        )
        answer_population = _answer_population(
            parsed.answer_population,
            membership_tests=membership_tests,
            path=f"{path}.answer_population",
        )
        answer_expression = _answer_expression(
            parsed.answer_expression,
            group_key_input_refs=input_uses.group_key_input_refs,
            group_key_derivation_input_refs=(
                input_uses.group_key_derivation_input_refs
            ),
            compute_input_refs=input_uses.compute_input_refs,
            limit_input_ref=input_uses.result_limit_input_ref,
            path=f"{path}.answer_expression",
        )
        output.append(
            RequestedFact(
                id=fact_id,
                description=_required_text(
                    parsed.answer_fact,
                    path=f"{path}.answer_fact",
                ),
                answer_expression=answer_expression,
                answer_subject=answer_subject,
                answer_population=answer_population,
                answer_outputs=answer_outputs,
                known_inputs=known_inputs,
                input_refs=input_uses.input_refs,
            )
        )
    return tuple(output)


def _answer_expression(
    item: provider_output.AnswerExpressionOutput,
    *,
    group_key_input_refs: tuple[str, ...],
    group_key_derivation_input_refs: tuple[str, ...],
    compute_input_refs: tuple[str, ...],
    limit_input_ref: str,
    path: str,
) -> RequestedFactAnswerExpression:
    family = RequestedFactAnswerExpressionFamily(
        _required_text(item.family, path=f"{path}.family")
    )
    selection_kind = (
        ResultSelectionKind(item.selection.kind) if item.selection is not None else None
    )
    ordering_basis = ""
    ordering_direction = None
    if item.ordering is not None:
        ordering_basis = _required_text(
            item.ordering.basis, path=f"{path}.ordering.basis"
        )
        ordering_direction = RequestedFactOrderingDirection(
            _required_text(item.ordering.direction, path=f"{path}.ordering.direction")
        )
    return RequestedFactAnswerExpression(
        family=family,
        group_key=_answer_expression_group_key(
            item.group_key,
            question_input_refs=group_key_input_refs,
            derivation_input_refs=group_key_derivation_input_refs,
            path=f"{path}.group_key",
        ),
        ordering_basis=ordering_basis,
        ordering_direction=ordering_direction,
        selection_kind=selection_kind,
        limit_input_ref=limit_input_ref,
        compute_input_refs=compute_input_refs,
    )


def _answer_expression_group_key(
    item: provider_output.GroupKeyOutput | None,
    *,
    question_input_refs: tuple[str, ...],
    derivation_input_refs: tuple[str, ...],
    path: str,
) -> RequestedFactGroupKey | None:
    if item is None:
        return None
    return RequestedFactGroupKey(
        description=_required_text(item.description, path=f"{path}.description"),
        domain=_group_key_domain(item, path=path),
        question_input_refs=question_input_refs,
        derivation_input_refs=derivation_input_refs,
    )


def _group_key_domain(
    item: provider_output.GroupKeyOutput,
    *,
    path: str,
) -> GroupKeyDomainKind:
    try:
        return GroupKeyDomainKind(_required_text(item.domain, path=f"{path}.domain"))
    except ValueError as exc:
        raise ValueError(f"{path}.domain is invalid") from exc


def _answer_subject(
    item: provider_output.AnswerSubjectOutput,
    *,
    question_context_texts: tuple[str, ...],
    path: str,
) -> RequestedFactAnswerSubject:
    return RequestedFactAnswerSubject(
        subject_text=_required_text(item.subject_text, path=f"{path}.subject_text"),
        instance_interpretation=_instance_interpretation(
            item.instance_interpretation,
            path=f"{path}.instance_interpretation",
        ),
    )


def _instance_interpretation(
    item: provider_output.AnswerSubjectInstanceInterpretationOutput,
    *,
    path: str,
) -> RequestedFactAnswerSubjectInstanceInterpretation:
    return RequestedFactAnswerSubjectInstanceInterpretation(
        kind=AnswerSubjectInstanceInterpretationKind(item.kind.strip())
    )


def _answer_population(
    item: provider_output.AnswerPopulationOutput,
    *,
    membership_tests: tuple[_ParsedMembershipTest, ...],
    path: str,
) -> RequestedFactAnswerPopulation:
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
            membership_tests,
        ),
    )


def _parsed_membership_tests(
    items: tuple[provider_output.AnswerPopulationMembershipTestOutput, ...],
    *,
    population_input_refs_by_use_id: dict[str, str],
    path: str,
) -> tuple[_ParsedMembershipTest, ...]:
    output: list[_ParsedMembershipTest] = []
    seen_ids: set[str] = set()
    consumed_use_ids: set[str] = set()
    for index, item in enumerate(items):
        item_path = f"{path}[{index}]"
        test_id = _required_text(item.test_id, path=f"{item_path}.test_id")
        if test_id in seen_ids:
            raise ValueError(f"{item_path}.test_id duplicates membership test")
        seen_ids.add(test_id)
        kind = AnswerPopulationMembershipTestKind(item.kind.strip())
        owned_input_refs: list[str] = []
        seen_use_refs: set[str] = set()
        for use_index, raw_use_ref in enumerate(item.question_input_use_refs):
            use_path = f"{item_path}.question_input_use_refs[{use_index}]"
            use_ref = _required_text(raw_use_ref, path=use_path)
            if use_ref in seen_use_refs:
                raise ValueError(f"{use_path} duplicates question input use")
            seen_use_refs.add(use_ref)
            input_ref = population_input_refs_by_use_id.get(use_ref)
            if input_ref is None:
                raise ValueError(f"{use_path} references unknown question input use")
            owned_input_refs.append(input_ref)
            consumed_use_ids.add(use_ref)
        if kind is AnswerPopulationMembershipTestKind.EXPLICIT_USER_CONSTRAINT:
            if not owned_input_refs:
                raise ValueError(
                    f"explicit membership test {test_id} requires at least one "
                    "question input"
                )
        elif owned_input_refs:
            raise ValueError(f"{item_path} non-explicit membership test has operands")
        output.append(
            _ParsedMembershipTest(
                id=test_id,
                kind=kind,
                polarity=AnswerPopulationMembershipTestPolarity(item.polarity.strip()),
                test_question=_required_text(
                    item.test_question,
                    path=f"{item_path}.test_question",
                ),
                owned_question_input_refs=tuple(owned_input_refs),
            )
        )
    if not output:
        raise ValueError(f"{path} must not be empty")
    unused_use_ids = set(population_input_refs_by_use_id) - consumed_use_ids
    if unused_use_ids:
        raise ValueError(
            "POPULATION_TESTS input uses must be consumed by a membership test: "
            + ", ".join(sorted(unused_use_ids))
        )
    return tuple(output)


def _answer_population_membership_tests(
    items: tuple[_ParsedMembershipTest, ...],
) -> tuple[RequestedFactAnswerPopulationMembershipTest, ...]:
    return tuple(
        RequestedFactAnswerPopulationMembershipTest(
            id=item.id,
            kind=item.kind,
            polarity=item.polarity,
            test_question=item.test_question,
            owned_question_input_refs=item.owned_question_input_refs,
        )
        for item in items
    )


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
    unowned = [known.id for known in question_inputs if known.id not in referenced]
    if unowned:
        raise ValueError(
            "question inputs must be owned by a requested fact: " + ", ".join(unowned)
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
        texts.append(fact.answer_population.counted_unit)
    return tuple(text for text in texts if text.strip())


def _same_question_text(left: str, right: str) -> bool:
    return _normalized_question_text(left) == _normalized_question_text(right)


def _normalized_question_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).casefold()


def _answer_outputs(
    items: tuple[provider_output.AnswerOutputOutput, ...],
    *,
    path: str,
) -> tuple[RequestedFactAnswerOutput, ...]:
    output: list[RequestedFactAnswerOutput] = []
    for output_index, item in enumerate(items, start=1):
        item_path = f"{path}[{output_index - 1}]"
        output.append(
            RequestedFactAnswerOutput(
                id=f"answer_{output_index}",
                description=_required_text(
                    item.description,
                    path=f"{item_path}.description",
                ),
                role=_required_text(
                    item.role,
                    path=f"{item_path}.role",
                ),
            )
        )
    if not output:
        raise ValueError(f"{path} must not be empty")
    return tuple(output)


def _question_input_uses(
    raw: tuple[ProviderObject, ...],
    *,
    inputs_by_id: dict[str, RequestedFactKnownInput],
    answer_expression: provider_output.AnswerExpressionOutput,
    expression_path: str,
    path: str,
) -> _ParsedQuestionInputUses:
    input_refs: list[str] = []
    group_key_input_refs: list[str] = []
    group_key_derivation_input_refs: list[str] = []
    population_input_refs_by_use_id: dict[str, str] = {}
    compute_input_refs: list[str] = []
    result_limit_input_ref = ""
    seen_inputs: set[str] = set()
    seen_use_ids: set[str] = set()
    group_domain = (
        None
        if answer_expression.group_key is None
        else _group_key_domain(
            answer_expression.group_key,
            path=f"{expression_path}.group_key",
        )
    )

    for index, raw_item in enumerate(raw):
        item_path = f"{path}[{index}]"
        item = raw_item.parse_as(provider_output.QuestionInputUseOutput)
        owner_kind = _question_input_owner_kind(item, path=item_path)
        input_ref = _required_text(item.input_ref, path=f"{item_path}.input_ref")
        if input_ref not in inputs_by_id:
            raise ValueError(f"{item_path}.input_ref references unknown question input")
        if input_ref in seen_inputs:
            raise ValueError(f"{item_path}.input_ref duplicates question input")
        seen_inputs.add(input_ref)
        known_input = inputs_by_id[input_ref]
        _validate_input_owner_kind(
            known_input,
            owner_kind=owner_kind,
            answer_expression=answer_expression,
            path=f"{item_path}.owner_kind",
        )
        input_refs.append(input_ref)

        if owner_kind is provider_output.QuestionInputOwnerKind.GROUP_KEY:
            if group_domain is not GroupKeyDomainKind.SPECIFIED_QUESTION_INPUTS:
                raise ValueError(
                    f"{item_path}.owner_kind GROUP_KEY requires "
                    "SPECIFIED_QUESTION_INPUTS"
                )
            group_key_input_refs.append(input_ref)
            continue

        if owner_kind is provider_output.QuestionInputOwnerKind.GROUP_KEY_DERIVATION:
            if group_domain is not GroupKeyDomainKind.SOURCE_RESULT_VALUES:
                raise ValueError(
                    f"{item_path}.owner_kind GROUP_KEY_DERIVATION requires "
                    "SOURCE_RESULT_VALUES"
                )
            group_key_derivation_input_refs.append(input_ref)
            continue

        if owner_kind is provider_output.QuestionInputOwnerKind.RESULT_LIMIT:
            if result_limit_input_ref:
                raise ValueError("answer request may use at most one result limit")
            result_limit_input_ref = input_ref
            continue

        if owner_kind is provider_output.QuestionInputOwnerKind.COMPUTE_EXPRESSION:
            compute_input_refs.append(input_ref)
            continue

        use_id = _required_text(item.use_id or "", path=f"{item_path}.use_id")
        if use_id in seen_use_ids:
            raise ValueError(f"{item_path}.use_id duplicates use ID")
        seen_use_ids.add(use_id)
        population_input_refs_by_use_id[use_id] = input_ref

    if (
        group_domain is GroupKeyDomainKind.SPECIFIED_QUESTION_INPUTS
        and not group_key_input_refs
    ):
        raise ValueError(
            "SPECIFIED_QUESTION_INPUTS requires at least one GROUP_KEY input use"
        )
    if len(group_key_derivation_input_refs) > 1:
        raise ValueError("group key may use at most one grouping grain")
    return _ParsedQuestionInputUses(
        input_refs=tuple(input_refs),
        group_key_input_refs=tuple(group_key_input_refs),
        group_key_derivation_input_refs=tuple(group_key_derivation_input_refs),
        population_input_refs_by_use_id=population_input_refs_by_use_id,
        compute_input_refs=tuple(compute_input_refs),
        result_limit_input_ref=result_limit_input_ref,
    )


def _question_input_owner_kind(
    item: provider_output.QuestionInputUseOutput,
    *,
    path: str,
) -> provider_output.QuestionInputOwnerKind:
    try:
        return provider_output.QuestionInputOwnerKind(item.owner_kind)
    except ValueError as exc:
        raise ValueError(f"{path}.owner_kind is invalid") from exc


def _validate_input_owner_kind(
    known_input: RequestedFactKnownInput,
    *,
    owner_kind: provider_output.QuestionInputOwnerKind,
    answer_expression: provider_output.AnswerExpressionOutput,
    path: str,
) -> None:
    if owner_kind is provider_output.QuestionInputOwnerKind.RESULT_LIMIT:
        if not known_input.is_result_limit:
            raise ValueError(f"{path} RESULT_LIMIT requires a result_limit input")
        return
    if owner_kind is provider_output.QuestionInputOwnerKind.COMPUTE_EXPRESSION:
        if not (
            isinstance(known_input, RequestedFactLiteralInput)
            and known_input.is_formula_value
        ):
            raise ValueError(
                f"{path} COMPUTE_EXPRESSION requires a formula_value input"
            )
        if answer_expression.family != (
            RequestedFactAnswerExpressionFamily.COMPUTED_SCALAR.value
        ):
            raise ValueError(f"{path} COMPUTE_EXPRESSION requires computed_scalar")
        return
    if owner_kind is provider_output.QuestionInputOwnerKind.GROUP_KEY_DERIVATION:
        if not (
            isinstance(known_input, RequestedFactLiteralInput)
            and known_input.is_grouping_grain
        ):
            raise ValueError(
                f"{path} GROUP_KEY_DERIVATION requires a grouping_grain input"
            )
        if answer_expression.family != (
            RequestedFactAnswerExpressionFamily.GROUPED_AGGREGATE.value
        ):
            raise ValueError(f"{path} GROUP_KEY_DERIVATION requires grouped_aggregate")
        return
    if (
        isinstance(known_input, RequestedFactLiteralInput)
        and known_input.is_formula_value
    ):
        raise ValueError(f"{path} for formula_value input must be COMPUTE_EXPRESSION")
    if (
        isinstance(known_input, RequestedFactLiteralInput)
        and known_input.is_grouping_grain
    ):
        raise ValueError(
            f"{path} for grouping_grain input must be GROUP_KEY_DERIVATION"
        )
    if known_input.is_result_limit:
        raise ValueError(f"{path} for result_limit input must be RESULT_LIMIT")


def _question_inputs(
    raw: tuple[ProviderObject, ...],
    *,
    current_question_texts: tuple[str, ...],
    question_context_texts: tuple[str, ...],
) -> tuple[RequestedFactKnownInput, ...]:
    output: list[RequestedFactKnownInput] = []
    seen_ids: set[str] = set()
    for index, raw_item in enumerate(raw):
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
        reference_text = _question_input_source_text(parsed)
        input_text_key = (
            "value_source_text" if kind == KnownInputKind.LITERAL else "reference_text"
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
    raw: ProviderObject,
    *,
    path: str,
) -> (
    provider_output.LiteralTextInputOutput | provider_output.RowSetReferenceInputOutput
):
    kind = _question_input_kind(raw.discriminator("kind"), path=f"{path}.kind")
    if kind == KnownInputKind.LITERAL:
        return raw.parse_as(provider_output.LiteralTextInputOutput)
    if kind == KnownInputKind.ROW_SET_REFERENCE:
        return raw.parse_as(provider_output.RowSetReferenceInputOutput)
    raise ValueError(f"{path}.kind is invalid")


def _question_input_source_text(
    item: provider_output.LiteralTextInputOutput
    | provider_output.RowSetReferenceInputOutput,
) -> str:
    match item:
        case provider_output.LiteralTextInputOutput():
            return item.value_source_text
        case provider_output.RowSetReferenceInputOutput():
            return item.reference_text
    raise AssertionError("unreachable question input contract")


def _question_input_inventory_check(
    item: provider_output.QuestionInputInventoryCheckOutput,
) -> None:
    if not item.all_input_like_phrases_declared:
        raise ValueError(
            "question_input_inventory_check.all_input_like_phrases_declared must be true"
        )


def _question_input_item_inventory_check(
    item: provider_output.QuestionInputItemInventoryCheckOutput, *, path: str
) -> None:
    _required_text(
        item.why_this_is_an_input,
        path=f"{path}.why_this_is_an_input",
    )


def _question_input(
    item: provider_output.LiteralTextInputOutput
    | provider_output.RowSetReferenceInputOutput,
    *,
    input_ref: str,
    kind: KnownInputKind,
    source: KnownInputSource,
    reference_text: str,
    question_context_texts: tuple[str, ...],
    path: str,
) -> RequestedFactKnownInput:
    match item:
        case provider_output.LiteralTextInputOutput():
            role = LiteralInputRole(_required_text(item.role, path=f"{path}.role"))
            raw_occurrence = item.occurrence
            return RequestedFactLiteralInput(
                id=input_ref,
                source=source,
                text=reference_text,
                resolved_value_text=_required_text(
                    item.operand_text,
                    path=f"{path}.operand_text",
                ),
                field_label_text=(item.field_label_text or "").strip(),
                value_meaning_hint=(item.value_meaning_hint or "").strip(),
                role=role,
                resolved_input_ref=(item.resolved_input_ref or "").strip(),
                occurrence=(
                    1
                    if raw_occurrence is None
                    else _positive_int(raw_occurrence, path=f"{path}.occurrence")
                ),
            )
        case provider_output.RowSetReferenceInputOutput():
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


def _question_input_kind(value: str, *, path: str) -> KnownInputKind:
    kind = _required_text(value, path=path)
    if kind == KnownInputKind.LITERAL.value:
        return KnownInputKind.LITERAL
    if kind == KnownInputKind.ROW_SET_REFERENCE.value:
        return KnownInputKind.ROW_SET_REFERENCE
    raise ValueError(f"{path} is invalid")


def _question_input_source(
    value: str,
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


def _positive_int(value: int, *, path: str) -> int:
    if value < 1:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _copied_text(
    value: str,
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


def _required_text(value: str, *, path: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError(f"{path} must not be empty")
    return text
