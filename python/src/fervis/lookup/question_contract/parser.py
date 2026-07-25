"""Parse the provider-authored relational Question Contract."""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Callable
from typing import Mapping, TypeVar

from fervis.lookup.expression_operators import (
    ExpressionBinaryOperator,
    ExpressionUnaryOperator,
)
from fervis.lookup.provider_contract import ProviderObject
import fervis.lookup.question_contract.clarification_provider_contract as clarification_output
from fervis.lookup.question_contract.clarification import (
    IncompleteFactualRequestItem,
    IncompleteFactualRequestKind,
    QuestionContractNeedsClarification,
)
import fervis.lookup.question_contract.provider_contract as output
from fervis.lookup.question_contract.analysis import (
    Groups,
    RequestedFactSemanticIndex,
    Singleton,
    analyze_requested_fact,
)
from fervis.lookup.question_contract.model import (
    Aggregate,
    AggregateFunction,
    AllResults,
    Arithmetic,
    AssociationTerm,
    BooleanComposition,
    BooleanCompositionOperator,
    Comparison,
    Coverage,
    ExpressionNode,
    FactTerm,
    FirstRankWithTies,
    InputDenotation,
    InputDenotationKind,
    InputTerm,
    InstanceInterpretation,
    NullCheck,
    Ordering,
    OrderingDirection,
    Quantifier,
    Quantify,
    QuestionContract,
    RequestedFact,
    RequestedOutput,
    SetTerm,
    Subject,
    ResultSelection,
    TakeWithBoundaryTies,
    TemporalBucket,
    TemporalGrain,
)
from fervis.lookup.semantic_types import (
    BooleanType,
    CollectionType,
    ContextualCurrency,
    CountMeasure,
    DateTimeType,
    DateType,
    DecimalType,
    DurationType,
    IdentifierType,
    IntegerType,
    ItemQuantityUnit,
    MoneyMeasure,
    Measure,
    PercentageMeasure,
    QuantityMeasure,
    RatioMeasure,
    SourceNamedCurrency,
    SourceNamedQuantityUnit,
    SourceOrigin,
    SourceOriginKind,
    ScalarType,
    TemporalScopeType,
    TextType,
    TimeUnit,
    UnitlessMeasure,
    ValueType,
    input_operand_matches_value_type,
)


@dataclass(frozen=True)
class ParsedSemanticQuestionContract:
    decision_basis: str
    contract: QuestionContract
    semantic_indexes: tuple[RequestedFactSemanticIndex, ...]


@dataclass(frozen=True)
class AnswerRequestMeaning:
    requested_fact_id: str
    result_kind: str
    candidate_set_origin: SourceOrigin
    grouping_origins: tuple[SourceOrigin, ...]
    grouping_kinds: tuple[str, ...]
    row_identity_origin: SourceOrigin | None
    ordering_origins: tuple[SourceOrigin, ...]
    output_origins: tuple[SourceOrigin, ...]
    selection_kind: str
    selection_limit_input_ref: str | None
    universal_shape: str


@dataclass(frozen=True)
class ParsedSemanticQuestionMeaning:
    decision_basis: str
    answer_requests: tuple[AnswerRequestMeaning, ...]
    inputs: tuple[InputTerm, ...]
    input_denotations: tuple[InputDenotation, ...]


def parse_semantic_question_frame(
    payload: dict[str, object],
    *,
    question_context_texts: tuple[str, ...],
    conversation_text_by_resolved_input_ref: Mapping[str, str] | None = None,
) -> ParsedSemanticQuestionMeaning | QuestionContractNeedsClarification:
    if not question_context_texts or any(not item for item in question_context_texts):
        raise ValueError("question context is required")
    decision = output.SemanticQuestionFrameDecisionOutput.parse(payload)
    if not decision.decision_basis.strip():
        raise ValueError("decision_basis is required")
    outcome_kind = decision.outcome.discriminator("kind")
    if outcome_kind in {"missing_requested_fact", "unresolved_prior_turn_references"}:
        return _incomplete_outcome(decision.outcome, outcome_kind=outcome_kind)
    if outcome_kind != "question_meaning":
        raise ValueError("unknown question meaning outcome")
    parsed = decision.outcome.parse_as(output.CompleteSemanticQuestionFrameOutput)
    conversation_text = dict(conversation_text_by_resolved_input_ref or {})
    input_interner = _InputInterner(
        question_context_texts=question_context_texts,
        conversation_text_by_ref=conversation_text,
    )
    answer_requests: list[AnswerRequestMeaning] = []
    denotations: list[InputDenotation] = []
    for result_index, result in enumerate(parsed.answer_requests, start=1):
        requested_fact_id = f"fact_{result_index}"
        candidate_set_origin = _frame_source_origin(
            result.qualifying_row_kind,
            conversation_text_by_ref=conversation_text,
        )
        grouping_origins = tuple(
            _frame_source_origin(
                item,
                conversation_text_by_ref=conversation_text,
            )
            for item in result.grouping_meanings
        )
        grouping_kinds = tuple(item.grouping_kind for item in result.grouping_meanings)
        row_identity_origin = (
            _frame_source_origin(
                result.returned_candidate_identity,
                conversation_text_by_ref=conversation_text,
            )
            if result.returned_candidate_identity is not None
            else None
        )
        if not result.return_request_basis.strip():
            raise ValueError("return_request_basis is required")
        answer_values = _frame_answer_values(
            result.answer_values,
            conversation_text_by_ref=conversation_text,
        )
        ordering_origins = _referenced_frame_values(
            result.ordering_value_refs,
            values_by_ref=answer_values,
            label="ordering value",
        )
        projected_non_key_origins = _referenced_frame_values(
            result.returned_value_refs,
            values_by_ref=answer_values,
            label="returned value",
        )
        returned_result = result.returned_result.parse_as(
            output.ReturnedResultOutput
        )
        if result.result_kind == "scalar" and (
            row_identity_origin is not None
            or grouping_origins
            or returned_result.kind != "values"
            or len(projected_non_key_origins) != 1
            or tuple(answer_values.values()) != projected_non_key_origins
            or ordering_origins
        ):
            raise ValueError("scalar requested meaning requires one output")
        if (result.result_kind == "grouped_results") != bool(grouping_origins):
            raise ValueError("grouping meanings must match grouped result meaning")
        if (result.result_kind == "qualifying_instances") != (
            row_identity_origin is not None
        ):
            raise ValueError(
                "row identity meaning must match qualifying-instance result"
            )
        if result.result_kind != "scalar":
            expected_returned_kind = (
                "identities_and_values"
                if projected_non_key_origins
                else "identities"
            )
            if returned_result.kind != expected_returned_kind:
                raise ValueError("returned result contradicts returned value refs")
        output_origins = (
            projected_non_key_origins
            if result.result_kind == "scalar"
            else (
                *((row_identity_origin,) if row_identity_origin is not None else ()),
                *grouping_origins,
                *projected_non_key_origins,
            )
        )
        if not output_origins:
            raise ValueError("requested meaning requires one output")
        selection = result.selection.parse_as(output.SelectionOutput)
        selection_limit_input_ref: str | None = None
        if selection.kind == "take_with_boundary_ties":
            limit = _required(selection.limit, field="selection.limit")
            selection_limit_input_ref = _append_supplied_value(
                limit,
                input_interner=input_interner,
                denotations=denotations,
            )
        elif selection.kind not in {"all_results", "first_rank_with_ties"}:
            raise ValueError("unknown requested result selection")
        if (
            selection.kind
            in {"first_rank_with_ties", "take_with_boundary_ties"}
            and not ordering_origins
        ):
            raise ValueError("ranked result requires an ordering value")
        if result.universal_shape not in {
            "none",
            "every_related_row",
            "every_required_member_has_observation",
        }:
            raise ValueError("unknown universal requirement shape")
        answer_requests.append(
            AnswerRequestMeaning(
                requested_fact_id=requested_fact_id,
                result_kind=result.result_kind,
                candidate_set_origin=candidate_set_origin,
                grouping_origins=grouping_origins,
                grouping_kinds=grouping_kinds,
                row_identity_origin=row_identity_origin,
                ordering_origins=ordering_origins,
                output_origins=output_origins,
                selection_kind=selection.kind,
                selection_limit_input_ref=selection_limit_input_ref,
                universal_shape=result.universal_shape,
            )
        )
    for item in parsed.supplied_values:
        _append_supplied_value(
            item,
            input_interner=input_interner,
            denotations=denotations,
        )
    if {item.input_ref for item in denotations} != set(input_interner.input_by_id):
        raise ValueError(
            "input denotations must cover every supplied input exactly once"
        )
    return ParsedSemanticQuestionMeaning(
        decision_basis=decision.decision_basis.strip(),
        answer_requests=tuple(answer_requests),
        inputs=input_interner.inputs,
        input_denotations=tuple(denotations),
    )


def _frame_answer_values(
    values: tuple[ProviderObject, ...],
    *,
    conversation_text_by_ref: Mapping[str, str],
) -> dict[str, SourceOrigin]:
    parsed: dict[str, SourceOrigin] = {}
    ref_by_origin: dict[SourceOrigin, str] = {}
    for value in values:
        item = value.parse_as(output.NonKeyFrameValueOutput)
        if item.value_ref in parsed:
            raise ValueError("non-key value refs must be unique")
        origin = _frame_source_origin(
            item,
            conversation_text_by_ref=conversation_text_by_ref,
        )
        if origin in ref_by_origin:
            raise ValueError("one non-key value is declared more than once")
        ref_by_origin[origin] = item.value_ref
        parsed[item.value_ref] = origin
    return parsed


def _referenced_frame_values(
    refs: tuple[str, ...],
    *,
    values_by_ref: Mapping[str, SourceOrigin],
    label: str,
) -> tuple[SourceOrigin, ...]:
    if len(refs) != len(set(refs)):
        raise ValueError(f"{label} refs must be unique")
    try:
        return tuple(values_by_ref[ref] for ref in refs)
    except KeyError as exc:
        raise ValueError(f"{label} references an undeclared value") from exc


def _append_supplied_value(
    item: ProviderObject,
    *,
    input_interner: _InputInterner,
    denotations: list[InputDenotation],
) -> str:
    has_identity = item.has_field("entity_reference")
    has_non_entity = item.has_field("non_entity_value")
    if has_identity == has_non_entity:
        raise ValueError("supplied value must choose exactly one denotation")
    if has_identity:
        identity_value = item.parse_as(output.EntityReferenceSuppliedValueOutput)
        identity_reference = identity_value.entity_reference
        operands = identity_reference.value.operands
        origin = identity_reference.value.origin
        meaning = identity_value.meaning
        denotation_basis = identity_value.denotation_basis
        value_type: ScalarType = TextType()
        kind = InputDenotationKind.IDENTITY_REFERENCE
        instance_kind = identity_reference.instance_kind.strip()
        if not instance_kind:
            raise ValueError("entity reference requires an instance kind")
    else:
        scalar_value = item.parse_as(output.NonEntitySuppliedValueOutput)
        non_entity_value = scalar_value.non_entity_value
        operands = non_entity_value.value.operands
        origin = non_entity_value.value.origin
        meaning = scalar_value.meaning
        denotation_basis = scalar_value.denotation_basis
        value_type = input_interner.scalar_value_type(
            non_entity_value.value.value_type
        )
        kind = InputDenotationKind.NON_IDENTITY_SCALAR
        instance_kind = None
    input_ref = input_interner.supplied_value_reference(
        meaning=meaning,
        operands=operands,
        value_type=value_type,
        origin=origin,
    )
    if any(value.input_ref == input_ref for value in denotations):
        raise ValueError("one supplied input has multiple denotations")
    denotations.append(
        InputDenotation(
            id=f"input_denotation_{len(denotations) + 1}",
            input_ref=input_ref,
            operand_meaning=meaning.strip(),
            denotation_basis=denotation_basis.strip(),
            denoted_instance_kind=instance_kind,
            kind=kind,
        )
    )
    return input_ref


def parse_semantic_question_contract(
    payload: dict[str, object],
    *,
    meaning: ParsedSemanticQuestionMeaning,
    question_context_texts: tuple[str, ...],
    conversation_text_by_resolved_input_ref: Mapping[str, str] | None = None,
) -> ParsedSemanticQuestionContract | QuestionContractNeedsClarification:
    if not question_context_texts or any(not item for item in question_context_texts):
        raise ValueError("question context is required")
    decision = output.SemanticQuestionContractDecisionOutput.parse(payload)
    if not decision.decision_basis.strip():
        raise ValueError("decision_basis is required")
    outcome_kind = decision.outcome.discriminator("kind")
    if outcome_kind in {"missing_requested_fact", "unresolved_prior_turn_references"}:
        return _incomplete_outcome(decision.outcome, outcome_kind=outcome_kind)
    if outcome_kind != "question_contract":
        raise ValueError("unknown question contract outcome")
    parsed = decision.outcome.parse_as(output.CompleteSemanticQuestionContractOutput)
    conversation_text = dict(conversation_text_by_resolved_input_ref or {})
    output_by_ref: dict[str, output.AnswerRequestOutput] = {}
    for item in parsed.answer_requests:
        if item.requested_fact_ref in output_by_ref:
            raise ValueError("relational contract repeats an answer request")
        output_by_ref[item.requested_fact_ref] = item
    expected_refs = tuple(item.requested_fact_id for item in meaning.answer_requests)
    if set(output_by_ref) != set(expected_refs):
        raise ValueError(
            "relational contract must cover every answer request exactly once"
        )
    input_by_id = {item.id: item for item in meaning.inputs}
    denotation_by_input_ref = {
        item.input_ref: item for item in meaning.input_denotations
    }
    requested_facts = tuple(
        _requested_fact(
            output_by_ref[request_meaning.requested_fact_id],
            request_meaning=request_meaning,
            question_context_texts=question_context_texts,
            conversation_text_by_ref=conversation_text,
            input_by_id=input_by_id,
            denotation_by_input_ref=denotation_by_input_ref,
        )
        for request_meaning in meaning.answer_requests
    )
    indexes = tuple(
        analyze_requested_fact(
            item,
            inputs=input_by_id,
            input_denotations=denotation_by_input_ref,
        )
        for item in requested_facts
    )
    used_input_refs = {
        use.input_ref for index in indexes for use in index.input_use_sites
    }
    declared_input_refs = set(input_by_id)
    if used_input_refs != declared_input_refs:
        raise ValueError("input inventory must contain exactly the used inputs")
    for request_meaning, requested_fact, index in zip(
        meaning.answer_requests, requested_facts, indexes, strict=True
    ):
        expected_result_kind = (
            "scalar"
            if isinstance(index.result_grain, Singleton)
            else "grouped_results"
            if isinstance(index.result_grain, Groups)
            else "qualifying_instances"
        )
        if request_meaning.result_kind != expected_result_kind:
            raise ValueError("result_kind does not match the requested result grain")
        expressions = requested_fact.expressions
        has_coverage = any(isinstance(item, Coverage) for item in expressions)
        has_quantifier = any(isinstance(item, Quantify) for item in expressions)
        if request_meaning.universal_shape == "every_required_member_has_observation":
            if not has_coverage:
                raise ValueError("required-member coverage expression is missing")
        elif has_coverage:
            raise ValueError("coverage contradicts the requested universal shape")
        if (
            request_meaning.universal_shape == "every_related_row"
            and not has_quantifier
        ):
            raise ValueError("related-row quantifier expression is missing")
    return ParsedSemanticQuestionContract(
        decision_basis=decision.decision_basis.strip(),
        contract=QuestionContract(
            inputs=meaning.inputs,
            requested_facts=requested_facts,
            input_denotations=meaning.input_denotations,
        ),
        semantic_indexes=indexes,
    )


def _incomplete_outcome(
    outcome: ProviderObject,
    *,
    outcome_kind: str,
) -> QuestionContractNeedsClarification:
    if outcome_kind == "missing_requested_fact":
        item = outcome.parse_as(clarification_output.MissingRequestedFactOutput)
        return QuestionContractNeedsClarification(
            missing=(
                IncompleteFactualRequestItem(
                    missing_kind=IncompleteFactualRequestKind.MISSING_REQUESTED_FACT,
                    source_text=item.source_text,
                    why_question_is_incomplete=item.why_question_is_incomplete,
                ),
            )
        )
    parsed = outcome.parse_as(clarification_output.UnresolvedPriorTurnReferencesOutput)
    return QuestionContractNeedsClarification(
        missing=tuple(
            IncompleteFactualRequestItem(
                missing_kind=(
                    IncompleteFactualRequestKind.UNRESOLVED_PRIOR_TURN_REFERENCE
                ),
                source_text=item.source_text,
                target_label=item.target_label,
                why_question_is_incomplete=item.why_question_is_incomplete,
            )
            for item in parsed.references
        )
    )


def _requested_fact(
    item: output.AnswerRequestOutput,
    *,
    request_meaning: AnswerRequestMeaning,
    question_context_texts: tuple[str, ...],
    conversation_text_by_ref: Mapping[str, str],
    input_by_id: Mapping[str, InputTerm],
    denotation_by_input_ref: Mapping[str, InputDenotation],
) -> RequestedFact:
    origin = _source_origin(
        item.origin,
        question_context_texts=question_context_texts,
        conversation_text_by_ref=conversation_text_by_ref,
    )

    def origin_for(value: output.SourceOriginOutput) -> SourceOrigin:
        return _source_origin(
            value,
            question_context_texts=question_context_texts,
            conversation_text_by_ref=conversation_text_by_ref,
        )

    candidate_set_ref = "s1"
    terms = _SemanticTermInterner(
        candidate_set=SetTerm(candidate_set_ref, request_meaning.candidate_set_origin),
        origin_for=origin_for,
    )
    grouping_outputs = tuple(item.grouping)
    if len(grouping_outputs) != len(request_meaning.grouping_origins):
        raise ValueError("relational grouping must match requested grouping meanings")
    for grouping, grouping_origin in zip(
        grouping_outputs,
        request_meaning.grouping_origins,
        strict=True,
    ):
        if grouping.discriminator("kind") != "related_instance_identity":
            continue
        parsed_grouping = grouping.parse_as(output.RelatedIdentityGroupingOutput)
        terms.add_derived_set(parsed_grouping.identified_set.id, grouping_origin)
        terms.add_derived_association(
            parsed_grouping.association.id,
            from_set_ref=candidate_set_ref,
            to_set_ref=parsed_grouping.identified_set.id,
            origin=origin_for(parsed_grouping.association.origin),
        )
    for set_output in item.other_sets:
        terms.add_set(set_output)
    for association_output in item.other_associations:
        terms.add_association(association_output)
    expression_parser = _NestedExpressionParser(
        default_origin=origin,
        input_by_id=input_by_id,
        denotation_by_input_ref=denotation_by_input_ref,
        candidate_set_ref=candidate_set_ref,
        terms=terms,
        origin_for=origin_for,
    )
    qualification_ref = (
        None
        if item.qualification is None
        else expression_parser.reference(item.qualification)
    )
    grouping_refs = tuple(
        _register_grouping(
            grouping,
            grouping_origin=grouping_origin,
            expected_grouping_kind=grouping_kind,
            expression_parser=expression_parser,
            candidate_set_ref=candidate_set_ref,
        )
        for grouping, grouping_origin, grouping_kind in zip(
            grouping_outputs,
            request_meaning.grouping_origins,
            request_meaning.grouping_kinds,
            strict=True,
        )
    )
    if len(item.outputs) != len(request_meaning.output_origins):
        raise ValueError("relational outputs must match requested output meanings")
    outputs = tuple(
        RequestedOutput(
            id=f"output_{index}",
            expression_ref=expression_parser.reference(value.expression),
            origin=output_origin,
        )
        for index, (value, output_origin) in enumerate(
            zip(
                item.outputs,
                request_meaning.output_origins,
                strict=True,
            ),
            start=1,
        )
    )
    if len(item.ordering) != len(request_meaning.ordering_origins):
        raise ValueError("relational ordering must match requested ordering meanings")
    ordering_values: list[Ordering] = []
    for value, ordering_origin in zip(
        item.ordering,
        request_meaning.ordering_origins,
        strict=True,
    ):
        ordering_values.append(
            Ordering(
                expression_ref=expression_parser.reference(value.expression),
                direction=OrderingDirection(value.direction),
                origin=ordering_origin,
            )
        )
    ordering = tuple(ordering_values)
    distinct_by = tuple(
        expression_parser.reference(value) for value in item.distinct_by
    )
    selection = _selection(item.selection, expression_parser=expression_parser)
    return RequestedFact(
        id=request_meaning.requested_fact_id,
        origin=origin,
        sets=expression_parser.sets,
        associations=terms.associations,
        facts=expression_parser.facts,
        expressions=expression_parser.expressions,
        subject=Subject(
            set_ref=candidate_set_ref,
            instance_interpretation=InstanceInterpretation(
                item.candidate_set.instance_interpretation
            ),
        ),
        qualification_ref=qualification_ref,
        grouping_refs=grouping_refs,
        outputs=outputs,
        ordering=ordering,
        selection=selection,
        distinct_by=distinct_by,
    )


def _register_grouping(
    grouping: ProviderObject,
    *,
    grouping_origin: SourceOrigin,
    expected_grouping_kind: str,
    expression_parser: _NestedExpressionParser,
    candidate_set_ref: str,
) -> str:
    kind = grouping.discriminator("kind")
    expected_provider_kind = {
        "qualifying_row_identity": "candidate_instance_identity",
        "related_entity_identity": "related_instance_identity",
        "non_identity_value": "value",
    }.get(expected_grouping_kind)
    if expected_provider_kind is None:
        raise ValueError("unknown frame grouping kind")
    if kind != expected_provider_kind:
        raise ValueError("grouping kind conflicts with the question frame")
    if kind == "candidate_instance_identity":
        candidate_grouping = grouping.parse_as(output.CandidateIdentityGroupingOutput)
        return expression_parser.register_identity_group(
            candidate_grouping.id,
            observed_for_ref=candidate_set_ref,
            identified_set_ref=candidate_set_ref,
            origin=grouping_origin,
        )
    if kind == "related_instance_identity":
        related_grouping = grouping.parse_as(output.RelatedIdentityGroupingOutput)
        return expression_parser.register_identity_group(
            related_grouping.id,
            observed_for_ref=related_grouping.association.id,
            identified_set_ref=related_grouping.identified_set.id,
            origin=grouping_origin,
        )
    if kind == "value":
        value_grouping = grouping.parse_as(output.ValueGroupingOutput)
        return expression_parser.register_group(
            value_grouping.id, value_grouping.expression
        )
    raise ValueError(f"unknown grouping kind: {kind}")


class _SemanticTermInterner:
    def __init__(
        self,
        *,
        candidate_set: SetTerm,
        origin_for: Callable[[output.SourceOriginOutput], SourceOrigin],
    ) -> None:
        self._candidate_set = candidate_set
        self._origin_for = origin_for
        self._sets = [candidate_set]
        self._canonical_set_ref_by_authored_ref = {candidate_set.id: candidate_set.id}
        self._associations: list[AssociationTerm] = []

    @property
    def sets(self) -> tuple[SetTerm, ...]:
        return tuple(self._sets)

    @property
    def associations(self) -> tuple[AssociationTerm, ...]:
        return tuple(self._associations)

    def add_set(self, value: output.SetTermOutput) -> None:
        self.add_derived_set(value.id, self._origin_for(value.origin))

    def add_derived_set(self, set_ref: str, origin: SourceOrigin) -> None:
        if set_ref in self._canonical_set_ref_by_authored_ref:
            return
        self._sets.append(SetTerm(set_ref, origin))
        self._canonical_set_ref_by_authored_ref[set_ref] = set_ref

    def set_ref(self, value: str) -> str:
        try:
            return self._canonical_set_ref_by_authored_ref[value]
        except KeyError as exc:
            raise ValueError(f"unknown semantic set reference: {value}") from exc

    def owner_ref(self, value: str) -> str:
        if value in self._canonical_set_ref_by_authored_ref:
            return self.set_ref(value)
        if value in {item.id for item in self._associations}:
            return value
        raise ValueError(f"unknown semantic owner reference: {value}")

    def add_association(self, value: output.AssociationTermOutput) -> None:
        self.add_derived_association(
            value.id,
            from_set_ref=value.from_set_ref,
            to_set_ref=value.to_set_ref,
            origin=self._origin_for(value.origin),
        )

    def add_derived_association(
        self,
        association_ref: str,
        *,
        from_set_ref: str,
        to_set_ref: str,
        origin: SourceOrigin,
    ) -> None:
        association = AssociationTerm(
            association_ref,
            self.set_ref(from_set_ref),
            self.set_ref(to_set_ref),
            origin,
        )
        existing = next(
            (item for item in self._associations if item.id == association_ref),
            None,
        )
        if existing is not None:
            if (
                existing.from_set_ref == association.from_set_ref
                and existing.to_set_ref == association.to_set_ref
            ):
                return
            raise ValueError(f"conflicting association reference: {association_ref}")
        self._associations.append(association)


class _NestedExpressionParser:
    def __init__(
        self,
        *,
        default_origin: SourceOrigin,
        input_by_id: Mapping[str, InputTerm],
        denotation_by_input_ref: Mapping[str, InputDenotation],
        candidate_set_ref: str,
        terms: _SemanticTermInterner,
        origin_for: Callable[[output.SourceOriginOutput], SourceOrigin],
    ) -> None:
        self._default_origin = default_origin
        self._input_by_id = input_by_id
        self._denotation_by_input_ref = denotation_by_input_ref
        self._candidate_set_ref = candidate_set_ref
        self._terms = terms
        self._origin_for = origin_for
        self._facts: list[FactTerm] = []
        self._fact_ref_by_value: dict[tuple[str, ValueType, SourceOrigin], str] = {}
        self._expressions: list[ExpressionNode] = []
        self._expression_ref_by_value: dict[ExpressionNode, str] = {}
        self._expression_ref_by_group_ref: dict[str, str] = {}

    @property
    def expressions(self) -> tuple[ExpressionNode, ...]:
        return tuple(self._expressions)

    @property
    def facts(self) -> tuple[FactTerm, ...]:
        return tuple(self._facts)

    @property
    def sets(self) -> tuple[SetTerm, ...]:
        return self._terms.sets

    def register_group(self, group_ref: str, expression: ProviderObject) -> str:
        if group_ref in self._expression_ref_by_group_ref:
            raise ValueError(f"duplicate grouping reference: {group_ref}")
        resolved_ref = self.reference(expression)
        self._expression_ref_by_group_ref[group_ref] = resolved_ref
        return resolved_ref

    def register_identity_group(
        self,
        group_ref: str,
        *,
        observed_for_ref: str,
        identified_set_ref: str,
        origin: SourceOrigin,
    ) -> str:
        if group_ref in self._expression_ref_by_group_ref:
            raise ValueError(f"duplicate grouping reference: {group_ref}")
        resolved_ref = self._register_fact(
            observed_for_ref=self._terms.owner_ref(observed_for_ref),
            value_type=IdentifierType(self._terms.set_ref(identified_set_ref)),
            origin=origin,
        )
        self._expression_ref_by_group_ref[group_ref] = resolved_ref
        return resolved_ref

    def reference(self, item: ProviderObject) -> str:
        kind = item.discriminator("kind")
        if kind == "fact":
            return self._fact_reference(item)
        if kind == "set_ref":
            return self._terms.set_ref(_required_text_field(item, "set_ref"))
        if kind == "group_ref":
            ref = _required_text_field(item, "ref")
            if ref not in self._expression_ref_by_group_ref:
                raise ValueError(f"unknown grouping reference: {ref}")
            return self._expression_ref_by_group_ref[ref]
        if kind == "input_ref":
            input_ref = _required_text_field(item, "input_ref")
            if input_ref not in self._input_by_id:
                raise ValueError(f"unknown input reference: {input_ref}")
            denotation = self._denotation_by_input_ref.get(input_ref)
            if denotation is None:
                raise ValueError(f"input lacks denotation: {input_ref}")
            return input_ref
        return self._intern_node(self._node(item, expression_id=""))

    def _intern_node(self, structural_node: ExpressionNode) -> str:
        existing_ref = self._expression_ref_by_value.get(structural_node)
        if existing_ref is not None:
            return existing_ref
        expression_id = f"e{len(self._expressions) + 1}"
        node = _with_expression_id(structural_node, expression_id)
        self._expressions.append(node)
        self._expression_ref_by_value[structural_node] = expression_id
        return expression_id

    def _fact_reference(self, item: ProviderObject) -> str:
        observed_for_ref = self._terms.owner_ref(
            _required_text_field(item, "observed_for_ref")
        )
        value_type_item = _provider_object(item.field("value_type"))
        origin = self._fact_origin(item)
        value_type: ValueType
        if value_type_item.discriminator("kind") == "identifier":
            value_type = IdentifierType(self._identifier_set_ref(value_type_item))
        else:
            value_type = _scalar_type(value_type_item, origin_for=self._origin_for)
        return self._register_fact(
            observed_for_ref=observed_for_ref,
            value_type=value_type,
            origin=origin,
        )

    def _fact_origin(self, item: ProviderObject) -> SourceOrigin:
        return self._origin_for(
            _provider_object(item.field("origin")).parse_as(output.SourceOriginOutput)
        )

    def _register_fact(
        self,
        *,
        observed_for_ref: str,
        value_type: ScalarType,
        origin: SourceOrigin,
    ) -> str:
        key = (observed_for_ref, value_type, origin)
        existing = self._fact_ref_by_value.get(key)
        if existing is not None:
            return existing
        fact_ref = f"f{len(self._facts) + 1}"
        self._facts.append(FactTerm(fact_ref, observed_for_ref, value_type, origin))
        self._fact_ref_by_value[key] = fact_ref
        return fact_ref

    def _identifier_set_ref(self, item: ProviderObject) -> str:
        return self._terms.set_ref(_required_text_field(item, "set_ref"))

    def _node(
        self,
        item: ProviderObject,
        *,
        expression_id: str,
    ) -> ExpressionNode:
        kind = item.discriminator("kind")
        if kind in {"and", "or"}:
            operator = BooleanCompositionOperator(kind)
            arguments = tuple(
                self.reference(value)
                for value in _provider_object_array(item.field("arguments"))
            )
            return BooleanComposition(
                expression_id, operator, arguments, self._default_origin
            )
        if kind == "not":
            return BooleanComposition(
                expression_id,
                BooleanCompositionOperator.NOT,
                (self.reference(_provider_object(item.field("argument"))),),
                self._default_origin,
            )
        if kind == "input_comparison":
            comparison_operator = ExpressionBinaryOperator(
                _required_text_field(item, "operator")
            )
            fact_item = _provider_object(item.field("fact"))
            input_item = _provider_object(item.field("input"))
            input_ref = _required_text_field(input_item, "input_ref")
            if input_ref not in self._input_by_id:
                raise ValueError(f"unknown input reference: {input_ref}")
            denotation = self._denotation_by_input_ref.get(input_ref)
            if denotation is None:
                raise ValueError(f"input lacks denotation: {input_ref}")
            fact_ref = self.reference(fact_item)
            return Comparison(
                expression_id,
                comparison_operator,
                fact_ref,
                input_ref,
                self._default_origin,
            )
        if kind == "value_comparison":
            comparison_operator = ExpressionBinaryOperator(
                _required_text_field(item, "operator")
            )
            return Comparison(
                expression_id,
                comparison_operator,
                self.reference(_provider_object(item.field("left"))),
                self.reference(_provider_object(item.field("right"))),
                self._default_origin,
            )
        if kind == "within":
            return Comparison(
                expression_id,
                ExpressionBinaryOperator.WITHIN,
                self.reference(_provider_object(item.field("value"))),
                self.reference(_provider_object(item.field("scope"))),
                self._default_origin,
            )
        if kind == "null_check":
            null_operator = ExpressionUnaryOperator(
                _required_text_field(item, "operator")
            )
            return NullCheck(
                expression_id,
                null_operator,
                self.reference(_provider_object(item.field("argument"))),
                self._default_origin,
            )
        if kind in {"add", "subtract", "multiply", "divide"}:
            return Arithmetic(
                expression_id,
                ExpressionBinaryOperator(kind),
                (
                    self.reference(_provider_object(item.field("left"))),
                    self.reference(_provider_object(item.field("right"))),
                ),
                self._default_origin,
            )
        if kind == "negate":
            return Arithmetic(
                expression_id,
                ExpressionUnaryOperator.NEGATE,
                (self.reference(_provider_object(item.field("argument"))),),
                self._default_origin,
            )
        if kind == "temporal_bucket":
            return TemporalBucket(
                expression_id,
                self.reference(_provider_object(item.field("value"))),
                TemporalGrain(_required_text_field(item, "grain")),
                self._default_origin,
            )
        if kind in {"aggregate", "filtered_aggregate"}:
            function = AggregateFunction(_required_text_field(item, "function"))
            filter_value = item.field("filter") if item.has_field("filter") else None
            filter_ref = (
                None
                if filter_value is None
                else self.reference(_provider_object(filter_value))
            )
            distinct = _required_bool_field(item, "distinct_argument")
            return Aggregate(
                expression_id,
                function,
                self.reference(_provider_object(item.field("argument"))),
                filter_ref,
                distinct,
                self._default_origin,
            )
        if kind == "quantify":
            quantifier = Quantifier(_required_text_field(item, "quantifier"))
            over_set_ref = self._terms.set_ref(
                _required_text_field(item, "over_set_ref")
            )
            associations = _text_array(item.field("association_refs"))
            return Quantify(
                expression_id,
                quantifier,
                over_set_ref,
                associations,
                self.reference(_provider_object(item.field("condition"))),
                self._default_origin,
            )
        if kind == "coverage":
            candidate = self._candidate_set_ref
            required = self._terms.set_ref(
                _required_text_field(item, "required_member_set_ref")
            )
            observation = self._terms.set_ref(
                _required_text_field(item, "observation_set_ref")
            )
            candidate_path = _unique_association_path(
                self._terms.associations,
                source=candidate,
                destination=observation,
            )
            dimension_path = _unique_association_path(
                self._terms.associations,
                source=required,
                destination=observation,
            )
            required_member_condition_value = item.field("required_member_condition")
            required_member_condition = (
                None
                if required_member_condition_value is None
                else self.reference(_provider_object(required_member_condition_value))
            )
            return Coverage(
                expression_id,
                candidate,
                required,
                observation,
                candidate_path,
                dimension_path,
                required_member_condition,
                self.reference(_provider_object(item.field("observation_condition"))),
                self._default_origin,
            )
        raise ValueError(f"unknown expression kind: {kind}")


def _unique_association_path(
    associations: tuple[AssociationTerm, ...],
    *,
    source: str,
    destination: str,
) -> tuple[str, ...]:
    adjacency: dict[str, list[tuple[str, str]]] = {}
    for association in associations:
        adjacency.setdefault(association.from_set_ref, []).append(
            (association.id, association.to_set_ref)
        )
        adjacency.setdefault(association.to_set_ref, []).append(
            (association.id, association.from_set_ref)
        )
    paths: list[tuple[str, ...]] = []

    def visit(current: str, *, visited: frozenset[str], path: tuple[str, ...]) -> None:
        if len(paths) > 1:
            return
        if current == destination:
            paths.append(path)
            return
        for association_ref, next_set_ref in adjacency.get(current, ()):
            if next_set_ref in visited:
                continue
            visit(
                next_set_ref,
                visited=visited | {next_set_ref},
                path=(*path, association_ref),
            )

    visit(source, visited=frozenset({source}), path=())
    if not paths:
        raise ValueError("coverage association path is missing")
    if len(paths) != 1:
        raise ValueError("coverage association path is ambiguous")
    return paths[0]


def _selection(
    item: ProviderObject | None,
    *,
    expression_parser: _NestedExpressionParser,
) -> ResultSelection:
    if item is None:
        return AllResults()
    parsed = item.parse_as(output.SelectionOutput)
    if parsed.kind == "first_rank_with_ties":
        return FirstRankWithTies()
    if parsed.kind == "take_with_boundary_ties":
        limit = _required(parsed.limit, field="selection.limit")
        if limit.discriminator("kind") != "input_ref":
            raise ValueError("take limit must reference an input")
        return TakeWithBoundaryTies(expression_parser.reference(limit))
    raise ValueError("unknown result selection")


def _source_origin(
    item: output.SourceOriginOutput,
    *,
    question_context_texts: tuple[str, ...],
    conversation_text_by_ref: Mapping[str, str],
) -> SourceOrigin:
    source = SourceOriginKind(item.source)
    if source is SourceOriginKind.CONVERSATION_RESOLUTION:
        resolved_ref = item.resolved_input_ref
        if resolved_ref is None or resolved_ref not in conversation_text_by_ref:
            raise ValueError("unknown conversation resolved input reference")
    del question_context_texts
    return SourceOrigin(
        source=source,
        meaning=item.meaning,
        resolved_input_ref=item.resolved_input_ref,
    )


def _frame_source_origin(
    item: (
        output.MeaningOriginOutput
        | output.GroupingMeaningOutput
        | output.NonKeyFrameValueOutput
    ),
    *,
    conversation_text_by_ref: Mapping[str, str],
) -> SourceOrigin:
    source, resolved_input_ref = _frame_origin(item.origin)
    if (
        source is SourceOriginKind.CONVERSATION_RESOLUTION
        and resolved_input_ref not in conversation_text_by_ref
    ):
        raise ValueError("unknown conversation resolved input reference")
    return SourceOrigin(
        source=source,
        meaning=item.meaning.strip(),
        resolved_input_ref=resolved_input_ref,
    )


def _frame_origin(
    item: output.FrameOriginOutput,
) -> tuple[SourceOriginKind, str | None]:
    if item.kind == "question":
        if item.resolved_input_ref is not None:
            raise ValueError("question origin cannot reference resolved input")
        return SourceOriginKind.QUESTION_CONTEXT, None
    if item.kind == "conversation_resolution":
        if item.resolved_input_ref is None:
            raise ValueError("conversation origin requires resolved input")
        return SourceOriginKind.CONVERSATION_RESOLUTION, item.resolved_input_ref
    raise ValueError("unknown frame origin")


def _value_type(
    item: ProviderObject,
    *,
    allow_collection: bool,
    origin_for: Callable[[output.SourceOriginOutput], SourceOrigin],
) -> ValueType:
    kind = item.discriminator("kind")
    if kind == "collection":
        if not allow_collection:
            raise ValueError("collection is valid only for input terms")
        element = item.field("element_type")
        return CollectionType(
            _scalar_type(_provider_object(element), origin_for=origin_for)
        )
    return _scalar_type(item, origin_for=origin_for)


def _scalar_type(
    item: ProviderObject,
    *,
    origin_for: Callable[[output.SourceOriginOutput], SourceOrigin],
) -> ScalarType:
    kind = item.discriminator("kind")
    if kind == "boolean":
        return BooleanType()
    if kind == "integer":
        return IntegerType()
    if kind in {"text", "property_value"}:
        return TextType()
    if kind == "date":
        return DateType()
    if kind == "datetime":
        return DateTimeType()
    if kind == "temporal_scope":
        return TemporalScopeType()
    if kind == "duration":
        return DurationType(TimeUnit(_required_text_field(item, "unit")))
    if kind == "identifier":
        return IdentifierType(_required_text_field(item, "set_ref"))
    if kind == "decimal":
        return DecimalType(
            _measure(_provider_object(item.field("measure")), origin_for=origin_for)
        )
    raise ValueError(f"unknown scalar type: {kind}")


def _measure(
    item: ProviderObject,
    *,
    origin_for: Callable[[output.SourceOriginOutput], SourceOrigin],
) -> Measure:
    kind = item.discriminator("kind")
    if kind == "unitless":
        return UnitlessMeasure()
    if kind == "count":
        return CountMeasure()
    if kind == "ratio":
        return RatioMeasure()
    if kind == "percentage":
        return PercentageMeasure()
    if kind == "money":
        currency = _provider_object(item.field("currency"))
        if currency.discriminator("kind") == "contextual":
            return MoneyMeasure(ContextualCurrency())
        return MoneyMeasure(
            SourceNamedCurrency(
                origin_for(_source_origin_output(currency, field="currency.origin"))
            )
        )
    if kind == "quantity":
        unit = _provider_object(item.field("unit"))
        if unit.discriminator("kind") == "item":
            return QuantityMeasure(ItemQuantityUnit())
        return QuantityMeasure(
            SourceNamedQuantityUnit(
                origin_for(_source_origin_output(unit, field="quantity.unit.origin"))
            )
        )
    raise ValueError(f"unknown measure: {kind}")


class _InputInterner:
    def __init__(
        self,
        *,
        question_context_texts: tuple[str, ...],
        conversation_text_by_ref: Mapping[str, str],
        existing_inputs: tuple[InputTerm, ...] = (),
    ) -> None:
        self._question_context_texts = question_context_texts
        self._conversation_text_by_ref = conversation_text_by_ref
        self._inputs = list(existing_inputs)
        self._ref_by_origin = {item.origin: item.id for item in existing_inputs}
        if tuple(item.id for item in existing_inputs) != tuple(
            f"i{index}" for index in range(1, len(existing_inputs) + 1)
        ):
            raise ValueError("existing input IDs are not canonical")

    @property
    def inputs(self) -> tuple[InputTerm, ...]:
        return tuple(self._inputs)

    @property
    def input_by_id(self) -> dict[str, InputTerm]:
        return {item.id: item for item in self._inputs}

    def scalar_value_type(self, value_type: ProviderObject) -> ScalarType:
        parsed = _value_type(
            value_type,
            allow_collection=False,
            origin_for=lambda value: _source_origin(
                value,
                question_context_texts=self._question_context_texts,
                conversation_text_by_ref=self._conversation_text_by_ref,
            ),
        )
        if isinstance(parsed, CollectionType):
            raise TypeError("input-role value type must be scalar")
        return parsed

    def supplied_value_reference(
        self,
        *,
        meaning: str,
        operands: tuple[str, ...],
        value_type: ScalarType,
        origin: output.FrameOriginOutput,
    ) -> str:
        parsed_operands = _input_operand(operands)
        operand = (
            parsed_operands[0]
            if len(parsed_operands) == 1
            else parsed_operands
        )
        input_value_type = (
            value_type if isinstance(operand, str) else CollectionType(value_type)
        )
        source, resolved_input_ref = _frame_origin(origin)
        return self._reference(
            source=source,
            meaning=meaning.strip(),
            resolved_input_ref=resolved_input_ref,
            operand=operand,
            value_type=input_value_type,
        )

    def _reference(
        self,
        *,
        source: SourceOriginKind,
        meaning: str,
        resolved_input_ref: str | None,
        operand: str | tuple[str, ...],
        value_type: ValueType,
    ) -> str:
        if source is SourceOriginKind.CONVERSATION_RESOLUTION:
            resolved_ref = resolved_input_ref
            if (
                resolved_ref is None
                or resolved_ref not in self._conversation_text_by_ref
            ):
                raise ValueError("unknown conversation resolved input reference")
        origin = SourceOrigin(
            source=source,
            meaning=meaning,
            resolved_input_ref=resolved_input_ref,
        )
        if not input_operand_matches_value_type(operand, value_type):
            raise ValueError("input operand does not match its declared value type")
        existing = self._ref_by_origin.get(origin)
        if existing is not None:
            existing_input = self.input_by_id[existing]
            if existing_input.value_type != value_type:
                raise ValueError("one supplied input has conflicting value types")
            return existing
        input_ref = f"i{len(self._inputs) + 1}"
        self._inputs.append(
            InputTerm(
                id=input_ref,
                origin=origin,
                operand=operand,
                value_type=value_type,
            )
        )
        self._ref_by_origin[origin] = input_ref
        return input_ref


def _input_operand(value: object) -> str | tuple[str, ...]:
    if isinstance(value, str):
        operand = value.strip()
        if not operand:
            raise ValueError("input operand is empty")
        return operand
    if not isinstance(value, tuple):
        raise ValueError("input operand must be a scalar or collection")
    operands = tuple(item.strip() for item in value if isinstance(item, str))
    if (
        len(operands) != len(value)
        or not operands
        or any(not item for item in operands)
    ):
        raise ValueError("collection input contains an invalid operand")
    if len(set(operands)) != len(operands):
        raise ValueError("collection input repeats an operand")
    return operands


def _provider_object(value: object) -> ProviderObject:
    if isinstance(value, ProviderObject):
        return value
    if not isinstance(value, dict):
        raise ValueError("provider value must be an object")
    return ProviderObject(value)


def _provider_object_array(value: object) -> tuple[ProviderObject, ...]:
    if not isinstance(value, list):
        raise ValueError("provider value must be an array")
    return tuple(_provider_object(item) for item in value)


def _text_array(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("provider value must be a text array")
    return tuple(value)


def _with_expression_id(node: ExpressionNode, node_ref: str) -> ExpressionNode:
    return replace(node, id=node_ref)


def _source_origin_output(
    item: ProviderObject, *, field: str
) -> output.SourceOriginOutput:
    try:
        return output.SourceOriginOutput.parse(item.field("origin"))
    except ValueError as exc:
        raise ValueError(f"{field} is invalid") from exc


_T = TypeVar("_T")


def _required(value: _T | None, *, field: str) -> _T:
    if value is None:
        raise ValueError(f"expression is missing {field}")
    return value


def _required_text_field(item: ProviderObject, field: str) -> str:
    value = item.field(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be text")
    return value


def _optional_text_field(item: ProviderObject, field: str) -> str | None:
    value = item.field(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be non-empty text or null")
    return value


def _required_bool_field(item: ProviderObject, field: str) -> bool:
    value = item.field(field)
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


__all__ = [
    "ParsedSemanticQuestionContract",
    "ParsedSemanticQuestionMeaning",
    "AnswerRequestMeaning",
    "parse_semantic_question_contract",
    "parse_semantic_question_frame",
]
