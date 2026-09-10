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
    RelatedRow,
    QuestionContract,
    RequestedFact,
    RequestedOutput,
    SetTerm,
    Subject,
    ResultSelection,
    TakeWithBoundaryTies,
    PositionWithTies,
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
    NumericType,
    OrderableType,
    PercentageMeasure,
    QuantityMeasure,
    RatioMeasure,
    SourceNamedCurrency,
    SourceNamedQuantityUnit,
    SourceOrigin,
    SourceOriginKind,
    ScalarType,
    TemporalScopeType,
    TemporalPointType,
    TextType,
    TimeUnit,
    UnitlessMeasure,
    UnspecifiedScalarType,
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
    return_request_basis: str
    relational_shape_basis: str
    result_grain_basis: str
    ordering_request_basis: str
    result_kind: str
    candidate_set_origin: SourceOrigin
    grouping_refs: tuple[str, ...]
    grouping_origins: tuple[SourceOrigin, ...]
    grouping_kinds: tuple[str, ...]
    grouping_value_shapes: tuple[tuple[str, str | None] | None, ...]
    ordering_group_refs: tuple[str | None, ...]
    ordering_value_refs: tuple[str | None, ...]
    ordering_origins: tuple[SourceOrigin, ...]
    output_origins: tuple[SourceOrigin, ...]
    output_kinds: tuple[str, ...]
    result_key_count: int
    requested_value_refs: tuple[str, ...]
    selection_kind: str
    selection_limit_input_ref: str | None
    relational_shape: str
    input_refs: tuple[str, ...] = ()


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
    input_collector = _InputCollector(
        question_context_texts=question_context_texts,
        conversation_text_by_ref=conversation_text,
    )
    answer_requests: list[AnswerRequestMeaning] = []
    denotations: list[InputDenotation] = []
    selection_limits = _selection_limit_inputs(
        parsed.supplied_values.selection_limits,
        answer_request_count=len(parsed.answer_requests),
        input_collector=input_collector,
        denotations=denotations,
    )
    result_kind_by_provider_value = {
        "one_value_for_population": "scalar",
        "one_result_per_qualifying_row": "qualifying_instances",
        "one_result_per_group": "grouped_results",
    }
    if not parsed.question_input_inventory_check.all_input_like_phrases_declared:
        raise ValueError("question input inventory must be complete")
    for result_index, frame in enumerate(parsed.answer_requests, start=1):
        if not frame.return_request_basis.strip():
            raise ValueError("return_request_basis is required")
        if not frame.relational_shape_basis.strip():
            raise ValueError("relational_shape_basis is required")
        request = frame.request.parse_as(output.QuestionFrameRequestOutput)
        if not request.result_grain_basis.strip():
            raise ValueError("result_grain_basis is required")
        result = request.result
        try:
            provider_result_kind = result.discriminator("kind")
            result_kind = result_kind_by_provider_value[provider_result_kind]
        except KeyError as error:
            raise ValueError("unknown requested result kind") from error
        requested_fact_id = f"fact_{result_index}"
        candidate_field = (
            "coverage_candidates"
            if request.relational_shape == "every_required_member_has_observation"
            else {
                "one_value_for_population": "population_rows",
                "one_result_per_qualifying_row": "result_candidates",
                "one_result_per_group": "grouped_observation_rows",
            }[provider_result_kind]
        )
        candidate_set_origin = _frame_row_source_origin(
            output.FrameRowSourceOutput.parse(result.field(candidate_field)),
            conversation_text_by_ref=conversation_text,
        )
        grouping_meanings = (
            tuple(
                item.parse_as(output.GroupingMeaningOutput)
                for item in _provider_object_array(
                    result.field("grouping_meanings"),
                )
            )
            if provider_result_kind == "one_result_per_group"
            else ()
        )
        if any(not item.grouping_basis.strip() for item in grouping_meanings):
            raise ValueError("grouping_basis is required")
        grouping_origins = tuple(
            _frame_source_origin(
                item,
                conversation_text_by_ref=conversation_text,
            )
            for item in grouping_meanings
        )
        grouping_kinds = tuple(item.grouping_kind for item in grouping_meanings)
        grouping_value_shapes = tuple(
            _frame_grouping_value_shape(item) for item in grouping_meanings
        )
        grouping_refs = tuple(item.group_ref for item in grouping_meanings)
        grouping_origin_by_ref = {
            item.group_ref: origin
            for item, origin in zip(
                grouping_meanings,
                grouping_origins,
                strict=True,
            )
        }
        if len(grouping_origin_by_ref) != len(grouping_meanings):
            raise ValueError("grouping refs must be unique")
        if result_kind == "scalar":
            returned_meanings = _frame_returned_meanings(
                _provider_object_array(result.field("returned_meanings")),
                conversation_text_by_ref=conversation_text,
            )
        else:
            returned_meanings = _frame_projection_meanings(
                _provider_object(result.field("projection")),
                candidate_origin=candidate_set_origin,
                grouping_origins=grouping_origins,
                grouping_kinds=grouping_kinds,
                conversation_text_by_ref=conversation_text,
            )
        result_order = output.FrameResultOrderOutput.parse(
            result.field("result_order")
        )
        if not result_order.ordering_request_basis.strip():
            raise ValueError("ordering_request_basis is required")
        ordering_kind = result_order.ordering.discriminator("kind")
        if ordering_kind == "no_ordering_requested":
            ordering_values: tuple[ProviderObject, ...] = ()
        elif ordering_kind == "ordered_by":
            ordering_values = result_order.ordering.parse_as(
                output.OrderedByOutput
            ).values
        else:
            raise ValueError("unknown ordering request")
        ordering_origins, ordering_group_refs, ordering_value_refs = _frame_ordering_meanings(
            ordering_values,
            returned_values_by_ref=returned_meanings.returned_values_by_ref,
            grouping_origins_by_ref=grouping_origin_by_ref,
            conversation_text_by_ref=conversation_text,
        )
        if result_kind == "scalar" and (
            grouping_origins
            or len(returned_meanings.output_origins) != 1
            or ordering_origins
        ):
            raise ValueError("scalar requested meaning requires one output")
        if (result_kind == "grouped_results") != bool(grouping_origins):
            raise ValueError("grouping meanings must match grouped result meaning")
        output_origins = returned_meanings.output_origins
        if not output_origins:
            raise ValueError("requested meaning requires one output")
        selection = result_order.selection.parse_as(output.SelectionOutput)
        selection_limit_input_ref: str | None = None
        if selection.kind in {"take_with_boundary_ties", "position_with_ties"}:
            selection_limit_input_ref = selection_limits.pop(result_index, None)
            if selection_limit_input_ref is None:
                raise ValueError("bounded selection requires one selection limit")
        elif selection.kind not in {"all_results", "first_rank_with_ties"}:
            raise ValueError("unknown requested result selection")
        elif result_index in selection_limits:
            raise ValueError("unbounded selection cannot own a selection limit")
        if (
            selection.kind in {"first_rank_with_ties", "take_with_boundary_ties", "position_with_ties"}
            and not ordering_origins
        ):
            raise ValueError("ranked result requires an ordering value")
        if request.relational_shape not in {
            "ordinary",
            "every_related_row",
            "every_required_member_has_observation",
            "same_related_row",
        }:
            raise ValueError("unknown relational requirement shape")
        answer_requests.append(
            AnswerRequestMeaning(
                requested_fact_id=requested_fact_id,
                return_request_basis=frame.return_request_basis.strip(),
                relational_shape_basis=frame.relational_shape_basis.strip(),
                result_grain_basis=request.result_grain_basis.strip(),
                ordering_request_basis=result_order.ordering_request_basis.strip(),
                result_kind=result_kind,
                candidate_set_origin=candidate_set_origin,
                grouping_refs=grouping_refs,
                grouping_origins=grouping_origins,
                grouping_kinds=grouping_kinds,
                grouping_value_shapes=grouping_value_shapes,
                ordering_group_refs=ordering_group_refs,
                ordering_value_refs=ordering_value_refs,
                ordering_origins=ordering_origins,
                output_origins=output_origins,
                output_kinds=returned_meanings.output_kinds,
                result_key_count=returned_meanings.result_key_count,
                requested_value_refs=returned_meanings.requested_value_refs,
                selection_kind=selection.kind,
                selection_limit_input_ref=selection_limit_input_ref,
                relational_shape=request.relational_shape,
            )
        )
    if selection_limits:
        raise ValueError("selection limit references no bounded answer request")
    input_refs_by_fact = {request.requested_fact_id: ([request.selection_limit_input_ref]
        if request.selection_limit_input_ref is not None else []) for request in answer_requests}
    for supplied_operand in parsed.supplied_values.operands:
        if supplied_operand.has_field("entity_reference"):
            entity_operand = supplied_operand.parse_as(output.SuppliedEntityReferenceOutput)
            owners = entity_operand.answer_request_numbers
            ref = _append_entity_reference(entity_operand, input_collector=input_collector, denotations=denotations)
        elif supplied_operand.has_field("non_entity_value"):
            scalar_operand = supplied_operand.parse_as(output.SuppliedNonEntityValueOutput)
            owners = scalar_operand.answer_request_numbers
            ref = _append_non_entity_value(scalar_operand, input_collector=input_collector, denotations=denotations)
        else:
            raise ValueError("supplied operand has no declared value branch")
        if not owners or len(owners)!=len(set(owners)) or any(type(number) is not int or not 1<=number<=len(answer_requests) for number in owners):
            raise ValueError("operand ownership must name existing answer requests exactly once")
        for number in owners:
            input_refs_by_fact[f'fact_{number}'].append(ref)
    answer_requests = [replace(request,input_refs=tuple(input_refs_by_fact[request.requested_fact_id]))
        for request in answer_requests]
    if {item.input_ref for item in denotations} != set(input_collector.input_by_id):
        raise ValueError(
            "input denotations must cover every supplied input exactly once"
        )
    return ParsedSemanticQuestionMeaning(
        decision_basis=decision.decision_basis.strip(),
        answer_requests=tuple(answer_requests),
        inputs=input_collector.inputs,
        input_denotations=tuple(denotations),
    )


def _frame_grouping_value_shape(
    grouping: output.GroupingMeaningOutput,
) -> tuple[str, str | None] | None:
    value = grouping.grouping_value
    if grouping.grouping_kind in {"related_entity_identity", "qualifying_row_identity"}:
        if value is not None:
            raise ValueError("entity grouping cannot declare a value shape")
        return None
    if grouping.grouping_kind != "non_identity_value" or value is None:
        raise ValueError("non-identity grouping requires a value shape")
    kind = value.discriminator("kind")
    if kind in {"observed_value", "computed_value", "condition"}:
        value.parse_as(output.NonTemporalGroupingValueOutput)
        return (kind, None)
    if kind == "temporal_bucket":
        parsed = value.parse_as(output.TemporalBucketGroupingValueOutput)
        return (kind, TemporalGrain(parsed.grain).value)
    raise ValueError("unknown grouping value shape")


def _frame_returned_meanings(
    values: tuple[ProviderObject, ...],
    *,
    conversation_text_by_ref: Mapping[str, str],
) -> _FrameProjection:
    parsed: dict[str, SourceOrigin] = {}
    for value in values:
        item = value.parse_as(output.ReturnedMeaningOutput)
        if item.meaning_ref in parsed:
            raise ValueError("returned meaning refs must be unique")
        origin = _frame_source_origin(
            item,
            conversation_text_by_ref=conversation_text_by_ref,
        )
        parsed[item.meaning_ref] = origin
    return _FrameProjection(
        output_origins=tuple(parsed.values()),
        output_kinds=tuple("value" for _ in parsed),
        result_key_count=0,
        requested_value_refs=tuple(parsed),
        returned_values_by_ref=parsed,
    )


@dataclass(frozen=True)
class _FrameProjection:
    output_origins: tuple[SourceOrigin, ...]
    output_kinds: tuple[str, ...]
    result_key_count: int
    requested_value_refs: tuple[str, ...]
    returned_values_by_ref: Mapping[str, SourceOrigin]


def _frame_projection_meanings(
    value: ProviderObject,
    *,
    candidate_origin: SourceOrigin,
    grouping_origins: tuple[SourceOrigin, ...],
    grouping_kinds: tuple[str, ...],
    conversation_text_by_ref: Mapping[str, str],
) -> _FrameProjection:
    if grouping_origins:
        group_projection = value.parse_as(output.GroupProjectionOutput)
        if group_projection.returned_grouping_keys != "all":
            raise ValueError("grouped projection must return every grouping key")
        candidate_identity = "returned"
        projection_basis = group_projection.projection_basis
        returned_values = group_projection.explicitly_requested_values
        if any(item.value_kind == "related_entity" for item in returned_values):
            raise ValueError("grouped projection cannot return an ungrouped entity")
    else:
        candidate_projection = value.parse_as(output.CandidateProjectionOutput)
        candidate_identity = candidate_projection.candidate_identity
        if candidate_identity not in {"returned", "omitted"}:
            raise ValueError("unknown candidate identity projection")
        projection_basis = candidate_projection.projection_basis
        returned_values = candidate_projection.explicitly_requested_values
    if not projection_basis.strip():
        raise ValueError("projection_basis is required")

    key_origins: tuple[SourceOrigin, ...] = ()
    key_kinds: tuple[str, ...] = ()
    if candidate_identity == "returned":
        if grouping_origins:
            key_origins = grouping_origins
            key_kinds = tuple(
                "identity"
                if grouping_kind
                in {"qualifying_row_identity", "related_entity_identity"}
                else "value"
                for grouping_kind in grouping_kinds
            )
        else:
            key_origins = (candidate_origin,)
            key_kinds = ("identity",)

    values_by_ref: dict[str, SourceOrigin] = {}
    value_kinds_by_ref: dict[str, str] = {}
    for item in returned_values:
        if item.value_ref in values_by_ref:
            raise ValueError("returned value refs must be unique")
        if not item.value_kind_basis.strip():
            raise ValueError("returned value kind basis is required")
        values_by_ref[item.value_ref] = _frame_source_origin(
            item,
            conversation_text_by_ref=conversation_text_by_ref,
        )
        value_kinds_by_ref[item.value_ref] = {
            "related_entity": "related_entity",
            "value": "value",
        }[item.value_kind]
    if not key_origins and not values_by_ref:
        raise ValueError("row projection must return identity or a requested value")
    return _FrameProjection(
        output_origins=(*key_origins, *values_by_ref.values()),
        output_kinds=(
            *key_kinds,
            *value_kinds_by_ref.values(),
        ),
        result_key_count=len(key_origins),
        requested_value_refs=tuple(values_by_ref),
        returned_values_by_ref=values_by_ref,
    )


def _frame_ordering_meanings(
    values: tuple[ProviderObject, ...],
    *,
    returned_values_by_ref: Mapping[str, SourceOrigin],
    grouping_origins_by_ref: Mapping[str, SourceOrigin],
    conversation_text_by_ref: Mapping[str, str],
) -> tuple[tuple[SourceOrigin, ...], tuple[str | None, ...], tuple[str | None, ...]]:
    origins: list[SourceOrigin] = []
    grouping_refs: list[str | None] = []
    value_refs: list[str | None] = []
    returned_refs: set[str] = set()
    for value in values:
        kind = value.discriminator("kind")
        value_ref = None
        if kind == "requested_value_ref":
            requested_value = value.parse_as(output.RequestedValueOrderingOutput)
            if not requested_value.ownership_basis.strip():
                raise ValueError("ordering ownership basis is required")
            try:
                origin = returned_values_by_ref[requested_value.value_ref]
            except KeyError as exc:
                raise ValueError(
                    "ordering references an undeclared returned value"
                ) from exc
            if requested_value.value_ref in returned_refs:
                raise ValueError("ordering returned value refs must be unique")
            returned_refs.add(requested_value.value_ref)
            value_ref = requested_value.value_ref
            grouping_ref = None
        elif kind == "group_ref":
            group_value = value.parse_as(output.GroupOrderingReferenceOutput)
            if not group_value.ownership_basis.strip():
                raise ValueError("ordering ownership basis is required")
            try:
                origin = grouping_origins_by_ref[group_value.group_ref]
            except KeyError as exc:
                raise ValueError(
                    "ordering references an undeclared grouping"
                ) from exc
            grouping_ref = group_value.group_ref
        elif kind == "unreturned_ordering_meaning":
            unreturned_value = value.parse_as(output.UnreturnedOrderingMeaningOutput)
            if not unreturned_value.ownership_basis.strip():
                raise ValueError("ordering ownership basis is required")
            origin = _frame_source_origin(
                unreturned_value,
                conversation_text_by_ref=conversation_text_by_ref,
            )
            grouping_ref = None
        else:
            raise ValueError("unknown ordering meaning kind")
        origins.append(origin)
        grouping_refs.append(grouping_ref)
        value_refs.append(value_ref)
    return tuple(origins), tuple(grouping_refs), tuple(value_refs)


def _selection_limit_inputs(
    limits: tuple[output.SelectionLimitOutput, ...],
    *,
    answer_request_count: int,
    input_collector: _InputCollector,
    denotations: list[InputDenotation],
) -> dict[int, str]:
    result: dict[int, str] = {}
    for limit in limits:
        request_number = limit.answer_request_number
        if request_number < 1 or request_number > answer_request_count:
            raise ValueError("selection limit references an unknown answer request")
        if request_number in result:
            raise ValueError("answer request has multiple selection limits")
        value = limit.non_entity_value.value
        result[request_number] = _record_supplied_value(
            meaning=limit.meaning,
            denotation_basis=limit.denotation_basis,
            operands=value.operands,
            origin=value.origin,
            value_type=input_collector.scalar_value_type(value.value_type),
            kind=InputDenotationKind.NON_IDENTITY_SCALAR,
            instance_kind=None,
            input_collector=input_collector,
            denotations=denotations,
        )
    return result


def _append_entity_reference(
    item: output.SuppliedEntityReferenceOutput,
    *,
    input_collector: _InputCollector,
    denotations: list[InputDenotation],
) -> str:
    reference = item.entity_reference
    instance_kind = reference.instance_kind.strip()
    if not instance_kind:
        raise ValueError("entity reference requires an instance kind")
    operands, origin, descriptions = _identity_value(reference.value)
    return _record_supplied_value(
        meaning=item.meaning,
        denotation_basis=item.denotation_basis,
        operands=operands,
        reference_descriptions=descriptions,
        origin=origin,
        value_type=TextType(),
        kind=InputDenotationKind.IDENTITY_REFERENCE,
        instance_kind=instance_kind,
        input_collector=input_collector,
        denotations=denotations,
    )


def _identity_value(value: ProviderObject):
    values: tuple[output.IdentityOperandOutput, ...]
    kind = value.discriminator("kind")
    if kind == "single_identity":
        single = value.parse_as(output.SingleIdentityValueOutput)
        values, origin = (single.identity_value,), single.origin
    elif kind == "identity_alternatives":
        alternatives = value.parse_as(output.IdentityAlternativesValueOutput)
        values, origin = alternatives.identity_values, alternatives.origin
    else:
        raise ValueError("unknown identity value kind")
    if any(item.kind not in {"literal", "description"} for item in values):
        raise ValueError("Entity reference operands require literal or description syntax")
    if len({(item.value.strip(), item.kind) for item in values}) != len({item.value.strip() for item in values}):
        raise ValueError("Equal reference values with different forms require separate operand declarations")
    return (tuple(item.value.strip() for item in values), origin,
            tuple(item.value.strip() for item in values if item.kind == "description"))


def _append_non_entity_value(
    item: output.SuppliedNonEntityValueOutput,
    *,
    input_collector: _InputCollector,
    denotations: list[InputDenotation],
) -> str:
    value = _non_entity_operand(item)
    return _record_supplied_value(
        meaning=item.meaning,
        denotation_basis=item.denotation_basis,
        operands=value.value.operands,
        origin=value.value.origin,
        value_type=_ledger_value_type(value),
        kind=InputDenotationKind.NON_IDENTITY_SCALAR,
        instance_kind=None,
        input_collector=input_collector,
        denotations=denotations,
    )


def _non_entity_operand(
    item: output.SuppliedNonEntityValueOutput,
) -> output.NonEntityOperandOutput | output.DurationOperandOutput:
    if item.non_entity_value.discriminator("kind") == "duration":
        return item.non_entity_value.parse_as(output.DurationOperandOutput)
    return item.non_entity_value.parse_as(output.NonEntityOperandOutput)


def _ledger_value_type(
    value: output.NonEntityOperandOutput | output.DurationOperandOutput,
) -> ScalarType:
    if value.kind == "categorical_value":
        return TextType()
    if value.kind == "temporal_scope":
        return TemporalScopeType()
    if value.kind == "number":
        return NumericType()
    if value.kind == "boolean":
        return BooleanType()
    if value.kind == "duration":
        if not isinstance(value, output.DurationOperandOutput):
            raise ValueError("duration value requires unit")
        return DurationType(TimeUnit(value.unit))
    raise ValueError("unknown non-entity value kind")


def _record_supplied_value(
    *,
    meaning: str,
    denotation_basis: str,
    operands: tuple[str, ...],
    origin: output.FrameOriginOutput,
    value_type: ScalarType,
    kind: InputDenotationKind,
    instance_kind: str | None,
    input_collector: _InputCollector,
    denotations: list[InputDenotation],
    reference_descriptions: tuple[str, ...] = (),
) -> str:
    input_ref = input_collector.supplied_value_reference(
        meaning=meaning,
        operands=operands,
        value_type=value_type,
        origin=origin,
    )
    denotations.append(
        InputDenotation(
            id=f"input_denotation_{len(denotations) + 1}",
            input_ref=input_ref,
            operand_meaning=meaning.strip(),
            denotation_basis=denotation_basis.strip(),
            denoted_instance_kind=instance_kind,
            kind=kind,
            reference_descriptions=reference_descriptions,
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
        has_related_row = any(isinstance(item, RelatedRow) for item in expressions)
        if request_meaning.relational_shape == "every_required_member_has_observation":
            if not has_coverage:
                raise ValueError("required-member coverage expression is missing")
        elif has_coverage:
            raise ValueError("coverage contradicts the requested relational shape")
        if (
            request_meaning.relational_shape == "every_related_row"
            and not _requires_universal_qualification(requested_fact, index)
        ):
            raise ValueError("universal qualification is missing from the required result")
        if request_meaning.relational_shape == "same_related_row":
            if not has_related_row:
                raise ValueError("same-related-row expression is missing")
        elif has_related_row:
            raise ValueError(
                "related-row expression contradicts the requested relational shape"
            )
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


def _requires_universal_qualification(fact: RequestedFact, index: RequestedFactSemanticIndex) -> bool:
    expressions = {item.id: item for item in fact.expressions}

    def required(ref: str, positive: bool = True) -> bool:
        node = expressions.get(ref)
        if isinstance(node, Quantify):
            return (positive and node.quantifier in {Quantifier.FORALL, Quantifier.NOT_EXISTS}) or (not positive and node.quantifier is Quantifier.EXISTS)
        if not isinstance(node, BooleanComposition):
            return False
        if node.operator is BooleanCompositionOperator.NOT:
            return required(node.argument_refs[0], not positive)
        conjunction = (node.operator is BooleanCompositionOperator.AND) == positive
        results = (required(child, positive) for child in node.argument_refs)
        return any(results) if conjunction else all(results)

    if fact.qualification_ref is not None:
        return required(fact.qualification_ref)
    if isinstance(index.result_grain, Singleton):
        return any(required(output.expression_ref) for output in fact.outputs)
    return False


def _requested_fact(
    item: output.AnswerRequestOutput,
    *,
    request_meaning: AnswerRequestMeaning,
    question_context_texts: tuple[str, ...],
    conversation_text_by_ref: Mapping[str, str],
    input_by_id: Mapping[str, InputTerm],
    denotation_by_input_ref: Mapping[str, InputDenotation],
) -> RequestedFact:
    candidate_instance_kind = item.candidate_set.instance_kind.strip()
    if candidate_instance_kind != request_meaning.candidate_set_origin.meaning:
        raise ValueError("candidate set type conflicts with the question frame")
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
        candidate_instance_kind=candidate_instance_kind,
        origin_for=origin_for,
    )
    grouping_outputs = tuple(item.grouping)
    if len(grouping_outputs) != len(request_meaning.grouping_origins):
        raise ValueError("relational grouping must match requested grouping meanings")
    if tuple(_required_text_field(group, "id") for group in grouping_outputs) != request_meaning.grouping_refs:
        raise ValueError("relational grouping must preserve frame references")
    for group, shape in zip(grouping_outputs, request_meaning.grouping_value_shapes, strict=True):
        if shape is None:
            continue
        expression = _provider_object(group.field("expression"))
        kind, grain = shape
        if kind == "observed_value" and expression.discriminator("kind") != "fact":
            raise ValueError("grouping expression conflicts with its declared recorded value")
        if kind == "computed_value" and expression.discriminator("kind") not in {
            "add", "subtract", "multiply", "divide", "negate"
        }:
            raise ValueError("grouping expression conflicts with its declared computation")
        if kind == "condition" and expression.discriminator("kind") not in {
            "and", "or", "not", "input_comparison", "value_comparison", "within",
            "null_check", "quantify", "coverage"
        }:
            raise ValueError("grouping expression conflicts with its declared condition")
        if kind == "temporal_bucket" and (
            expression.discriminator("kind") != "temporal_bucket"
            or expression.field("grain") != grain
        ):
            raise ValueError("grouping expression conflicts with its declared calendar grain")
    expected_selection = request_meaning.selection_kind
    if item.selection is None:
        actual_selection = "all_results"
    else:
        actual_selection = item.selection.discriminator("kind")
    if actual_selection != expected_selection:
        raise ValueError("selection conflicts with the question frame")
    if expected_selection in {"take_with_boundary_ties", "position_with_ties"}:
        assert item.selection is not None
        limit = _provider_object(item.selection.field("limit"))
        if limit.discriminator("kind") != "input_ref" or limit.field("input_ref") != request_meaning.selection_limit_input_ref:
            raise ValueError("selection limit conflicts with the question frame")
    terms.add_graph(item.set_graph.parse_as(output.SetGraphOutput))
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
        else expression_parser.reference(item.qualification, expected_type=BooleanType())
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
    result_key_outputs = iter(item.outputs.result_key_outputs)
    requested_value_outputs_by_ref = {
        _required_text_field(value, "output_ref"): value
        for value in item.outputs.requested_value_outputs
    }
    if len(requested_value_outputs_by_ref) != len(item.outputs.requested_value_outputs):
        raise ValueError("relational requested value refs must be unique")
    if tuple(requested_value_outputs_by_ref) != (
        request_meaning.requested_value_refs
    ):
        raise ValueError("relational requested values must preserve frame references")
    requested_value_outputs = iter(requested_value_outputs_by_ref.values())
    if (
        len(item.outputs.result_key_outputs)
        + len(item.outputs.requested_value_outputs)
        != len(request_meaning.output_origins)
    ):
        raise ValueError("relational outputs must match requested output meanings")
    outputs: list[RequestedOutput] = []
    for index, (output_origin, output_kind) in enumerate(
        zip(
            request_meaning.output_origins,
            request_meaning.output_kinds,
            strict=True,
        ),
        start=1,
    ):
        is_result_key = index <= request_meaning.result_key_count
        if is_result_key:
            result_key = next(result_key_outputs)
            expression_ref = expression_parser.reference(result_key.expression)
        elif output_kind == "related_entity":
            value = next(requested_value_outputs)
            related_value = value.parse_as(
                output.RequestedRelatedEntityOutputOutput
            )
            expression_ref = expression_parser.register_returned_identity(
                output_ref=related_value.output_ref,
                instance_kind=related_value.instance_kind,
                origin=output_origin,
            )
        else:
            value = next(requested_value_outputs)
            parsed_value = value.parse_as(output.RequestedValueOutputOutput)
            expression_ref = expression_parser.reference(parsed_value.expression)
        if (is_result_key and request_meaning.result_kind == "qualifying_instances"
            and output_kind == "identity"
            and expression_parser.identified_set_ref(expression_ref) != candidate_set_ref):
            raise ValueError("candidate identity output must identify the candidate set")
        is_identity = expression_parser.is_identity_ref(expression_ref)
        if output_kind in {"identity", "related_entity"} and not is_identity:
            raise ValueError("result identity output must reference an identity")
        if output_kind == "value" and is_identity:
            raise ValueError("result value output cannot reference an identity")
        outputs.append(
            RequestedOutput(
                id=f"output_{index}",
                expression_ref=expression_ref,
                origin=output_origin,
            )
        )
    if len(item.ordering) != len(request_meaning.ordering_origins):
        raise ValueError("relational ordering must match requested ordering meanings")
    ordering_values: list[Ordering] = []
    for ordering_value, ordering_origin, ordering_group_ref, ordering_value_ref in zip(
        item.ordering,
        request_meaning.ordering_origins,
        request_meaning.ordering_group_refs,
        request_meaning.ordering_value_refs,
        strict=True,
    ):
        if ordering_group_ref is not None and (
            ordering_value.expression.discriminator("kind") != "group_ref"
            or ordering_value.expression.field("ref") != ordering_group_ref
        ):
            raise ValueError("ordering must reference its declared grouping")
        if (
            ordering_group_ref is None
            and ordering_value.expression.discriminator("kind") == "group_ref"
        ):
            raise ValueError("ordering must realize its declared non-key value")
        if ordering_value.expression.discriminator("kind") == "requested_value_ref":
            if ordering_value_ref is None or ordering_value.expression.field("value_ref") != ordering_value_ref:
                raise ValueError("ordering must reference its declared requested value")
            from .output_references import requested_value_output_index
            expression_ref = outputs[requested_value_output_index(request_meaning, ordering_value_ref)].expression_ref
        else:
            if ordering_value_ref is not None:
                raise ValueError("ordering must reference its declared requested value")
            expression_ref = expression_parser.reference(
                ordering_value.expression, expected_type=OrderableType(),
            )
        ordering_values.append(
            Ordering(
                expression_ref=expression_ref,
                direction=OrderingDirection(ordering_value.direction),
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
        outputs=tuple(outputs),
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
        return expression_parser.register_related_identity_group(
            related_grouping.id,
            set_ref=related_grouping.set_ref,
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
        candidate_instance_kind: str,
        origin_for: Callable[[output.SourceOriginOutput], SourceOrigin],
    ) -> None:
        self._candidate_set = candidate_set
        self._origin_for = origin_for
        self._sets = [candidate_set]
        self._canonical_set_ref_by_authored_ref = {candidate_set.id: candidate_set.id}
        self._instance_kind_by_set_ref = {
            candidate_set.id: candidate_instance_kind,
        }
        self._associations: list[AssociationTerm] = []
        self._set_ref_by_output_ref: dict[str, str] = {}

    @property
    def sets(self) -> tuple[SetTerm, ...]:
        return tuple(self._sets)

    @property
    def associations(self) -> tuple[AssociationTerm, ...]:
        return tuple(self._associations)

    def add_set(self, value: output.SetTermOutput) -> None:
        self.add_derived_set(
            value.id,
            self._origin_for(value.origin),
            instance_kind=value.instance_kind,
        )

    def add_derived_set(
        self,
        set_ref: str,
        origin: SourceOrigin,
        *,
        instance_kind: str,
    ) -> None:
        if set_ref in self._canonical_set_ref_by_authored_ref:
            raise ValueError(f"duplicate semantic set declaration: {set_ref}")
        self._sets.append(SetTerm(set_ref, origin))
        self._canonical_set_ref_by_authored_ref[set_ref] = set_ref
        self._instance_kind_by_set_ref[set_ref] = instance_kind.strip()

    def set_ref(self, value: str) -> str:
        try:
            return self._canonical_set_ref_by_authored_ref[value]
        except KeyError as exc:
            raise ValueError(f"unknown semantic set reference: {value}") from exc

    def require_instance_kind(self, set_ref: str, expected: str) -> None:
        canonical_ref = self.set_ref(set_ref)
        if self._instance_kind_by_set_ref[canonical_ref] != expected.strip():
            raise ValueError("entity type conflicts with the identified set")

    def identity_owner_for_set(self, set_ref: str) -> tuple[str, str]:
        canonical_set_ref = self.set_ref(set_ref)
        incoming = tuple(
            association
            for association in self._associations
            if association.to_set_ref == canonical_set_ref
        )
        if len(incoming) != 1:
            raise ValueError(
                "related requested value requires one incoming association"
            )
        return incoming[0].id, canonical_set_ref

    def owner_ref(self, value: str) -> str:
        if value in self._canonical_set_ref_by_authored_ref:
            return self.set_ref(value)
        if value in {item.id for item in self._associations}:
            return value
        raise ValueError(f"unknown semantic owner reference: {value}")

    def direct_association_refs(
        self,
        *,
        from_set_ref: str,
        to_set_ref: str,
    ) -> tuple[str, ...]:
        refs = tuple(
            item.id
            for item in self._associations
            if item.from_set_ref == from_set_ref and item.to_set_ref == to_set_ref
        )
        if not refs:
            raise ValueError("related row lacks a direct association")
        return refs

    def association_target(
        self,
        association_ref: str,
        *,
        from_set_ref: str,
    ) -> str:
        association = next(
            (item for item in self._associations if item.id == association_ref),
            None,
        )
        if association is None:
            raise ValueError(f"unknown semantic association: {association_ref}")
        if association.from_set_ref != self.set_ref(from_set_ref):
            raise ValueError(
                "identity path association does not start at the current row"
            )
        return association.to_set_ref

    def set_ref_for_output(self, output_ref: str) -> str:
        try:
            return self._set_ref_by_output_ref[output_ref]
        except KeyError as exc:
            raise ValueError(
                f"related output has no relation: {output_ref}"
            ) from exc

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
            raise ValueError(
                f"duplicate semantic association declaration: {association_ref}"
            )
        self._associations.append(association)

    def add_graph(
        self,
        graph: output.SetGraphOutput,
    ) -> None:
        for relation in graph.identity_input_relations.values():
            if relation is None:
                continue
            self._add_role_relation(relation)
        for output_ref, relation in graph.requested_output_relations.items():
            _, set_ref = self._add_role_relation(relation)
            self._set_ref_by_output_ref[output_ref] = set_ref
        self._add_related_sets(
            graph.other_related_sets,
            parent_set_ref=self._candidate_set.id,
        )

    def _add_role_relation(
        self,
        relation: output.RoleRelationOutput,
    ) -> tuple[str, str]:
        self.add_set(relation.set)
        association_ref = _required_text_field(
            relation.association,
            "id",
        )
        self.add_derived_association(
            association_ref,
            from_set_ref=self._candidate_set.id,
            to_set_ref=relation.set.id,
            origin=self._origin_for(
                _provider_object(relation.association.field("origin")).parse_as(
                    output.SourceOriginOutput
                )
            ),
        )
        self._add_related_sets(
            relation.related_sets,
            parent_set_ref=relation.set.id,
        )
        return association_ref, relation.set.id

    def _add_related_sets(
        self,
        values: tuple[ProviderObject, ...],
        *,
        parent_set_ref: str,
    ) -> None:
        for item in values:
            set_value = _provider_object(item.field("set"))
            association_values = (
                _provider_object_array(item.field("associations"))
                if item.has_field("associations")
                else (_provider_object(item.field("association")),)
            )
            set_output = set_value.parse_as(output.SetTermOutput)
            self.add_set(set_output)
            set_ref = set_output.id
            for association in association_values:
                self.add_derived_association(
                    _required_text_field(association, "id"),
                    from_set_ref=parent_set_ref,
                    to_set_ref=set_ref,
                    origin=self._origin_for(
                        _provider_object(association.field("origin")).parse_as(
                            output.SourceOriginOutput
                        )
                    ),
                )
            self._add_related_sets(
                _provider_object_array(item.field("related_sets")),
                parent_set_ref=set_ref,
            )


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
        self._fact_refs_by_origin: dict[tuple[str, SourceOrigin, bool], list[str]] = {}
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

    def identified_set_ref(self, ref: str) -> str | None:
        if ref in {item.id for item in self._terms.sets}:
            return ref
        return next((item.value_type.set_ref for item in self._facts
                     if item.id == ref and isinstance(item.value_type, IdentifierType)), None)

    def is_identity_ref(self, ref: str) -> bool:
        return self.identified_set_ref(ref) is not None

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

    def register_related_identity_group(
        self,
        group_ref: str,
        *,
        set_ref: str,
        origin: SourceOrigin,
    ) -> str:
        association_ref, identified_set_ref = self._terms.identity_owner_for_set(
            set_ref
        )
        return self.register_identity_group(
            group_ref,
            observed_for_ref=association_ref,
            identified_set_ref=identified_set_ref,
            origin=origin,
        )

    def register_returned_identity(
        self,
        *,
        output_ref: str,
        instance_kind: str,
        origin: SourceOrigin,
    ) -> str:
        set_ref = self._terms.set_ref_for_output(output_ref)
        self._terms.require_instance_kind(set_ref, instance_kind)
        association_ref, identified_set_ref = self._terms.identity_owner_for_set(set_ref)
        return self._register_fact(
            observed_for_ref=association_ref,
            value_type=IdentifierType(identified_set_ref),
            origin=origin,
        )

    def reference(
        self,
        item: ProviderObject,
        *,
        expected_type: ScalarType | None = None,
        row_set_ref: str | None = None,
        identity_input_ref: str | None = None,
    ) -> str:
        kind = item.discriminator("kind")
        if kind == "fact":
            return self._fact_reference(
                item,
                expected_type=expected_type or UnspecifiedScalarType(),
                row_set_ref=row_set_ref,
                identity_input_ref=identity_input_ref,
            )
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
        return self._intern_node(
            self._node(item, expression_id="", row_set_ref=row_set_ref)
        )

    def _intern_node(self, structural_node: ExpressionNode) -> str:
        existing_ref = self._expression_ref_by_value.get(structural_node)
        if existing_ref is not None:
            return existing_ref
        expression_id = f"e{len(self._expressions) + 1}"
        node = _with_expression_id(structural_node, expression_id)
        self._expressions.append(node)
        self._expression_ref_by_value[structural_node] = expression_id
        return expression_id

    def _fact_reference(
        self,
        item: ProviderObject,
        *,
        expected_type: ScalarType,
        row_set_ref: str | None,
        identity_input_ref: str | None,
    ) -> str:
        identity_contract = self._identity_fact_contract(
            item,
            row_set_ref=row_set_ref,
            identity_input_ref=identity_input_ref,
        )
        observed_for_ref = (
            identity_contract[0]
            if identity_contract is not None
            else (
                self._terms.owner_ref(_required_text_field(item, "observed_for_ref"))
                if item.has_field("observed_for_ref")
                else self._terms.set_ref(row_set_ref or _raise_missing_row_scope())
            )
        )
        origin = self._fact_origin(item)
        value_type = (
            IdentifierType(identity_contract[1])
            if identity_contract is not None
            else expected_type
        )
        return self._register_fact(
            observed_for_ref=observed_for_ref,
            value_type=_merge_fact_types(value_type, expected_type),
            origin=origin,
        )

    def _identity_fact_contract(
        self,
        item: ProviderObject,
        *,
        row_set_ref: str | None,
        identity_input_ref: str | None = None,
    ) -> tuple[str, str] | None:
        if not item.has_field("identity_path"):
            return None
        current_set_ref = self._terms.set_ref(row_set_ref or self._candidate_set_ref)
        identity_path = _provider_object(item.field("identity_path"))
        path_kind = _required_text_field(identity_path, "kind")
        if path_kind == "candidate_instance":
            return current_set_ref, current_set_ref
        if path_kind != "related_instance":
            raise ValueError(f"unknown identity path kind: {path_kind}")
        association_ref = _required_text_field(identity_path, "association_ref")
        identified_set_ref = self._terms.association_target(
            association_ref,
            from_set_ref=current_set_ref,
        )
        return association_ref, identified_set_ref

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
        # Identity and observed-property terms remain different even when their
        # prose origin is equal. Type refinement must not turn a returned value
        # into an identity after output validation has already accepted it.
        key = (observed_for_ref, origin, isinstance(value_type, IdentifierType))
        for existing in self._fact_refs_by_origin.get(key, ()):
            index = next(
                index for index, fact in enumerate(self._facts) if fact.id == existing
            )
            fact = self._facts[index]
            try:
                merged_type = _merge_fact_types(fact.value_type, value_type)
            except ValueError:
                continue
            if merged_type != fact.value_type:
                self._facts[index] = FactTerm(
                    fact.id,
                    fact.owner_ref,
                    merged_type,
                    fact.origin,
                )
            return existing
        fact_ref = f"f{len(self._facts) + 1}"
        self._facts.append(FactTerm(fact_ref, observed_for_ref, value_type, origin))
        self._fact_refs_by_origin.setdefault(key, []).append(fact_ref)
        return fact_ref

    def _node(
        self,
        item: ProviderObject,
        *,
        expression_id: str,
        row_set_ref: str | None,
    ) -> ExpressionNode:
        kind = item.discriminator("kind")
        if kind in {"and", "or"}:
            operator = BooleanCompositionOperator(kind)
            arguments = tuple(
                self.reference(
                    value,
                    expected_type=BooleanType(),
                    row_set_ref=row_set_ref,
                )
                for value in _provider_object_array(item.field("arguments"))
            )
            return BooleanComposition(
                expression_id, operator, arguments, self._default_origin
            )
        if kind == "not":
            return BooleanComposition(
                expression_id,
                BooleanCompositionOperator.NOT,
                (
                    self.reference(
                        _provider_object(item.field("argument")),
                        expected_type=BooleanType(),
                        row_set_ref=row_set_ref,
                    ),
                ),
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
            if denotation.kind is InputDenotationKind.IDENTITY_REFERENCE:
                operand_meaning = _required_text_field(
                    input_item,
                    "operand_meaning",
                )
                instance_kind = _required_text_field(input_item, "instance_kind")
                if operand_meaning != denotation.operand_meaning:
                    raise ValueError(
                        "identity operand meaning conflicts with its declaration"
                    )
                if instance_kind != denotation.denoted_instance_kind:
                    raise ValueError(
                        "identity type conflicts with its declaration"
                    )
            identity_contract = self._identity_fact_contract(
                fact_item,
                row_set_ref=row_set_ref,
                identity_input_ref=input_ref,
            )
            if denotation.kind is InputDenotationKind.IDENTITY_REFERENCE:
                if identity_contract is None:
                    raise ValueError("identity comparison lacks an identity path")
                self._terms.require_instance_kind(
                    identity_contract[1],
                    instance_kind,
                )
            fact_type = self._comparison_fact_type(
                input_ref,
                identified_set_ref=(
                    identity_contract[1] if identity_contract is not None else None
                ),
            )
            fact_ref = self.reference(
                fact_item,
                expected_type=fact_type,
                row_set_ref=row_set_ref,
                identity_input_ref=input_ref,
            )
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
            comparison_type: ScalarType = (
                OrderableType()
                if comparison_operator
                in {
                    ExpressionBinaryOperator.LT,
                    ExpressionBinaryOperator.LTE,
                    ExpressionBinaryOperator.GT,
                    ExpressionBinaryOperator.GTE,
                }
                else TextType()
                if comparison_operator is ExpressionBinaryOperator.CONTAINS
                else UnspecifiedScalarType()
            )
            # Carry declared operand types across the comparison. An abstract
            # orderable fact must inherit a numeric input/computation's constraint.
            for operand_name in ("left", "right"):
                operand = _provider_object(item.field(operand_name))
                operand_kind = operand.discriminator("kind")
                if operand_kind == "input_ref":
                    input_ref = _required_text_field(operand, "input_ref")
                    value_type = self._input_by_id[input_ref].value_type
                    if not isinstance(value_type, (IdentifierType, CollectionType, TemporalScopeType)):
                        comparison_type = value_type
                        break
                elif operand_kind in {"add", "subtract", "multiply", "divide", "negate"}:
                    comparison_type = NumericType()
                    break
            return Comparison(
                expression_id,
                comparison_operator,
                self.reference(
                    _provider_object(item.field("left")),
                    expected_type=comparison_type,
                    row_set_ref=row_set_ref,
                ),
                self.reference(
                    _provider_object(item.field("right")),
                    expected_type=comparison_type,
                    row_set_ref=row_set_ref,
                ),
                self._default_origin,
            )
        if kind == "within":
            return Comparison(
                expression_id,
                ExpressionBinaryOperator.WITHIN,
                self.reference(
                    _provider_object(item.field("value")),
                    expected_type=TemporalPointType(),
                    row_set_ref=row_set_ref,
                ),
                self.reference(
                    _provider_object(item.field("scope")),
                    row_set_ref=row_set_ref,
                ),
                self._default_origin,
            )
        if kind == "null_check":
            null_operator = ExpressionUnaryOperator(
                _required_text_field(item, "operator")
            )
            return NullCheck(
                expression_id,
                null_operator,
                self.reference(
                    _provider_object(item.field("argument")),
                    row_set_ref=row_set_ref,
                ),
                self._default_origin,
            )
        if kind in {"add", "subtract", "multiply", "divide"}:
            return Arithmetic(
                expression_id,
                ExpressionBinaryOperator(kind),
                (
                    self.reference(
                        _provider_object(item.field("left")),
                        expected_type=NumericType(),
                        row_set_ref=row_set_ref,
                    ),
                    self.reference(
                        _provider_object(item.field("right")),
                        expected_type=NumericType(),
                        row_set_ref=row_set_ref,
                    ),
                ),
                self._default_origin,
            )
        if kind == "negate":
            return Arithmetic(
                expression_id,
                ExpressionUnaryOperator.NEGATE,
                (
                    self.reference(
                        _provider_object(item.field("argument")),
                        expected_type=NumericType(),
                        row_set_ref=row_set_ref,
                    ),
                ),
                self._default_origin,
            )
        if kind == "temporal_bucket":
            return TemporalBucket(
                expression_id,
                self.reference(
                    _provider_object(item.field("value")),
                    expected_type=TemporalPointType(),
                    row_set_ref=row_set_ref,
                ),
                TemporalGrain(_required_text_field(item, "grain")),
                self._default_origin,
            )
        if kind in {"aggregate", "filtered_aggregate"}:
            function = AggregateFunction(_required_text_field(item, "function"))
            filter_value = item.field("filter") if item.has_field("filter") else None
            filter_ref = (
                None
                if filter_value is None
                else self.reference(
                    _provider_object(filter_value),
                    expected_type=BooleanType(),
                )
            )
            distinct = _required_bool_field(item, "distinct_argument")
            aggregate_argument_types: dict[AggregateFunction, ScalarType] = {
                AggregateFunction.SUM: NumericType(),
                AggregateFunction.AVERAGE: NumericType(),
                AggregateFunction.MINIMUM: OrderableType(),
                AggregateFunction.MAXIMUM: OrderableType(),
            }
            argument_type = aggregate_argument_types.get(function)
            return Aggregate(
                expression_id,
                function,
                self.reference(
                    _provider_object(item.field("argument")),
                    expected_type=argument_type,
                ),
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
                self.reference(
                    _provider_object(item.field("condition")),
                    expected_type=BooleanType(),
                    row_set_ref=over_set_ref,
                ),
                self._default_origin,
            )
        if kind == "related_row":
            related_set_ref = self._terms.set_ref(_required_text_field(item, "set_ref"))
            condition_value = item.field("condition")
            condition_ref = (
                None
                if condition_value is None
                else self.reference(
                    _provider_object(condition_value),
                    expected_type=BooleanType(),
                    row_set_ref=related_set_ref,
                )
            )
            return RelatedRow(
                expression_id,
                related_set_ref,
                self._terms.direct_association_refs(
                    from_set_ref=self._candidate_set_ref,
                    to_set_ref=related_set_ref,
                ),
                condition_ref,
                self._default_origin,
            )
        if kind == "coverage":
            candidate = self._candidate_set_ref
            required = self._terms.set_ref(
                _required_text_field(item, "required_member_set_ref")
            )
            observation_value = _provider_object(item.field("observation"))
            observation = self._terms.set_ref(
                _required_text_field(observation_value, "set_ref")
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
                else self.reference(
                    _provider_object(required_member_condition_value),
                    expected_type=BooleanType(),
                    row_set_ref=required,
                )
            )
            coverage = Coverage(
                expression_id,
                candidate,
                required,
                observation,
                candidate_path,
                dimension_path,
                required_member_condition,
                self.reference(
                    _provider_object(observation_value.field("condition")),
                    expected_type=BooleanType(),
                    row_set_ref=observation,
                ),
                self._default_origin,
            )
            candidate_condition_value = item.field("candidate_condition")
            if candidate_condition_value is None:
                return coverage
            candidate_condition_ref = self.reference(
                _provider_object(candidate_condition_value),
                expected_type=BooleanType(),
                row_set_ref=candidate,
            )
            coverage_ref = self._intern_node(coverage)
            return BooleanComposition(
                expression_id,
                BooleanCompositionOperator.AND,
                (candidate_condition_ref, coverage_ref),
                self._default_origin,
            )
        raise ValueError(f"unknown expression kind: {kind}")

    def _comparison_fact_type(
        self,
        input_ref: str,
        *,
        identified_set_ref: str | None,
    ) -> ScalarType:
        denotation = self._denotation_by_input_ref[input_ref]
        if denotation.kind is not InputDenotationKind.IDENTITY_REFERENCE:
            value_type = self._input_by_id[input_ref].value_type
            return (
                value_type.element_type
                if isinstance(value_type, CollectionType)
                else value_type
            )
        if identified_set_ref is None:
            raise ValueError("identity comparison lacks identified_set_ref")
        return IdentifierType(self._terms.set_ref(identified_set_ref))


def _merge_fact_types(left: ScalarType, right: ScalarType) -> ScalarType:
    if left == right:
        return left
    if isinstance(left, UnspecifiedScalarType):
        return right
    if isinstance(right, UnspecifiedScalarType):
        return left
    if isinstance(left, OrderableType):
        return _merge_orderable_type(right)
    if isinstance(right, OrderableType):
        return _merge_orderable_type(left)
    if isinstance(left, NumericType) and isinstance(
        right,
        (IntegerType, DecimalType),
    ):
        return right
    if isinstance(right, NumericType) and isinstance(
        left,
        (IntegerType, DecimalType),
    ):
        return left
    if isinstance(left, TemporalPointType) and isinstance(
        right,
        (DateType, DateTimeType),
    ):
        return right
    if isinstance(right, TemporalPointType) and isinstance(
        left,
        (DateType, DateTimeType),
    ):
        return left
    raise ValueError("one observed fact has incompatible operator requirements")


def _merge_orderable_type(value_type: ScalarType) -> ScalarType:
    if isinstance(
        value_type,
        (
            NumericType,
            TemporalPointType,
            IntegerType,
            DecimalType,
            DateType,
            DateTimeType,
            TextType,
        ),
    ):
        return value_type
    raise ValueError("one observed fact has incompatible operator requirements")


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
    if parsed.kind in {"take_with_boundary_ties", "position_with_ties"}:
        limit = _required(parsed.limit, field="selection.limit")
        if limit.discriminator("kind") != "input_ref":
            raise ValueError("take limit must reference an input")
        return (PositionWithTies if parsed.kind == "position_with_ties" else TakeWithBoundaryTies)(expression_parser.reference(limit))
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
        | output.ReturnedMeaningOutput
        | output.ReturnedProjectionValueOutput
        | output.UnreturnedOrderingMeaningOutput
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


def _frame_row_source_origin(
    item: output.FrameRowSourceOutput,
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
        meaning=item.instance_kind.strip(),
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
    if kind in {"text", "categorical_value"}:
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


class _InputCollector:
    def __init__(
        self,
        *,
        question_context_texts: tuple[str, ...],
        conversation_text_by_ref: Mapping[str, str],
    ) -> None:
        self._question_context_texts = question_context_texts
        self._conversation_text_by_ref = conversation_text_by_ref
        self._inputs: list[InputTerm] = []

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
        operand = parsed_operands[0] if len(parsed_operands) == 1 else parsed_operands
        input_value_type = (
            value_type if isinstance(operand, str) else CollectionType(value_type)
        )
        source, resolved_input_ref = _frame_origin(origin)
        if source is SourceOriginKind.CONVERSATION_RESOLUTION:
            resolved_ref = resolved_input_ref
            if (
                resolved_ref is None
                or resolved_ref not in self._conversation_text_by_ref
            ):
                raise ValueError("unknown conversation resolved input reference")
            if parsed_operands != (self._conversation_text_by_ref[resolved_ref],):
                raise ValueError(
                    "conversation input operand must copy its resolved value"
                )
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
        origin = SourceOrigin(
            source=source,
            meaning=meaning,
            resolved_input_ref=resolved_input_ref,
        )
        if not input_operand_matches_value_type(operand, value_type):
            raise ValueError("input operand does not match its declared value type")
        input_ref = f"i{len(self._inputs) + 1}"
        self._inputs.append(
            InputTerm(
                id=input_ref,
                origin=origin,
                operand=operand,
                value_type=value_type,
            )
        )
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


def _raise_missing_row_scope() -> str:
    raise ValueError("row-scoped fact requires an enclosing row")


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
