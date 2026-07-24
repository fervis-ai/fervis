"""Semantic validity assertion for standalone Question Contract stability runs."""

from __future__ import annotations

from typing import Any

from fervis.lookup.question_contract.semantic_parser import (
    ParsedSemanticQuestionContract,
    ParsedSemanticQuestionMeaning,
    parse_semantic_question_contract,
    parse_semantic_question_frame,
)
from fervis.lookup.question_contract.semantic_analysis import (
    Groups,
    Singleton,
    SubjectRows,
)
from fervis.lookup.question_contract.semantic_model import (
    Aggregate,
    AllResults,
    Arithmetic,
    BooleanComposition,
    Comparison,
    Coverage,
    FirstRankWithTies,
    NullCheck,
    Quantify,
    TakeWithBoundaryTies,
    TemporalBucket,
)
from fervis.lookup.semantic_types import CollectionType, IdentifierType


def validate(arguments: dict[str, Any], context: dict[str, Any]) -> list[str]:
    try:
        meaning = parse_semantic_question_frame(
            context["frame_payload"],
            question_context_texts=tuple(context["question_context_texts"]),
        )
        if not isinstance(meaning, ParsedSemanticQuestionMeaning):
            return ["question frame did not produce complete question meaning"]
        parsed = parse_semantic_question_contract(
            arguments,
            meaning=meaning,
            question_context_texts=tuple(context["question_context_texts"]),
        )
    except ValueError as exc:
        return [f"semantic parser rejected output: {exc}"]
    if not isinstance(parsed, ParsedSemanticQuestionContract):
        return ["model returned an incomplete outcome"]
    expected_count = context.get("expected_request_count")
    if expected_count is not None and len(parsed.contract.requested_facts) != int(
        expected_count
    ):
        return [
            "requested fact count is "
            f"{len(parsed.contract.requested_facts)}, expected {expected_count}"
        ]
    errors: list[str] = []
    expected_interpretation = context.get("expected_instance_interpretation")
    if expected_interpretation is not None:
        interpretations = [
            fact.subject.instance_interpretation.value
            for fact in parsed.contract.requested_facts
        ]
        if interpretations != [expected_interpretation] * len(interpretations):
            errors.append(
                "instance interpretations are "
                f"{interpretations}, expected {expected_interpretation!r}"
            )
    for operand, expected_kind in (context.get("expected_input_kinds") or {}).items():
        actual_kind = _meaning_input_kinds(meaning).get(operand)
        if actual_kind != expected_kind:
            errors.append(
                f"input {operand!r} is {actual_kind!r}; expected {expected_kind!r}"
            )
    expected_grain = context.get("expected_result_grain")
    if expected_grain is not None:
        grains = [_grain(item.result_grain) for item in parsed.semantic_indexes]
        if grains != [expected_grain] * len(grains):
            errors.append(f"result grains are {grains}, expected {expected_grain}")
    expected_inputs = sorted(context.get("expected_input_texts") or [])
    actual_inputs = sorted(
        operand
        for item in parsed.contract.inputs
        for operand in (
            (item.operand,) if isinstance(item.operand, str) else item.operand
        )
    )
    accepted_inventories = [
        sorted(inventory)
        for inventory in context.get("accepted_input_inventories") or []
    ]
    if accepted_inventories and actual_inputs not in accepted_inventories:
        errors.append(
            f"input inventory is {actual_inputs}, expected one of "
            f"{accepted_inventories}"
        )
    elif expected_inputs and actual_inputs != expected_inputs:
        errors.append(f"input inventory is {actual_inputs}, expected {expected_inputs}")
    for term in context.get("required_input_terms") or []:
        if not any(str(term).casefold() in item.casefold() for item in actual_inputs):
            errors.append(f"input inventory does not represent {term!r}")
    reference_input_refs = {
        use.input_ref
        for index in parsed.semantic_indexes
        for use in index.input_use_sites
        if use.reference_fact_ref is not None
    }
    reference_operands = {
        operand
        for item in parsed.contract.inputs
        if item.id in reference_input_refs
        for operand in (
            (item.operand,) if isinstance(item.operand, str) else item.operand
        )
    }
    for term in context.get("required_reference_input_terms") or []:
        if term not in reference_operands:
            errors.append(f"input {term!r} has no reference-grounding use site")
    expected_collection = tuple(context.get("expected_collection_operands") or ())
    if expected_collection:
        collection_inputs = [
            item
            for item in parsed.contract.inputs
            if isinstance(item.value_type, CollectionType)
        ]
        if len(collection_inputs) != 1:
            errors.append(
                "collection input count is "
                f"{len(collection_inputs)}, expected exactly one"
            )
        elif collection_inputs[0].operand != expected_collection:
            errors.append(
                "collection operands are "
                f"{collection_inputs[0].operand}, expected {expected_collection}"
            )
    actual_expression_kinds = {
        _expression_kind(node)
        for fact in parsed.contract.requested_facts
        for node in fact.expressions
    }
    for kind in context.get("required_expression_kinds") or []:
        if kind not in actual_expression_kinds:
            errors.append(f"expression kind {kind!r} is missing")
    actual_boolean_use_sites = {
        requirement.use_site.value
        for index in parsed.semantic_indexes
        for requirement in index.boolean_requirements
    }
    for use_site in context.get("required_boolean_use_sites") or []:
        if use_site not in actual_boolean_use_sites:
            errors.append(f"Boolean use site {use_site!r} is missing")
    expected_selection = context.get("expected_selection")
    if expected_selection is not None:
        selections = [_selection(item.selection) for item in parsed.semantic_indexes]
        if selections != [expected_selection] * len(selections):
            errors.append(
                f"result selections are {selections}, expected {expected_selection}"
            )
    expected_grouping_kinds = tuple(
        context.get("expected_grouping_value_kinds") or ()
    )
    if expected_grouping_kinds:
        actual_grouping_kinds = tuple(
            _value_type_kind(index.inferred_type_by_ref[ref])
            for index in parsed.semantic_indexes
            for ref in index.grouping_refs
        )
        if actual_grouping_kinds != expected_grouping_kinds:
            errors.append(
                "grouping value kinds are "
                f"{actual_grouping_kinds}, expected {expected_grouping_kinds}"
            )
    grouping_identifier_set = context.get("grouping_identifier_set")
    if grouping_identifier_set is not None:
        for index in parsed.semantic_indexes:
            candidate_set_ref = index.subject_obligation.subject_set_ref.local_id
            for ref in index.grouping_refs:
                value_type = index.inferred_type_by_ref[ref]
                if not isinstance(value_type, IdentifierType):
                    continue
                is_candidate = value_type.set_ref == candidate_set_ref
                if grouping_identifier_set == "candidate" and not is_candidate:
                    errors.append(
                        f"grouping identifier {ref.token} identifies "
                        f"{value_type.set_ref}, expected candidate set {candidate_set_ref}"
                    )
                if grouping_identifier_set == "distinct_from_candidate" and is_candidate:
                    errors.append(
                        f"grouping identifier {ref.token} incorrectly identifies "
                        f"candidate set {candidate_set_ref}"
                    )
    return errors


def _meaning_input_kinds(
    meaning: ParsedSemanticQuestionMeaning,
) -> dict[str, str]:
    inputs = {item.id: item for item in meaning.inputs}
    return {
        str(inputs[item.input_ref].operand): item.kind.value
        for item in meaning.input_denotations
    }


def _grain(value: object) -> str:
    if isinstance(value, SubjectRows):
        return "subject_rows"
    if isinstance(value, Groups):
        return "groups"
    if isinstance(value, Singleton):
        return "singleton"
    raise TypeError("unknown result grain")


def _selection(value: object) -> str:
    if isinstance(value, AllResults):
        return "all_results"
    if isinstance(value, FirstRankWithTies):
        return "first_rank_with_ties"
    if isinstance(value, TakeWithBoundaryTies):
        return "take_with_boundary_ties"
    raise TypeError("unknown result selection")


def _value_type_kind(value: object) -> str:
    if isinstance(value, IdentifierType):
        return "identifier"
    return type(value).__name__.removesuffix("Type").casefold()


def _expression_kind(value: object) -> str:
    if isinstance(value, BooleanComposition):
        return "boolean"
    if isinstance(value, Comparison):
        return "comparison"
    if isinstance(value, NullCheck):
        return "null_check"
    if isinstance(value, Arithmetic):
        return "arithmetic"
    if isinstance(value, TemporalBucket):
        return "temporal_bucket"
    if isinstance(value, Aggregate):
        return "aggregate"
    if isinstance(value, Quantify):
        return "quantify"
    if isinstance(value, Coverage):
        return "coverage"
    raise TypeError("unknown expression kind")


__all__ = ["validate"]
