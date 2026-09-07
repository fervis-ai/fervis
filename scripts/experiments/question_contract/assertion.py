"""Outcome assertions for isolated production Question Contract invocations."""

from __future__ import annotations

from typing import Any

from fervis.lookup.question_contract.parser import (
    ParsedSemanticQuestionContract,
    ParsedSemanticQuestionMeaning,
    parse_semantic_question_contract,
    parse_semantic_question_frame,
)
from fervis.lookup.question_contract.analysis import (
    Groups,
    Singleton,
    SubjectRows,
)
from fervis.lookup.question_contract.model import (
    Aggregate,
    AllResults,
    Arithmetic,
    BooleanComposition,
    Comparison,
    Coverage,
    FirstRankWithTies,
    NullCheck,
    Quantify,
    RelatedRow,
    TakeWithBoundaryTies,
    PositionWithTies,
    TemporalBucket,
)
from fervis.lookup.semantic_types import BooleanType, CollectionType, IdentifierType


def validate(arguments: dict[str, Any], context: dict[str, Any]) -> list[str]:
    try:
        meaning = parse_semantic_question_frame(
            context["frame_payload"],
            question_context_texts=tuple(context["question_context_texts"]),
            conversation_text_by_resolved_input_ref=context.get(
                "conversation_text_by_resolved_input_ref"
            ),
        )
        if not isinstance(meaning, ParsedSemanticQuestionMeaning):
            return ["question frame did not produce complete question meaning"]
        parsed = parse_semantic_question_contract(
            arguments,
            meaning=meaning,
            question_context_texts=tuple(context["question_context_texts"]),
            conversation_text_by_resolved_input_ref=context.get(
                "conversation_text_by_resolved_input_ref"
            ),
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
    expected_aggregate_functions = tuple(
        context.get("expected_aggregate_functions") or ()
    )
    if expected_aggregate_functions:
        actual_aggregate_functions = tuple(
            node.function.value
            for fact in parsed.contract.requested_facts
            for node in fact.expressions
            if isinstance(node, Aggregate)
        )
        if actual_aggregate_functions != expected_aggregate_functions:
            errors.append(
                "aggregate functions are "
                f"{actual_aggregate_functions}, expected "
                f"{expected_aggregate_functions}"
            )
    actual_boolean_use_sites = {
        requirement.use_site.value
        for index in parsed.semantic_indexes
        for requirement in index.boolean_requirements
    }
    for use_site in context.get("required_boolean_use_sites") or []:
        if use_site not in actual_boolean_use_sites:
            errors.append(f"Boolean use site {use_site!r} is missing")
    errors.extend(_coverage_observation_errors(arguments, context=context))
    expected_selection = context.get("expected_selection")
    if expected_selection is not None:
        selections = [_selection(item.selection) for item in parsed.semantic_indexes]
        if selections != [expected_selection] * len(selections):
            errors.append(
                f"result selections are {selections}, expected {expected_selection}"
            )
    expected_operations = context.get("expected_grouping_operations")
    if expected_operations is not None:
        actual_operations = []
        for index in parsed.semantic_indexes:
            for ref in index.grouping_refs:
                node = index.expression_by_ref.get(ref)
                if not isinstance(node, (Arithmetic, Comparison)):
                    actual_operations.append(None)
                    continue
                operands = node.argument_refs if isinstance(node, Arithmetic) else (node.left_ref, node.right_ref)
                meanings = [
                    str(index.input_by_ref[operand].operand).lower()
                    if operand in index.input_by_ref
                    else (index.term_by_ref[index.fact_local_ref_by_local_id[operand]].origin.meaning.lower()
                          if index.fact_local_ref_by_local_id[operand] in index.term_by_ref
                          else "<computed expression>")
                    for operand in operands
                ]
                actual_operations.append((node.operator.value, meanings))
        if len(actual_operations) != len(expected_operations):
            errors.append("grouping operation count differs")
        else:
            for actual, expected in zip(actual_operations, expected_operations, strict=True):
                if actual is None or actual[0] != expected["operator"] or len(actual[1]) != len(expected["operand_meanings"]) or any(
                    hint.lower() not in meaning
                    for hint, meaning in zip(expected["operand_meanings"], actual[1], strict=True)
                ):
                    errors.append(f"grouping operation {actual!r} does not preserve {expected!r}")
    expected_grouping_expressions = context.get("expected_grouping_expression_kinds")
    if expected_grouping_expressions is not None:
        from collections import Counter
        actual = [
            _expression_kind(index.expression_by_ref[ref])
            if ref in index.expression_by_ref else "fact"
            for index in parsed.semantic_indexes for ref in index.grouping_refs
        ]
        if Counter(actual) != Counter(expected_grouping_expressions):
            errors.append(f"grouping expressions are {actual}; expected {expected_grouping_expressions}")
    expected_grouping_kinds = tuple(context.get("expected_grouping_value_kinds") or ())
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
                if (
                    grouping_identifier_set == "distinct_from_candidate"
                    and is_candidate
                ):
                    errors.append(
                        f"grouping identifier {ref.token} incorrectly identifies "
                        f"candidate set {candidate_set_ref}"
                    )
    reference_identifier_set = context.get("reference_identifier_set")
    if reference_identifier_set is not None:
        for index in parsed.semantic_indexes:
            candidate_set_ref = index.subject_obligation.subject_set_ref.local_id
            grouping_set_refs = {
                value_type.set_ref
                for ref in index.grouping_refs
                if isinstance(
                    value_type := index.inferred_type_by_ref[ref],
                    IdentifierType,
                )
            }
            for use in index.input_use_sites:
                identity_set_ref = use.identity_set_ref
                if identity_set_ref is None:
                    continue
                identity_set_id = identity_set_ref.local_id
                is_candidate = identity_set_id == candidate_set_ref
                if reference_identifier_set == "candidate" and not is_candidate:
                    errors.append(
                        f"reference input {use.input_ref} identifies "
                        f"{identity_set_ref.token}, expected candidate set "
                        f"{candidate_set_ref}"
                    )
                if (
                    reference_identifier_set == "distinct_from_candidate"
                    and is_candidate
                ):
                    errors.append(
                        f"reference input {use.input_ref} incorrectly identifies "
                        f"candidate set {candidate_set_ref}"
                    )
                if (
                    reference_identifier_set == "grouping"
                    and identity_set_id not in grouping_set_refs
                ):
                    errors.append(
                        f"reference input {use.input_ref} identifies "
                        f"{identity_set_ref.token}, expected one of the grouping "
                        f"sets {sorted(grouping_set_refs)}"
                    )
    output_identifier_set = context.get("output_identifier_set")
    if output_identifier_set is not None:
        for index in parsed.semantic_indexes:
            candidate_set_ref = index.subject_obligation.subject_set_ref.local_id
            reference_set_refs = {
                use.identity_set_ref.local_id
                for use in index.input_use_sites
                if use.identity_set_ref is not None
            }
            for requirement in index.output_requirements:
                value_type = index.value_type(requirement.value_ref)
                if not isinstance(value_type, IdentifierType):
                    errors.append(
                        f"output {requirement.output_ref.token} is not an identity"
                    )
                    continue
                if (
                    output_identifier_set
                    in {
                        "distinct_from_candidate",
                        "distinct_from_candidate_and_reference",
                    }
                    and value_type.set_ref == candidate_set_ref
                ):
                    errors.append(
                        f"output {requirement.output_ref.token} identifies candidate "
                        f"set {candidate_set_ref}, expected a related entity"
                    )
                if (
                    output_identifier_set
                    == "distinct_from_candidate_and_reference"
                    and value_type.set_ref in reference_set_refs
                ):
                    errors.append(
                        f"output {requirement.output_ref.token} identifies input "
                        f"entity set {value_type.set_ref}, expected a distinct "
                        "related entity"
                    )
    if context.get("forbid_identifier_ordering"):
        for index in parsed.semantic_indexes:
            for ordering_ref in index.ordering_refs:
                if isinstance(index.inferred_type_by_ref[ordering_ref], IdentifierType):
                    errors.append(
                        f"ordering {ordering_ref.token} incorrectly uses entity identity"
                    )
    if context.get("require_boolean_outputs"):
        for index in parsed.semantic_indexes:
            for output in index.requested_fact.outputs:
                ref = index.fact_local_ref_by_local_id[output.expression_ref]
                if not isinstance(index.inferred_type_by_ref[ref], BooleanType):
                    errors.append("The requested existence result must be Boolean.")
    for scope, correlated in (("global", False), ("related", True)):
        actual = {
            kind
            for index in parsed.semantic_indexes
            for is_correlated, kind in _quantifier_meanings(index)
            if is_correlated == correlated
        }
        for required in context.get(f"required_{scope}_quantifiers", []):
            if required not in actual:
                errors.append(f"Required {scope} quantifier {required!r} is missing.")
    return errors


def _quantifier_meanings(index):
    """Inspect used quantifiers modulo Boolean negation, without shape aliases."""
    meanings = set()

    def visit(ref, negated=False):
        node = index.expression_by_ref.get(ref)
        if isinstance(node, BooleanComposition):
            for child in node.argument_refs:
                visit(index.fact_local_ref_by_local_id[child], negated ^ (node.operator.value == "not"))
        elif isinstance(node, Quantify):
            kind = node.quantifier.value
            condition_negated = False
            if negated:
                condition_negated = kind == "forall"
                kind = {"exists": "not_exists", "not_exists": "exists", "forall": "exists"}[kind]
            meanings.add((bool(node.association_refs), kind))
            visit(index.fact_local_ref_by_local_id[node.condition_ref], condition_negated)
        else:
            for child in index.direct_dependencies_by_ref.get(ref, ()):
                if child in index.expression_by_ref:
                    visit(child)

    for output in index.requested_fact.outputs:
        visit(index.fact_local_ref_by_local_id[output.expression_ref])
    if index.requested_fact.qualification_ref is not None:
        visit(index.fact_local_ref_by_local_id[index.requested_fact.qualification_ref])
    return meanings


def _coverage_observation_errors(
    arguments: dict[str, Any],
    *,
    context: dict[str, Any],
) -> list[str]:
    required_input_ref = context.get("required_coverage_observation_input_ref")
    if not isinstance(required_input_ref, str):
        return []
    outcome = arguments.get("outcome")
    requests = outcome.get("answer_requests") if isinstance(outcome, dict) else None
    if not isinstance(requests, list):
        return ["coverage output lacks answer_requests"]
    coverage_nodes = [
        qualification
        for request in requests
        if isinstance(request, dict)
        for qualification in [request.get("qualification")]
        if isinstance(qualification, dict) and qualification.get("kind") == "coverage"
    ]
    if len(coverage_nodes) != 1:
        return [f"coverage node count is {len(coverage_nodes)}, expected 1"]
    observation = coverage_nodes[0].get("observation")
    if not isinstance(observation, dict):
        return ["coverage observation is not locally owned"]
    observation_refs = _input_refs(observation.get("condition"))
    if required_input_ref not in observation_refs:
        return [f"coverage observation does not consume {required_input_ref}"]
    outside_refs = _input_refs(
        {key: value for key, value in coverage_nodes[0].items() if key != "observation"}
    )
    if required_input_ref in outside_refs:
        return [
            f"coverage observation input {required_input_ref} is also used "
            "outside its row scope"
        ]
    return []


def _input_refs(value: object) -> set[str]:
    if isinstance(value, list):
        return set().union(*(_input_refs(item) for item in value), set())
    if not isinstance(value, dict):
        return set()
    refs = (
        {str(value["input_ref"])}
        if value.get("kind") == "input_ref" and isinstance(value.get("input_ref"), str)
        else set()
    )
    return refs | set().union(
        *(_input_refs(item) for item in value.values()),
        set(),
    )


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
    if isinstance(value, PositionWithTies):
        return "position_with_ties"
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
    if isinstance(value, RelatedRow):
        return "related_row"
    raise TypeError("unknown expression kind")


__all__ = ["validate"]
