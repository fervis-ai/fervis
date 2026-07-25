from __future__ import annotations

from typing import Any

from jsonschema import Draft7Validator

from fervis.lookup.grounding.semantic import (
    deterministic_scalar_values,
    grounding_partitions,
)
from fervis.lookup.question_contract.analysis import (
    Groups,
    Singleton,
    SubjectRows,
)
from fervis.lookup.question_contract.parser import (
    ParsedSemanticQuestionContract,
    ParsedSemanticQuestionMeaning,
    parse_semantic_question_contract,
    parse_semantic_question_frame,
)
from fervis.lookup.question_contract.schema import (
    build_semantic_question_contract_schema_for_meaning,
)
from fervis.lookup.query_enrichment.semantic import (
    reference_input_recall_tasks,
    semantic_recall_requirements,
)
from fervis.lookup.semantic_types import value_type_kind
from tests.testkit.assertions import (
    expects_rejection,
    status_mismatches,
    subset_mismatches,
)


def run_semantic_question_contract_case(payload: dict[str, Any]) -> list[str]:
    request = payload["input"]
    frame_payload = request["frame_payload"]
    model_payload = request["payload"]
    try:
        meaning = parse_semantic_question_frame(
            frame_payload,
            question_context_texts=tuple(request["question_context_texts"]),
            conversation_text_by_resolved_input_ref=(
                request.get("conversation_text_by_resolved_input_ref") or {}
            ),
        )
    except ValueError as exc:
        if expects_rejection(payload["expect"]):
            return status_mismatches(
                actual_status="rejected", expected=payload["expect"]
            )
        return [f"meaning parser rejected payload: {exc}"]
    if not isinstance(meaning, ParsedSemanticQuestionMeaning):
        return ["expected a complete semantic question meaning"]
    schema_errors = sorted(
        Draft7Validator(
            build_semantic_question_contract_schema_for_meaning(meaning)
        ).iter_errors(model_payload),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    )
    if schema_errors:
        if expects_rejection(payload["expect"]):
            return status_mismatches(
                actual_status="rejected", expected=payload["expect"]
            )
        return [f"schema rejected payload: {schema_errors[0].message}"]
    try:
        parsed = parse_semantic_question_contract(
            model_payload,
            meaning=meaning,
            question_context_texts=tuple(request["question_context_texts"]),
            conversation_text_by_resolved_input_ref=(
                request.get("conversation_text_by_resolved_input_ref") or {}
            ),
        )
    except ValueError as exc:
        if expects_rejection(payload["expect"]):
            return status_mismatches(
                actual_status="rejected", expected=payload["expect"]
            )
        return [f"parser rejected payload: {exc}"]
    if expects_rejection(payload["expect"]):
        return status_mismatches(actual_status="accepted", expected=payload["expect"])
    if not isinstance(parsed, ParsedSemanticQuestionContract):
        return ["expected a complete semantic question contract"]
    input_by_id = {item.id: item for item in parsed.contract.inputs}
    use_sites = tuple(
        use for index in parsed.semantic_indexes for use in index.input_use_sites
    )
    partitions = grounding_partitions(use_sites)
    deterministic_values = deterministic_scalar_values(partitions, inputs=input_by_id)
    recall_requirements = tuple(
        requirement
        for index in parsed.semantic_indexes
        for requirement in semantic_recall_requirements(index)
    )
    reference_recall = reference_input_recall_tasks(
        parsed.semantic_indexes, inputs=input_by_id
    )
    actual = {
        "input_refs": [item.id for item in parsed.contract.inputs],
        "inputs": [
            {
                "input_ref": item.id,
                "operand": (
                    list(item.operand)
                    if isinstance(item.operand, tuple)
                    else item.operand
                ),
                "value_kind": value_type_kind(item.value_type),
            }
            for item in parsed.contract.inputs
        ],
        "requested_facts": [
            {
                "id": index.requested_fact_id,
                "result_grain": _grain(index.result_grain),
                "grouping_refs": [item.token for item in index.grouping_refs],
                "ordering_refs": [item.token for item in index.ordering_refs],
                "input_use_refs": [item.input_ref for item in index.input_use_sites],
                "input_uses": [
                    {
                        "input_ref": item.input_ref,
                        "operand_meaning": item.operand_meaning,
                    }
                    for item in index.input_use_sites
                ],
                "identity_input_refs": [
                    item.input_ref
                    for item in index.input_use_sites
                    if item.identity_set_ref is not None
                ],
                "population_atoms": [
                    requirement.atom_ref.value_ref
                    for requirement in index.boolean_requirements
                    if requirement.use_site.value == "population"
                ],
                "scoped_boolean_requirements": [
                    {
                        "atom": requirement.atom_ref.value_ref,
                        "use_site": requirement.use_site.value,
                        "owner_expression_ref": requirement.owner_expression_ref,
                    }
                    for requirement in index.boolean_requirements
                    if requirement.use_site.value != "population"
                ],
                "qualification_clauses": [
                    sorted(
                        ("!" if atom.polarity.value == "negative" else "")
                        + atom.value_ref
                        for atom in clause.atom_refs
                    )
                    for clause in index.qualification.clauses
                ],
                "output_refs": [
                    item.output_ref.token for item in index.output_requirements
                ],
                "output_source_requirement_refs": [
                    {
                        "output_ref": item.output_ref.token,
                        "source_requirement_refs": sorted(
                            ref.token
                            for dependency in item.dependencies
                            for ref in index.transitive_dependencies_by_ref.get(
                                dependency, frozenset((dependency,))
                            )
                            if ref in index.source_requirement_refs
                        ),
                    }
                    for item in index.output_requirements
                ],
                "source_requirement_refs": sorted(
                    item.token for item in index.source_requirement_refs
                ),
                "source_support_closures": {
                    ref.token: sorted(
                        item.token
                        for item in index.source_support_closure(frozenset((ref,)))
                    )
                    for ref in sorted(index.term_requirement_refs)
                },
            }
            for index in parsed.semantic_indexes
        ],
        "grounding_partitions": [
            {
                "input_ref": item.input_ref,
                "use_refs": list(item.use_refs),
                "expected_set_ref": (
                    item.expected_set_ref.token
                    if item.expected_set_ref is not None
                    else None
                ),
                "operand_meaning": item.operand_meaning,
            }
            for item in partitions
        ],
        "deterministic_input_values": [
            {
                "input_ref": item.input_ref,
                "use_refs": list(item.use_refs),
                "value_kind": item.typed_value.kind.value,
            }
            for item in deterministic_values
        ],
        "semantic_recall_requirements": [
            {
                "requirement_ref": item.requirement_ref.token,
                "kind": item.kind.value,
                "owner_refs": [ref.token for ref in item.owner_refs],
                "identity_set_ref": (
                    item.identity_set_ref.token
                    if item.identity_set_ref is not None
                    else None
                ),
            }
            for item in recall_requirements
        ],
        "reference_input_recall_tasks": [
            {
                "input_use_ref": item.input_use_ref,
                "input_ref": item.input_ref,
                "operand_meaning": item.operand_meaning,
                "expected_set_ref": (
                    item.expected_set_ref.token
                    if item.expected_set_ref is not None
                    else None
                ),
            }
            for item in reference_recall
        ],
    }
    return subset_mismatches(
        actual=actual,
        expected_subset=payload["expect"].get("result_contains") or {},
    )


def _grain(value: object) -> str:
    if isinstance(value, SubjectRows):
        return "subject_rows"
    if isinstance(value, Groups):
        return "groups"
    if isinstance(value, Singleton):
        return "singleton"
    raise TypeError("unknown result grain")


__all__ = ["run_semantic_question_contract_case"]
