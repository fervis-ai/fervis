"""Shared model-facing projections of semantic requirements."""

from __future__ import annotations

from fervis.lookup.question_contract import (
    AssociationTerm,
    BooleanComposition,
    Comparison,
    Coverage,
    FactLocalRef,
    FactLocalKind,
    FactTerm,
    NullCheck,
    Quantify,
    RequestedFactSemanticIndex,
)
from fervis.lookup.semantic_types import IdentifierType


def semantic_requirements_prompt_payload(
    index: RequestedFactSemanticIndex,
) -> dict[str, object]:
    """Project one requested fact's logical requirements for downstream turns."""

    return {
        "requested_fact_id": index.requested_fact_id,
        "term_requirements": [
            _term_requirement_prompt_item(index, ref)
            for ref in sorted(index.term_requirement_refs, key=lambda item: item.token)
        ],
        "boolean_requirements": list(boolean_requirement_prompt_items(index)),
        "association_requirements": [
            _association_requirement_prompt_item(index, ref)
            for ref in sorted(
                index.association_requirement_refs,
                key=lambda item: item.token,
            )
        ],
        "grouping_refs": [ref.token for ref in index.grouping_refs],
        "ordering_refs": [ref.token for ref in index.ordering_refs],
        "output_refs": [
            item.output_ref.token for item in index.output_requirements
        ],
    }


def plan_selection_fact_prompt_payload(
    index: RequestedFactSemanticIndex,
) -> dict[str, object]:
    """Project the requested result meaning needed to assess source alignment."""

    return {
        "requested_fact_id": index.requested_fact_id,
        "fact_text": index.requested_fact.origin.meaning,
        "candidate_instance_kind": index.term_by_ref[
            index.subject_obligation.subject_set_ref
        ].origin.meaning,
        "required_facts": [
            _term_requirement_prompt_item(index, ref)
            for ref in sorted(
                index.term_requirement_refs,
                key=lambda item: item.token,
            )
            if isinstance(index.term_by_ref[ref], FactTerm)
        ],
        "answer_outputs": [
            _plan_selection_output_prompt_item(index, position)
            for position, _ in enumerate(index.requested_fact.outputs)
        ],
        "qualification_clauses": [
            {
                "clause_ref": clause.clause_ref,
                "conditions": [
                    _local_value_meaning(
                        index,
                        FactLocalRef.from_token(atom.value_ref),
                    )
                    for atom in clause.atom_refs
                ],
            }
            for clause in index.qualification.clauses
        ],
        "required_associations": [
            {
                "association_ref": ref.token,
                "meaning": index.term_by_ref[ref].origin.meaning,
            }
            for ref in sorted(
                index.association_requirement_refs,
                key=lambda item: item.token,
            )
        ],
        "grouping": [
            {
                "group_ref": ref.token,
                "meaning": _local_value_meaning(index, ref),
            }
            for ref in index.grouping_refs
        ],
        "ordering": [
            {
                "ordering_ref": ref.token,
                "meaning": _local_value_meaning(index, ref),
            }
            for ref in index.ordering_refs
        ],
    }


def _plan_selection_output_prompt_item(
    index: RequestedFactSemanticIndex,
    position: int,
) -> dict[str, str]:
    output = index.requested_fact.outputs[position]
    requirement = index.output_requirements[position]
    payload = {
        "answer_output_id": output.id,
        "meaning": output.origin.meaning,
    }
    identified_set_ref: FactLocalRef | None = None
    if (
        isinstance(requirement.value_ref, FactLocalRef)
        and requirement.value_ref.kind is FactLocalKind.SET
    ):
        identified_set_ref = requirement.value_ref
    else:
        value_type = index.inferred_type_by_ref[requirement.value_ref]
        if isinstance(value_type, IdentifierType):
            identified_set_ref = index.fact_local_ref_by_local_id[
                value_type.set_ref
            ]
    if identified_set_ref is not None:
        payload.update(
            value_kind="canonical_identity",
            identified_set_meaning=index.term_by_ref[
                identified_set_ref
            ].origin.meaning,
        )
    return payload


def _local_value_meaning(index: RequestedFactSemanticIndex, ref) -> str:
    value = index.expression_by_ref.get(ref) or index.term_by_ref.get(ref)
    if value is None:
        raise KeyError(ref)
    return value.origin.meaning


def _term_requirement_prompt_item(index, ref) -> dict[str, object]:
    term = index.term_by_ref[ref]
    payload: dict[str, object] = {
        "requirement_ref": ref.token,
        "kind": ref.kind.value,
        "meaning": term.origin.meaning,
    }
    if isinstance(term, FactTerm):
        payload["observed_for_ref"] = index.fact_local_ref_by_local_id[
            term.owner_ref
        ].token
        if isinstance(term.value_type, IdentifierType):
            payload["identified_set_ref"] = index.fact_local_ref_by_local_id[
                term.value_type.set_ref
            ].token
    return payload


def _association_requirement_prompt_item(index, ref) -> dict[str, str]:
    term = index.term_by_ref[ref]
    if not isinstance(term, AssociationTerm):
        raise TypeError("association requirement requires an association term")
    return {
        "requirement_ref": ref.token,
        "from_set_ref": index.fact_local_ref_by_local_id[term.from_set_ref].token,
        "to_set_ref": index.fact_local_ref_by_local_id[term.to_set_ref].token,
        "meaning": term.origin.meaning,
    }


def boolean_requirement_prompt_items(
    index: RequestedFactSemanticIndex,
) -> tuple[dict[str, object], ...]:
    """Describe each Boolean obligation with its meaning and direct operands."""

    return tuple(
        {
            "requirement_ref": item.requirement_ref,
            "expression_ref": item.atom_ref.value_ref,
            "polarity": item.atom_ref.polarity.value,
            "use_site": item.use_site.value,
            "owner_expression_ref": item.owner_expression_ref,
            "condition": _boolean_condition_prompt_item(
                index,
                FactLocalRef.from_token(item.atom_ref.value_ref),
            ),
        }
        for item in index.boolean_requirements
    )


def _boolean_condition_prompt_item(
    index: RequestedFactSemanticIndex,
    ref: FactLocalRef,
) -> dict[str, object]:
    if ref in index.term_by_ref:
        return {
            "operator": "is_true",
            "operands": [_semantic_operand_prompt_item(index, ref)],
        }
    node = index.expression_by_ref.get(ref)
    operand_refs: tuple[str | FactLocalRef, ...]
    if isinstance(node, Comparison):
        operator = node.operator.value
        operand_refs = (node.left_ref, node.right_ref)
    elif isinstance(node, NullCheck):
        operator = node.operator.value
        operand_refs = (node.argument_ref,)
    elif isinstance(node, BooleanComposition):
        operator = node.operator.value
        operand_refs = node.argument_refs
    elif isinstance(node, Quantify):
        operator = node.quantifier.value
        operand_refs = (node.condition_ref,)
    elif isinstance(node, Coverage):
        operator = "coverage"
        operand_refs = (
            (() if node.required_member_condition_ref is None else (
                node.required_member_condition_ref,
            ))
            + (node.condition_ref,)
        )
    else:
        raise TypeError("Boolean requirement references a non-Boolean expression")
    return {
        "operator": operator,
        "operands": [
            _semantic_operand_prompt_item(index, operand_ref)
            for operand_ref in operand_refs
        ],
    }


def _semantic_operand_prompt_item(
    index: RequestedFactSemanticIndex,
    ref: str | FactLocalRef,
) -> dict[str, str]:
    local_ref = (
        ref
        if isinstance(ref, FactLocalRef)
        else index.fact_local_ref_by_local_id.get(ref)
    )
    if local_ref is not None:
        return {
            "ref": local_ref.token,
            "meaning": _local_value_meaning(index, local_ref),
        }
    if isinstance(ref, FactLocalRef):
        raise KeyError(ref.token)
    denotation = index.input_denotation_by_ref.get(ref)
    if denotation is None:
        raise KeyError(ref)
    return {
        "ref": ref,
        "meaning": denotation.operand_meaning,
    }


__all__ = [
    "boolean_requirement_prompt_items",
    "plan_selection_fact_prompt_payload",
    "semantic_requirements_prompt_payload",
]
