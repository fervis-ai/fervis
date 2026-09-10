"""Small semantic contracts for behavior-focused runtime tests."""

from __future__ import annotations

from collections.abc import Mapping

from fervis.lookup.answer_program.model import RelationGuaranteeDeclaration
from fervis.lookup.qualification import QualificationGuarantee, SubjectGuarantee
from fervis.lookup.question_contract.analysis import analyze_requested_fact
from fervis.lookup.question_contract.model import (
    Aggregate,
    AggregateFunction,
    AllResults,
    FactTerm,
    InstanceInterpretation,
    QuestionContract,
    RequestedFact,
    RequestedOutput,
    SetTerm,
    Subject,
)
from fervis.lookup.semantic_types import SourceOrigin, SourceOriginKind, TextType


def semantic_question_contract(
    *,
    requested_fact_id: str,
    output_ids: tuple[str, ...],
    description: str,
    scalar: bool = False,
) -> QuestionContract:
    fact = semantic_requested_fact(
        requested_fact_id=requested_fact_id,
        output_ids=output_ids,
        description=description,
        scalar=scalar,
    )
    return QuestionContract(inputs=(), requested_facts=(fact,))


def semantic_requested_fact(
    *,
    requested_fact_id: str,
    output_ids: tuple[str, ...],
    description: str,
    scalar: bool = False,
) -> RequestedFact:
    origin = SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, description)
    expressions = (
        (
            Aggregate(
                id="e1",
                function=AggregateFunction.COUNT,
                argument_ref="s1",
                filter_ref=None,
                distinct_argument=False,
                origin=origin,
            ),
        )
        if scalar
        else ()
    )
    return RequestedFact(
        id=requested_fact_id,
        origin=origin,
        sets=(SetTerm("s1", origin),),
        associations=(),
        facts=(() if scalar else (FactTerm("f1", "s1", TextType(), origin),)),
        expressions=expressions,
        subject=Subject("s1", InstanceInterpretation.RESOURCE_POPULATION),
        qualification_ref=None,
        grouping_refs=(),
        outputs=tuple(
            RequestedOutput(output_id, "e1" if scalar else "f1", origin)
            for output_id in output_ids
        ),
        ordering=(),
        selection=AllResults(),
        distinct_by=(),
    )


def semantic_relation_guarantees(
    fact: RequestedFact,
    *,
    relation_ids: tuple[str, ...],
    proof_refs_by_relation: Mapping[str, tuple[str, ...]],
) -> tuple[RelationGuaranteeDeclaration, ...]:
    index = analyze_requested_fact(fact, inputs={}, input_denotations={})
    return tuple(
        RelationGuaranteeDeclaration(
            relation_id=relation_id,
            qualification=QualificationGuarantee(
                requested_fact_id=fact.id,
                formula=index.qualification,
                atom_proofs=(),
            ),
            subject=SubjectGuarantee(
                requested_fact_id=fact.id,
                subject_set_ref=index.subject_obligation.subject_set_ref.token,
                interpretation=type(index.subject_obligation).__name__,
                proof_refs=proof_refs_by_relation[relation_id],
            ),
        )
        for relation_id in relation_ids
    )


__all__ = [
    "semantic_question_contract",
    "semantic_relation_guarantees",
    "semantic_requested_fact",
]
