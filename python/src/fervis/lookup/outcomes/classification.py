"""Classify deterministic execution output into factual terminal outcomes."""

from __future__ import annotations

from fervis.lookup.clarification import (
    ClarificationOption,
    TargetReferenceAmbiguous,
    TargetReferenceUnsupported,
    clarify,
)
from fervis.lookup.answer_program.model import AnswerProgram
from fervis.lookup.answer_program.operations import AggregateSpec
from fervis.lookup.plan_execution.operation_runtime import RelationEngineOutput
from fervis.lookup.plan_execution.relations import (
    CompletenessStatus,
    RelationRows,
)
from fervis.lookup.outcomes.errors import (
    ExecutionIssue,
    ExecutionIssueKind,
)
from fervis.lookup.outcomes.model import (
    AnswerResult,
    EmptyRelation,
    EmptyRelationKind,
    FactResult,
    NoData,
    NeedsClarification,
)
from fervis.lookup.outcomes.operation_semantics import (
    empty_relation_kind_for_output_relation,
)


def classify_answer_result(
    answer: AnswerProgram,
    *,
    engine_output: RelationEngineOutput,
    final_relation_id: str = "",
) -> FactResult | ExecutionIssue:
    if engine_output.issue is not None:
        return engine_output.issue
    if engine_output.undefined is not None:
        no_data = _undefined_empty_aggregation_outcome(
            answer,
            engine_output=engine_output,
        )
        if no_data is not None:
            return no_data
        return FactResult(outcome=engine_output.undefined)

    relation_id = final_relation_id or _render_relation_id(answer)
    if relation_id:
        relation = engine_output.relation(relation_id)
        if relation.completeness.status != CompletenessStatus.COMPLETE:
            return ExecutionIssue(
                kind=ExecutionIssueKind.INCOMPLETE_EVIDENCE,
                message="answer relation evidence is incomplete",
                relation_id=relation.id,
                proof_refs=relation.completeness.proof_refs,
            )
        no_data = _empty_relation_outcome(
            relation,
            kind=_empty_relation_kind(answer, relation_id),
            requested_fact_ids=_fulfilled_requested_fact_ids(answer, relation_id),
        )
        if no_data is not None:
            return no_data

    return FactResult(
        outcome=AnswerResult(
            render_spec=answer.render_spec,
            relations=engine_output.relations,
            scalars=_rendered_scalars(answer, engine_output=engine_output),
            proof_refs=_proof_refs(
                engine_output.relations,
                scalar_proofs=_rendered_scalar_proofs(
                    answer,
                    engine_output=engine_output,
                ),
            ),
        )
    )


def classify_empty_relation(
    relation: RelationRows,
    *,
    kind: EmptyRelationKind,
    requested_fact_ids: tuple[str, ...] = (),
) -> FactResult | ExecutionIssue | None:
    return _empty_relation_outcome(
        relation,
        kind=kind,
        requested_fact_ids=requested_fact_ids,
    )


def classify_binding_candidates(
    *,
    requested_fact_id: str,
    binding_target_id: str,
    known_input_id: str,
    candidate_relation: RelationRows,
    display_fields: tuple[str, ...] = (),
) -> FactResult | ExecutionIssue | None:
    proof = candidate_relation.completeness
    if proof.status != CompletenessStatus.COMPLETE:
        return ExecutionIssue(
            kind=ExecutionIssueKind.INCOMPLETE_EVIDENCE,
            message="binding candidate relation is not complete",
            relation_id=candidate_relation.id,
            proof_refs=proof.proof_refs,
        )
    candidate_count = len(candidate_relation.rows)
    if candidate_count == 0:
        return FactResult(
            outcome=NeedsClarification(
                clarifications=(
                    clarify(
                        _unsupported_binding_candidate_cause(
                            requested_fact_id=requested_fact_id,
                            binding_target_id=binding_target_id,
                            known_input_id=known_input_id,
                            proof_refs=proof.proof_refs,
                        )
                    ),
                ),
                proof_refs=(f"known_input:{known_input_id}", *proof.proof_refs),
            )
        )
    if candidate_count == 1:
        return None
    return FactResult(
        outcome=NeedsClarification(
            clarifications=(
                clarify(
                    _ambiguous_binding_candidate_cause(
                        requested_fact_id=requested_fact_id,
                        binding_target_id=binding_target_id,
                        known_input_id=known_input_id,
                        candidate_relation=candidate_relation,
                        proof_refs=proof.proof_refs,
                    )
                ),
            ),
            proof_refs=(f"known_input:{known_input_id}", *proof.proof_refs),
        )
    )


def _unsupported_binding_candidate_cause(
    *,
    requested_fact_id: str,
    binding_target_id: str,
    known_input_id: str,
    proof_refs: tuple[str, ...],
) -> TargetReferenceUnsupported:
    return TargetReferenceUnsupported(
        clarification_id=f"unsupported_{binding_target_id}",
        requested_fact_id=requested_fact_id,
        known_input_id=known_input_id,
        source_text=binding_target_id,
        target_label="reference",
        proof_refs=proof_refs,
    )


def _ambiguous_binding_candidate_cause(
    *,
    requested_fact_id: str,
    binding_target_id: str,
    known_input_id: str,
    candidate_relation: RelationRows,
    proof_refs: tuple[str, ...],
) -> TargetReferenceAmbiguous:
    return TargetReferenceAmbiguous(
        clarification_id=f"clarify_{binding_target_id}",
        requested_fact_id=requested_fact_id,
        known_input_id=known_input_id,
        source_text=binding_target_id,
        target_label="reference",
        options=tuple(
            ClarificationOption(
                id=f"{candidate_relation.id}:{index}",
                label=f"Candidate {index}",
            )
            for index, _row in enumerate(candidate_relation.rows, start=1)
        ),
        proof_refs=proof_refs,
    )


def _empty_relation_outcome(
    relation: RelationRows,
    *,
    kind: EmptyRelationKind,
    requested_fact_ids: tuple[str, ...] = (),
) -> FactResult | ExecutionIssue | None:
    proof = relation.completeness
    if proof.status != CompletenessStatus.COMPLETE:
        return ExecutionIssue(
            kind=ExecutionIssueKind.INCOMPLETE_EVIDENCE,
            message="empty relation is not complete",
            relation_id=relation.id,
            proof_refs=proof.proof_refs,
        )
    if relation.rows:
        return None
    empty_relation = EmptyRelation(
        kind=kind,
        relation_id=relation.id,
        grain_keys=relation.grain_keys,
        requested_fact_ids=requested_fact_ids,
        scope_ref=proof.scope_fingerprint,
        proof_refs=proof.proof_refs,
    )
    return FactResult(
        outcome=NoData(
            empty_relation=empty_relation,
            proof_refs=proof.proof_refs,
        )
    )


def _undefined_empty_aggregation_outcome(
    answer: AnswerProgram,
    *,
    engine_output: RelationEngineOutput,
) -> FactResult | ExecutionIssue | None:
    undefined = engine_output.undefined
    if undefined is None:
        return None
    operation = next(
        (
            item
            for item in answer.operations
            if item.id == undefined.operation.operation_id
        ),
        None,
    )
    if operation is None or not isinstance(operation.spec, AggregateSpec):
        return None
    input_relation = engine_output.relation(operation.spec.input_relation)
    return _empty_relation_outcome(
        input_relation,
        kind=_empty_relation_kind(answer, operation.output_relation),
        requested_fact_ids=_fulfilled_requested_fact_ids(
            answer,
            operation.output_relation,
        ),
    )


def _render_relation_id(answer: AnswerProgram) -> str:
    if answer.render_spec is None or not answer.render_spec.relation_outputs:
        return ""
    relations = {
        relation_output.relation_id
        for relation_output in answer.render_spec.relation_outputs
    }
    if len(relations) != 1:
        return ""
    return next(iter(relations))


def _fulfilled_requested_fact_ids(
    answer: AnswerProgram,
    relation_id: str,
) -> tuple[str, ...]:
    if answer.render_spec is None:
        return ()
    render_output_ids = {
        relation_output.id
        for relation_output in answer.render_spec.relation_outputs
        if relation_output.relation_id == relation_id
    }
    ids: list[str] = []
    for item in answer.fulfillment:
        if (
            item.render_output_id in render_output_ids
            and item.requested_fact_id not in ids
        ):
            ids.append(item.requested_fact_id)
    return tuple(ids)


def _empty_relation_kind(
    answer: AnswerProgram,
    relation_id: str,
) -> EmptyRelationKind:
    return empty_relation_kind_for_output_relation(answer.operations, relation_id)


def _rendered_scalars(
    answer: AnswerProgram,
    *,
    engine_output: RelationEngineOutput,
) -> dict[str, object]:
    scalars = dict(engine_output.scalars or {})
    rendered: dict[str, object] = {}
    for scalar_output in tuple(getattr(answer.render_spec, "scalar_outputs", ()) or ()):
        scalar_id = str(getattr(scalar_output, "scalar_id", "") or "")
        if scalar_id in scalars:
            rendered[str(scalar_output.id)] = scalars[scalar_id]
    return rendered


def _rendered_scalar_proofs(
    answer: AnswerProgram,
    *,
    engine_output: RelationEngineOutput,
) -> tuple[tuple[str, ...], ...]:
    scalar_proofs = dict(engine_output.scalar_proofs or {})
    proofs: list[tuple[str, ...]] = []
    for scalar_output in tuple(getattr(answer.render_spec, "scalar_outputs", ()) or ()):
        scalar_id = str(getattr(scalar_output, "scalar_id", "") or "")
        if scalar_id in scalar_proofs:
            proofs.append(tuple(scalar_proofs[scalar_id]))
    return tuple(proofs)


def _proof_refs(
    relations: tuple[RelationRows, ...],
    *,
    scalar_proofs: tuple[tuple[str, ...], ...] = (),
) -> tuple[str, ...]:
    refs: list[str] = []
    for relation in relations:
        for ref in relation.completeness.proof_refs:
            if ref not in refs:
                refs.append(ref)
    for proof in scalar_proofs:
        for ref in proof:
            if ref not in refs:
                refs.append(ref)
    return tuple(refs)
