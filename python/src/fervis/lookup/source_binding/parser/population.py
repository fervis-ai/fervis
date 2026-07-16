"""Answer-population and relation-source parsing."""

from __future__ import annotations

from dataclasses import replace
from fervis.lookup.source_binding.compiler_ir import (
    DraftEndpointParamBinding,
    DraftRelationSource,
    DraftRelationSourcePopulationChoice,
)
from fervis.lookup.answer_program.relations import (
    PopulationCoverageClaim,
    PopulationCoverageRole,
    RelationSourcePopulationBinding,
    SourceKind,
)
from fervis.lookup.source_binding import provider_contract as provider_output
from fervis.lookup.source_binding.model import AnswerPopulation, SourceBindingRequest
from fervis.lookup.source_binding.candidates import SourceCandidate
from fervis.lookup.source_binding.candidates.model import CandidatePopulationBinding
from fervis.lookup.source_binding.parser_common import _text
from fervis.lookup.source_binding.population_bindings import PopulationBindingIndex
from fervis.lookup.source_binding.membership_tests import membership_tests_by_key
from fervis.lookup.source_binding.plan_targets import SourceBindingTarget
from fervis.lookup.source_binding.population_effects import population_coverage_claims
from fervis.lookup.source_binding.review_scope import SourceBindingReviewScope


__all__ = [
    "bound_relation_source",
    "parse_answer_population",
]


def parse_answer_population(
    raw: provider_output.AnswerPopulationOutput,
    *,
    request: SourceBindingRequest,
    target: SourceBindingTarget,
    candidate: SourceCandidate,
    review_scope: SourceBindingReviewScope,
) -> tuple[
    AnswerPopulation,
    CandidatePopulationBinding,
    tuple[PopulationCoverageClaim, ...],
]:
    population_binding_id = _text(raw.population_binding_id)
    binding = _candidate_population_binding(
        population_binding_id,
        candidate=candidate,
    )
    PopulationBindingIndex.from_request(request).validate_selection(
        requested_fact_id=target.requested_fact_id,
        candidate=candidate,
        population_binding_id=population_binding_id,
    )
    intent_text = _text(raw.intent_text)
    claims = _answer_population_coverage_claims(
        raw,
        binding=binding,
        candidate=candidate,
        request=request,
        target=target,
        review_scope=review_scope,
    )
    return (
        AnswerPopulation(
            population_binding_id=population_binding_id,
            intent_text=intent_text,
            match_basis_explanation=_text(raw.match_basis_explanation),
        ),
        binding,
        claims,
    )


def _answer_population_coverage_claims(
    raw: provider_output.AnswerPopulationOutput,
    *,
    binding: CandidatePopulationBinding,
    candidate: SourceCandidate,
    request: SourceBindingRequest,
    target: SourceBindingTarget,
    review_scope: SourceBindingReviewScope,
) -> tuple[PopulationCoverageClaim, ...]:
    fact = next(
        item for item in request.requested_facts if item.id == target.requested_fact_id
    )
    if fact.answer_population is None:
        if raw.population_test_results:
            raise ValueError("answer population has no membership tests")
        return ()
    tests_by_key = membership_tests_by_key(fact.answer_population.membership_tests)
    tests = tuple(
        tests_by_key[test_id]
        for test_id in review_scope.population_binding_test_ids(
            target.binding_target_id
        )
    )
    coverage_role = (
        PopulationCoverageRole.ROW_POPULATION
        if target.requires_answer_fulfillment
        else PopulationCoverageRole.OPERATION_CONDITION
    )
    return population_coverage_claims(
        raw.population_test_results,
        tests=tests,
        requested_fact_id=target.requested_fact_id,
        role_text=target.requirement_id,
        coverage_role=coverage_role,
        proof_refs=_population_binding_proof_refs(
            binding,
            candidate=candidate,
            request=request,
        ),
    )


def _population_binding_proof_refs(
    binding: CandidatePopulationBinding,
    *,
    candidate: SourceCandidate,
    request: SourceBindingRequest,
) -> tuple[str, ...]:
    if candidate.source is None:
        value = next(
            (item for item in request.available_values if item.id == candidate.value_id),
            None,
        )
        if value is None:
            return binding.proof_refs
        return tuple(dict.fromkeys((*binding.proof_refs, *value.proof_refs)))
    return RelationSourcePopulationBinding(
        id=binding.id,
        supporting_proof_refs=binding.proof_refs,
    ).proof_refs


def bound_relation_source(
    *,
    candidate: SourceCandidate,
    population_binding: CandidatePopulationBinding,
    param_binding_sets: tuple[tuple[DraftEndpointParamBinding, ...], ...],
    population_choices: tuple[DraftRelationSourcePopulationChoice, ...],
    population_coverage_claims: tuple[PopulationCoverageClaim, ...],
    row_source_id: str = "",
) -> tuple[DraftRelationSource | None, tuple[DraftRelationSource, ...]]:
    if (
        population_binding.kind == "exact_row_set"
        and candidate.kind == "prior_answer_rows"
    ):
        return (
            DraftRelationSource(
                kind=SourceKind.MEMORY_READ,
                memory_relation_id=population_binding.memory_relation_id,
                population_choices=population_choices,
                population_binding=_relation_population_binding(population_binding),
                population_coverage_claims=population_coverage_claims,
                proof_refs=population_binding.proof_refs,
            ),
            (),
        )
    source = candidate.source
    source_invocations: tuple[DraftRelationSource, ...] = ()
    if source is not None:
        source_invocations = tuple(
            replace(
                source,
                row_source_id=row_source_id or source.row_source_id,
                param_bindings=param_bindings,
                population_choices=population_choices,
                population_binding=_relation_population_binding(population_binding),
                population_coverage_claims=_invocation_population_coverage_claims(
                    population_coverage_claims,
                    param_bindings=param_bindings,
                ),
            )
            for param_bindings in param_binding_sets
        )
        source = source_invocations[0]
    return source, source_invocations


def _relation_population_binding(
    binding: CandidatePopulationBinding,
) -> RelationSourcePopulationBinding:
    return RelationSourcePopulationBinding(
        id=binding.id,
        supporting_proof_refs=binding.proof_refs,
    )


def _invocation_population_coverage_claims(
    claims: tuple[PopulationCoverageClaim, ...],
    *,
    param_bindings: tuple[DraftEndpointParamBinding, ...],
) -> tuple[PopulationCoverageClaim, ...]:
    param_refs = {f"source_param:{binding.param_id}" for binding in param_bindings}
    binding_proof_refs = {
        proof_ref for binding in param_bindings for proof_ref in binding.proof_refs
    }
    output: list[PopulationCoverageClaim] = []
    for claim in claims:
        claimed_param_refs = {
            proof_ref
            for proof_ref in claim.proof_refs
            if proof_ref.startswith("source_param:")
        }
        if not claimed_param_refs:
            output.append(claim)
            continue
        if not claimed_param_refs <= param_refs:
            continue
        scoped_proof_refs = tuple(
            proof_ref
            for proof_ref in claim.proof_refs
            if not proof_ref.startswith("known_input:")
            or proof_ref in binding_proof_refs
        )
        output.append(replace(claim, proof_refs=scoped_proof_refs))
    return tuple(output)


def _candidate_population_binding(
    population_binding_id: str,
    *,
    candidate: SourceCandidate,
) -> CandidatePopulationBinding:
    bindings = {item.id: item for item in candidate.population_bindings if item.id}
    binding = bindings.get(population_binding_id)
    if binding is None:
        raise ValueError("answer population references unknown population binding")
    return binding
