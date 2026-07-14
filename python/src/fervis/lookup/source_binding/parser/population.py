"""Answer-population and relation-source parsing."""

from __future__ import annotations

from dataclasses import replace
from fervis.lookup.source_binding.compiler_ir import (
    DraftEndpointParamBinding,
    DraftRelationSource,
    DraftRelationSourcePopulationChoice,
)
from fervis.lookup.answer_program.relations import SourceKind
from fervis.lookup.source_binding import provider_contract as provider_output
from fervis.lookup.source_binding.model import AnswerPopulation, SourceBindingRequest
from fervis.lookup.source_binding.candidates import SourceCandidate
from fervis.lookup.source_binding.candidates.model import CandidatePopulationBinding
from fervis.lookup.source_binding.parser_common import _text
from fervis.lookup.source_binding.population_bindings import PopulationBindingIndex


__all__ = [
    "bound_relation_source",
    "parse_answer_population",
]


def parse_answer_population(
    raw: provider_output.AnswerPopulationOutput,
    *,
    request: SourceBindingRequest,
    requested_fact_id: str,
    candidate: SourceCandidate,
) -> tuple[AnswerPopulation, CandidatePopulationBinding]:
    population_binding_id = _text(raw.population_binding_id)
    binding = _candidate_population_binding(
        population_binding_id,
        candidate=candidate,
    )
    PopulationBindingIndex.from_request(request).validate_selection(
        requested_fact_id=requested_fact_id,
        candidate=candidate,
        population_binding_id=population_binding_id,
    )
    intent_text = _text(raw.intent_text)
    return (
        AnswerPopulation(
            population_binding_id=population_binding_id,
            intent_text=intent_text,
            match_basis_explanation=_text(raw.match_basis_explanation),
        ),
        binding,
    )


def bound_relation_source(
    *,
    candidate: SourceCandidate,
    population_binding: CandidatePopulationBinding,
    param_binding_sets: tuple[tuple[DraftEndpointParamBinding, ...], ...],
    population_choices: tuple[DraftRelationSourcePopulationChoice, ...],
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
            )
            for param_bindings in param_binding_sets
        )
        source = source_invocations[0]
    return source, source_invocations


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
