"""Answer-population and relation-source parsing."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from fervis.lookup.source_binding.compiler_ir import (
    DraftEndpointParamBinding,
    DraftRelationSource,
    DraftRelationSourcePopulationChoice,
)
from fervis.lookup.answer_program.relations import SourceKind
from fervis.lookup.source_binding import provider_contract as provider_output
from fervis.lookup.source_binding.model import AnswerPopulation, SourceBindingRequest
from fervis.lookup.source_binding.parser_common import _dict, _required_strings, _text
from fervis.lookup.source_binding.population_bindings import PopulationBindingIndex


__all__ = [
    "bound_relation_source",
    "parse_answer_population",
]


def parse_answer_population(
    raw_value: Any,
    *,
    request: SourceBindingRequest,
    requested_fact_id: str,
    candidate: Any,
) -> tuple[AnswerPopulation, dict[str, Any]]:
    raw = provider_output.AnswerPopulationOutput.parse(raw_value)
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
    candidate: Any,
    population_binding: dict[str, Any],
    param_binding_sets: tuple[tuple[DraftEndpointParamBinding, ...], ...],
    population_choices: tuple[DraftRelationSourcePopulationChoice, ...],
) -> tuple[Any, tuple[Any, ...]]:
    if (
        str(population_binding.get("kind") or "") == "exact_row_set"
        and str(getattr(candidate, "kind", "") or "") == "prior_answer_rows"
    ):
        basis = _dict(population_binding.get("basis"), "answer_population.basis")
        memory_relation_id = _text(basis.get("memory_relation_id"))
        return (
            DraftRelationSource(
                kind=SourceKind.MEMORY_READ,
                memory_relation_id=memory_relation_id,
                population_choices=population_choices,
                proof_refs=_required_strings(
                    basis.get("proof_refs"),
                    "answer_population.basis.proof_refs",
                ),
            ),
            (),
        )
    source = candidate.source
    source_invocations: tuple[Any, ...] = ()
    if source is not None:
        source_invocations = tuple(
            replace(
                source,
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
    candidate: Any,
) -> dict[str, Any]:
    bindings = {
        binding_id: item
        for item in getattr(candidate, "population_bindings", ())
        if isinstance(item, dict)
        for binding_id in (str(item.get("population_binding_id") or ""),)
        if binding_id
    }
    binding = bindings.get(population_binding_id)
    if binding is None:
        raise ValueError("answer population references unknown population binding")
    return binding
