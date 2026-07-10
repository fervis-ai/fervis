"""Materialize explicit execution scope into the requested-fact contract."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from fervis.lookup.plan_execution.errors import VerificationError
from fervis.lookup.question_contract import (
    RequestedFact,
    RequestedFactPopulationConstraint,
)

if TYPE_CHECKING:
    from fervis.lookup.answer_program.expression_instantiation import (
        ResolvedPopulationChoice,
    )


def materialize_requested_facts(
    facts: tuple[RequestedFact, ...],
    *,
    population_choices: tuple[ResolvedPopulationChoice, ...],
) -> tuple[RequestedFact, ...]:
    constraints_by_fact = {
        fact.id: {constraint.id: constraint for constraint in fact.population_constraints}
        for fact in facts
    }
    for choice in population_choices:
        constraint = RequestedFactPopulationConstraint(
            id=choice.semantic_control_ref,
            included_values=choice.included_values,
            excluded_values=choice.excluded_values,
        )
        for fact_id in choice.requested_fact_ids:
            if fact_id not in constraints_by_fact:
                raise VerificationError(
                    f"population choice references unknown requested fact {fact_id}"
                )
            existing = constraints_by_fact[fact_id].get(constraint.id)
            if existing is not None and existing != constraint:
                raise VerificationError(
                    f"requested fact {fact_id} has conflicting population constraints"
                )
            constraints_by_fact[fact_id][constraint.id] = constraint
    return tuple(
        replace(
            fact,
            population_constraints=tuple(constraints_by_fact[fact.id].values()),
        )
        for fact in facts
    )
