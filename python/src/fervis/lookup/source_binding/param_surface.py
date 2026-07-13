"""Param decision helpers for source binding."""

from __future__ import annotations

from typing import Any

from fervis.lookup.source_binding.candidates.model import SourceCandidate


def param_decision_ids_by_effective_param(
    candidate: SourceCandidate,
    *,
    effective_param_ids: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    effective = set(effective_param_ids)
    output: dict[str, list[str]] = {}
    for param in candidate.params:
        param_id = param.id
        if param_id not in effective:
            continue
        decision_ids = [
            decision_id
            for option in param.decision_options
            for decision_id in (option.id,)
            if decision_id
        ]
        if decision_ids:
            output[param_id] = decision_ids
    return {param_id: tuple(ids) for param_id, ids in output.items()}


def param_requires_finite_choice_review(param: dict[str, Any]) -> bool:
    if not param.get("choices"):
        return False
    population_contract = param.get("population_contract")
    if not isinstance(population_contract, dict):
        return False
    omission_behavior = population_contract.get("omission_behavior")
    if not isinstance(omission_behavior, dict):
        return bool(param.get("required"))
    return str(omission_behavior.get("kind") or "") == "all_values"
