"""Param decision helpers for source binding."""

from __future__ import annotations

from typing import Any


def param_decision_ids_by_effective_param(
    candidate: Any,
    *,
    effective_param_ids: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    effective = set(effective_param_ids)
    output: dict[str, list[str]] = {}
    for param in candidate.params:
        if not isinstance(param, dict):
            continue
        param_id = str(param.get("param_id") or "")
        if param_id not in effective:
            continue
        decision_ids = [
            decision_id
            for option in param.get("decision_options") or ()
            if isinstance(option, dict)
            for decision_id in (str(option.get("param_decision_id") or ""),)
            if decision_id
        ]
        if decision_ids:
            output[param_id] = decision_ids
    return {param_id: tuple(ids) for param_id, ids in output.items()}


def choice_values_for_effective_params(
    candidate: Any,
    *,
    effective_param_ids: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    effective = set(effective_param_ids)
    return {
        param_id: tuple(
            str(choice) for choice in param.get("choices") or () if str(choice)
        )
        for param in candidate.params
        if isinstance(param, dict)
        for param_id in (str(param.get("param_id") or ""),)
        if param_id in effective and param.get("choices")
    }


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


def param_has_default_value(param: dict[str, Any]) -> bool:
    return "default" in param and param.get("default") is not None
