"""Deterministic source-binding population role handles."""

from __future__ import annotations

from fervis.lookup.source_binding.param_surface import (
    param_requires_finite_choice_review,
)

from ._shared import Any
from .candidate_tree import CandidateTreeContext, map_source_candidate_tree


def _with_source_population_roles(
    payload: dict[str, Any],
    *,
    request: Any,
) -> dict[str, Any]:
    facts_by_id = {fact.id: fact for fact in request.requested_facts}
    return map_source_candidate_tree(
        payload,
        lambda candidate, context: _candidate_with_population_roles_for_tree(
            candidate,
            context=context,
            facts_by_id=facts_by_id,
        ),
        top_level_keys=(),
    )


def _candidate_with_population_roles_for_tree(
    candidate: dict[str, Any],
    *,
    context: CandidateTreeContext,
    facts_by_id: dict[str, Any],
) -> dict[str, Any] | None:
    fact = facts_by_id.get(context.requested_fact_id)
    if fact is None or not _needs_population_roles(candidate):
        return candidate
    membership_test_ids = (
        tuple(test.id for test in fact.answer_population.membership_tests)
        if fact.answer_population is not None
        else ()
    )
    if not membership_test_ids:
        return candidate
    roles = _population_roles_for_candidate(candidate)
    if not roles:
        return None
    output = dict(candidate)
    output["population_roles"] = roles
    return output


def _needs_population_roles(candidate: dict[str, Any]) -> bool:
    return any(
        isinstance(param, dict) and param_requires_finite_choice_review(param)
        for param in candidate.get("params") or ()
    )


def _population_roles_for_candidate(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    grains = tuple(
        item for item in candidate.get("result_grains") or () if isinstance(item, dict)
    )
    if not grains:
        grains = (
            {
                "row_path_id": candidate.get("row_path_id") or "root",
                "grain_id": candidate.get("row_source_id")
                or candidate.get("row_path_id")
                or "root",
                "cardinality": candidate.get("cardinality") or "",
            },
        )
    roles: list[dict[str, Any]] = []
    for index, grain in enumerate(_role_grains(grains, candidate=candidate), start=1):
        str(grain.get("row_path_id") or grain.get("path") or "root")
        roles.append(_population_role(candidate, grain=grain, index=index))
    return roles


def _role_grains(
    grains: tuple[dict[str, Any], ...],
    *,
    candidate: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    many_or_root = tuple(
        grain
        for grain in grains
        if str(grain.get("cardinality") or "") == "many"
        or str(grain.get("row_path_id") or "") == "root"
    )
    candidate_grains = many_or_root or grains[:1]
    controlled_row_path_ids = _finite_choice_axis_row_path_ids(candidate)
    if controlled_row_path_ids:
        controlled_grains = tuple(
            grain
            for grain in candidate_grains
            if str(grain.get("row_path_id") or grain.get("path") or "root")
            in controlled_row_path_ids
        )
        if controlled_grains:
            return controlled_grains
    return candidate_grains


def _finite_choice_axis_row_path_ids(candidate: dict[str, Any]) -> set[str]:
    evidence_items_by_id = {
        str(item.get("evidence_id") or ""): item
        for item in candidate.get("evidence_items") or ()
        if isinstance(item, dict) and str(item.get("evidence_id") or "")
    }
    output: set[str] = set()
    for param in candidate.get("params") or ():
        if not isinstance(param, dict) or not param_requires_finite_choice_review(
            param
        ):
            continue
        population_contract = param.get("population_contract")
        if not isinstance(population_contract, dict):
            continue
        axis_field = population_contract.get("axis_field")
        if not isinstance(axis_field, dict):
            continue
        evidence_id = str(axis_field.get("evidence_id") or "")
        evidence_item = evidence_items_by_id.get(evidence_id)
        row_path_id = str((evidence_item or {}).get("row_path_id") or "")
        if row_path_id:
            output.add(row_path_id)
    return output


def _population_role(
    candidate: dict[str, Any],
    *,
    grain: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    read_id = str(candidate.get("read_id") or "")
    row_path_id = str(grain.get("row_path_id") or grain.get("path") or "root")
    row_grain_id = str(grain.get("grain_id") or row_path_id)
    return {
        "role_id": f"role_{index}",
        "row_grain_id": row_grain_id,
        "row_path_id": row_path_id,
        "role_kind": "SOURCE_RESULT_ROWS",
        "role_text": f"{read_id} rows at row path {row_path_id}",
    }
