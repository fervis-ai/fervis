"""Lower graph-owned output refs, then reuse the production assertion."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from scripts.experiments.question_contract.assertion import validate as validate_contract


def validate(arguments: dict[str, Any], context: dict[str, Any]) -> list[str]:
    expected = context.get("expected_requested_output_kinds") or ()
    actual = _output_kinds(arguments)
    errors = [] if actual == expected else [
        f"requested output kinds are {actual!r}; expected {expected!r}"
    ]
    production_arguments = deepcopy(arguments)
    try:
        _lower_output_refs(production_arguments)
    except ValueError as exc:
        return [*errors, f"output-ref lowering rejected output: {exc}"]
    return [*errors, *validate_contract(production_arguments, context)]


def _output_kinds(arguments: dict[str, Any]) -> list[str]:
    return [
        str(output.get("output_kind"))
        for request in _answer_requests(arguments)
        for output in _requested_outputs(request)
    ]


def _lower_output_refs(arguments: dict[str, Any]) -> None:
    for request in _answer_requests(arguments):
        associations = _returned_entity_associations(request.get("set_graph"))
        for output in _requested_outputs(request):
            kind = output.pop("output_kind", None)
            output.pop("output_kind_basis", None)
            if kind == "value":
                continue
            if kind != "related_entity":
                raise ValueError(f"unknown output kind: {kind!r}")
            returned_ref = output.pop("returned_entity_ref", None)
            origin = output.pop("origin", None)
            association_ref = associations.get(returned_ref)
            if association_ref is None:
                raise ValueError(
                    f"returned entity {returned_ref!r} has no graph owner"
                )
            output["expression"] = {
                "kind": "fact",
                "identity_path": {
                    "kind": "related_instance",
                    "association_ref": association_ref,
                },
                "origin": origin,
            }


def _answer_requests(arguments: dict[str, Any]) -> list[dict[str, Any]]:
    outcome = arguments.get("outcome")
    requests = outcome.get("answer_requests") if isinstance(outcome, dict) else None
    if not isinstance(requests, list):
        return []
    return [item for item in requests if isinstance(item, dict)]


def _requested_outputs(request: dict[str, Any]) -> list[dict[str, Any]]:
    outputs = request.get("outputs")
    values = outputs.get("requested_value_outputs") if isinstance(outputs, dict) else None
    if not isinstance(values, list):
        return []
    return [item for item in values if isinstance(item, dict)]


def _returned_entity_associations(value: object) -> dict[object, str]:
    associations: dict[object, str] = {}
    if not isinstance(value, dict):
        return associations
    for related in value.get("related_sets") or ():
        if not isinstance(related, dict):
            continue
        association = related.get("association")
        related_set = related.get("set")
        if isinstance(association, dict) and isinstance(related_set, dict):
            returned_ref = related_set.pop("returned_entity_ref", None)
            association_ref = association.get("id")
            if returned_ref is not None and isinstance(association_ref, str):
                if returned_ref in associations:
                    raise ValueError(
                        f"returned entity {returned_ref!r} has multiple graph owners"
                    )
                associations[returned_ref] = association_ref
        associations.update(_returned_entity_associations(related))
    return associations


__all__ = ["validate"]
