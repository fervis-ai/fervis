"""Source-binding assessment basis projected into fact-planning choices."""

from __future__ import annotations

from typing import Any

from fervis.lookup.source_binding.model import SourceFulfillment


def metric_fit_bases_by_evidence_id(
    fulfillments: tuple[SourceFulfillment, ...],
) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for fulfillment in fulfillments:
        for basis in fulfillment.metric_fit_bases:
            output.setdefault(
                basis.evidence_id,
                {
                    "metric_meaning": basis.metric_meaning,
                    "fit_basis": basis.fit_basis,
                },
            )
    return output


def attach_metric_fit_basis(
    payload: dict[str, Any],
    *,
    evidence_id: str,
    bases_by_evidence_id: dict[str, dict[str, str]],
) -> dict[str, Any]:
    basis = bases_by_evidence_id.get(evidence_id)
    if not basis:
        return payload
    return {**payload, "source_binding_basis": basis}
