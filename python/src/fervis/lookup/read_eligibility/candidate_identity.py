"""Stable read-candidate identity shared across read eligibility and binding."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def read_candidate_signature(
    candidate: Mapping[str, Any],
    *,
    requested_fact_id: str,
) -> str:
    """Return the stable identity for one API-read candidate variant."""

    return (
        "read_candidate_"
        + hashlib.sha256(
            _canonical_candidate_identity(
                candidate,
                requested_fact_id=requested_fact_id,
            ).encode("utf-8")
        ).hexdigest()
    )


def _canonical_candidate_identity(
    candidate: Mapping[str, Any],
    *,
    requested_fact_id: str,
) -> str:
    payload = {
        "requested_fact_id": str(requested_fact_id or ""),
        "read_id": str(candidate.get("read_id") or ""),
        "bound_params": [
            {"param_id": param_id, "value": value}
            for param_id, value in _stable_bound_params(candidate)
        ],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _stable_bound_params(
    candidate: Mapping[str, Any],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (
                str(
                    param.get("param_id")
                    or param.get("param_ref")
                    or param.get("name")
                    or ""
                ),
                _stable_value(param.get("value")),
            )
            for param in candidate.get("bound_params") or ()
            if isinstance(param, Mapping)
            and str(
                param.get("param_id")
                or param.get("param_ref")
                or param.get("name")
                or ""
            )
        )
    )


def _stable_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
