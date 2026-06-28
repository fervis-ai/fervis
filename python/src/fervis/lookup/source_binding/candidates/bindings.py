"""Bound endpoint parameter helpers for source-binding candidates."""

from ._shared import Any, EndpointParamBinding


def _bound_param_bindings(value: Any) -> tuple[EndpointParamBinding, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        EndpointParamBinding(
            param_id=str(item.get("param_id") or ""),
            value=item.get("value"),
            proof_refs=tuple(
                str(ref) for ref in item.get("proof_refs") or () if str(ref)
            ),
        )
        for item in value
        if isinstance(item, dict)
        and str(item.get("param_id") or "")
        and "value" in item
        and item.get("value") is not None
    )
