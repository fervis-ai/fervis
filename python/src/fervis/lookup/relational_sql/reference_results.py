"""Certify complete reference-query results before publishing scalar key values."""

from fervis.lookup.identity_types import (
    IdentityExecutionFailureReason,
    ReferenceResolutionFailure,
)
from fervis.lookup.outcomes.errors import UnresolvedReferenceError


def require_unique_reference(spec, rows, *, relation_id, proof_refs):
    projection = spec.entity_keys[0]
    keys = {}
    try:
        for row in rows:
            keys.setdefault(projection.project(row), row)
    except (ValueError, TypeError) as exc:
        raise UnresolvedReferenceError(
            ReferenceResolutionFailure(
                spec.reference_input_ref,
                IdentityExecutionFailureReason.INVALID_RESOLVER_RESULT,
                operand=spec.reference_operand,
            ),
            relation_id=relation_id,
            proof_refs=proof_refs,
        ) from exc
    if len(keys) != 1:
        reason = (
            IdentityExecutionFailureReason.NOT_FOUND
            if not keys
            else IdentityExecutionFailureReason.AMBIGUOUS_RESULT
        )
        raise UnresolvedReferenceError(
            ReferenceResolutionFailure(
                spec.reference_input_ref, reason, tuple(keys), spec.reference_operand
            ),
            relation_id=relation_id,
            proof_refs=proof_refs,
        )
    return (next(iter(keys.values())),)
