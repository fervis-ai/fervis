"""Source-binding parser runtime context."""

from __future__ import annotations

from dataclasses import dataclass
from fervis.lookup.source_binding.candidates import SourceCandidate, source_candidates
from fervis.lookup.source_binding.model import SourceBindingRequest
from fervis.lookup.source_binding.plan_targets import (
    SourceBindingTargetIndex,
    source_binding_target_index,
)
from fervis.lookup.source_binding.review_scope import (
    SourceBindingReviewScope,
    source_binding_review_scope,
)


__all__ = [
    "SourceBindingParseContext",
    "source_binding_parse_context",
]


@dataclass(frozen=True)
class SourceBindingParseContext:
    request: SourceBindingRequest
    target_index: SourceBindingTargetIndex
    candidates: dict[str, SourceCandidate]
    review_scope: SourceBindingReviewScope


def source_binding_parse_context(
    request: SourceBindingRequest,
) -> SourceBindingParseContext:
    target_index = source_binding_target_index(request)
    candidates = source_candidates(request)
    return SourceBindingParseContext(
        request=request,
        target_index=target_index,
        candidates=candidates,
        review_scope=source_binding_review_scope(
            request,
            candidates_by_id=candidates,
            target_index=target_index,
        ),
    )
