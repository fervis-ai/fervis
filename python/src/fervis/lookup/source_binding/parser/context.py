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
from fervis.lookup.source_binding.input_applications import (
    ResolvedInputApplicationSurface,
    resolved_input_application_surfaces,
)
from fervis.lookup.source_binding.closed_key_params import closed_key_param_binding_index


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
    input_application_surfaces: dict[str, ResolvedInputApplicationSurface]


def source_binding_parse_context(
    request: SourceBindingRequest,
) -> SourceBindingParseContext:
    target_index = source_binding_target_index(request)
    candidates = source_candidates(request)
    closed_key_bindings = closed_key_param_binding_index(
        request,
        targets=target_index.targets,
        candidates_by_id=candidates,
    )
    return SourceBindingParseContext(
        request=request,
        target_index=target_index,
        candidates=candidates,
        review_scope=source_binding_review_scope(
            request,
            candidates_by_id=candidates,
            target_index=target_index,
        ),
        input_application_surfaces=resolved_input_application_surfaces(
            request,
            targets=target_index.targets,
            candidates_by_id=candidates,
            closed_key_bindings=closed_key_bindings,
        ),
    )
