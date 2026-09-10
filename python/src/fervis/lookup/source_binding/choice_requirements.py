"""Finite source values with explicit question requirement owners."""

from fervis.lookup.available_sources import SourceChoiceSurfaceKind


def requirement_choice_surfaces(request, branch_id):
    branch = next(b for b in request.strategy.branches if b.branch_id == branch_id)
    return tuple(
        surface
        for surface in request.source_catalog.choice_surfaces
        if surface.source_ref in branch.source_refs
        and surface.kind is SourceChoiceSurfaceKind.RETURNED_FIELD
        and any(
            request.explicit_subject_requirement_refs(choice, branch_id=branch_id)
            for choice in surface.values
        )
    )
