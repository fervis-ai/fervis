"""Strict provider schema for semantic Source Binding."""

from __future__ import annotations

from fervis.lookup.question_contract import (
    AssociationTerm,
    FactTerm,
    SetTerm,
)
from fervis.lookup.question_contract import RawDataRecord
from fervis.lookup.source_binding import semantic_provider_contract as output
from fervis.lookup.source_binding.semantic import (
    InvocationProjectionOption,
    SemanticSourceBindingRequest,
)
from fervis.lookup.source_binding.subject_obligations import (
    NormalInstanceExcludedStateRole,
)
from fervis.lookup.semantic_types import IdentifierType


def build_semantic_source_binding_schema(
    request: SemanticSourceBindingRequest,
) -> dict[str, object]:
    branch_ids = tuple(item.branch_id for item in request.strategy.branches)
    source_refs = tuple(source.id for source in request.source_catalog.sources)
    relation_refs = tuple(
        item.evidence_ref for item in request.source_catalog.relation_evidence
    )
    identity_refs = tuple(
        item.identity_ref for item in request.source_catalog.identity_evidence
    )
    set_refs = tuple(
        ref.token
        for ref in request.index.source_requirement_refs
        if isinstance(request.index.term_by_ref.get(ref), SetTerm)
    )
    fact_refs = tuple(
        ref.token
        for ref in request.index.source_requirement_refs
        if isinstance(request.index.term_by_ref.get(ref), FactTerm)
    )
    association_refs = tuple(
        ref.token
        for ref in request.index.association_requirement_refs
        if isinstance(request.index.term_by_ref.get(ref), AssociationTerm)
    )
    return output.SemanticSourceBindingOutput.schema(
        {
            "set_bindings": _closed_object(
                {
                    ref: _exact_realizations(
                        branch_ids,
                        output.SetRealizationOutput.schema(
                            {
                                "branch_id": {"enum": list(branch_ids)},
                                "mapping_basis": _text(),
                                "source_ref": {"enum": list(source_refs)},
                                "identity_ref": _nullable_enum(identity_refs),
                            }
                        ),
                    )
                    for ref in set_refs
                }
            ),
            "resolved_input_applications": (
                _resolved_input_applications_schema(request)
            ),
            "fact_bindings": _closed_object(
                {
                    ref: _optional_realizations(
                        branch_ids,
                        _fact_realization_schema(
                            request,
                            fact_ref=ref,
                            branch_ids=branch_ids,
                            source_refs=source_refs,
                        ),
                    )
                    for ref in fact_refs
                }
            ),
            "association_bindings": _closed_object(
                {
                    ref: _exact_realizations(
                        branch_ids,
                        _association_realization_schema(
                            branch_ids=branch_ids,
                            source_refs=source_refs,
                            relation_refs=relation_refs,
                        ),
                    )
                    for ref in association_refs
                }
            ),
            "subject_binding": _subject_binding_schema(
                request,
                branch_ids=branch_ids,
            ),
        }
    )


def _association_realization_schema(
    *,
    branch_ids: tuple[str, ...],
    source_refs: tuple[str, ...],
    relation_refs: tuple[str, ...],
) -> dict[str, object]:
    return output.AssociationRealizationOutput.schema(
        {
            "branch_id": {"enum": list(branch_ids)},
            "mapping_basis": _text(),
            "realization_ref": {"enum": [*relation_refs, *source_refs]},
        }
    )


def _fact_realization_schema(
    request: SemanticSourceBindingRequest,
    *,
    fact_ref: str,
    branch_ids: tuple[str, ...],
    source_refs: tuple[str, ...],
) -> dict[str, object]:
    semantic_ref = next(
        ref
        for ref in request.index.inferred_type_by_ref
        if getattr(ref, "token", None) == fact_ref
    )
    is_identity = isinstance(
        request.index.inferred_type_by_ref[semantic_ref], IdentifierType
    )
    field_refs_for_sources = request.returned_field_refs_for_fact(fact_ref)
    common = {
        "branch_id": {"enum": list(branch_ids)},
        "mapping_basis": _text(),
        "source_ref": {"enum": list(source_refs)},
    }
    if is_identity:
        return output.IdentifierFactRealizationOutput.schema(common)
    return output.ReturnedFactRealizationOutput.schema(
        {
            **common,
            "field_ref": {"type": "string", "enum": list(field_refs_for_sources)},
        }
    )


def _resolved_input_applications_schema(
    request: SemanticSourceBindingRequest,
) -> dict[str, object]:
    return _closed_object(
        {
            branch.branch_id: _branch_resolved_input_applications_schema(
                request,
                branch_id=branch.branch_id,
            )
            for branch in request.strategy.branches
        }
    )


def _branch_resolved_input_applications_schema(
    request: SemanticSourceBindingRequest,
    *,
    branch_id: str,
) -> dict[str, object]:
    variants = [
        output.ResolvedInputApplicationOutput.schema(
            {
                "mapping_basis": _text(),
                "owner_ref": {"enum": [owner_ref]},
                "value_ref": {"enum": [option.value_ref]},
                "value_component": {"enum": [_projection_component(option)]},
                "target_ref": {"enum": [option.target_ref]},
            }
        )
        for owner_ref in request.invocation_application_owner_refs
        for option in request.direct_value_options_for_owner(
            owner_ref,
            branch_id=branch_id,
        )
    ]
    item_schema = (
        variants[0]
        if len(variants) == 1
        else {"oneOf": variants}
        if variants
        else _closed_object({})
    )
    return {
        "type": "array",
        "items": item_schema,
        "maxItems": len(variants),
    }


def _projection_component(option: InvocationProjectionOption) -> str:
    return option.component_ref or option.projection.value


def _subject_binding_schema(
    request: SemanticSourceBindingRequest,
    *,
    branch_ids: tuple[str, ...],
) -> dict[str, object]:
    subject_ref = request.index.subject_obligation.subject_set_ref.token
    raw_records = isinstance(request.index.subject_obligation, RawDataRecord)
    return output.SubjectObligationBindingOutput.schema(
        {
            "subject_ref": {"enum": [subject_ref]},
            "branch_realizations": {
                "type": "array",
                "minItems": len(branch_ids),
                "maxItems": len(branch_ids),
                "items": {
                    "oneOf": [
                        _subject_branch_schema(
                            request,
                            branch_id=branch_id,
                            raw_records=raw_records,
                        )
                        for branch_id in branch_ids
                    ]
                },
            },
        }
    )


def _subject_branch_schema(
    request: SemanticSourceBindingRequest,
    *,
    branch_id: str,
    raw_records: bool,
) -> dict[str, object]:
    branch = next(
        item for item in request.strategy.branches if item.branch_id == branch_id
    )
    surfaces = tuple(
        surface
        for source_ref in branch.source_refs
        for surface in request.source_catalog.choice_surfaces
        if surface.source_ref == source_ref
    )
    return output.SubjectObligationRealizationOutput.schema(
        {
            "branch_id": {"enum": [branch_id]},
            "finite_choice_reviews": _closed_object(
                {}
                if raw_records
                else {
                    surface.surface_ref: _subject_surface_review_schema(
                        request,
                        branch_id=branch_id,
                        surface=surface,
                    )
                    for surface in surfaces
                }
            ),
        }
    )


def _subject_surface_review_schema(
    request: SemanticSourceBindingRequest,
    *,
    branch_id: str,
    surface,
) -> dict[str, object]:
    role_values = [
        "NONE",
        *(item.value for item in NormalInstanceExcludedStateRole),
    ]
    common = {
        "choice_domain_meaning": _text(),
        "role_match_basis": _text(),
        "matched_excluded_role": {"enum": role_values},
        "choice_inclusion_basis": _text(),
        "choice_inclusion": {"enum": ["INCLUDE", "EXCLUDE"]},
    }
    return output.SubjectSurfaceReviewOutput.schema(
        {
            "surface_mapping_basis": _text(),
            "choice_reviews": _closed_object(
                {
                    choice.value: _subject_choice_review_schema(
                        common=common,
                        requirement_refs=request.choice_value_requirement_refs(
                            choice,
                            branch_id=branch_id,
                        ),
                    )
                    for choice in surface.values
                }
            ),
        }
    )


def _subject_choice_review_schema(
    *,
    common: dict[str, object],
    requirement_refs: tuple[str, ...],
) -> dict[str, object]:
    review_without_requirement = output.SubjectChoiceReviewOutput.schema(
        {
            **common,
            "requirement_mapping_basis": {"type": "null"},
            "requirement_ref": {"type": "null"},
        }
    )
    if not requirement_refs:
        return review_without_requirement
    return {
        "oneOf": [
            review_without_requirement,
            output.SubjectChoiceReviewOutput.schema(
                {
                    **common,
                    "requirement_mapping_basis": _text(),
                    "requirement_ref": {"enum": list(requirement_refs)},
                }
            ),
        ]
    }


def _exact_realizations(
    branch_ids: tuple[str, ...], item_schema: dict[str, object]
) -> dict[str, object]:
    return {
        "type": "array",
        "minItems": len(branch_ids),
        "maxItems": len(branch_ids),
        "items": item_schema,
    }


def _optional_realizations(
    branch_ids: tuple[str, ...], item_schema: dict[str, object]
) -> dict[str, object]:
    return {
        "type": "array",
        "minItems": 0,
        "maxItems": len(branch_ids),
        "items": item_schema,
    }


def _bounded_array(values: tuple[str, ...], *, min_items: int = 0) -> dict[str, object]:
    return {
        "type": "array",
        "minItems": min_items,
        "maxItems": len(values),
        "items": {"type": "string", "enum": list(values)},
    }


def _closed_object(properties: dict[str, object]) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


def _nullable_enum(values: tuple[str, ...]) -> dict[str, object]:
    return {
        "oneOf": [
            *(({"type": "string", "enum": list(values)},) if values else ()),
            {"type": "null"},
        ]
    }


def _text() -> dict[str, object]:
    return {"type": "string", "minLength": 1}


__all__ = ["build_semantic_source_binding_schema"]
