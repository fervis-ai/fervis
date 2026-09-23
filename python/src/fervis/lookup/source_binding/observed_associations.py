"""Typed equality joins over observed properties, without identity assertions."""

from fervis.lookup.plan_execution.declared_values import (
    declared_comparison_types_compatible,
)
from fervis.lookup.available_sources import SourceFieldBinding
from fervis.lookup.relation_catalog.row_sources.model import (
    RowSourceField,
    row_source_value_type_is_scalar,
)
from .model import SemanticSourceBindingRequest, SetRealization
from .association_choices import association_endpoints


def validate_reference_proxy_associations(request, set_bindings, association_bindings):
    """A proxy's referent value must be the equality witness at every edge."""
    from fervis.lookup.question_contract.model import FactLocalRef
    from .model import AssociationRealizationKind

    for set_ref, realizations in set_bindings.items():
        set_local = request.index.fact_local_ref_by_local_id
        proxy_set = FactLocalRef.from_token(set_ref).local_id
        for realization in realizations:
            proxy_field = realization.reference_proxy_field_ref
            if not proxy_field:
                continue
            proxy_ref = request.index.fact_local_ref_by_local_id[proxy_set]
            if any(
                proxy_ref in output.dependencies
                for output in request.index.output_requirements
            ) or proxy_ref in request.index.grouping_refs:
                raise ValueError("Reference proxy cannot supply an entity output or grouping")
            for association in request.index.requested_fact.associations:
                side = (
                    0 if association.from_set_ref == proxy_set else
                    1 if association.to_set_ref == proxy_set else None
                )
                if side is None:
                    continue
                association_ref = set_local[association.id].token
                matches = [
                    item for item in association_bindings.get(association_ref, ())
                    if item.branch_id == realization.branch_id
                ]
                if (
                    len(matches) != 1
                    or matches[0].kind is not AssociationRealizationKind.OBSERVED_EQUALITY
                    or not matches[0].field_pairs
                    or not any(pair[side] == proxy_field for pair in matches[0].field_pairs)
                ):
                    raise ValueError("Reference proxy association must compare its declared value")


def comparable_fields(left: RowSourceField, right: RowSourceField) -> bool:
    return (
        row_source_value_type_is_scalar(left.type)
        and row_source_value_type_is_scalar(right.type)
        and declared_comparison_types_compatible(left.type.value, right.type.value)
    )


def observed_pairs(
    request: SemanticSourceBindingRequest,
    *,
    association_ref: str,
    branch_id: str,
    set_bindings: dict[str, tuple[SetRealization, ...]],
    field_pairs: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, ...], tuple[tuple[SourceFieldBinding, SourceFieldBinding], ...]]:
    """Validate orientation and ownership once for parsing and verification."""
    if not field_pairs or len(set(field_pairs)) != len(field_pairs):
        raise ValueError("observed association requires distinct field pairs")
    endpoints = association_endpoints(request, association_ref)
    selected = tuple(
        tuple(
            value for value in set_bindings.get(ref, ()) if value.branch_id == branch_id
        )
        for ref in endpoints
    )
    if any(len(values) != 1 for values in selected):
        raise ValueError("observed association requires one carrier per endpoint")
    sources = tuple(values[0].source_ref for values in selected)
    result = []
    for left_ref, right_ref in field_pairs:
        left = request.source_catalog.field_binding(left_ref)
        right = request.source_catalog.field_binding(right_ref)
        if (left.source_ref, right.source_ref) != sources:
            raise ValueError(
                "observed association fields must belong to assigned endpoints"
            )
        if not comparable_fields(left.field, right.field):
            raise ValueError(
                "observed association fields require comparable scalar types"
            )
        result.append((left, right))
    return sources, tuple(result)
