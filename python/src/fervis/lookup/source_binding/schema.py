"""Strict provider schema for Source Binding."""

from __future__ import annotations


from fervis.model_io.structured_output.schema import without_unreferenced_definitions

from fervis.lookup.question_contract import (
    AssociationTerm,
    SetTerm,
)
from fervis.lookup.source_binding import provider_contract as output
from fervis.lookup.source_binding.association_choices import association_choices
from fervis.lookup.source_binding.model import (
    InvocationProjectionOption,
    SemanticSourceBindingRequest,
)


def _record_fields_schema(request, set_ref):
    from .record_outputs import is_record_output_set
    from fervis.lookup.available_sources import SourceFieldBinding
    is_output = is_record_output_set(request, set_ref)
    rows = set(request.row_references_for_set(set_ref))
    anonymous = tuple(source for source in request.source_catalog.sources if source.id in rows)
    fields = sorted({SourceFieldBinding(source.id,field).ref for source in anonymous for field in source.fields}) if is_output else []
    if not fields:
        return {"type":"array","maxItems":0,"items":_closed_object({})}
    return {"type":"array","minItems":1 if rows == {source.id for source in anonymous} else 0,
        "items":output.RecordFieldRealizationOutput.schema({"name":_text(),"field_ref":{"enum":fields}})}


def _set_realization_schema(request, set_ref, branch_ids):
    available = tuple(request.row_references_for_set(set_ref))
    proxies = request.reference_proxy_options_for_set(set_ref)
    addresses = request.address_scope_options_for_set(set_ref)
    address_sources = {item.source_ref for item in addresses}
    normal = tuple(ref for ref in available if ref not in proxies and ref not in address_sources)
    common = {
        "branch_id": {"enum": list(branch_ids)},
        "mapping_basis": _text(),
    }
    variants = []
    if normal:
        variants.append(output.SetRealizationOutput.schema({
            **common,
            "rows_ref": {"enum": list(normal)},
            "record_fields": _record_fields_schema(request, set_ref),
        }))
    for source_ref, fields in proxies.items():
        proxy_variant = output.SetRealizationOutput.schema({
            **common,
            "rows_ref": {"enum": [source_ref]},
            "record_fields": {"type": "array", "maxItems": 0, "items": _closed_object({})},
            "reference_proxy_field_ref": {"enum": list(fields)},
        })
        proxy_variant["required"] = list(proxy_variant["properties"])
        variants.append(proxy_variant)
    for option in addresses:
        address_variant = output.SetRealizationOutput.schema({
            **common,
            "rows_ref": {"enum": [option.source_ref]},
            "record_fields": {"type": "array", "maxItems": 0, "items": _closed_object({})},
            "address_parameter_ref": {"enum": [option.parameter_ref]},
            "address_input_use_ref": {"enum": [option.input_use_ref]},
        })
        address_variant["required"] = list(address_variant["properties"])
        variants.append(address_variant)
    if not variants:
        raise ValueError("logical set has no source realization")
    return _exact_realizations(
        branch_ids, variants[0] if len(variants) == 1 else {"oneOf": variants}
    )


def build_unavailable_source_realization_schema(
    requirement_refs: tuple[str, ...],
) -> dict[str, object]:
    refs = tuple(dict.fromkeys(requirement_refs))
    if not refs:
        raise ValueError(
            "unavailable realization requires a declared requirement scope"
        )
    return output.SourceRealizationUnavailableOutput.schema(
        {
            "kind": {"enum": ["unavailable_source_realization"]},
            "unmet_requirement_refs": {
                "type": "array",
                "minItems": 1,
                "maxItems": len(refs),
                "items": {"enum": list(refs)},
            },
            "explanation": {"type": "string", "minLength": 1},
        }
    )


def build_semantic_source_realization_schema(
    request: SemanticSourceBindingRequest,
) -> dict[str, object]:
    branch_ids = tuple(item.branch_id for item in request.strategy.branches)
    source_refs = tuple(source.id for source in request.source_catalog.sources)
    set_refs = tuple(
        ref.token
        for ref in request.index.source_requirement_refs
        if isinstance(request.index.term_by_ref.get(ref), SetTerm)
    )
    association_refs = tuple(
        ref.token
        for ref in request.index.association_requirement_refs
        if isinstance(request.index.term_by_ref.get(ref), AssociationTerm)
    )
    return output.SourceRealizationOutput.schema(
        {
            "set_bindings": _closed_object(
                {
                    ref: _set_realization_schema(request, ref, branch_ids)
                    for ref in set_refs
                }
            ),
            "fact_bindings": _closed_object(
                {
                    ref: _fact_realizations_schema(
                        request,
                        fact_ref=ref,
                        branch_ids=branch_ids,
                        source_refs=source_refs,
                    )
                    for ref in request.model_authored_fact_refs
                }
            ),
            "association_bindings": _closed_object(
                {
                    ref: _exact_realizations(
                        branch_ids,
                        _association_realization_schema(
                            request=request,
                            association_ref=ref,
                            branch_ids=branch_ids,
                        ),
                    )
                    for ref in association_refs
                }
            ),
        }
    )



def build_semantic_source_binding_schema(
    request: SemanticSourceBindingRequest,
) -> dict[str, object]:
    """Bind inputs and population controls to an already realized source graph."""
    schema = output.SemanticSourceBindingOutput.schema(
        {
            "resolved_input_applications": _resolved_input_applications_schema(request),
            "finite_choice_applications": _finite_choice_applications_schema(request),
            "choice_requirement_applications": _choice_requirement_applications_schema(
                request
            ),
        }
    )
    return without_unreferenced_definitions(schema)


def _association_realization_schema(
    *,
    request: SemanticSourceBindingRequest,
    association_ref: str,
    branch_ids: tuple[str, ...],
) -> dict[str, object]:
    choices = tuple(
        dict.fromkeys(
            (choice.realization_ref, choice.reference_from_set_ref)
            for choice in association_choices(request, association_ref)
        )
    )
    variants = []
    for evidence, orientation in choices:
        variant = output.AssociationRealizationOutput.schema(
            {
                "branch_id": {"enum": list(branch_ids)},
                "mapping_basis": _text(),
                "realization_ref": {"enum": [evidence]},
                "reference_from_set_ref": {"enum": [orientation]},
                "field_pairs": {
                    "type": "array", "minItems": 1 if evidence is None else 0,
                    **({} if evidence is None else {"maxItems": 0}),
                    "items": output.AssociationFieldPairOutput.schema({
                        key: {"enum": [field.ref for field in request.source_catalog.field_bindings]}
                        for key in ("from_field_ref", "to_field_ref")
                    }),
                },
            }
        )
        if orientation is not None:
            variant["required"] = [
                "branch_id",
                "mapping_basis",
                "realization_ref",
                "reference_from_set_ref",
                "field_pairs",
            ]
        variants.append(variant)
    if not variants:
        raise ValueError("association has no structurally compatible row realization")
    return variants[0] if len(variants) == 1 else {"oneOf": variants}


def _fact_realizations_schema(
    request: SemanticSourceBindingRequest,
    *,
    fact_ref: str,
    branch_ids: tuple[str, ...],
    source_refs: tuple[str, ...],
) -> dict[str, object]:
    required_branches = request.required_returned_fact_branches().get(fact_ref, ())
    if not request.returned_field_refs_for_fact(fact_ref):
        if required_branches:
            raise ValueError("required fact has no returned-field realization")
        return {
            "type": "array",
            "minItems": 0,
            "maxItems": 0,
            "items": _closed_object({}),
        }
    result = _optional_realizations(
        branch_ids,
        _fact_realization_schema(
            request, fact_ref=fact_ref, branch_ids=branch_ids, source_refs=source_refs
        ),
    )
    result["minItems"] = len(required_branches)
    return result


def _fact_realization_schema(
    request: SemanticSourceBindingRequest,
    *,
    fact_ref: str,
    branch_ids: tuple[str, ...],
    source_refs: tuple[str, ...],
) -> dict[str, object]:
    field_refs_for_sources = request.returned_field_refs_for_fact(fact_ref)
    common = {
        "branch_id": {"enum": list(branch_ids)},
        "mapping_basis": _text(),
    }
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


def unapplied_input_application_schema(
    owner_ref: str, value_ref: str
) -> dict[str, object]:
    return output.UnappliedInputOutput.schema(
        {
            "kind": {"enum": ["no_request_application"]},
            "mapping_basis": _text(),
            "owner_ref": {"enum": [owner_ref]},
            "value_ref": {"enum": [value_ref]},
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
                "kind": {"enum": ["request_application"]},
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
    variants.extend(
        unapplied_input_application_schema(owner_ref, value_ref)
        for owner_ref in request.invocation_application_owner_refs
        for value_ref in request.unapplied_input_value_refs_for_owner(
            owner_ref, branch_id=branch_id
        )
    )
    # Multiple response row paths can expose the same endpoint parameter.
    # The authored value/target projection is still one legal choice.
    variants = list({repr(variant): variant for variant in variants}.values())
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


def _finite_choice_applications_schema(
    request: SemanticSourceBindingRequest,
) -> dict[str, object]:
    return _closed_object(
        {
            branch.branch_id: _branch_finite_choice_applications_schema(
                request,
                branch_id=branch.branch_id,
            )
            for branch in request.strategy.branches
        }
    )


def _branch_finite_choice_applications_schema(
    request: SemanticSourceBindingRequest,
    *,
    branch_id: str,
) -> dict[str, object]:
    return _closed_object(
        {
            owner_ref: (
                {
                    "anyOf": [
                        _finite_choice_owner_application_schema(options),
                        {"type": "null"},
                    ]
                }
                if owner_ref
                in {item.requirement_ref for item in request.index.boolean_requirements}
                else _finite_choice_owner_application_schema(options)
            )
            for owner_ref in request.invocation_application_owner_refs
            if (
                options := request.finite_choice_options_for_owner(
                    owner_ref,
                    branch_id=branch_id,
                )
            )
        }
    )


def _finite_choice_owner_application_schema(options) -> dict[str, object]:
    variants = [
        output.FiniteChoiceApplicationOutput.schema(
            {
                "application_basis": _text(),
                "surface_ref": {"enum": [surface.surface_ref]},
                "selected_choice_values": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": len(choices),
                    "items": {"enum": [choice.value for choice in choices]},
                },
            }
        )
        for surface, choices in options
    ]
    return variants[0] if len(variants) == 1 else {"oneOf": variants}


def _choice_requirement_applications_schema(request: SemanticSourceBindingRequest):
    from fervis.lookup.source_binding.choice_requirements import (
        requirement_choice_surfaces,
    )

    return _closed_object(
        {
            branch.branch_id: _closed_object(
                {
                    surface.surface_ref: _closed_object(
                        {
                            choice.value: output.ChoiceRequirementApplicationOutput.schema(
                                {
                                    "mapping_basis": _text(),
                                    "selected_by_requirements": {
                                        "type": "array",
                                        "uniqueItems": True,
                                        "maxItems": len(refs),
                                        "items": {"enum": list(refs)}
                                        if refs
                                        else {"type": "string"},
                                    },
                                }
                            )
                            for choice in surface.values
                            for refs in (
                                request.explicit_subject_requirement_refs(
                                    choice, branch_id=branch.branch_id
                                ),
                            )
                        }
                    )
                    for surface in requirement_choice_surfaces(
                        request, branch.branch_id
                    )
                }
            )
            for branch in request.strategy.branches
        }
    )


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


__all__ = [
    "build_semantic_source_binding_schema",
    "build_semantic_source_realization_schema",
    "build_unavailable_source_realization_schema",
]
