"""Revalidate user-supplied API arguments against their current source contract."""

from fervis.lookup.available_sources import AvailableSourceCatalog, source_value_literal
from fervis.lookup.clarification.context import clarification_response_ref
from fervis.lookup.clarification.model import (
    ClarificationOwnerResponse,
    SourceBindingCatalogInputResponse,
)
from fervis.lookup.answer_program.values import ValueProjectionKind
from .param_values import fact_value_parameter_projection
from .model import CatalogProvidedValue


def catalog_response_values(
    *,
    requested_fact_id: str,
    source_catalog: AvailableSourceCatalog,
    responses: tuple[ClarificationOwnerResponse, ...],
) -> tuple[CatalogProvidedValue, ...]:
    values = []
    targets = set()
    for response in responses:
        if (
            not isinstance(response, SourceBindingCatalogInputResponse)
            or response.requested_fact_id != requested_fact_id
        ):
            continue
        target = response.target
        try:
            source = source_catalog.source(target.row_source_id)
        except KeyError as exc:
            raise ValueError(
                "Catalog clarification source is no longer available"
            ) from exc
        parameter = next(
            (item for item in source.params if item.param_ref == target.param_ref), None
        )
        if parameter is None or (
            parameter.id,
            parameter.type.value,
            parameter.choices,
        ) != (target.param_id, target.value_type, target.choices):
            raise ValueError(
                "Catalog clarification target no longer matches its source"
            )
        identity = (source.id, parameter.param_ref)
        if identity in targets:
            raise ValueError("Catalog clarification repeats a supplied API argument")
        targets.add(identity)
        value_id = f"catalog_value:{response.response_id}"
        proof_ref = clarification_response_ref(response.response_id)
        typed_value = source_value_literal(
            value_ref=value_id,
            value=response.value,
            declared_type=parameter.type,
            label=parameter.name,
            source_ref=source.id,
            proof_refs=(proof_ref,),
        )
        fact_value_parameter_projection(
            typed_value,
            projection=ValueProjectionKind.WHOLE_VALUE,
            component_id=None,
            type_name=parameter.type.value,
            choices=parameter.choices,
        )
        values.append(
            CatalogProvidedValue(
                catalog_input_ref=response.clarification_id,
                target_ref=parameter.param_ref,
                value_id=value_id,
                typed_value=typed_value,
                certification_refs=(proof_ref,),
            )
        )
    return tuple(values)
