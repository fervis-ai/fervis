"""One shared prompt surface for named-reference resolver candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from fervis.lookup.grounding.model import GroundingRequest, InputBindingOption
from fervis.lookup.relation_catalog.parameter_values import (
    CatalogParameterValueError,
    CatalogScalarParameterValue,
    parse_catalog_parameter_text,
)
from fervis.lookup.relation_catalog import (
    EndpointRead,
    RelationCatalog,
)
from fervis.lookup.fact_plan.row_sources import (
    RowSource,
    RowSourceField,
    RowSourceParam,
    RowSourceParamSemantics,
    RowSourceValueType,
)
from fervis.lookup.turn_prompts.projections.response_shape import (
    ApiReadResponseShapeProjector,
)


@dataclass(frozen=True)
class ResolverOptionSurface:
    """Catalog authority shown for one resolver read and canonical result."""

    option: InputBindingOption
    read: EndpointRead
    source: RowSource
    request_parameters: tuple[RowSourceParam, ...]
    response_match_fields: tuple[RowSourceField, ...]

    def prompt_payload(self) -> dict[str, object]:
        candidate = self.option.candidate
        read_payload = ApiReadResponseShapeProjector(self.read).prompt_payload(
            row_path_ids=(self.source.row_path_id,),
        )
        read_payload["row_source_id"] = self.source.id
        shared_input_params = cast(
            list[dict[str, object]], read_payload["input_params"]
        )
        shared_params_by_ref = {
            str(parameter["param_ref"]): parameter
            for parameter in shared_input_params
        }
        read_payload["input_params"] = [
            _with_row_source_parameter_overlay(
                shared_params_by_ref[parameter.param_ref],
                parameter,
            )
            for parameter in self.request_parameters
        ]
        selected_field_paths = {field.path for field in self.source.fields}
        response_rows = cast(list[dict[str, object]], read_payload["response_rows"])
        for row in response_rows:
            fields = cast(list[dict[str, object]], row["fields"])
            row["fields"] = [
                field for field in fields if field["path"] in selected_field_paths
            ]
        return {
            "binding_option_id": self.option.id,
            "resource_type": candidate.entity_kind,
            "api_read": read_payload,
            "canonical_result": {
                "entity_kind": candidate.entity_kind,
                "key_id": candidate.key_id,
                "components": [
                    {
                        "component_id": component.component_id,
                        "field_path": self.source_field(component.field_ref).path,
                    }
                    for component in candidate.key_components
                ],
            },
        }

    def source_field(self, field_ref: str) -> RowSourceField:
        for field in self.source.fields:
            if field.field_ref == field_ref:
                return field
        raise ValueError("resolver candidate references an unknown response field")

    def parameter(self, param_ref: str) -> RowSourceParam:
        for parameter in self.selectable_request_parameters:
            if parameter.param_ref == param_ref:
                return parameter
        raise ValueError("resolver review selected an unavailable request parameter")

    @property
    def selectable_request_parameters(self) -> tuple[RowSourceParam, ...]:
        return tuple(
            parameter
            for parameter in self.request_parameters
            if parameter.semantics is not RowSourceParamSemantics.RESPONSE_SHAPE
        )

    def compiled_request_value(
        self,
        param_ref: str,
        *,
        lookup_text: str,
    ) -> tuple[RowSourceParam, CatalogScalarParameterValue]:
        parameter = self.parameter(param_ref)
        try:
            value = parse_catalog_parameter_text(
                lookup_text,
                type_name=parameter.type.value,
                choices=parameter.choices,
            )
        except CatalogParameterValueError as exc:
            raise ValueError(
                "resolver request parameter does not accept the lookup text"
            ) from exc
        return parameter, value

    def compatible_request_parameters(
        self,
        *,
        lookup_text: str,
    ) -> tuple[RowSourceParam, ...]:
        return tuple(
            parameter
            for parameter in self.selectable_request_parameters
            if self._request_parameter_accepts(
                parameter,
                lookup_text=lookup_text,
            )
        )

    def required_request_parameters_accept(
        self,
        *,
        lookup_text: str,
    ) -> bool:
        return all(
            not parameter.required
            or parameter.default is not None
            or self._request_parameter_accepts(
                parameter,
                lookup_text=lookup_text,
            )
            for parameter in self.request_parameters
        )

    @staticmethod
    def _request_parameter_accepts(
        parameter: RowSourceParam,
        *,
        lookup_text: str,
    ) -> bool:
        try:
            parse_catalog_parameter_text(
                lookup_text,
                type_name=parameter.type.value,
                choices=parameter.choices,
            )
        except CatalogParameterValueError:
            return False
        return True

    def match_field(self, field_path: str) -> RowSourceField:
        for field in self.response_match_fields:
            if field.path == field_path:
                return field
        raise ValueError("resolver review selected an unavailable response field")

    def compiled_match_value(
        self,
        field_path: str,
        *,
        lookup_text: str,
    ) -> CatalogScalarParameterValue:
        field = self.match_field(field_path)
        try:
            return parse_catalog_parameter_text(
                lookup_text,
                type_name=field.type.value,
                choices=field.choices,
            )
        except CatalogParameterValueError as exc:
            raise ValueError(
                "resolver response field does not accept the lookup text"
            ) from exc


def resolver_option_surface(
    request: GroundingRequest,
    option: InputBindingOption,
) -> ResolverOptionSurface:
    return resolver_option_surface_from_catalog(request.resolver_catalog, option)


def resolver_option_surface_from_catalog(
    catalog: RelationCatalog,
    option: InputBindingOption,
) -> ResolverOptionSurface:
    read = catalog.read(option.candidate.resolver_read_id)
    source = option.candidate.resolver_source
    related_resource_field_ids = source.related_resource_field_ids
    return ResolverOptionSurface(
        option=option,
        read=read,
        source=source,
        request_parameters=source.params,
        response_match_fields=tuple(
            field
            for field in source.fields
            if field.id not in related_resource_field_ids
            and _field_supports_exact_match(field)
        ),
    )


def _field_supports_exact_match(field: RowSourceField) -> bool:
    return field.type not in {
        RowSourceValueType.ARRAY,
        RowSourceValueType.JSON,
        RowSourceValueType.LIST,
        RowSourceValueType.OBJECT,
    }


def _with_row_source_parameter_overlay(
    shared_payload: dict[str, object],
    parameter: RowSourceParam,
) -> dict[str, object]:
    payload = dict(shared_payload)
    if parameter.default is not None:
        payload["default"] = parameter.default
    else:
        payload.pop("default", None)
    if parameter.default_source:
        payload["default_source"] = parameter.default_source
    else:
        payload.pop("default_source", None)
    if parameter.semantics is not RowSourceParamSemantics.OPAQUE_QUERY_PARAM:
        payload["semantics"] = parameter.semantics.value
    else:
        payload.pop("semantics", None)
    return payload


__all__ = [
    "ResolverOptionSurface",
    "resolver_option_surface",
    "resolver_option_surface_from_catalog",
]
