"""DRF serializer introspection for Fervis endpoint contracts."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import GenericAlias, NoneType, UnionType
from typing import TypeAlias, Union, get_args, get_origin, get_type_hints

from django.db import models
from django.db.models.options import Options
from rest_framework import serializers

from fervis.host_api.contracts import (
    EntityKeyComponentTargetContract,
    CandidateKeyAuthorityComponentContract,
    CandidateKeyAuthorityContract,
    CandidateKeyComponentContract,
    CandidateKeyContract,
    EntityReferenceComponentContract,
    EntityReferenceContract,
    ParameterContract,
    ResponseFieldContract,
)
from fervis.host_api.contracts.values import ContractValue


@dataclass(frozen=True)
class _ModelFieldBinding:
    output_path: str
    relation_model: type
    model: type
    model_field: models.Field
    output_type: str


@dataclass(frozen=True)
class SerializerInspection:
    response_fields: tuple[ResponseFieldContract, ...]
    response_schema: dict[str, "ContractValue"]
    model_field_bindings: tuple[_ModelFieldBinding, ...]

    @property
    def relation_model(self) -> type | None:
        models = tuple(
            dict.fromkeys(
                binding.relation_model for binding in self.model_field_bindings
            )
        )
        return models[0] if len(models) == 1 else None

    @property
    def candidate_keys(self) -> tuple[CandidateKeyContract, ...]:
        return _relation_keys_from_bindings(self.model_field_bindings)

    @property
    def entity_references(self) -> tuple[EntityReferenceContract, ...]:
        return _entity_references_from_bindings(self.model_field_bindings)

    @property
    def candidate_key_authorities(self) -> tuple[CandidateKeyAuthorityContract, ...]:
        authorities = tuple(
            authority
            for binding in self.model_field_bindings
            for authority in (_referenced_candidate_key_authority(binding),)
            if authority is not None
        )
        return tuple(dict.fromkeys(authorities))


_ModelCandidateKey: TypeAlias = tuple[str, tuple[models.Field, ...], bool]
_PythonAnnotation: TypeAlias = type | UnionType | GenericAlias | str | None


def relation_keys_from_serializer(
    serializer_class: type | None,
    *,
    model_context: type | None = None,
) -> tuple[CandidateKeyContract, ...]:
    inspection = inspect_response_serializer(
        serializer_class,
        model_context=model_context,
    )
    bindings = inspection.model_field_bindings
    return _relation_keys_from_bindings(bindings)


def _relation_keys_from_bindings(
    bindings: tuple[_ModelFieldBinding, ...],
) -> tuple[CandidateKeyContract, ...]:
    keys = tuple(
        key
        for model in dict.fromkeys(binding.relation_model for binding in bindings)
        for key in _relation_keys_for_model(model, bindings=bindings)
    )
    return tuple(dict.fromkeys(keys))


def _relation_keys_for_model(
    model: type,
    *,
    bindings: tuple[_ModelFieldBinding, ...],
) -> tuple[CandidateKeyContract, ...]:
    model_bindings = tuple(
        binding
        for binding in bindings
        if binding.relation_model is model and binding.model is model
    )
    keys: list[CandidateKeyContract] = []
    for key_id, model_fields, primary in _model_candidate_keys(model):
        key = _relation_key_contract(
            key_id=key_id,
            model=model,
            model_fields=model_fields,
            primary=primary,
            model_bindings=model_bindings,
        )
        if key is not None:
            keys.append(key)
    return tuple(keys)


def _relation_key_contract(
    *,
    key_id: str,
    model: type,
    model_fields: tuple[models.Field, ...],
    primary: bool,
    model_bindings: tuple[_ModelFieldBinding, ...],
) -> CandidateKeyContract | None:
    component_bindings = _bindings_for_model_fields(
        model_bindings,
        model_fields=model_fields,
    )
    if len(component_bindings) != len(model_fields):
        return None
    components = tuple(
        CandidateKeyComponentContract(
            component_id=_model_field_component_id(model_field),
            field_path=binding.output_path,
        )
        for model_field, binding in zip(
            model_fields,
            component_bindings,
            strict=True,
        )
    )
    return CandidateKeyContract(
        key_id=key_id,
        entity_kind=_model_identity_type(model),
        components=components,
        primary=primary,
        stable=True,
        context_field_paths=(),
    )


def entity_references_from_serializer(
    serializer_class: type | None,
    *,
    model_context: type | None = None,
) -> tuple[EntityReferenceContract, ...]:
    inspection = inspect_response_serializer(
        serializer_class,
        model_context=model_context,
    )
    return _entity_references_from_bindings(inspection.model_field_bindings)


def _entity_references_from_bindings(
    bindings: tuple[_ModelFieldBinding, ...],
) -> tuple[EntityReferenceContract, ...]:
    references: list[EntityReferenceContract] = []
    for binding in bindings:
        reference = _entity_reference_contract(
            binding,
            bindings=bindings,
        )
        if reference is not None:
            references.append(reference)
    return tuple(dict.fromkeys(references))


def _entity_reference_contract(
    binding: _ModelFieldBinding,
    *,
    bindings: tuple[_ModelFieldBinding, ...],
) -> EntityReferenceContract | None:
    if not _binding_has_scalar_value(binding):
        return None
    target_model = _related_model(binding.model_field)
    target_key = _related_target_field(binding.model_field)
    if target_model is None or not isinstance(target_key, models.Field):
        return None
    reference_id = f"{binding.output_path.replace('.', '_')}_reference"
    component = EntityReferenceComponentContract(
        target_component_id=_model_field_component_id(target_key),
        local_field_path=binding.output_path,
    )
    return EntityReferenceContract(
        reference_id=reference_id,
        target_entity_kind=_model_identity_type(target_model),
        target_key_id=_model_field_key_id(target_key),
        components=(component,),
        context_field_paths=_nested_reference_context_paths(
            binding,
            target_model=target_model,
            bindings=bindings,
        ),
    )


def _nested_reference_context_paths(
    binding: _ModelFieldBinding,
    *,
    target_model: type,
    bindings: tuple[_ModelFieldBinding, ...],
) -> tuple[str, ...]:
    parent_path, separator, _ = binding.output_path.rpartition(".")
    if not separator:
        return ()
    nested_prefix = f"{parent_path}."
    return tuple(
        other.output_path
        for other in bindings
        if other is not binding
        and other.model is target_model
        and other.output_path.startswith(nested_prefix)
        and _binding_has_scalar_value(other)
    )


def _related_target_field(model_field: models.Field) -> models.Field | None:
    if isinstance(model_field, models.ForeignKey):
        return model_field.target_field
    return None


def _referenced_candidate_key_authority(
    binding: _ModelFieldBinding,
) -> CandidateKeyAuthorityContract | None:
    if not _binding_has_scalar_value(binding):
        return None
    target_model = _related_model(binding.model_field)
    target_field = _related_target_field(binding.model_field)
    if target_model is None or target_field is None:
        return None
    component = CandidateKeyAuthorityComponentContract(
        component_id=_model_field_component_id(target_field),
        type=_model_field_type(target_field),
    )
    return CandidateKeyAuthorityContract(
        key_id=_model_field_key_id(target_field),
        entity_kind=_model_identity_type(target_model),
        components=(component,),
        primary=target_field.primary_key,
    )


def _model_field_key_id(model_field: models.Field) -> str:
    if model_field.primary_key:
        return "primary_key"
    return f"unique_{model_field.name}"


def inspect_response_serializer(
    serializer_class: type | None,
    *,
    model_context: type | None = None,
) -> SerializerInspection:
    if serializer_class is None:
        return SerializerInspection((), {}, ())
    serializer = _instantiate_serializer(serializer_class)
    if serializer is None:
        return SerializerInspection((), {}, ())
    fields: list[ResponseFieldContract] = []
    bindings: list[_ModelFieldBinding] = []
    schema = _inspect_serializer(
        serializer,
        fields,
        bindings,
        prefix="",
        model_context=_serializer_model(serializer_class) or model_context,
        relation_model=_serializer_model(serializer_class) or model_context,
    )
    return SerializerInspection(tuple(fields), schema, tuple(bindings))


def _inspect_serializer(
    serializer: serializers.Serializer,
    fields: list[ResponseFieldContract],
    bindings: list[_ModelFieldBinding],
    *,
    prefix: str,
    model_context: type | None,
    relation_model: type | None,
    parent_relation_binding: tuple[type, models.Field] | None = None,
    depth: int = 0,
) -> dict[str, ContractValue]:
    if depth > 4:
        return {"_truncated": True}
    model = _serializer_model(serializer.__class__) or model_context
    schema: dict[str, ContractValue] = {}
    for output_name, serializer_field in serializer.fields.items():
        output_path = f"{prefix}.{output_name}" if prefix else output_name
        nested = _nested_serializer(serializer_field)
        if nested is not None:
            related_model = _related_model_for_serializer_field(
                model,
                output_name=output_name,
                field=serializer_field,
            )
            field_type = (
                "array"
                if isinstance(serializer_field, serializers.ListSerializer)
                else "object"
            )
            fields.append(
                ResponseFieldContract(
                    name=output_name,
                    type=field_type,
                    path=output_path,
                )
            )
            nested_schema = _inspect_serializer(
                nested,
                fields,
                bindings,
                prefix=output_path,
                model_context=related_model,
                relation_model=(
                    related_model
                    if isinstance(serializer_field, serializers.ListSerializer)
                    else relation_model
                ),
                parent_relation_binding=(
                    None
                    if isinstance(serializer_field, serializers.ListSerializer)
                    else _nested_relation_binding(
                        model,
                        output_name=output_name,
                        field=serializer_field,
                    )
                ),
                depth=depth + 1,
            )
            schema[output_name] = (
                [nested_schema]
                if isinstance(serializer_field, serializers.ListSerializer)
                else nested_schema
            )
            continue
        response_field = _response_field(
            output_name,
            serializer_field,
            path=output_path,
        )
        fields.append(response_field)
        schema[output_name] = _schema_for_field(serializer_field, depth=depth)
        if isinstance(serializer_field, serializers.SerializerMethodField):
            continue
        if model is None:
            continue
        source_path = _serializer_field_source(output_name, serializer_field)
        owner_model, model_field = _resolve_model_field(model, source_path=source_path)
        if owner_model is None or model_field is None:
            continue
        owner_model, model_field = _nested_key_binding(
            owner_model,
            model_field,
            parent_relation_binding=parent_relation_binding,
        )
        if relation_model is None:
            continue
        bindings.append(
            _ModelFieldBinding(
                output_path=output_path,
                relation_model=relation_model,
                model=owner_model,
                model_field=model_field,
                output_type=response_field.type,
            )
        )
    return schema


def _nested_relation_binding(
    model: type | None,
    *,
    output_name: str,
    field: serializers.Field,
) -> tuple[type, models.Field] | None:
    if model is None:
        return None
    source_path = _serializer_field_source(output_name, field)
    owner_model, model_field = _resolve_model_field(model, source_path=source_path)
    if owner_model is None or model_field is None:
        return None
    if _related_model(model_field) is None:
        return None
    return owner_model, model_field


def _nested_key_binding(
    owner_model: type,
    model_field: models.Field,
    *,
    parent_relation_binding: tuple[type, models.Field] | None,
) -> tuple[type, models.Field]:
    if parent_relation_binding is None:
        return owner_model, model_field
    relation_owner, relation_field = parent_relation_binding
    if _related_target_field(relation_field) is model_field:
        return relation_owner, relation_field
    return owner_model, model_field


def _response_field(
    name: str,
    field: serializers.Field,
    *,
    path: str,
) -> ResponseFieldContract:
    return ResponseFieldContract(
        name=name,
        type=_field_type(field),
        path=path,
        description=str(getattr(field, "help_text", "") or ""),
        choices=_choices(field),
    )


def _nested_serializer(
    field: serializers.Field,
) -> serializers.Serializer | None:
    if isinstance(field, serializers.ListSerializer):
        child = getattr(field, "child", None)
        return child if isinstance(child, serializers.Serializer) else None
    return field if isinstance(field, serializers.Serializer) else None


def _serializer_field_source(
    output_name: str,
    field: serializers.Field,
) -> str:
    source = str(getattr(field, "source", "") or "")
    return output_name if not source or source == "*" else source


def _resolve_model_field(
    model: type,
    *,
    source_path: str,
) -> tuple[type | None, models.Field | None]:
    owner_model = model
    relation_binding: tuple[type, models.Field] | None = None
    segments = tuple(part for part in source_path.split(".") if part)
    for index, segment in enumerate(segments):
        meta = getattr(owner_model, "_meta", None)
        if meta is None:
            return None, None
        model_field = _model_field(meta, segment) or _model_field_by_attname(
            meta,
            segment,
        )
        if model_field is None:
            return None, None
        if index == len(segments) - 1:
            if relation_binding is not None:
                relation_owner, relation_field = relation_binding
                if _related_target_field(relation_field) is model_field:
                    return relation_owner, relation_field
            return owner_model, model_field
        related_model = _related_model(model_field)
        if related_model is None:
            return None, None
        relation_binding = (owner_model, model_field)
        owner_model = related_model
    return None, None


def _model_candidate_keys(
    model: type,
) -> tuple[_ModelCandidateKey, ...]:
    meta = getattr(model, "_meta", None)
    if meta is None:
        return ()
    keys: list[_ModelCandidateKey] = []
    primary_key = getattr(meta, "pk", None)
    if isinstance(primary_key, models.Field):
        keys.append(("primary_key", (primary_key,), True))
    else:
        primary_key = None
    keys.extend(_single_field_unique_keys(meta, primary_key=primary_key))
    keys.extend(_unique_together_keys(meta))
    keys.extend(_unique_constraint_keys(meta))
    return tuple(dict.fromkeys(keys))


def _single_field_unique_keys(
    meta: Options,
    *,
    primary_key: models.Field | None,
) -> tuple[_ModelCandidateKey, ...]:
    return tuple(
        (f"unique_{field.name}", (field,), False)
        for field in meta.fields
        if field.unique and not field.null and field is not primary_key
    )


def _unique_together_keys(meta: Options) -> tuple[_ModelCandidateKey, ...]:
    keys: list[_ModelCandidateKey] = []
    for names in tuple(meta.unique_together or ()):
        fields = _declared_model_fields(meta, names=tuple(str(name) for name in names))
        if _fields_form_total_key(fields):
            keys.append((f"unique_{'_'.join(names)}", fields, False))
    return tuple(keys)


def _unique_constraint_keys(meta: Options) -> tuple[_ModelCandidateKey, ...]:
    keys: list[_ModelCandidateKey] = []
    for constraint in meta.constraints:
        if not isinstance(constraint, models.UniqueConstraint):
            continue
        if constraint.condition is not None:
            continue
        names = tuple(str(name) for name in constraint.fields)
        fields = _declared_model_fields(meta, names=names)
        if not _fields_form_total_key(fields):
            continue
        key_id = str(constraint.name or f"unique_{'_'.join(names)}")
        keys.append((key_id, fields, False))
    return tuple(keys)


def _fields_form_total_key(fields: tuple[models.Field, ...]) -> bool:
    return bool(fields) and all(not field.null for field in fields)


def _declared_model_fields(
    meta: Options,
    *,
    names: tuple[str, ...],
) -> tuple[models.Field, ...]:
    fields = tuple(_model_field(meta, name) for name in names)
    if any(field is None for field in fields):
        return ()
    return tuple(field for field in fields if field is not None)


def _bindings_for_model_fields(
    bindings: tuple[_ModelFieldBinding, ...],
    *,
    model_fields: tuple[models.Field, ...],
) -> tuple[_ModelFieldBinding, ...]:
    selected: list[_ModelFieldBinding] = []
    for model_field in model_fields:
        matches = tuple(
            binding
            for binding in bindings
            if binding.model_field is model_field and _binding_has_scalar_value(binding)
        )
        if len(matches) != 1:
            return ()
        selected.append(matches[0])
    return tuple(selected)


def _binding_has_scalar_value(binding: _ModelFieldBinding) -> bool:
    return binding.output_type not in {"array", "json", "object"}


def _model_field_component_id(model_field: models.Field) -> str:
    return str(getattr(model_field, "attname", "") or getattr(model_field, "name", ""))


def _model_field_type(model_field: models.Field) -> str:
    if isinstance(model_field, models.UUIDField):
        return "uuid"
    if isinstance(model_field, models.BooleanField):
        return "boolean"
    if isinstance(model_field, models.DecimalField):
        return "decimal"
    if isinstance(model_field, models.FloatField):
        return "float"
    if isinstance(model_field, models.DateTimeField):
        return "datetime"
    if isinstance(model_field, models.DateField):
        return "date"
    if isinstance(model_field, models.TimeField):
        return "time"
    if isinstance(model_field, models.IntegerField):
        return "integer"
    return "string"


_FIELD_TYPE_MAP = {
    "BooleanField": "boolean",
    "CharField": "string",
    "ChoiceField": "choice",
    "DateField": "date",
    "DateTimeField": "datetime",
    "DecimalField": "decimal",
    "DictField": "object",
    "DurationField": "duration",
    "EmailField": "string",
    "FloatField": "float",
    "IntegerField": "integer",
    "JSONField": "json",
    "ListField": "array",
    "PrimaryKeyRelatedField": "pk",
    "ReadOnlyField": "any",
    "SerializerMethodField": "any",
    "SlugField": "string",
    "TimeField": "time",
    "URLField": "string",
    "UUIDField": "uuid",
}


def query_params_from_serializer(
    serializer_class: type | None,
    *,
    model_context: type | None = None,
    view_class: type | None = None,
) -> tuple[ParameterContract, ...]:
    if serializer_class is None:
        return ()

    instance = _instantiate_serializer(serializer_class)
    if instance is None:
        return ()

    params: list[ParameterContract] = []
    for name, field in instance.fields.items():
        params.append(
            ParameterContract(
                name=name,
                type=_field_type(field),
                required=bool(getattr(field, "required", False)),
                description=str(getattr(field, "help_text", "") or ""),
                choices=_choices(field),
                choice_labels=_choice_labels(field),
                default=_json_safe_default(getattr(field, "default", None)),
                source="query",
                entity_target=_query_param_entity_target(
                    name,
                    field,
                    model_context=model_context,
                    view_class=view_class,
                ),
            )
        )
    return tuple(params)


def path_param_entity_target(
    model: type | None,
    *,
    param_name: str,
) -> EntityKeyComponentTargetContract | None:
    identity = _path_param_identity(model, param_name=param_name)
    if identity is None:
        return None
    target_model, target_field = identity
    return EntityKeyComponentTargetContract(
        entity_kind=_model_identity_type(target_model),
        key_id=_model_field_key_id(target_field),
        component_id=_model_field_component_id(target_field),
    )


def path_param_candidate_key_authority(
    model: type | None,
    *,
    param_name: str,
) -> CandidateKeyAuthorityContract | None:
    identity = _path_param_identity(model, param_name=param_name)
    if identity is None:
        return None
    target_model, target_field = identity
    component = CandidateKeyAuthorityComponentContract(
        component_id=_model_field_component_id(target_field),
        type=_model_field_type(target_field),
    )
    return CandidateKeyAuthorityContract(
        key_id=_model_field_key_id(target_field),
        entity_kind=_model_identity_type(target_model),
        components=(component,),
        primary=target_field.primary_key,
    )


def _path_param_identity(
    model: type | None,
    *,
    param_name: str,
) -> tuple[type, models.Field] | None:
    if not isinstance(model, type):
        return None
    meta = getattr(model, "_meta", None)
    if not isinstance(meta, Options):
        return None
    model_field = _model_field(meta, param_name) or _model_field_by_attname(
        meta,
        param_name,
    )
    if model_field is None:
        return None
    return _model_field_identity(model_field)


def response_fields_from_serializer(
    serializer_class: type | None,
    *,
    model_context: type | None,
) -> tuple[ResponseFieldContract, ...]:
    inspection = inspect_response_serializer(
        serializer_class,
        model_context=model_context,
    )
    return inspection.response_fields


def conditional_response_roots_from_serializer(
    serializer_class: type | None,
) -> tuple[str, ...]:
    if serializer_class is None:
        return ()
    declared_roots = getattr(serializer_class, "fervis_conditional_response_roots", ())
    if isinstance(declared_roots, str):
        declared_roots = (declared_roots,)
    if not isinstance(declared_roots, (list, tuple, set, frozenset)):
        declared_roots = ()
    roots = {str(root).strip() for root in declared_roots if str(root).strip()}
    return tuple(sorted(roots))


def optional_full_response_projection_param_names(
    serializer_class: type | None,
    *,
    query_params: tuple[ParameterContract, ...],
) -> frozenset[str]:
    if serializer_class is None:
        return frozenset()
    serializer = _instantiate_serializer(serializer_class)
    if serializer is None:
        return frozenset()
    field_names = tuple(serializer.fields)
    if len(field_names) < 2:
        return frozenset()
    selected_field = field_names[0]
    projection_params: set[str] = set()
    for param in query_params:
        if param.required:
            continue
        try:
            projected = serializer_class(**{param.name: (selected_field,)})
        except (TypeError, ValueError):
            continue
        if not isinstance(projected, serializers.Serializer):
            continue
        if tuple(projected.fields) == (selected_field,):
            projection_params.add(param.name)
    return frozenset(projection_params)


def _instantiate_serializer(serializer_class: type) -> serializers.Serializer | None:
    try:
        return serializer_class()
    except Exception as exc:
        raise ValueError(
            f"Could not instantiate DRF serializer {serializer_class!r}."
        ) from exc


def _schema_for_field(field: serializers.Field, *, depth: int) -> ContractValue:
    if isinstance(field, serializers.ListSerializer):
        child = getattr(field, "child", None)
        if isinstance(child, serializers.Serializer):
            return ["object"]
        return [_field_type(child) if child is not None else "any"]
    if isinstance(field, serializers.Serializer):
        return "object"
    if isinstance(field, serializers.ListField):
        child = getattr(field, "child", None)
        if child is not None:
            return [_schema_for_field(child, depth=depth + 1)]
        return ["any"]
    choices = _choices(field)
    if choices:
        return {"type": _field_type(field), "choices": list(choices)}
    return _field_type(field)


def _field_type(field: serializers.Field) -> str:
    if isinstance(field, serializers.SerializerMethodField):
        method_type = _serializer_method_field_type(field)
        if method_type:
            return method_type
    return _FIELD_TYPE_MAP.get(field.__class__.__name__, "any")


def _serializer_method_field_type(field: serializers.SerializerMethodField) -> str:
    parent = getattr(field, "parent", None)
    method_name = str(
        getattr(field, "method_name", "") or f"get_{getattr(field, 'field_name', '')}"
    )
    method = getattr(parent, method_name, None)
    if not callable(method):
        return ""
    annotation = _return_annotation(method)
    return _python_type_name(annotation)


def _return_annotation(
    method: Callable[..., _PythonAnnotation],
) -> _PythonAnnotation:
    try:
        return get_type_hints(method).get("return")
    except Exception:
        return getattr(method, "__annotations__", {}).get("return")


def _python_type_name(annotation: _PythonAnnotation) -> str:
    origin = get_origin(annotation)
    if origin in {Union, UnionType}:
        members = tuple(item for item in get_args(annotation) if item is not NoneType)
        if len(members) == 1:
            return _python_type_name(members[0])
        return ""
    if annotation is bool:
        return "boolean"
    if annotation is str:
        return "string"
    if annotation is int:
        return "integer"
    if annotation is float:
        return "float"
    if annotation is dict:
        return "object"
    if annotation in {list, tuple}:
        return "array"
    return ""


def _choices(field: serializers.Field) -> tuple[str, ...]:
    if not isinstance(field, serializers.ChoiceField):
        return ()
    choices = getattr(field, "choices", None)
    if not choices:
        return ()
    return tuple(str(key) for key in choices.keys())


def _choice_labels(field: serializers.Field) -> dict[str, str]:
    if not isinstance(field, serializers.ChoiceField):
        return {}
    choices = getattr(field, "choices", None)
    if not choices:
        return {}
    return {
        str(key): str(value) for key, value in choices.items() if str(key) != str(value)
    }


def _json_safe_default(
    value: ContractValue | Callable[[], ContractValue],
) -> ContractValue:
    if value is serializers.empty:
        return None
    if callable(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _query_param_entity_target(
    param_name: str,
    field: serializers.Field,
    *,
    model_context: type | None,
    view_class: type | None,
) -> EntityKeyComponentTargetContract | None:
    queryset = getattr(field, "queryset", None)
    model = getattr(queryset, "model", None)
    if isinstance(model, type):
        target_field = _query_param_target_field(field, model=model)
        if target_field is not None:
            return _entity_target_contract(model=model, target_field=target_field)
    filtered_field = _declared_query_param_field(
        param_name,
        model_context=model_context,
        view_class=view_class,
    )
    if filtered_field is None:
        return None
    identity = _model_field_identity(filtered_field)
    if identity is None:
        return None
    target_model, target_field = identity
    return _entity_target_contract(model=target_model, target_field=target_field)


def _entity_target_contract(
    *,
    model: type,
    target_field: models.Field,
) -> EntityKeyComponentTargetContract:
    return EntityKeyComponentTargetContract(
        entity_kind=_model_identity_type(model),
        key_id=_model_field_key_id(target_field),
        component_id=_model_field_component_id(target_field),
    )


def _declared_query_param_field(
    param_name: str,
    *,
    model_context: type | None,
    view_class: type | None,
) -> models.Field | None:
    if not isinstance(model_context, type) or not isinstance(view_class, type):
        return None
    fields = tuple(
        model_field
        for filterset_class in _declared_filterset_classes(
            view_class,
            model_context=model_context,
        )
        for model_field in (
            _declared_filter_field(
                filterset_class,
                param_name=param_name,
                model_context=model_context,
            ),
        )
        if model_field is not None
    )
    unique_fields = tuple(dict.fromkeys(fields))
    return unique_fields[0] if len(unique_fields) == 1 else None


def _declared_filterset_classes(
    view_class: type,
    *,
    model_context: type,
) -> tuple[type, ...]:
    view = view_class()
    model_manager = getattr(model_context, "_default_manager", None)
    if model_manager is None:
        return ()
    queryset = model_manager.none()
    filtersets: list[type] = []
    for backend_class in tuple(getattr(view_class, "filter_backends", ()) or ()):
        backend = backend_class()
        get_filterset_class = getattr(backend, "get_filterset_class", None)
        if not callable(get_filterset_class):
            continue
        filterset_class = get_filterset_class(view, queryset)
        if isinstance(filterset_class, type):
            filtersets.append(filterset_class)
    return tuple(dict.fromkeys(filtersets))


def _declared_filter_field(
    filterset_class: type,
    *,
    param_name: str,
    model_context: type,
) -> models.Field | None:
    filterset_options = getattr(filterset_class, "_meta", None)
    if getattr(filterset_options, "model", None) is not model_context:
        return None
    filters = getattr(filterset_class, "base_filters", None)
    if not isinstance(filters, Mapping):
        return None
    declared_filter = filters.get(param_name)
    if declared_filter is None:
        return None
    if str(getattr(declared_filter, "lookup_expr", "exact")) != "exact":
        return None
    if bool(getattr(declared_filter, "exclude", False)):
        return None
    if getattr(declared_filter, "method", None) is not None:
        return None
    field_name = str(getattr(declared_filter, "field_name", "") or "")
    model_options = getattr(model_context, "_meta", None)
    if not isinstance(model_options, Options):
        return None
    return _model_field(model_options, field_name) or _model_field_by_attname(
        model_options,
        field_name,
    )


def _model_field_identity(
    model_field: models.Field,
) -> tuple[type, models.Field] | None:
    target_model = _related_model(model_field)
    target_field = _related_target_field(model_field)
    if target_model is not None and target_field is not None:
        return target_model, target_field
    owner_model = getattr(model_field, "model", None)
    if not isinstance(owner_model, type):
        return None
    if not (model_field.primary_key or model_field.unique):
        return None
    return owner_model, model_field


def _query_param_target_field(
    field: serializers.Field,
    *,
    model: type,
) -> models.Field | None:
    meta = getattr(model, "_meta", None)
    if not isinstance(meta, Options):
        return None
    if isinstance(field, serializers.SlugRelatedField):
        return _model_field(meta, str(field.slug_field))
    primary_key = meta.pk
    return primary_key if isinstance(primary_key, models.Field) else None


def _serializer_model(serializer_class: type | None) -> type | None:
    meta = getattr(serializer_class, "Meta", None)
    return getattr(meta, "model", None)


def _related_model_for_serializer_field(
    model: type | None,
    *,
    output_name: str,
    field: serializers.Field,
) -> type | None:
    if model is None:
        return None
    raw_source = str(getattr(field, "source", "") or "")
    source = output_name if not raw_source or raw_source == "*" else raw_source
    return _related_model_for_model_path(model=model, source_path=source)


def _related_model_for_model_path(
    *,
    model: type,
    source_path: str,
) -> type | None:
    meta = getattr(model, "_meta", None)
    if meta is None:
        return None
    if "." in source_path:
        relation_name, remainder = source_path.split(".", 1)
        relation_field = _model_field(meta, relation_name)
        related_model = _related_model(relation_field)
        if related_model is None:
            return None
        return _related_model_for_model_path(
            model=related_model,
            source_path=remainder,
        )
    field = _model_field(meta, source_path) or _model_field_by_attname(
        meta, source_path
    )
    return _related_model(field)


def _model_field(meta: Options, name: str) -> models.Field | None:
    try:
        field = meta.get_field(name)
    except Exception:
        return None
    return field if isinstance(field, models.Field) else None


def _model_field_by_attname(meta: Options, attname: str) -> models.Field | None:
    for field in meta.fields:
        if str(getattr(field, "attname", "") or "") == attname:
            return field
    return None


def _related_model(field: models.Field | None) -> type | None:
    remote = getattr(field, "remote_field", None)
    model = getattr(remote, "model", None)
    return model if isinstance(model, type) else None


def _model_identity_type(model: type) -> str:
    object_name = str(
        getattr(getattr(model, "_meta", None), "object_name", "")
        or getattr(model, "__name__", "")
    )
    return "_".join(_camel_words(object_name).split()) or str(
        getattr(getattr(model, "_meta", None), "model_name", "") or ""
    )


def _camel_words(value: str) -> str:
    parts = re.findall(r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|\d+", str(value or ""))
    return " ".join(part.lower() for part in parts if part)
