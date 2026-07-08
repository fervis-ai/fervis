"""DRF serializer introspection for Fervis endpoint contracts."""

from __future__ import annotations

import re
from types import NoneType, UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

from rest_framework import serializers

from fervis.host_api.contracts import (
    ParameterContract,
    ResponseFieldContract,
)


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
                identity=_query_param_identity(name=name, field=field),
            )
        )
    return tuple(params)


def response_fields_from_serializer(
    serializer_class: type | None,
    *,
    model_context: type | None,
) -> tuple[ResponseFieldContract, ...]:
    if serializer_class is None:
        return ()

    instance = _instantiate_serializer(serializer_class)
    if instance is None:
        return ()

    fields: list[ResponseFieldContract] = []
    _collect_response_fields(
        instance,
        fields,
        prefix="",
        model_context=model_context,
    )
    return tuple(fields)


def response_schema_from_serializer(
    serializer_class: type | None,
) -> dict[str, Any]:
    if serializer_class is None:
        return {}

    instance = _instantiate_serializer(serializer_class)
    if instance is None:
        return {}

    return _schema_for_serializer(instance)


def conditional_response_roots_from_serializer(
    serializer_class: type | None,
) -> tuple[str, ...]:
    if serializer_class is None:
        return ()
    declared_roots = getattr(
        serializer_class, "fervis_conditional_response_roots", ()
    )
    if isinstance(declared_roots, str):
        declared_roots = (declared_roots,)
    if not isinstance(declared_roots, (list, tuple, set, frozenset)):
        declared_roots = ()
    roots = {str(root).strip() for root in declared_roots if str(root).strip()}
    return tuple(sorted(roots))


def _instantiate_serializer(serializer_class: type) -> serializers.Serializer | None:
    try:
        return serializer_class()
    except Exception as exc:
        raise ValueError(
            f"Could not instantiate DRF serializer {serializer_class!r}."
        ) from exc


def _collect_response_fields(
    serializer: serializers.Serializer,
    fields: list[ResponseFieldContract],
    *,
    prefix: str,
    model_context: type | None,
    depth: int = 0,
) -> None:
    if depth > 4:
        return
    model = _serializer_model(serializer.__class__) or model_context
    for name, field in serializer.fields.items():
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(field, serializers.ListSerializer):
            fields.append(ResponseFieldContract(name=name, type="array", path=path))
            child = getattr(field, "child", None)
            if isinstance(child, serializers.Serializer):
                _collect_response_fields(
                    child,
                    fields,
                    prefix=path,
                    model_context=_related_model_for_serializer_field(
                        model,
                        output_name=name,
                        field=field,
                    ),
                    depth=depth + 1,
                )
            continue
        if isinstance(field, serializers.Serializer):
            fields.append(ResponseFieldContract(name=name, type="object", path=path))
            _collect_response_fields(
                field,
                fields,
                prefix=path,
                model_context=_related_model_for_serializer_field(
                    model,
                    output_name=name,
                    field=field,
                ),
                depth=depth + 1,
            )
            continue
        fields.append(
            ResponseFieldContract(
                name=name,
                type=_field_type(field),
                path=path,
                description=str(getattr(field, "help_text", "") or ""),
                choices=_choices(field),
                identity=_field_identity(
                    model=model,
                    output_name=name,
                    field=field,
                ),
            )
        )


def _schema_for_serializer(
    serializer: serializers.Serializer,
    *,
    depth: int = 0,
) -> dict[str, Any]:
    if depth > 4:
        return {"_truncated": True}
    return {
        name: _schema_for_field(field, depth=depth)
        for name, field in serializer.fields.items()
    }


def _schema_for_field(field: serializers.Field, *, depth: int) -> Any:
    if isinstance(field, serializers.ListSerializer):
        child = getattr(field, "child", None)
        if isinstance(child, serializers.Serializer):
            return [_schema_for_serializer(child, depth=depth + 1)]
        return [_field_type(child) if child is not None else "any"]
    if isinstance(field, serializers.Serializer):
        return _schema_for_serializer(field, depth=depth + 1)
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
    annotation = _return_annotation(method)
    return _python_type_name(annotation)


def _return_annotation(method: Any) -> Any:
    try:
        return get_type_hints(method).get("return")
    except Exception:
        return getattr(method, "__annotations__", {}).get("return")


def _python_type_name(annotation: Any) -> str:
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


def _json_safe_default(value: Any) -> Any:
    if value is serializers.empty:
        return None
    if callable(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _query_param_identity(name: str, field: serializers.Field) -> dict[str, Any]:
    explicit = getattr(field, "fervis_identity", None)
    if isinstance(explicit, dict):
        entity_ref = str(explicit.get("entityRef") or "")
        id_field = str(explicit.get("idField") or name)
        if entity_ref and id_field:
            return {"entityRef": entity_ref, "idField": id_field, "primaryKey": True}
    queryset = getattr(field, "queryset", None)
    model = getattr(queryset, "model", None)
    if isinstance(model, type):
        return _identity_payload(model=model, id_field=name)
    return {}


def _serializer_model(serializer_class: type | None) -> type | None:
    meta = getattr(serializer_class, "Meta", None)
    return getattr(meta, "model", None)


def _field_identity(
    *,
    model: type | None,
    output_name: str,
    field: serializers.Field,
) -> dict[str, Any]:
    explicit = _explicit_identity(field, output_name=output_name)
    if explicit:
        return explicit
    if model is None:
        return {}
    raw_source = str(getattr(field, "source", "") or "")
    source = output_name if not raw_source or raw_source == "*" else raw_source
    return _identity_for_model_path(
        model=model,
        source_path=source,
        output_name=output_name,
    )


def _explicit_identity(
    field: serializers.Field,
    *,
    output_name: str,
) -> dict[str, Any]:
    explicit = getattr(field, "fervis_identity", None)
    if not isinstance(explicit, dict):
        return {}
    entity_ref = str(explicit.get("entityRef") or "")
    id_field = str(explicit.get("idField") or output_name)
    if not entity_ref or not id_field:
        return {}
    return {"entityRef": entity_ref, "idField": id_field, "primaryKey": True}


def _identity_for_model_path(
    *,
    model: type,
    source_path: str,
    output_name: str,
) -> dict[str, Any]:
    meta = getattr(model, "_meta", None)
    if meta is None:
        return {}
    if "." in source_path:
        relation_name, remainder = source_path.split(".", 1)
        relation_field = _model_field(meta, relation_name)
        related_model = _related_model(relation_field)
        if related_model is None:
            return {}
        return _identity_for_model_path(
            model=related_model,
            source_path=remainder,
            output_name=output_name,
        )
    direct_field = _model_field(meta, source_path)
    if direct_field is not None:
        if bool(getattr(direct_field, "primary_key", False)):
            return _identity_payload(model=model, id_field=output_name)
        related_model = _related_model(direct_field)
        if related_model is not None:
            return _identity_payload(model=related_model, id_field=output_name)
    attname_field = _model_field_by_attname(meta, source_path)
    if attname_field is not None:
        if bool(getattr(attname_field, "primary_key", False)):
            return _identity_payload(model=model, id_field=output_name)
        related_model = _related_model(attname_field)
        if related_model is not None:
            return _identity_payload(model=related_model, id_field=output_name)
    return {}


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
    field = _model_field(meta, source_path) or _model_field_by_attname(meta, source_path)
    return _related_model(field)


def _model_field(meta: Any, name: str) -> Any:
    try:
        return meta.get_field(name)
    except Exception:
        return None


def _model_field_by_attname(meta: Any, attname: str) -> Any:
    for field in tuple(getattr(meta, "fields", ()) or ()):
        if str(getattr(field, "attname", "") or "") == attname:
            return field
    return None


def _related_model(field: Any) -> type | None:
    remote = getattr(field, "remote_field", None)
    model = getattr(remote, "model", None)
    return model if isinstance(model, type) else None


def _identity_payload(*, model: type, id_field: str) -> dict[str, Any]:
    return {
        "entityRef": _model_identity_type(model),
        "idField": id_field,
        "primaryKey": True,
    }


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
