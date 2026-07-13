"""Canonical, portable codec and content identity for answer programs."""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from enum import Enum
import hashlib
import json
import math
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

from . import (
    capability_contracts,
    model,
    operations,
    relations,
    result_projection,
    values,
)
from fervis.lookup.answer_program.model import (
    ANSWER_PROGRAM_SCHEMA_REVISION,
    AnswerProgram,
    ProgramCompatibility,
)
from fervis.lookup.answer_program.errors import AnswerProgramContractError
from fervis.lookup.answer_program.values import BindingPatch, BindingSet
from fervis.lookup.question_contract import model as question_contract_model
from fervis.lookup import question_inputs


def canonicalize_answer_program(program: AnswerProgram) -> AnswerProgram:
    """Normalize declaration order without changing graph execution order."""

    return replace(
        program,
        fact_template=tuple(sorted(program.fact_template, key=lambda item: item.id)),
        fulfillment=tuple(
            sorted(
                program.fulfillment,
                key=lambda item: (
                    item.requested_fact_id,
                    item.answer_output_id,
                    item.result_output_id,
                ),
            )
        ),
        parameters=tuple(sorted(program.parameters, key=lambda item: item.id)),
        capabilities=tuple(sorted(program.capabilities, key=lambda item: item.id)),
        relations=tuple(sorted(program.relations, key=lambda item: item.id)),
        result_projection=replace(
            program.result_projection,
            relation_outputs=tuple(
                sorted(
                    program.result_projection.relation_outputs, key=lambda item: item.id
                )
            ),
            scalar_outputs=tuple(
                sorted(
                    program.result_projection.scalar_outputs, key=lambda item: item.id
                )
            ),
        ),
        compatibility=_canonical_compatibility(program.compatibility),
    )


def canonical_answer_program_payload(program: AnswerProgram) -> dict[str, Any]:
    canonical = canonicalize_answer_program(program)
    return {
        "schema_revision": ANSWER_PROGRAM_SCHEMA_REVISION,
        "program": _encode(canonical, registered_types_only=True),
    }


def canonical_answer_program_json(program: AnswerProgram) -> str:
    return json.dumps(
        canonical_answer_program_payload(program),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def answer_program_id(program: AnswerProgram) -> str:
    digest = hashlib.sha256(canonical_answer_program_json(program).encode()).hexdigest()
    return f"ap_{digest}"


def canonical_binding_set_json(bindings: BindingSet) -> str:
    return json.dumps(
        {
            "schema_revision": ANSWER_PROGRAM_SCHEMA_REVISION,
            "bindings": _encode(bindings, registered_types_only=True),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def decode_binding_set(payload: str) -> BindingSet:
    raw = json.loads(payload)
    if not isinstance(raw, dict) or set(raw) != {"schema_revision", "bindings"}:
        raise ValueError("binding-set payload fields do not match schema")
    if raw["schema_revision"] != ANSWER_PROGRAM_SCHEMA_REVISION:
        raise ValueError("unsupported binding-set schema revision")
    return _decode_as(raw["bindings"], BindingSet)


def canonical_binding_patch_json(patch: BindingPatch) -> str:
    canonical = replace(
        patch,
        operations=tuple(
            sorted(patch.operations, key=lambda operation: operation.parameter_id)
        ),
        provenance_refs=tuple(sorted(set(patch.provenance_refs))),
    )
    return json.dumps(
        {
            "schema_revision": ANSWER_PROGRAM_SCHEMA_REVISION,
            "patch": _encode(canonical, registered_types_only=True),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def binding_patch_id(patch: BindingPatch) -> str:
    digest = hashlib.sha256(canonical_binding_patch_json(patch).encode()).hexdigest()
    return f"bp_{digest}"


def decode_binding_patch(payload: str) -> BindingPatch:
    raw = json.loads(payload)
    if not isinstance(raw, dict) or set(raw) != {"schema_revision", "patch"}:
        raise ValueError("binding-patch payload fields do not match schema")
    if raw["schema_revision"] != ANSWER_PROGRAM_SCHEMA_REVISION:
        raise ValueError("unsupported binding-patch schema revision")
    return _decode_as(raw["patch"], BindingPatch)


def decode_answer_program(payload: dict[str, Any] | str) -> AnswerProgram:
    try:
        raw = json.loads(payload) if isinstance(payload, str) else payload
        if not isinstance(raw, dict) or set(raw) != {"schema_revision", "program"}:
            raise ValueError("answer-program payload fields do not match schema")
        if raw.get("schema_revision") != ANSWER_PROGRAM_SCHEMA_REVISION:
            raise ValueError("unsupported answer-program schema revision")
        decoded = _decode_as(raw.get("program"), AnswerProgram)
        return canonicalize_answer_program(decoded)
    except AnswerProgramContractError:
        raise
    except (TypeError, ValueError) as exc:
        raise AnswerProgramContractError(
            "invalid_answer_program",
            "answer program does not match the canonical executable contract",
        ) from exc


def canonical_contract_fingerprint(contract: object) -> str:
    payload = json.dumps(
        _encode(contract),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _canonical_compatibility(
    compatibility: ProgramCompatibility,
) -> ProgramCompatibility:
    return replace(
        compatibility,
        function_semantics=tuple(
            sorted(
                compatibility.function_semantics,
                key=lambda item: item.function_key,
            )
        ),
        source_contracts=tuple(
            sorted(
                compatibility.source_contracts,
                key=lambda item: (item.kind.value, item.source_id),
            )
        ),
    )


def _encode(value: Any, *, registered_types_only: bool = False) -> Any:
    if isinstance(value, Enum):
        return {"$enum": _type_key(type(value)), "value": value.value}
    if isinstance(value, float) and not math.isfinite(value):
        raise TypeError("canonical floats must be finite")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        item_type = type(value)
        if (
            registered_types_only
            and _DATACLASS_TYPES.get(_type_key(item_type)) is not item_type
        ):
            raise AnswerProgramContractError(
                "invalid_answer_program",
                "only canonical contract types can be encoded",
            )
        annotations = get_type_hints(item_type)
        return {
            "$type": _type_key(item_type),
            "fields": {
                item.name: _encode_as(
                    getattr(value, item.name),
                    annotations[item.name],
                    registered_types_only=registered_types_only,
                )
                for item in fields(value)
            },
        }
    if isinstance(value, tuple):
        return {
            "$tuple": [
                _encode(item, registered_types_only=registered_types_only)
                for item in value
            ]
        }
    if isinstance(value, list):
        return {
            "$list": [
                _encode(item, registered_types_only=registered_types_only)
                for item in value
            ]
        }
    if isinstance(value, dict):
        return {
            "$map": [
                [
                    _encode(key, registered_types_only=registered_types_only),
                    _encode(item, registered_types_only=registered_types_only),
                ]
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            ]
        }
    raise TypeError(f"unsupported canonical value {type(value).__name__}")


def _encode_as(
    value: Any,
    expected: Any,
    *,
    registered_types_only: bool,
) -> Any:
    """Encode a value through the type declared by its owning contract field."""

    if expected is Any or expected is object:
        return _encode_json_value(value)
    origin = get_origin(expected)
    if origin in {Union, UnionType}:
        errors: list[TypeError] = []
        for variant in get_args(expected):
            try:
                return _encode_as(
                    value,
                    variant,
                    registered_types_only=registered_types_only,
                )
            except TypeError as exc:
                errors.append(exc)
        raise TypeError("value does not match its declared union") from errors[-1]
    if origin is tuple:
        if type(value) is not tuple:
            raise TypeError("canonical tuple field must be a tuple")
        arguments = get_args(expected)
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            item_types = (arguments[0],) * len(value)
        elif len(arguments) == len(value):
            item_types = arguments
        else:
            raise TypeError("canonical fixed tuple length does not match schema")
        return {
            "$tuple": [
                _encode_as(
                    item,
                    item_type,
                    registered_types_only=registered_types_only,
                )
                for item, item_type in zip(value, item_types, strict=True)
            ]
        }
    if origin is list:
        if type(value) is not list:
            raise TypeError("canonical list field must be a list")
        item_type = get_args(expected)[0]
        return {
            "$list": [
                _encode_as(
                    item,
                    item_type,
                    registered_types_only=registered_types_only,
                )
                for item in value
            ]
        }
    if origin is dict:
        if type(value) is not dict:
            raise TypeError("canonical map field must be a dict")
        key_type, item_type = get_args(expected)
        return {
            "$map": [
                [
                    _encode_as(
                        key,
                        key_type,
                        registered_types_only=registered_types_only,
                    ),
                    _encode_as(
                        item,
                        item_type,
                        registered_types_only=registered_types_only,
                    ),
                ]
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            ]
        }
    if expected is type(None):
        if value is not None:
            raise TypeError("canonical null field must be null")
        return None
    if isinstance(expected, type) and issubclass(expected, Enum):
        if type(value) is not expected:
            raise TypeError("canonical enum does not match declared type")
        return _encode(value, registered_types_only=registered_types_only)
    if isinstance(expected, type) and is_dataclass(expected):
        if type(value) is not expected:
            raise TypeError("canonical contract does not match declared type")
        return _encode(value, registered_types_only=registered_types_only)
    if expected is bool and type(value) is not bool:
        raise TypeError("canonical boolean field must be bool")
    if expected is int and type(value) is not int:
        raise TypeError("canonical integer field must be int")
    if expected is float and (type(value) is not float or not math.isfinite(value)):
        raise TypeError("canonical float field must be a finite float")
    if expected is str and type(value) is not str:
        raise TypeError("canonical string field must be str")
    return _encode(value, registered_types_only=registered_types_only)


def _encode_json_value(value: Any) -> Any:
    """Encode the closed JSON value vocabulary used by open payload fields."""

    if isinstance(value, float) and not math.isfinite(value):
        raise TypeError("JSON numbers must be finite")
    if value is None or type(value) in {bool, int, float, str}:
        return value
    if type(value) is list:
        return {"$list": [_encode_json_value(item) for item in value]}
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise TypeError("JSON object keys must be strings")
        return {
            "$map": [
                [key, _encode_json_value(item)] for key, item in sorted(value.items())
            ]
        }
    raise TypeError("open contract fields accept JSON values only")


def _decode_json_value(value: Any) -> Any:
    """Parse a closed JSON value without dispatching contract type tags."""

    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON numbers must be finite")
    if value is None or type(value) in {bool, int, float, str}:
        return value
    if not isinstance(value, dict):
        raise ValueError("invalid canonical JSON value")
    if set(value) == {"$list"} and isinstance(value["$list"], list):
        return [_decode_json_value(item) for item in value["$list"]]
    if set(value) == {"$map"} and isinstance(value["$map"], list):
        pairs = value["$map"]
        if any(not isinstance(pair, list) or len(pair) != 2 for pair in pairs):
            raise ValueError("canonical JSON map entries must be key-value pairs")
        decoded = [
            (_decode_as(pair[0], str), _decode_json_value(pair[1])) for pair in pairs
        ]
        output = dict(decoded)
        if len(output) != len(decoded):
            raise ValueError("canonical JSON map cannot contain duplicate keys")
        return output
    raise ValueError("open contract fields accept JSON values only")


def _decode_as(value: Any, expected: Any) -> Any:
    """Parse one canonical value according to its declared contract type."""

    if expected is Any or expected is object:
        return _decode_json_value(value)
    origin = get_origin(expected)
    if origin in {Union, UnionType}:
        return _decode_union(value, get_args(expected))
    if origin is tuple:
        return _decode_tuple(value, get_args(expected))
    if origin is list:
        return _decode_list(value, get_args(expected))
    if origin is dict:
        return _decode_map(value, get_args(expected))
    if expected is type(None):
        if value is not None:
            raise ValueError("canonical value must be null")
        return None
    if isinstance(expected, type) and issubclass(expected, Enum):
        return _decode_enum_as(value, expected)
    if isinstance(expected, type) and is_dataclass(expected):
        return _decode_contract_as(value, expected)
    if expected is bool:
        if type(value) is not bool:
            raise ValueError("canonical value must be boolean")
        return value
    if expected is int:
        if type(value) is not int:
            raise ValueError("canonical value must be integer")
        return value
    if expected is float:
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("canonical value must be finite float")
        return value
    if expected is str:
        if type(value) is not str:
            raise ValueError("canonical value must be string")
        return value
    raise ValueError(f"unsupported canonical contract annotation {expected!r}")


def _decode_union(value: Any, variants: tuple[Any, ...]) -> Any:
    errors: list[ValueError] = []
    for variant in variants:
        try:
            return _decode_as(value, variant)
        except ValueError as exc:
            errors.append(exc)
    raise ValueError("canonical value does not match its declared union") from errors[
        -1
    ]


def _decode_tuple(value: Any, arguments: tuple[Any, ...]) -> tuple[Any, ...]:
    if (
        not isinstance(value, dict)
        or set(value) != {"$tuple"}
        or not isinstance(value["$tuple"], list)
    ):
        raise ValueError("canonical tuple fields do not match schema")
    items = value["$tuple"]
    if len(arguments) == 2 and arguments[1] is Ellipsis:
        return tuple(_decode_as(item, arguments[0]) for item in items)
    if len(items) != len(arguments):
        raise ValueError("canonical fixed tuple length does not match schema")
    return tuple(
        _decode_as(item, expected)
        for item, expected in zip(items, arguments, strict=True)
    )


def _decode_list(value: Any, arguments: tuple[Any, ...]) -> list[Any]:
    if (
        not isinstance(value, dict)
        or set(value) != {"$list"}
        or not isinstance(value["$list"], list)
    ):
        raise ValueError("canonical list fields do not match schema")
    item_type = arguments[0] if arguments else Any
    return [_decode_as(item, item_type) for item in value["$list"]]


def _decode_map(value: Any, arguments: tuple[Any, ...]) -> dict[Any, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != {"$map"}
        or not isinstance(value["$map"], list)
    ):
        raise ValueError("canonical map fields do not match schema")
    key_type, item_type = arguments or (Any, Any)
    pairs = value["$map"]
    if any(not isinstance(pair, list) or len(pair) != 2 for pair in pairs):
        raise ValueError("canonical map entries must be key-value pairs")
    decoded = [
        (_decode_as(pair[0], key_type), _decode_as(pair[1], item_type))
        for pair in pairs
    ]
    output = dict(decoded)
    if len(output) != len(decoded):
        raise ValueError("canonical map cannot contain duplicate keys")
    return output


def _decode_enum_as(value: Any, enum_type: type[Enum]) -> Enum:
    if (
        not isinstance(value, dict)
        or set(value) != {"$enum", "value"}
        or value["$enum"] != _type_key(enum_type)
        or _ENUM_TYPES.get(_type_key(enum_type)) is not enum_type
    ):
        raise ValueError("canonical enum does not match declared type")
    return enum_type(value["value"])


def _decode_contract_as(value: Any, item_type: type) -> Any:
    if (
        not isinstance(value, dict)
        or set(value) != {"$type", "fields"}
        or value["$type"] != _type_key(item_type)
        or _DATACLASS_TYPES.get(_type_key(item_type)) is not item_type
    ):
        raise ValueError("canonical contract does not match declared type")
    raw_fields = value["fields"]
    if not isinstance(raw_fields, dict):
        raise ValueError("canonical contract fields must be an object")
    contract_fields = fields(item_type)
    if set(raw_fields) != {item.name for item in contract_fields}:
        raise ValueError("canonical contract fields do not match schema")
    annotations = get_type_hints(item_type)
    decoded_fields = {
        item.name: _decode_as(raw_fields[item.name], annotations[item.name])
        for item in contract_fields
    }
    instance = item_type(
        **{
            item.name: decoded_fields[item.name]
            for item in contract_fields
            if item.init
        }
    )
    if any(
        getattr(instance, item.name) != decoded_fields[item.name]
        for item in contract_fields
        if not item.init
    ):
        raise ValueError("canonical fixed contract field does not match schema")
    return instance


def _type_key(item_type: type) -> str:
    # Type tags are part of the language-neutral wire schema.  Python module paths
    # are implementation details and would give another runtime a different hash.
    return item_type.__name__


_CONTRACT_TYPES = (
    capability_contracts.CapabilityApplication,
    capability_contracts.CapabilityKind,
    capability_contracts.NarrowPopulationCapability,
    model.AnswerProgram,
    model.FactFulfillment,
    model.FunctionSemanticVersion,
    model.ProgramCompatibility,
    model.SourceContractKind,
    model.SourceContractPin,
    operations.AggregateSpec,
    operations.AggregationFunction,
    operations.AggregationSpec,
    operations.AntiJoinSpec,
    operations.ComputeBinary,
    operations.ComputeBinaryOperator,
    operations.ComputeExpressionReferences,
    operations.ComputeNegation,
    operations.ComputeSpec,
    operations.CrossJoinSpec,
    operations.FilterSpec,
    operations.JoinKey,
    operations.JoinSpec,
    operations.Operation,
    operations.OperationKind,
    operations.Predicate,
    operations.PredicateOperator,
    operations.ProjectField,
    operations.ProjectSpec,
    operations.ProjectToKeySpec,
    operations.RankSpec,
    operations.RelationRole,
    operations.RelationRoleRef,
    operations.RoleExpandSpec,
    operations.RoleMapping,
    operations.SortDirection,
    operations.SortKey,
    operations.TiePolicy,
    operations.UnionSpec,
    operations.UniversalConditionSpec,
    relations.EndpointParamBinding,
    relations.FieldBindingRole,
    relations.PopulationChoiceControllerKind,
    relations.Relation,
    relations.RelationField,
    relations.RelationSource,
    relations.RelationSourceAppliedFilter,
    relations.RelationSourcePopulationChoice,
    relations.RelationSourceReviewScopeDecision,
    relations.RelationSourceRowFilter,
    relations.ReviewScopeDecisionKind,
    relations.SourceKind,
    result_projection.EntityKeyProjection,
    result_projection.EntityKeyProjectionComponent,
    result_projection.RelationResultOutput,
    result_projection.ScalarResultOutput,
    result_projection.ResultProjection,
    values.BindingPatch,
    values.BindingPatchOperationKind,
    values.BindingProvenance,
    values.BindingProvenanceKind,
    values.BindingSet,
    values.ConstantRef,
    values.EnvironmentRef,
    values.FactValue,
    values.IdentitySetValuePayload,
    values.IdentityValuePayload,
    values.LiteralType,
    values.LiteralValuePayload,
    values.NamedValueExpression,
    values.NamedValuePayload,
    values.NodeOutputRef,
    values.ParameterBinding,
    values.ParameterDeclaration,
    values.ParameterRef,
    values.ParameterRole,
    values.ParameterValueType,
    values.ProgramInputs,
    values.SetParameter,
    values.StringSetValuePayload,
    values.TimeComponent,
    values.TimeGranularity,
    values.TimeValuePayload,
    values.UnsetParameter,
    values.ValueComponent,
    values.ValueDependency,
    values.ValueDependencyKind,
    values.ValueExpressionOrigin,
    values.ValueFilterOperator,
    values.ValueKind,
    question_contract_model.AnswerPopulationMembershipTestKind,
    question_contract_model.AnswerPopulationMembershipTestPolarity,
    question_contract_model.AnswerSubjectInstanceInterpretationKind,
    question_contract_model.GroupKeyDomainKind,
    question_contract_model.KnownInputSource,
    question_inputs.LiteralInputRole,
    question_contract_model.IncompleteFactualRequestItem,
    question_contract_model.IncompleteFactualRequestKind,
    question_contract_model.NormalInstanceExcludedStateRole,
    question_contract_model.NormalInstanceExcludedStateRoleDefinition,
    question_contract_model.NormalInstanceExplicitOverrideReason,
    question_contract_model.NormalInstanceProfile,
    question_contract_model.NormalInstanceProfileId,
    question_contract_model.QuestionContract,
    question_contract_model.QuestionContractNeedsClarification,
    question_contract_model.QuestionContractRequest,
    question_contract_model.QuestionContractResult,
    question_contract_model.RequestedFact,
    question_contract_model.RequestedFactAnswerExpression,
    question_contract_model.RequestedFactAnswerExpressionFamily,
    question_contract_model.RequestedFactAnswerOutput,
    question_contract_model.RequestedFactAnswerPopulation,
    question_contract_model.RequestedFactAnswerPopulationMembershipTest,
    question_contract_model.RequestedFactAnswerSubject,
    question_contract_model.RequestedFactAnswerSubjectInstanceInterpretation,
    question_contract_model.RequestedFactGroupKey,
    question_contract_model.RequestedFactLiteralInput,
    question_contract_model.RequestedFactPopulationConstraint,
    question_contract_model.RequestedFactRowSetReferenceInput,
)
_DATACLASS_TYPES = {
    _type_key(item): item for item in _CONTRACT_TYPES if is_dataclass(item)
}
_ENUM_TYPES = {
    _type_key(item): item for item in _CONTRACT_TYPES if issubclass(item, Enum)
}

if len(_DATACLASS_TYPES) + len(_ENUM_TYPES) != len(_CONTRACT_TYPES):
    raise RuntimeError("canonical answer-program type keys must be globally unique")
