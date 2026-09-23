"""Executable logical-set membership over a covering source population.

The expression algebra is shared with AnswerProgram. This boundary supplies
only source-owned fields and current-run values, never invented literals.
"""

from dataclasses import dataclass

from fervis.lookup.answer_program.expressions import (
    BinaryExpression,
    Expression,
    FieldRef,
    UnaryExpression,
    expression_references,
    fold_expression,
)
from fervis.lookup.answer_program.values import (
    ParameterRef,
    project_fact_value,
    ValueProjectionKind,
    TimeValuePayload,
)
from fervis.lookup.available_sources import SourceFieldBinding, SourceChoiceSurfaceKind
from fervis.lookup.expression_operators import (
    ExpressionBinaryOperator,
    ExpressionUnaryOperator,
    infer_operator_result,
    operator_signature,
    OperatorKind,
)
from fervis.lookup.provider_contract import ProviderObject, ProviderOutput
from fervis.lookup.relation_catalog.row_sources import (
    semantic_type_for_row_source_type,
    RowSourceValueType,
)
from fervis.lookup.semantic_types import BooleanType, TemporalScopeType


@dataclass(frozen=True)
class ExactPopulationOutput(ProviderOutput):
    kind: str


@dataclass(frozen=True)
class RestrictedPopulationOutput(ProviderOutput):
    kind: str
    condition: ProviderObject


@dataclass(frozen=True)
class MembershipFieldOutput(ProviderOutput):
    kind: str
    field_ref: str


@dataclass(frozen=True)
class MembershipBooleanOutput(ProviderOutput):
    kind: str
    field_ref: str
    expected_value: bool


@dataclass(frozen=True)
class MembershipValueOutput(ProviderOutput):
    kind: str
    value_ref: str


@dataclass(frozen=True)
class MembershipUnaryOutput(ProviderOutput):
    kind: str
    operator: str
    operand: ProviderObject


@dataclass(frozen=True)
class MembershipBinaryOutput(ProviderOutput):
    kind: str
    operator: str
    left: ProviderObject
    right: ProviderObject


def _has_whole_projection(value):
    try:
        project_fact_value(value, projection=ValueProjectionKind.WHOLE_VALUE)
    except ValueError:
        return False
    return True


def membership_choice_values(request):
    return tuple(
        item
        for item in request.source_catalog.choice_values
        if request.source_catalog.choice_surface(item.surface_ref).kind
        is SourceChoiceSurfaceKind.RETURNED_FIELD
    )


def membership_schema(request):
    exact = ExactPopulationOutput.schema({"kind": {"enum": ["exact_population"]}})
    if not any(source.fields for source in request.source_catalog.sources):
        return exact
    return {
        "anyOf": [
            exact,
            RestrictedPopulationOutput.schema(
                {
                    "kind": {"enum": ["restricted_population"]},
                    "condition": {"$ref": "#/$defs/membership_condition_3"},
                }
            ),
        ]
    }


def membership_definitions(request):
    fields = [
        SourceFieldBinding(source.id, field).ref
        for source in request.source_catalog.sources
        for field in source.fields
    ]
    values = [
        item.canonical_value_id
        for item in request.canonical_values
        if _has_whole_projection(item.typed_value)
        or isinstance(item.typed_value.payload, TimeValuePayload)
    ]
    values.extend(item.value_ref for item in membership_choice_values(request))
    if not fields:
        return {}
    definitions = {
        "membership_field": MembershipFieldOutput.schema(
            {"kind": {"enum": ["field"]}, "field_ref": {"enum": fields}}
        )
    }
    if values:
        definitions["membership_value"] = MembershipValueOutput.schema(
            {"kind": {"enum": ["value"]}, "value_ref": {"enum": values}}
        )

    def unary(operand, operators):
        return MembershipUnaryOutput.schema(
            {
                "kind": {"enum": ["unary"]},
                "operator": {"enum": operators},
                "operand": operand,
            }
        )

    def binary(left, right, operators):
        return MembershipBinaryOutput.schema(
            {
                "kind": {"enum": ["binary"]},
                "operator": {"enum": operators},
                "left": left,
                "right": right,
            }
        )

    arithmetic = [
        item.value
        for item in ExpressionBinaryOperator
        if operator_signature(item).kind is OperatorKind.ARITHMETIC
    ]
    comparisons = [
        item.value
        for item in ExpressionBinaryOperator
        if operator_signature(item).kind
        in {OperatorKind.COMPARISON, OperatorKind.MEMBERSHIP}
    ]
    boolean_fields = [
        SourceFieldBinding(source.id, field).ref
        for source in request.source_catalog.sources
        for field in source.fields
        if field.type is RowSourceValueType.BOOLEAN
    ]
    for depth in range(4):
        variants = [{"$ref": "#/$defs/membership_field"}]
        row_variants = list(variants)
        if values:
            variants.append({"$ref": "#/$defs/membership_value"})
        if depth:
            child = {"$ref": f"#/$defs/membership_expression_{depth - 1}"}
            row_child = {"$ref": f"#/$defs/membership_row_expression_{depth - 1}"}
            variants.extend(
                (unary(child, ["negate"]), binary(child, child, arithmetic))
            )
            row_variants.extend(
                (
                    unary(row_child, ["negate"]),
                    binary(row_child, child, arithmetic),
                    binary(child, row_child, arithmetic),
                )
            )
        definitions[f"membership_expression_{depth}"] = {"anyOf": variants}
        definitions[f"membership_row_expression_{depth}"] = {"anyOf": row_variants}
        row = {"$ref": "#/$defs/membership_row_expression_2"}
        value = {"$ref": "#/$defs/membership_expression_2"}
        conditions = [
            unary(row, ["is_null", "not_null"]),
            binary(row, value, comparisons),
            binary(value, row, comparisons),
        ]
        if boolean_fields:
            conditions.append(
                MembershipBooleanOutput.schema(
                    {
                        "kind": {"enum": ["boolean_field"]},
                        "field_ref": {"enum": boolean_fields},
                        "expected_value": {"type": "boolean"},
                    }
                )
            )
        if depth:
            child_condition = {"$ref": f"#/$defs/membership_condition_{depth - 1}"}
            conditions.extend(
                (
                    unary(child_condition, ["not"]),
                    binary(child_condition, child_condition, ["and", "or"]),
                )
            )
        definitions[f"membership_condition_{depth}"] = {"anyOf": conditions}
    return definitions


def parse_membership(payload, *, source, request):
    if payload.discriminator("kind") == "exact_population":
        payload.parse_as(ExactPopulationOutput)
        return None
    if payload.discriminator("kind") != "restricted_population":
        raise ValueError("unknown set population realization")
    parsed = payload.parse_as(RestrictedPopulationOutput)

    def expression(value):
        kind = value.discriminator("kind")
        if kind == "boolean_field":
            boolean = value.parse_as(MembershipBooleanOutput)
            field = next(
                (
                    item
                    for item in source.fields
                    if SourceFieldBinding(source.id, item).ref == boolean.field_ref
                ),
                None,
            )
            if field is None or field.type is not RowSourceValueType.BOOLEAN:
                raise ValueError(
                    "membership Boolean predicate requires a declared Boolean field"
                )
            operand = FieldRef(field.id)
            return (
                operand
                if boolean.expected_value
                else UnaryExpression(ExpressionUnaryOperator.NOT, operand)
            )
        if kind == "field":
            field = value.parse_as(MembershipFieldOutput)
            match = next(
                (
                    item
                    for item in source.fields
                    if SourceFieldBinding(source.id, item).ref == field.field_ref
                ),
                None,
            )
            if match is None:
                raise ValueError("membership field belongs to another source")
            return FieldRef(match.id)
        if kind == "value":
            return ParameterRef(value.parse_as(MembershipValueOutput).value_ref)
        if kind == "unary":
            unary = value.parse_as(MembershipUnaryOutput)
            return UnaryExpression(
                ExpressionUnaryOperator(unary.operator), expression(unary.operand)
            )
        if kind == "binary":
            binary = value.parse_as(MembershipBinaryOutput)
            return BinaryExpression(
                ExpressionBinaryOperator(binary.operator),
                expression(binary.left),
                expression(binary.right),
            )
        raise ValueError("unknown membership expression")

    result = expression(parsed.condition)
    validate_membership(result, source=source, request=request)
    return result


def map_membership(expression: Expression, *, field, value, binary=None):
    def unsupported(_):
        raise ValueError("membership requires source fields and authorized values")

    return fold_expression(
        expression,
        field=field,
        parameter=value,
        output=unsupported,
        constant=unsupported,
        environment=unsupported,
        unary=lambda node, operand: UnaryExpression(node.operator, operand),
        binary=binary
        or (lambda node, left, right: BinaryExpression(node.operator, left, right)),
        function=lambda node, arguments: unsupported(node),
    )


def validate_membership(expression, *, source, request):
    refs = expression_references(expression)
    if not refs.fields:
        raise ValueError("restricted population requires a source-row predicate")
    fields = {item.id: item for item in source.fields}
    canonical_values = {
        item.canonical_value_id: item for item in request.canonical_values
    }
    choices = {
        item.value_ref: item
        for item in membership_choice_values(request)
        if item.source_ref == source.id
    }

    def field_type(ref):
        if ref.field_id not in fields:
            raise ValueError("membership field belongs to another source")
        return semantic_type_for_row_source_type(fields[ref.field_id].type), False

    def value_type(ref):
        if (
            ref.component != "value"
            or ref.item_index is not None
            or ref.parameter_id not in canonical_values
            and ref.parameter_id not in choices
        ):
            raise ValueError("membership value lacks current-run authority")
        if ref.parameter_id in canonical_values:
            if isinstance(
                canonical_values[ref.parameter_id].typed_value.payload, TimeValuePayload
            ):
                return TemporalScopeType(), True
            if not _has_whole_projection(
                canonical_values[ref.parameter_id].typed_value
            ):
                raise ValueError(
                    "membership operand lacks an executable whole-value projection"
                )
            return request.index.value_type(
                canonical_values[ref.parameter_id].input_ref
            ), False
        return semantic_type_for_row_source_type(
            choices[ref.parameter_id].declared_type
        ), False

    def unsupported(_):
        raise ValueError("unsupported membership expression leaf")

    def unary_type(node, operand):
        if operand[1]:
            raise ValueError("temporal scope requires within")
        return infer_operator_result(node.operator, (operand[0],)), False

    def binary_type(node, left, right):
        if left[1] or right[1]:
            if (
                left[1]
                or node.operator is not ExpressionBinaryOperator.WITHIN
                or not isinstance(node.right, ParameterRef)
            ):
                raise ValueError("temporal scope requires within")
        return infer_operator_result(node.operator, (left[0], right[0])), False

    result = fold_expression(
        expression,
        field=field_type,
        parameter=value_type,
        output=unsupported,
        constant=unsupported,
        environment=unsupported,
        unary=unary_type,
        binary=binary_type,
        function=lambda node, arguments: unsupported(node),
    )
    if not isinstance(result[0], BooleanType) or result[1]:
        raise ValueError("set membership must be Boolean")
