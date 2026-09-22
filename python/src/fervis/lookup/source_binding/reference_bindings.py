"""Typed reference selection within an already assigned logical carrier."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fervis.lookup.answer_program.values import NamedValuePayload, StringSetValuePayload
from fervis.lookup.available_sources import SourceFieldBinding
from fervis.lookup.expression_operators import ExpressionBinaryOperator
from fervis.lookup.question_contract.model import Comparison, InputDenotationKind
from fervis.lookup.relation_catalog.parameter_values import parse_catalog_parameter_text
from fervis.lookup.relation_catalog.row_sources.model import RowSourceValueType
from fervis.lookup.relation_catalog.model import RowCardinality
from fervis.types.enums import StrEnum

if TYPE_CHECKING:
    from .model import SemanticSourceBindingRequest


class ReferenceMatchKind(StrEnum):
    LITERAL = "literal"
    DECLARED_CHOICE = "declared_choice"
    SINGLETON_VALUE = "singleton_value"


@dataclass(frozen=True)
class ReferenceBinding:
    branch_id: str
    input_use_ref: str
    field_refs: tuple[str, ...]
    mapping_basis: str
    match_kind: ReferenceMatchKind
    choice_value: str | None = None
    member_index: int | None = None


@dataclass(frozen=True)
class AddressReferenceOption:
    input_use_ref: str
    source_ref: str
    parameter_ref: str
    declared_type: RowSourceValueType


def address_reference_options(request: "SemanticSourceBindingRequest"):
    """Offer declared address parameters that accept the supplied scalar."""
    subject_ref = request.index.subject_obligation.subject_set_ref.token
    subject_sources = set(request._local_row_references_for_set(subject_ref))
    options = []
    for use in literal_reference_uses(request):
        node = request.index.expression_by_ref.get(use.expression_ref)
        if not isinstance(node, Comparison) or node.operator is not ExpressionBinaryOperator.EQUALS:
            continue
        supplied = request.index.input_by_ref[use.input_ref].operand
        if not isinstance(supplied, str):
            continue
        for source in request.source_catalog.sources:
            if source.id not in subject_sources or not source.read_id:
                continue
            for param in source.params:
                if param.source != "path" or not param.required:
                    continue
                try:
                    parse_catalog_parameter_text(
                        supplied, type_name=param.type.value, choices=param.choices
                    )
                except ValueError:
                    continue
                options.append(AddressReferenceOption(
                    use.use_ref, source.id, param.param_ref, param.type
                ))
    return tuple(options)


def validate_address_scope_bindings(request, set_bindings):
    allowed = set(address_reference_options(request))
    for set_ref, values in set_bindings.items():
        for value in values:
            if not value.address_parameter_ref and not value.address_input_use_ref:
                continue
            marker = (
                value.address_input_use_ref,
                value.source_ref,
                value.address_parameter_ref,
            )
            if (
                not value.address_parameter_ref
                or not value.address_input_use_ref
                or value.identity_ref is not None
                or value.record_fields
                or set_ref == request.index.subject_obligation.subject_set_ref.token
                or not any(
                    (item.input_use_ref, item.source_ref, item.parameter_ref) == marker
                    and any(
                        use.use_ref == item.input_use_ref
                        and use.identity_set_ref is not None
                        and use.identity_set_ref.token == set_ref
                        for use in request.index.input_use_sites
                    )
                    for item in allowed
                )
            ):
                raise ValueError("Address scope lacks a declared compatible input and path target")


def validated_address_applications(request, set_bindings, applications):
    """Identify qualifiers discharged by the exact supplied path value."""
    from fervis.lookup.answer_program.values import ValueProjectionKind
    from fervis.lookup.qualification import BooleanRequirementUseSite

    owners = set()
    for values in set_bindings.values():
        for carrier in values:
            if not carrier.address_parameter_ref:
                continue
            use = next(
                item for item in request.index.input_use_sites
                if item.use_ref == carrier.address_input_use_ref
            )
            requirements = tuple(
                requirement for requirement in request.index.boolean_requirements
                if requirement.atom_ref.value_ref == use.expression_ref.token
                and requirement.use_site is BooleanRequirementUseSite.POPULATION
                and requirement.atom_ref.polarity.value == "positive"
            )
            matches = tuple(
                (requirement.requirement_ref, application)
                for requirement in requirements
                for application in applications
                if application.owner_ref == requirement.requirement_ref
                and application.branch_id == carrier.branch_id
                and application.source_ref == carrier.source_ref
                and any(
                    target.target_ref == carrier.address_parameter_ref
                    and target.projection is ValueProjectionKind.WHOLE_VALUE
                    and any(
                        value.canonical_value_id == target.value_ref
                        and value.input_ref == use.input_ref
                        and use.use_ref in value.use_refs
                        for value in request.canonical_values
                    )
                    for target in application.target_applications
                )
            )
            if len(matches) != 1:
                raise ValueError("Address scope requires one owned request admission")
            owners.add((carrier.branch_id, matches[0][0]))
    return frozenset(owners)


def runtime_reference_uses(request: "SemanticSourceBindingRequest"):
    supplied = {
        ref
        for value in request.canonical_values
        if isinstance(value.typed_value.payload, (NamedValuePayload, StringSetValuePayload))
        for ref in value.use_refs
    }
    result = []
    for use in request.index.input_use_sites:
        if (
            use.use_ref not in supplied
            or use.identity_set_ref is None
            or use.reference_fact_ref is None
            or use.expression_ref is None
        ):
            continue
        denotation = request.index.input_denotation_by_ref[use.input_ref]
        node = request.index.expression_by_ref.get(use.expression_ref)
        if (
            denotation.kind is InputDenotationKind.IDENTITY_REFERENCE
            and isinstance(request.index.input_by_ref[use.input_ref].operand, (str, tuple))
            and isinstance(node, Comparison)
            and (
                node.operator in {ExpressionBinaryOperator.EQUALS, ExpressionBinaryOperator.NOT_EQUALS}
                if isinstance(request.index.input_by_ref[use.input_ref].operand, str)
                else node.operator is ExpressionBinaryOperator.IN
            )
        ):
            result.append(use)
    return tuple(result)


def literal_reference_uses(request: "SemanticSourceBindingRequest"):
    return tuple(
        use for use in runtime_reference_uses(request)
        if any(
            operand not in request.index.input_denotation_by_ref[use.input_ref].reference_descriptions
            for operand in _reference_operands(request, use)
        )
    )


def described_reference_uses(request: "SemanticSourceBindingRequest"):
    return tuple(
        use for use in runtime_reference_uses(request)
        if any(
            operand in request.index.input_denotation_by_ref[use.input_ref].reference_descriptions
            for operand in _reference_operands(request, use)
        )
    )


def _reference_operands(request, use) -> tuple[str, ...]:
    value = request.index.input_by_ref[use.input_ref].operand
    return (value,) if isinstance(value, str) else value


def reference_fields(request, use, source_ref, *, identity_ref=None, operand=None):
    literals = (operand,) if operand is not None else _reference_operands(request, use)
    source = request.source_catalog.source(source_ref)
    from fervis.lookup.relation_catalog.row_sources.model import RowSourceIdentityKind

    identity = (
        request.source_catalog.identity(identity_ref)
        if identity_ref is not None
        else None
    )
    key_only = (
        identity is not None and identity.kind is RowSourceIdentityKind.ENTITY_REFERENCE
    )
    fields = []
    for field in source.fields:
        if key_only and field.field_ref not in identity.field_refs:
            continue
        if field.declared_entity_kind:
            continue
        if any(
            not _field_accepts_literal(literal, field)
            for literal in literals
        ):
            continue
        fields.append(SourceFieldBinding(source_ref, field).ref)
    return tuple(fields)


def _field_accepts_literal(literal, field) -> bool:
    try:
        parse_catalog_parameter_text(
            literal, type_name=field.type.value, choices=field.choices
        )
    except ValueError:
        return False
    return True


def descriptor_choices(request, source_ref, *, identity_ref=None):
    source = request.source_catalog.source(source_ref)
    identity = request.source_catalog.identity(identity_ref) if identity_ref else None
    from fervis.lookup.relation_catalog.row_sources.model import RowSourceIdentityKind

    key_only = identity is not None and identity.kind is RowSourceIdentityKind.ENTITY_REFERENCE
    return {
        SourceFieldBinding(source_ref, field).ref: field.finite_choices
        for field in source.fields
        if field.type in {RowSourceValueType.BOOLEAN, RowSourceValueType.CHOICE}
        and field.finite_choices
        and not field.declared_entity_kind
        and (not key_only or field.field_ref in identity.field_refs)
    }


def descriptor_options(request, source_ref, *, identity_ref=None):
    choices = descriptor_choices(request, source_ref, identity_ref=identity_ref)
    source = request.source_catalog.source(source_ref)
    if identity_ref is not None or source.row_cardinality is not RowCardinality.ONE:
        return choices
    return {
        **{
            SourceFieldBinding(source_ref, field).ref: ()
            for field in source.fields
            if field.type in {
                RowSourceValueType.INTEGER, RowSourceValueType.UUID,
                RowSourceValueType.STRING, RowSourceValueType.NUMBER,
                RowSourceValueType.DECIMAL,
            }
            and not field.declared_entity_kind
        },
        **choices,
    }


def reference_match_options(request, use, source_ref, *, identity_ref=None):
    if request.index.input_denotation_by_ref[use.input_ref].reference_descriptions:
        return descriptor_options(request, source_ref, identity_ref=identity_ref)
    return reference_fields(request, use, source_ref, identity_ref=identity_ref)


def validate_reference_bindings(request, set_bindings, bindings, *, complete=True):
    uses = {use.use_ref: use for use in runtime_reference_uses(request)}
    addressed = {
        (item.branch_id, item.address_input_use_ref)
        for values in set_bindings.values()
        for item in values
        if item.address_parameter_ref
    }
    required = {
        (branch.branch_id, ref, index)
        for branch in request.strategy.branches
        for ref, use in uses.items()
        if (branch.branch_id, ref) not in addressed
        for index in (
            range(len(_reference_operands(request, use)))
            if isinstance(request.index.input_by_ref[use.input_ref].operand, tuple)
            else (None,)
        )
    }
    actual = [(binding.branch_id, binding.input_use_ref, binding.member_index) for binding in bindings]
    if len(actual) != len(set(actual)) or (
        set(actual) != required if complete else not set(actual) <= required
    ):
        raise ValueError(
            "Runtime references must cover every pending input use and branch once"
        )
    shared = {}
    for binding in bindings:
        use = uses[binding.input_use_ref]
        operands = _reference_operands(request, use)
        if binding.member_index is None:
            if isinstance(request.index.input_by_ref[use.input_ref].operand, tuple):
                raise ValueError("Collection reference requires one indexed member per binding")
            operand = operands[0]
        else:
            if binding.member_index < 0 or binding.member_index >= len(operands):
                raise ValueError("Reference member index is outside the original input")
            operand = operands[binding.member_index]
        owners = [
            value
            for value in set_bindings[use.identity_set_ref.token]
            if value.branch_id == binding.branch_id
        ]
        if (
            len(owners) != 1
            or not binding.mapping_basis.strip()
            or not binding.field_refs
            or len(set(binding.field_refs)) != len(binding.field_refs)
        ):
            raise ValueError(
                "Runtime reference requires one carrier and distinct literal match fields"
            )
        described = operand in request.index.input_denotation_by_ref[use.input_ref].reference_descriptions
        if described:
            choices = descriptor_options(
                request, owners[0].source_ref, identity_ref=owners[0].identity_ref
            )
            if len(binding.field_refs) != 1 or binding.field_refs[0] not in choices:
                raise ValueError("Descriptive reference must select a declared carrier choice")
            if binding.choice_value is None:
                if (
                    binding.match_kind is not ReferenceMatchKind.SINGLETON_VALUE
                    or choices[binding.field_refs[0]]
                    or owners[0].reference_proxy_field_ref != binding.field_refs[0]
                ):
                    raise ValueError("Singleton reference must use its declared proxy value")
            elif (
                binding.match_kind is not ReferenceMatchKind.DECLARED_CHOICE
                or binding.choice_value not in choices[binding.field_refs[0]]
            ):
                raise ValueError("Descriptive reference must select a declared carrier choice")
        else:
            allowed = reference_fields(
                request, use, owners[0].source_ref, identity_ref=owners[0].identity_ref,
                operand=operand,
            )
            if (
                binding.match_kind is not ReferenceMatchKind.LITERAL
                or binding.choice_value is not None
                or not set(binding.field_refs) <= set(allowed)
            ):
                raise ValueError(
                    "Runtime reference fields must belong to the selected carrier and accept the literal"
                )
        key = (binding.branch_id, use.input_ref, use.identity_set_ref, binding.member_index)
        fields = (frozenset(binding.field_refs), binding.choice_value)
        if key in shared and shared[key] != fields:
            raise ValueError(
                "One input reference cannot change its matching fields between consumers"
            )
        shared[key] = fields
