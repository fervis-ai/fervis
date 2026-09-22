"""Select literal matching properties on the already bound instance carrier."""

from dataclasses import dataclass, replace

from fervis.lookup.provider_contract import ProviderOutput
from fervis.lookup.turn_prompts import (
    TurnPromptBase,
    ProviderToolContract,
    ProviderResponseContract,
)
from fervis.model_io.structured_output.specs import required_tool_spec
from .reference_bindings import (
    ReferenceBinding,
    ReferenceMatchKind,
    literal_reference_uses,
    described_reference_uses,
    descriptor_options,
    reference_fields,
    validate_reference_bindings,
)
from .model import SourceRealizationUnavailable


@dataclass(frozen=True)
class LiteralMatchOutput(ProviderOutput):
    mapping_basis: str
    field_refs: tuple[str, ...]


@dataclass(frozen=True)
class LiteralReferencesOutput(ProviderOutput):
    references: dict[str, LiteralMatchOutput]


def reference_tasks(realization):
    tasks = {}
    request = realization.request
    for use in literal_reference_uses(request):
        supplied = request.index.input_by_ref[use.input_ref].operand
        operands = (supplied,) if isinstance(supplied, str) else supplied
        for branch in request.strategy.branches:
            carrier = next(
                value
                for value in realization.set_bindings[use.identity_set_ref.token]
                if value.branch_id == branch.branch_id
            )
            if carrier.address_parameter_ref:
                continue
            for index, operand in enumerate(operands):
                if operand in request.index.input_denotation_by_ref[use.input_ref].reference_descriptions:
                    continue
                member_index = index if isinstance(supplied, tuple) else None
                ref = f"{branch.branch_id}:{use.use_ref}"
                if member_index is not None:
                    ref += f":member:{member_index}"
                tasks[ref] = (
                    use,
                    branch.branch_id,
                    carrier,
                    reference_fields(
                        request, use, carrier.source_ref,
                        identity_ref=carrier.identity_ref, operand=operand,
                    ),
                    member_index,
                    operand,
                )
    return tasks


def _object(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


class LiteralReferenceTurnPrompt(TurnPromptBase):
    turn_name = "literal references"
    turn_task = (
        "bind each supplied literal to identifying properties on its selected carrier"
    )
    include_current_question = False

    def __init__(self, realization):
        self.realization = realization
        self.request = realization.request

    def schema(self):
        references = {}
        for ref, (_, _, _, fields, _, _) in reference_tasks(self.realization).items():
            items = {"enum": list(fields)} if fields else {"type": "string"}
            references[ref] = LiteralMatchOutput.schema(
                {
                    "mapping_basis": {"type": "string", "minLength": 1},
                    "field_refs": {
                        "type": "array",
                        "items": items,
                        **({} if fields else {"maxItems": 0}),
                    },
                }
            )
        return _object({"references": _object(references)})

    def instruction_sections(self, builder):
        return (
            builder.instruction_block(
                "Literal reference matching",
                (
                    "The input already denotes an instance. Its literal text and meaning are fixed. Select only observed properties on the assigned carrier that can identify it in that meaning.",
                    "Matching uses the exact supplied text interpreted by each selected property scalar type. It does not extract digits, change names, invent keys, or reinterpret a literal as a role description.",
                    "Multiple selected properties are alternative exact matches. Every matching record remains a candidate; execution checks uniqueness before using the reference.",
                    "Do not choose measurements, totals, or unrelated status fields just because their scalar type accepts the text. Respect whether the supplied reference is a name, code, or identifier.",
                    "Return an empty field_refs list when this carrier cannot resolve the stated literal from its observed properties. One input on one carrier must use the same properties across its consumers.",
                ),
            ),
        )

    def data_sections(self, builder):
        items = {}
        for ref, (use, branch, carrier, fields, _, operand) in reference_tasks(
            self.realization
        ).items():
            source = self.request.source_catalog.source(carrier.source_ref)
            items[ref] = {
                "input_ref": use.input_ref,
                "literal": operand,
                "meaning": use.operand_meaning,
                "source_ref": source.id,
                "source_description": source.description,
                "fields": [
                    {
                        "ref": field_ref,
                        "path": self.request.source_catalog.field_binding(
                            field_ref
                        ).field.path,
                        "type": self.request.source_catalog.field_binding(
                            field_ref
                        ).field.type.value,
                        "description": self.request.source_catalog.field_binding(
                            field_ref
                        ).field.description,
                    }
                    for field_ref in fields
                ],
            }
        return (builder.json_section("Fixed reference tasks:", items, indent=2),)

    def tool_contract(self):
        return ProviderToolContract(
            tool_specs=(
                required_tool_spec(
                    tool_name="submit_literal_references",
                    tool_description="Choose identifying properties for the supplied reference literals.",
                    input_schema=self.schema(),
                ),
            )
        )

    def response_contract(self):
        return ProviderResponseContract(
            provider_schema={"submit_literal_references": self.schema()}
        )


def bind_literal_references(payload, *, realization):
    parsed = LiteralReferencesOutput.parse(payload)
    tasks = reference_tasks(realization)
    if set(parsed.references) != set(tasks):
        raise ValueError("Literal reference binding must cover the exact use scope")
    bindings = []
    for ref, value in parsed.references.items():
        use, branch, carrier, fields, member_index, _ = tasks[ref]
        if not value.mapping_basis.strip():
            raise ValueError("Literal reference mapping requires its evidence basis")
        if not value.field_refs:
            return SourceRealizationUnavailable(
                realization.request.index.requested_fact_id,
                (use.reference_fact_ref.token,),
                value.mapping_basis,
            )
        bindings.append(
            ReferenceBinding(
                branch, use.use_ref, value.field_refs, value.mapping_basis,
                ReferenceMatchKind.LITERAL, member_index=member_index,
            )
        )
    validate_reference_bindings(
        realization.request, realization.set_bindings, tuple(bindings), complete=False
    )
    return replace(
        realization,
        request=replace(realization.request, reference_bindings=(
            *realization.request.reference_bindings, *bindings
        )),
    )


@dataclass(frozen=True)
class DescriptorMatchOutput(ProviderOutput):
    mapping_basis: str
    field_ref: str | None
    choice_value: str | None


@dataclass(frozen=True)
class DescriptorReferencesOutput(ProviderOutput):
    references: dict[str, DescriptorMatchOutput]


def descriptor_tasks(realization):
    tasks = {}
    request = realization.request
    for use in described_reference_uses(request):
        supplied = request.index.input_by_ref[use.input_ref].operand
        operands = (supplied,) if isinstance(supplied, str) else supplied
        for branch in request.strategy.branches:
            carrier = next(
                value for value in realization.set_bindings[use.identity_set_ref.token]
                if value.branch_id == branch.branch_id
            )
            for index, operand in enumerate(operands):
                if operand not in request.index.input_denotation_by_ref[use.input_ref].reference_descriptions:
                    continue
                member_index = index if isinstance(supplied, tuple) else None
                ref = f"{branch.branch_id}:{use.use_ref}"
                if member_index is not None:
                    ref += f":member:{member_index}"
                tasks[ref] = (
                    use, branch.branch_id, carrier,
                    descriptor_options(
                        request, carrier.source_ref, identity_ref=carrier.identity_ref
                    ),
                    member_index, operand,
                )
    return tasks


class DescriptorReferenceTurnPrompt(TurnPromptBase):
    turn_name = "descriptive references"
    turn_task = "bind a described instance to a declared property choice"
    include_current_question = False

    def __init__(self, realization):
        self.realization = realization
        self.request = realization.request

    def schema(self):
        references = {}
        for ref, (_, _, _, choices, _, _) in descriptor_tasks(self.realization).items():
            references[ref] = DescriptorMatchOutput.schema({
                "mapping_basis": {"type": "string", "minLength": 1},
                "field_ref": {"enum": [None, *choices]},
                "choice_value": {"enum": [
                    None, *sorted({value for values in choices.values() for value in values})
                ]},
            })
        return _object({"references": _object(references)})

    def instruction_sections(self, builder):
        return (builder.instruction_block("Descriptive reference selection", (
            "The supplied phrase denotes one instance by a role or description. It is not a literal name unless its input contract says so.",
            "Choose one declared property and one of that property's declared choices only when their meanings select the described instance on the assigned carrier.",
            "A complete one-row reference proxy may instead project its declared scalar value field with choice_value=null; this identifies the referent without asserting that the source row is the entity row.",
            "A matching row remains a candidate; execution proves complete source coverage and uniqueness before using it.",
            "Choose null for both fields when no property choice on this carrier states the described role. Do not guess an identifier or borrow a property from a different source.",
        )),)

    def data_sections(self, builder):
        items = {}
        for ref, (use, _, carrier, choices, _, operand) in descriptor_tasks(self.realization).items():
            source = self.request.source_catalog.source(carrier.source_ref)
            items[ref] = {
                "input_ref": use.input_ref,
                "description": operand,
                "meaning": use.operand_meaning,
                "source_ref": source.id,
                "source_description": source.description,
                "choices": [
                    {
                        "field_ref": field_ref,
                        "path": self.request.source_catalog.field_binding(field_ref).field.path,
                        "description": self.request.source_catalog.field_binding(field_ref).field.description,
                        "value": value,
                    }
                    for field_ref, values in choices.items()
                    for value in values
                ],
                "singleton_value_fields": [
                    {
                        "field_ref": field_ref,
                        "path": self.request.source_catalog.field_binding(field_ref).field.path,
                        "description": self.request.source_catalog.field_binding(field_ref).field.description,
                        "type": self.request.source_catalog.field_binding(field_ref).field.type.value,
                    }
                    for field_ref, values in choices.items()
                    if not values
                ],
            }
        return (builder.json_section("Fixed descriptor tasks:", items, indent=2),)

    def tool_contract(self):
        return ProviderToolContract(tool_specs=(required_tool_spec(
            tool_name="submit_descriptor_references",
            tool_description="Choose declared property values for descriptive references.",
            input_schema=self.schema(),
        ),))

    def response_contract(self):
        return ProviderResponseContract(provider_schema={
            "submit_descriptor_references": self.schema()
        })


def bind_descriptor_references(payload, *, realization):
    parsed = DescriptorReferencesOutput.parse(payload)
    tasks = descriptor_tasks(realization)
    if set(parsed.references) != set(tasks):
        raise ValueError("Descriptor reference binding must cover the exact use scope")
    bindings = []
    for ref, value in parsed.references.items():
        use, branch, _, _, member_index, _ = tasks[ref]
        if not value.mapping_basis.strip():
            raise ValueError("Descriptor binding requires its evidence basis")
        if value.field_ref is None:
            if value.choice_value is not None:
                raise ValueError("Unavailable descriptor must omit both selection values")
            return SourceRealizationUnavailable(
                realization.request.index.requested_fact_id,
                (use.reference_fact_ref.token,),
                value.mapping_basis,
            )
        bindings.append(ReferenceBinding(
            branch, use.use_ref, (value.field_ref,), value.mapping_basis,
            (ReferenceMatchKind.SINGLETON_VALUE if value.choice_value is None
             else ReferenceMatchKind.DECLARED_CHOICE), value.choice_value,
            member_index,
        ))
    combined = (*realization.request.reference_bindings, *bindings)
    validate_reference_bindings(
        realization.request, realization.set_bindings, combined, complete=False
    )
    return replace(
        realization, request=replace(realization.request, reference_bindings=combined)
    )
