"""Compile verified source values into the existing program-input contract."""

from fervis.lookup.answer_program.compiler_inputs import (
    CompilerInputContext,
    compiler_input_context_from_program_inputs,
)
from fervis.lookup.answer_program.contracts import (
    BindingProvenance,
    BindingProvenanceKind,
    BindingSet,
    ParameterBinding,
    ParameterDeclaration,
    ParameterRole,
    ProgramInputs,
    parameter_value_type,
)
from fervis.lookup.answer_program.values import FactValue, LiteralType
from fervis.lookup.source_binding import VerifiedSourceStrategy
from fervis.lookup.contract_codec import canonical_contract_fingerprint


def source_choice_literal(choice, *, snapshot_ref: str) -> FactValue:
    """Preserve the declared primitive type of a catalog choice token."""
    from fervis.lookup.relation_catalog.row_sources import RowSourceValueType

    if choice.declared_type is RowSourceValueType.BOOLEAN:
        literal_type, value = (
            LiteralType.BOOLEAN,
            "true" if choice.boolean_value else "false",
        )
    elif choice.declared_type in {
        RowSourceValueType.INTEGER,
        RowSourceValueType.DECIMAL,
    }:
        literal_type, value = LiteralType.NUMBER, str(choice.value)
    else:
        literal_type, value = LiteralType.STRING, str(choice.value)
    return FactValue.literal(
        id=choice.value_ref,
        literal_type=literal_type,
        value=value,
        label=choice.label,
        proof_refs=(
            snapshot_ref,
            choice.source_ref,
            choice.surface_ref,
            choice.value_ref,
        ),
        source_refs=(choice.source_ref,),
    )


def semantic_compiler_inputs(
    verified: VerifiedSourceStrategy,
) -> CompilerInputContext:
    """Declare each verified typed value once for program reuse and invocation."""

    request = verified.request
    mapped_value_refs = {
        value_ref
        for branch in verified.binding_plan.subject_binding.branch_realizations
        for review in branch.surface_reviews
        for choice in review.choice_reviews
        for owner in choice.selection_requirement_refs
        for value_ref in request.requirement_value_refs(owner)
    }
    parameters: list[ParameterDeclaration] = []
    bindings: list[ParameterBinding] = []
    for value in request.canonical_values:
        parameter_id = _parameter_id(value.canonical_value_id)
        parameters.append(
            ParameterDeclaration(
                id=parameter_id,
                role=ParameterRole.QUESTION_INPUT,
                value_type=parameter_value_type(value.typed_value),
                input_ref=value.input_ref,
                input_use_refs=value.use_refs,
                fixed_value_fingerprint=canonical_contract_fingerprint(
                    value.typed_value.payload
                )
                if value.canonical_value_id in mapped_value_refs
                else "",
            )
        )
        bindings.append(
            ParameterBinding(
                parameter_id=parameter_id,
                value=value.typed_value,
                provenance=BindingProvenance(
                    kind=BindingProvenanceKind.QUESTION_INPUT,
                    refs=tuple(
                        dict.fromkeys((value.input_ref, *value.certification_refs))
                    ),
                ),
            )
        )
    selected_choice_refs = {
        target.value_ref
        for application in verified.binding_plan.invocation_applications
        for target in application.target_applications
    }
    source_choice_values = tuple(
        (
            value.value_ref,
            source_choice_literal(
                value, snapshot_ref=request.source_catalog.contract_snapshot.ref
            ),
            (
                request.source_catalog.contract_snapshot.ref,
                value.source_ref,
                value.surface_ref,
            ),
        )
        for value in request.source_catalog.choice_values
        if value.value_ref in selected_choice_refs
    )
    plan_values = (
        tuple(
            (
                value.value_id,
                value.typed_value,
                (value.catalog_input_ref, *value.certification_refs),
            )
            for value in request.catalog_values
        )
        + source_choice_values
    )
    for value_id, typed_value, provenance_refs in plan_values:
        parameter_id = _parameter_id(value_id)
        parameters.append(
            ParameterDeclaration(
                id=parameter_id,
                role=ParameterRole.PLAN_CONTROL,
                value_type=parameter_value_type(typed_value),
                fixed_value_fingerprint=canonical_contract_fingerprint(
                    typed_value.payload
                ),
            )
        )
        bindings.append(
            ParameterBinding(
                parameter_id=parameter_id,
                value=typed_value,
                provenance=BindingProvenance(
                    kind=BindingProvenanceKind.PLAN_CHOICE,
                    refs=tuple(dict.fromkeys(provenance_refs)),
                ),
            )
        )
    return compiler_input_context_from_program_inputs(
        ProgramInputs(
            parameters=tuple(parameters),
            bindings=BindingSet.from_bindings(tuple(bindings)),
        )
    )


def _parameter_id(value_ref: str) -> str:
    return f"value.{value_ref}"


__all__ = ["semantic_compiler_inputs"]
