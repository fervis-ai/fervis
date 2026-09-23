"""Compile verified source values into the existing program-input contract."""

from fervis.lookup.answer_program.compiler_inputs import (
    CompilerInputContext,
    grounded_program_inputs,
    program_value_parameter_id,
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
from fervis.lookup.available_sources import source_choice_literal
from fervis.lookup.source_binding import VerifiedSourceStrategy
from fervis.lookup.contract_codec import canonical_contract_fingerprint


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
    initial = grounded_program_inputs(request.canonical_values,fixed_value_refs=frozenset(mapped_value_refs))
    parameters = list(initial.parameters)
    bindings = list(initial.bindings.bindings)
    selected_choice_refs = {
        target.value_ref
        for application in verified.binding_plan.invocation_applications
        for target in application.target_applications
    }
    from fervis.lookup.answer_program.expressions import expression_references
    selected_choice_refs.update(ref.parameter_id
        for realizations in verified.binding_plan.set_bindings.values()
        for realization in realizations if realization.membership is not None
        for ref in expression_references(realization.membership).parameters)
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
    from fervis.lookup.source_binding.population_values import population_control_values
    population_values = tuple(
        (ref, value, value.proof_refs)
        for ref, value in population_control_values(request).items()
        if ref in selected_choice_refs
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
        + population_values
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
    return program_value_parameter_id(value_ref)


__all__ = ["semantic_compiler_inputs"]
