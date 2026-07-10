"""Render-reference checks for fact-plan verification."""

from ._shared import (
    AnswerProgram,
    ComputeSpec,
    FieldBindingRole,
    Operation,
    VerificationError,
)
from .contract_types import ProofLineage, RelationContract
from .operations import _operation_input_refs
from .scalars import _operation_scalar_inputs
from fervis.lookup.plan_execution.operation_runtime import ResolvedOperationInput
from fervis.lookup.answer_program.render_spec import (
    RenderRelationOutput,
    RenderScalarOutput,
)


def _render_output_fact_refs(
    answer: AnswerProgram,
    *,
    relation_contracts: dict[str, RelationContract],
    operation_inputs: tuple[ResolvedOperationInput, ...],
) -> dict[str, frozenset[str]]:
    refs: dict[str, frozenset[str]] = {}
    for render_output in answer.render_spec.relation_outputs:
        contract = relation_contracts.get(render_output.relation_id)
        if contract is None:
            continue
        refs[render_output.id] = contract.field_proofs.get(
            render_output.field_id, ProofLineage()
        ).fulfillment_refs()
    scalar_refs = _compute_scalar_fact_refs(
        answer,
        operation_inputs=operation_inputs,
    )
    for scalar_output in answer.render_spec.scalar_outputs:
        refs[scalar_output.id] = scalar_refs.get(scalar_output.scalar_id, frozenset())
    return refs


def _compute_scalar_fact_refs(
    answer: AnswerProgram,
    *,
    operation_inputs: tuple[ResolvedOperationInput, ...],
) -> dict[str, frozenset[str]]:
    scalar_inputs = {
        (item.operation_id, item.input_id): frozenset(item.proof_refs)
        for item in operation_inputs
    }
    output: dict[str, frozenset[str]] = {}
    for operation in answer.operations:
        if not isinstance(operation.spec, ComputeSpec):
            continue
        refs: set[str] = set()
        refs.update(
            proof_ref
            for (operation_id, _input_id), proof_refs in scalar_inputs.items()
            if operation_id == operation.id
            for proof_ref in proof_refs
        )
        for input_id in _operation_scalar_inputs(operation):
            refs.update(output.get(input_id, frozenset()))
        output[operation.spec.output_scalar] = frozenset(refs)
    return output


def _verify_render_references(
    answer: AnswerProgram,
    *,
    relation_contracts: dict[str, RelationContract],
) -> None:
    _verify_render_output_targets(answer)
    render_outputs = tuple(answer.render_spec.relation_outputs)
    for relation_output in render_outputs:
        if not relation_output.field_id:
            raise VerificationError(
                f"render output {relation_output.id} requires field id"
            )
        contract = relation_contracts.get(relation_output.relation_id)
        if contract is None or relation_output.field_id not in contract.fields:
            raise VerificationError(
                f"render output {relation_output.id} references unknown output field"
            )
        if FieldBindingRole.OUTPUT not in contract.fields[relation_output.field_id]:
            raise VerificationError(
                f"render output {relation_output.id} requires factual output field"
            )


def _verify_render_output_targets(
    answer: AnswerProgram,
    *,
    require_output: bool = True,
) -> None:
    operation_outputs = {
        operation.output_relation
        for operation in answer.operations
        if operation.output_relation
    }
    terminal_outputs = operation_outputs - set(
        _operation_input_refs_for_all(answer.operations)
    )
    render_outputs = tuple(answer.render_spec.relation_outputs)
    scalar_outputs = answer.render_spec.scalar_outputs
    _verify_unique_render_output_ids(render_outputs, scalar_outputs)
    if not render_outputs and not scalar_outputs and not require_output:
        return
    if not render_outputs and not scalar_outputs and require_output:
        raise VerificationError("render spec requires at least one render output")
    if render_outputs:
        render_relations = {
            relation_output.relation_id for relation_output in render_outputs
        }
        unknown_render_relations = render_relations - operation_outputs
        if unknown_render_relations:
            raise VerificationError(
                f"render output {render_outputs[0].id} references unknown operation output"
            )
        non_terminal_render_relations = render_relations - terminal_outputs
        if non_terminal_render_relations:
            raise VerificationError("render spec requires terminal final relation")
        if terminal_outputs - render_relations:
            raise VerificationError("render spec cannot leave terminal relation output")
    elif terminal_outputs:
        raise VerificationError("render spec cannot leave terminal relation output")
    _verify_render_scalar_references(answer, scalar_outputs=scalar_outputs)


def _verify_unique_render_output_ids(
    relation_outputs: tuple[RenderRelationOutput, ...],
    scalar_outputs: tuple[RenderScalarOutput, ...],
) -> None:
    seen: set[str] = set()
    for output in (*relation_outputs, *scalar_outputs):
        output_id = output.id
        if not output_id:
            raise VerificationError("render output requires id")
        if output_id in seen:
            raise VerificationError(f"duplicate render output {output_id}")
        seen.add(output_id)


def _verify_render_scalar_references(
    answer: AnswerProgram,
    *,
    scalar_outputs: tuple[RenderScalarOutput, ...],
) -> None:
    rendered_scalars = {
        scalar_output.scalar_id
        for scalar_output in scalar_outputs
    }
    compute_outputs = {
        operation.spec.output_scalar
        for operation in answer.operations
        if isinstance(operation.spec, ComputeSpec)
    }
    missing = rendered_scalars - compute_outputs
    if missing:
        raise VerificationError(
            "render scalar output references unknown scalar "
            + ", ".join(sorted(missing))
        )
    consumed_scalars = {
        scalar_input
        for operation in answer.operations
        for scalar_input in _operation_scalar_inputs(operation)
        if scalar_input in compute_outputs
    }
    unrendered = compute_outputs - rendered_scalars - consumed_scalars
    if unrendered:
        raise VerificationError(
            "unrendered scalar output " + ", ".join(sorted(unrendered))
        )


def _operation_input_refs_for_all(operations: tuple[Operation, ...]) -> tuple[str, ...]:
    refs: list[str] = []
    for operation in operations:
        refs.extend(_operation_input_refs(operation))
    return tuple(refs)
