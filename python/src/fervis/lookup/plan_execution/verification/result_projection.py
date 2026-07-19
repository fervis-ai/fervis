"""Render-reference checks for fact-plan verification."""

from ._shared import (
    AnswerProgram,
    ComputeSpec,
    FieldBindingRole,
    Operation,
    VerificationError,
)
from .contract_types import (
    PopulationCoverage,
    ProofLineage,
    RelationContract,
    RelationEntityKey,
    RelationEntityKeyComponent,
)
from .contracts import _scalar_contracts
from .operations import _operation_input_refs
from .scalars import _operation_scalar_inputs
from fervis.lookup.plan_execution.operation_runtime import ResolvedOperationInput
from fervis.lookup.answer_program.result_projection import (
    RelationResultOutput,
    ScalarResultOutput,
)


def _result_output_fact_refs(
    answer: AnswerProgram,
    *,
    relation_contracts: dict[str, RelationContract],
    operation_inputs: tuple[ResolvedOperationInput, ...],
) -> dict[str, frozenset[str]]:
    proofs = _result_output_proofs(
        answer,
        relation_contracts=relation_contracts,
        operation_inputs=operation_inputs,
    )
    return {
        output_id: proof.fulfillment_refs()
        for output_id, proof in proofs.items()
    }


def _result_output_proofs(
    answer: AnswerProgram,
    *,
    relation_contracts: dict[str, RelationContract],
    operation_inputs: tuple[ResolvedOperationInput, ...],
) -> dict[str, ProofLineage]:
    proofs: dict[str, ProofLineage] = {}
    for result_output in answer.result_projection.relation_outputs:
        contract = relation_contracts.get(result_output.relation_id)
        if contract is None:
            continue
        field_ids = _result_output_field_ids(result_output)
        field_proofs = tuple(
            contract.field_proofs.get(field_id, ProofLineage())
            for field_id in field_ids
        )
        proofs[result_output.id] = ProofLineage(
            value_refs=frozenset(
                ref for field_proof in field_proofs for ref in field_proof.value_refs
            ),
            population_coverage=PopulationCoverage.guaranteed_by_every(
                tuple(
                    field_proof.population_coverage for field_proof in field_proofs
                )
            ),
        )
    scalar_contracts = _scalar_contracts(
        answer,
        relation_contracts=relation_contracts,
        operation_inputs=operation_inputs,
    )
    for scalar_output in answer.result_projection.scalar_outputs:
        scalar_contract = scalar_contracts.get(scalar_output.scalar_id)
        proofs[scalar_output.id] = (
            scalar_contract.proof
            if scalar_contract is not None
            else ProofLineage()
        )
    return proofs


def _verify_result_references(
    answer: AnswerProgram,
    *,
    relation_contracts: dict[str, RelationContract],
) -> None:
    _verify_result_output_targets(answer)
    result_outputs = tuple(answer.result_projection.relation_outputs)
    for relation_output in result_outputs:
        contract = relation_contracts.get(relation_output.relation_id)
        field_ids = _result_output_field_ids(relation_output)
        if contract is None or any(
            field_id not in contract.fields for field_id in field_ids
        ):
            raise VerificationError(
                f"result output {relation_output.id} references unknown output field"
            )
        if relation_output.entity_key is not None:
            _verify_declared_entity_key(relation_output, contract=contract)
        if relation_output.entity_key is None and any(
            FieldBindingRole.IDENTITY in contract.fields[field_id]
            and FieldBindingRole.OUTPUT not in contract.fields[field_id]
            for field_id in field_ids
        ):
            raise VerificationError(
                f"result output {relation_output.id} requires entity key metadata"
            )
        if any(
            not (
                {FieldBindingRole.OUTPUT, FieldBindingRole.IDENTITY}
                & set(contract.fields[field_id])
            )
            for field_id in field_ids
        ):
            raise VerificationError(
                f"result output {relation_output.id} requires factual output field"
            )


def _verify_declared_entity_key(
    output: RelationResultOutput,
    *,
    contract: RelationContract,
) -> None:
    projection = output.entity_key
    assert projection is not None
    projected_key = RelationEntityKey(
        entity_kind=projection.entity_kind,
        key_id=projection.key_id,
        components=tuple(
            RelationEntityKeyComponent(
                component_id=component.component_id,
                field_id=component.field_id,
            )
            for component in projection.components
        ),
    )
    if projected_key not in contract.entity_keys:
        raise VerificationError(
            f"result output {output.id} requires declared entity key"
        )


def _verify_result_output_targets(
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
    result_outputs = tuple(answer.result_projection.relation_outputs)
    scalar_outputs = answer.result_projection.scalar_outputs
    _verify_unique_result_output_ids(result_outputs, scalar_outputs)
    if not result_outputs and not scalar_outputs and not require_output:
        return
    if not result_outputs and not scalar_outputs and require_output:
        raise VerificationError("result projection requires at least one result output")
    if result_outputs:
        result_relations = {
            relation_output.relation_id for relation_output in result_outputs
        }
        unknown_result_relations = result_relations - operation_outputs
        if unknown_result_relations:
            raise VerificationError(
                f"result output {result_outputs[0].id} references unknown operation output"
            )
        non_terminal_result_relations = result_relations - terminal_outputs
        if non_terminal_result_relations:
            raise VerificationError(
                "result projection requires terminal final relation"
            )
        if terminal_outputs - result_relations:
            raise VerificationError(
                "result projection cannot leave terminal relation output"
            )
    elif terminal_outputs:
        raise VerificationError(
            "result projection cannot leave terminal relation output"
        )
    _verify_result_scalar_references(answer, scalar_outputs=scalar_outputs)


def _verify_unique_result_output_ids(
    relation_outputs: tuple[RelationResultOutput, ...],
    scalar_outputs: tuple[ScalarResultOutput, ...],
) -> None:
    output_ids = (
        *(output.id for output in relation_outputs),
        *(output.id for output in scalar_outputs),
    )
    seen: set[str] = set()
    for output_id in output_ids:
        if not output_id:
            raise VerificationError("result output requires id")
        if output_id in seen:
            raise VerificationError(f"duplicate result output {output_id}")
        seen.add(output_id)


def _verify_result_scalar_references(
    answer: AnswerProgram,
    *,
    scalar_outputs: tuple[ScalarResultOutput, ...],
) -> None:
    projected_scalars = {scalar_output.scalar_id for scalar_output in scalar_outputs}
    compute_outputs = {
        operation.spec.output_scalar
        for operation in answer.operations
        if isinstance(operation.spec, ComputeSpec)
    }
    missing = projected_scalars - compute_outputs
    if missing:
        raise VerificationError(
            "result scalar output references unknown scalar "
            + ", ".join(sorted(missing))
        )
    consumed_scalars = {
        scalar_input
        for operation in answer.operations
        for scalar_input in _operation_scalar_inputs(operation)
        if scalar_input in compute_outputs
    }
    unprojected = compute_outputs - projected_scalars - consumed_scalars
    if unprojected:
        raise VerificationError(
            "unprojected scalar output " + ", ".join(sorted(unprojected))
        )


def _operation_input_refs_for_all(operations: tuple[Operation, ...]) -> tuple[str, ...]:
    refs: list[str] = []
    for operation in operations:
        refs.extend(_operation_input_refs(operation))
    return tuple(refs)


def _result_output_field_ids(output: RelationResultOutput) -> tuple[str, ...]:
    if output.entity_key is not None:
        return tuple(component.field_id for component in output.entity_key.components)
    return (output.field_id,)
