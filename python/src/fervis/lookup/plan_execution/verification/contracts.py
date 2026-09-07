"""Relation contracts for answer-program verification."""

from ._shared import (
    Relation,
    RelationCatalog,
    RowSourceCatalog,
    SourceKind,
    read_evidence_ref,
    read_field_evidence_ref,
    row_source_evidence_ref,
    row_source_field_evidence_ref,
)
from fervis.lookup.answer_program.model import RelationProgram, RelationGuaranteeDeclaration
from .contract_types import (
    ProofLineage,
    RelationContract,
    RelationEntityKey,
    RelationEntityKeyComponent,
    RelationSemanticGuarantee,
    ScalarContract,
)
from .execution_proof import ExecutionProofContext
from .operation_contracts import _operation_relation_contract
from fervis.lookup.plan_execution.expression_schema import expression_value_type
from fervis.lookup.answer_program.inputs import parameter_runtime_type
from fervis.lookup.answer_program.operations import operation_scalar_output_ids
from fervis.lookup.answer_program.operations import (
    ComputeSpec,
    Operation,
)
from fervis.lookup.answer_program.expressions import fold_expression
from fervis.lookup.answer_program.expressions import expression_input_id
from fervis.lookup.answer_program.values import NodeOutputRef
from fervis.lookup.plan_execution.operation_runtime import ResolvedOperationInput
from .sources import _row_source_for_relation, _source_mechanic_proof_refs


def _relation_contracts(
    answer: RelationProgram,
    *,
    catalog: RelationCatalog | None,
    row_sources: RowSourceCatalog,
    proof_context: ExecutionProofContext,
    guarantee_declarations: tuple[RelationGuaranteeDeclaration,...] = (),
) -> dict[str, RelationContract]:
    declarations = {item.relation_id: item for item in guarantee_declarations}
    if len(declarations) != len(guarantee_declarations):
        raise ValueError("answer program repeats a relation guarantee")
    from dataclasses import replace
    from fervis.lookup.answer_program.dependencies import execution_schedule
    from fervis.lookup.answer_program.expressions import expression_references
    from fervis.lookup.plan_execution.errors import VerificationError
    contracts: dict[str, RelationContract] = {}
    scalar_types = {f"parameter:{p.id}":parameter_runtime_type(p.value_type) for p in answer.parameters}
    node_output_types: dict[str, dict[str, str]] = {}
    for item in execution_schedule(answer):
        if isinstance(item, Relation):
            contract = _base_relation_contract(item, catalog=catalog, row_sources=row_sources, proof_context=proof_context)
            if item.source.argument_relation_id:
                parent = contracts[item.source.argument_relation_id]
                field_ids = tuple(ref.field_id for binding in item.source.param_bindings
                                  for ref in expression_references(binding.value_expr).fields)
                if any(field_id not in parent.fields for field_id in field_ids):
                    raise VerificationError('dependent argument references an unavailable parent field')
                argument_proof = parent.row_proof.merge(*(parent.field_proofs[field_id] for field_id in field_ids))
                contract = replace(contract,
                    row_proof=contract.row_proof.merge(argument_proof),
                    field_proofs={key:proof.merge(argument_proof) for key,proof in contract.field_proofs.items()})
            contracts[item.id] = _with_declared_semantic_guarantee(contract, declarations.pop(item.id, None))
            continue
        operation = item
        if isinstance(operation.spec, ComputeSpec):
            node_output_types[operation.id] = {operation.spec.output_scalar:expression_value_type(operation.spec.expression, scalar_types=scalar_types, node_output_types=node_output_types)}
        if not operation.output_relation:
            continue
        contract = _operation_relation_contract(
            operation, contracts, proof_context=proof_context,
            scalar_types=scalar_types, node_output_types=node_output_types,
        )
        node_output_types[operation.id] = {key:contract.field_types.get(key, "") for key in operation_scalar_output_ids(operation.spec)}
        contracts[operation.output_relation] = _with_declared_semantic_guarantee(
            contract, declarations.pop(operation.output_relation, None),
        )
    if declarations:
        raise ValueError("relation guarantee references an unknown relation")
    return contracts


def _with_declared_semantic_guarantee(contract, declaration):
    if declaration is None:
        return contract
    fact_id = declaration.qualification.requested_fact_id
    if fact_id in contract.semantic_guarantees:
        raise ValueError("relation repeats a fact guarantee")
    required_refs = {
        ref
        for proof in declaration.qualification.atom_proofs
        for ref in proof.proof_refs
    } | set(declaration.subject.proof_refs)
    if not required_refs <= set(contract.row_proof.value_refs):
        raise ValueError("relation guarantee lacks executable proof evidence")
    return RelationContract(
        fields=dict(contract.fields),
        grain_keys=contract.grain_keys,
        field_proofs=dict(contract.field_proofs),
        field_types=dict(contract.field_types),
        entity_keys=contract.entity_keys,
        row_proof=contract.row_proof,
        semantic_guarantees={
            **contract.semantic_guarantees,
            fact_id: RelationSemanticGuarantee(
                qualification=declaration.qualification,
                subject=declaration.subject,
            ),
        },
    )


def _scalar_contracts(
    answer: RelationProgram,
    *,
    relation_contracts: dict[str, RelationContract],
    operation_inputs: tuple[ResolvedOperationInput, ...],
) -> dict[str, ScalarContract]:
    """Fold scalar proof through the existing compute-expression graph."""

    inputs: dict[tuple[str, str], ScalarContract] = {}
    for item in operation_inputs:
        key = (item.operation_id, item.input_id)
        declared = inputs.get(key, ScalarContract())
        inputs[key] = ScalarContract(
            proof=ProofLineage(
                value_refs=frozenset({*declared.proof.value_refs, *item.proof_refs})
            ),
            semantic_guarantees=dict(declared.semantic_guarantees),
        )
    operations = {operation.id: operation for operation in answer.operations}
    scalars: dict[str, ScalarContract] = {}

    for operation in answer.operations:
        spec = operation.spec
        if not isinstance(spec, ComputeSpec):
            continue
        proof = fold_expression(
            spec.expression,
            field=lambda _item: ScalarContract(),
            parameter=lambda item: _scalar_input_contract(
                inputs,
                operation_id=operation.id,
                input_id=expression_input_id(item),
            ),
            constant=lambda item: _scalar_input_contract(
                inputs,
                operation_id=operation.id,
                input_id=expression_input_id(item),
            ),
            environment=lambda _item: ScalarContract(),
            output=lambda item: _node_output_contract(
                item,
                operations=operations,
                relation_contracts=relation_contracts,
                scalar_contracts=scalars,
            ),
            unary=lambda _item, operand: operand,
            binary=lambda _item, left, right: left.combine(right),
            function=lambda _item, arguments: _combine_scalar_contracts(arguments),
        )
        scalars[spec.output_scalar] = proof
    return scalars


def _combine_scalar_contracts(
    contracts: tuple[ScalarContract, ...],
) -> ScalarContract:
    output = ScalarContract()
    for contract in contracts:
        output = output.combine(contract)
    return output


def _scalar_input_contract(
    inputs: dict[tuple[str, str], ScalarContract],
    *,
    operation_id: str,
    input_id: str,
) -> ScalarContract:
    return inputs.get((operation_id, input_id), ScalarContract())


def _node_output_contract(
    ref: NodeOutputRef,
    *,
    operations: dict[str, Operation],
    relation_contracts: dict[str, RelationContract],
    scalar_contracts: dict[str, ScalarContract],
) -> ScalarContract:
    operation = operations.get(ref.node_id)
    if operation is None:
        return ScalarContract()
    spec = operation.spec
    if isinstance(spec, ComputeSpec):
        if spec.output_scalar != ref.output_id:
            return ScalarContract()
        return scalar_contracts.get(ref.output_id, ScalarContract())
    output_relation = operation.output_relation
    relation = relation_contracts.get(output_relation)
    if relation is None:
        return ScalarContract()
    return ScalarContract(
        proof=relation.field_proofs.get(ref.output_id, ProofLineage()),
        semantic_guarantees=dict(relation.semantic_guarantees),
    )


def _base_relation_contract(
    relation: Relation,
    *,
    catalog: RelationCatalog | None,
    row_sources: RowSourceCatalog,
    proof_context: ExecutionProofContext,
) -> RelationContract:
    fields = {field.field_id: frozenset(field.roles) for field in relation.fields}
    row_proof = _relation_source_row_proof(
        relation,
        catalog=catalog,
        row_sources=row_sources,
        endpoint_arg_scope_refs=proof_context.endpoint_arg_scope_refs,
    )
    field_proofs = {
        field.field_id: _binding_proof(
            relation,
            field.field_id,
            catalog=catalog,
            row_sources=row_sources,
        ).merge(row_proof)
        for field in relation.fields
    }
    field_types: dict[str, str] = {}
    if relation.source.kind in {
        SourceKind.API_READ,
        SourceKind.GENERATED_CALENDAR,
        SourceKind.MEMORY_READ,
    }:
        try:
            row_source = _row_source_for_relation(relation, row_sources=row_sources)
            field_types = {
                field.field_id: row_source.field(field.field_id).type.value
                for field in relation.fields
            }
        except KeyError:
            field_types = {}
    return RelationContract(
        fields=fields,
        grain_keys=relation.grain_keys,
        field_proofs=field_proofs,
        field_types=field_types,
        entity_keys=_relation_entity_keys(relation, row_sources=row_sources),
        row_proof=row_proof,
        semantic_guarantees={},
    )


def _relation_entity_keys(
    relation: Relation,
    *,
    row_sources: RowSourceCatalog,
) -> tuple[RelationEntityKey, ...]:
    if relation.source.kind not in {
        SourceKind.API_READ,
        SourceKind.GENERATED_CALENDAR,
        SourceKind.MEMORY_READ,
    }:
        return ()
    row_source = _row_source_for_relation(relation, row_sources=row_sources)
    relation_field_ids = {field.field_id for field in relation.fields}
    keys = [
        RelationEntityKey(
            entity_kind=key.entity_kind,
            key_id=key.id,
            components=tuple(
                RelationEntityKeyComponent(
                    component_id=component.id,
                    field_id=component.field_id,
                )
                for component in key.components
            ),
        )
        for key in row_source.candidate_keys
        if all(component.field_id in relation_field_ids for component in key.components)
    ]
    keys.extend(
        RelationEntityKey(
            entity_kind=reference.target_entity_kind,
            key_id=reference.target_key_id,
            components=tuple(
                RelationEntityKeyComponent(
                    component_id=component.target_component_id,
                    field_id=component.local_field_id,
                )
                for component in reference.components
            ),
        )
        for reference in row_source.entity_references
        if all(
            component.local_field_id in relation_field_ids
            for component in reference.components
        )
    )
    return tuple(dict.fromkeys(keys))


def _relation_source_row_proof(
    relation: Relation,
    *,
    catalog: RelationCatalog | None,
    row_sources: RowSourceCatalog,
    endpoint_arg_scope_refs: dict[str, frozenset[str]],
) -> ProofLineage:
    if catalog is None or relation.source.kind not in {
        SourceKind.API_READ,
        SourceKind.GENERATED_CALENDAR,
        SourceKind.MEMORY_READ,
    }:
        return ProofLineage()
    try:
        row_source = _row_source_for_relation(relation, row_sources=row_sources)
    except KeyError:
        return ProofLineage()
    value_refs: set[str] = set()
    proof_refs = _source_mechanic_proof_refs(relation)
    if row_source.read_id:
        value_refs.add(read_evidence_ref(row_source.read_id))
    else:
        value_refs.add(row_source_evidence_ref(row_source.id))
    proof_refs.update(endpoint_arg_scope_refs.get(relation.id, frozenset()))
    return ProofLineage(value_refs=frozenset({*value_refs, *proof_refs}))


def _binding_proof(
    relation: Relation,
    field_id: str,
    *,
    catalog: RelationCatalog | None,
    row_sources: RowSourceCatalog,
) -> ProofLineage:
    refs = {field_id}
    if catalog is None or relation.source.kind not in {
        SourceKind.API_READ,
        SourceKind.GENERATED_CALENDAR,
        SourceKind.MEMORY_READ,
    }:
        return ProofLineage.value(frozenset(refs))
    try:
        row_source = _row_source_for_relation(relation, row_sources=row_sources)
        row_source_field = row_source.field(field_id)
    except KeyError:
        return ProofLineage.value(frozenset(refs))
    if row_source_field.request_parameter_ref:
        from fervis.lookup.relation_catalog.row_sources import row_source_param_evidence_ref
        param = next(param for param in row_source.params if param.param_ref == row_source_field.request_parameter_ref)
        refs.add(row_source_param_evidence_ref(row_source_id=row_source.id,param_id=param.id))
        return ProofLineage.value(frozenset(refs))
    refs.add(
        read_field_evidence_ref(
            read_id=row_source.read_id, field_id=row_source_field.id
        )
        if row_source.read_id and not row_source_field.declared_entity_kind
        else row_source_field_evidence_ref(
            row_source_id=row_source.id,
            field_id=row_source_field.id,
        )
    )
    refs.update(row_source_field.fact_refs)
    return ProofLineage.value(frozenset(refs))
