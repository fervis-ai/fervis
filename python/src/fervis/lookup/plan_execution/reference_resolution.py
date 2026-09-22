"""Certify complete reference-query results before publishing scalar key values."""

from fervis.lookup.identity_types import (
    IdentityExecutionFailureReason,
    ReferenceResolutionFailure,
)
from fervis.lookup.outcomes.errors import UnresolvedReferenceError


def require_unique_reference(rows, *, input_ref, operand, entity_key, relation_id, proof_refs):
    if entity_key is None:
        if len(rows) == 1:
            return rows
        raise UnresolvedReferenceError(ReferenceResolutionFailure(input_ref,
            IdentityExecutionFailureReason.NOT_FOUND if not rows else IdentityExecutionFailureReason.AMBIGUOUS_RESULT,
            operand=operand), relation_id=relation_id, proof_refs=proof_refs)
    projection = entity_key
    keys = {}
    try:
        for row in rows:
            keys.setdefault(projection.project(row), row)
    except (ValueError, TypeError) as exc:
        raise UnresolvedReferenceError(
            ReferenceResolutionFailure(
                input_ref,
                IdentityExecutionFailureReason.INVALID_RESOLVER_RESULT,
                operand=operand,
            ),
            relation_id=relation_id,
            proof_refs=proof_refs,
        ) from exc
    if len(keys) != 1:
        reason = (
            IdentityExecutionFailureReason.NOT_FOUND
            if not keys
            else IdentityExecutionFailureReason.AMBIGUOUS_RESULT
        )
        raise UnresolvedReferenceError(
            ReferenceResolutionFailure(
                input_ref, reason, tuple(keys), operand
            ),
            relation_id=relation_id,
            proof_refs=proof_refs,
        )
    return (next(iter(keys.values())),)


def execute_reference_guard(operation, relations, *, operation_refs=()):
    from fervis.lookup.plan_execution.relations import CompletenessStatus
    from fervis.lookup.outcomes.errors import IncompleteEvidenceError
    from fervis.lookup.plan_execution.operation_engine.shared import _operation_relation
    spec = operation.spec
    source = relations[spec.input_relation]
    if source.completeness.status is not CompletenessStatus.COMPLETE:
        raise IncompleteEvidenceError(relation_id=source.id, proof_refs=source.completeness.proof_refs)
    rows = tuple({field:row[field] for field in spec.fields} for row in source.rows)
    rows = require_unique_reference(rows, input_ref=spec.reference_input_ref, operand=spec.reference_operand,
        entity_key=spec.entity_key, relation_id=operation.output_relation,
        proof_refs=tuple(dict.fromkeys((*operation_refs,*source.evidence.proof_refs))))
    return _operation_relation(operation,rows,grain_keys=spec.fields if spec.entity_key is not None else (),inputs=(source,),
        field_types={field:source.field_types[field] for field in spec.fields},scalar_refs=operation_refs)


def reference_guard_contract(operation, contracts, proof_context):
    from fervis.lookup.plan_execution.errors import VerificationError
    from fervis.lookup.plan_execution.verification.contract_types import (
        RelationContract,RelationEntityKey,RelationEntityKeyComponent,ProofLineage,
    )
    spec = operation.spec
    source = contracts[spec.input_relation]
    if not set(spec.fields) <= set(source.fields):
        raise VerificationError('Reference guard requires observed input fields')
    keys = ()
    if spec.entity_key is not None:
        key = RelationEntityKey(spec.entity_key.entity_kind,spec.entity_key.key_id,
            tuple(RelationEntityKeyComponent(item.component_id,item.field_id) for item in spec.entity_key.components))
        if not any(candidate.entity_kind==key.entity_kind and candidate.key_id==key.key_id
                   and set(candidate.components)==set(key.components) for candidate in source.entity_keys):
            raise VerificationError('Reference guard key must preserve its declared input identity authority')
        keys = (key,)
    proof = source.row_proof.merge(ProofLineage.value(frozenset(proof_context.operation_refs.get(operation.id, ()))))
    return RelationContract(fields={field:source.fields[field] for field in spec.fields},
        grain_keys=spec.fields if keys else (),entity_keys=keys,
        field_proofs={field:source.field_proofs[field].merge(proof) for field in spec.fields},
        field_types={field:source.field_types[field] for field in spec.fields},row_proof=proof,
        semantic_guarantees=dict(source.semantic_guarantees))


def verify_reference_candidate_completeness(program):
    """Recheck persisted reference ancestry before any source read."""
    from fervis.lookup.answer_program.operations import SqlQuerySpec, ReferenceGuardSpec, OrderSpec, KeepAll
    from fervis.lookup.plan_execution.errors import VerificationError
    from fervis.lookup.relational_sql.reference_matching import reject_reference_truncation
    from fervis.lookup.relational_sql.execution import QueryValidationError
    producers = {operation.output_relation: operation for operation in program.operations if operation.output_relation}
    pending = [operation for operation in program.operations
               if isinstance(operation.spec, ReferenceGuardSpec)]
    seen = set()
    while pending:
        operation = pending.pop()
        if operation.id in seen:
            continue
        seen.add(operation.id)
        if isinstance(operation.spec, OrderSpec) and not isinstance(operation.spec.selection, KeepAll):
            raise VerificationError("Reference candidates cannot be truncated before uniqueness is checked")
        if isinstance(operation.spec, SqlQuerySpec):
            try:
                reject_reference_truncation(operation.spec.query)
            except QueryValidationError as exc:
                raise VerificationError(str(exc)) from exc
        pending.extend(producers[relation] for relation in operation.input_relation_ids if relation in producers)


def verify_observed_reference_guards(program):
    from types import SimpleNamespace
    from fervis.lookup.answer_program.operations import ReferenceGuardSpec
    from fervis.lookup.plan_execution.verification.record_lineage import verify_record_projection, occurrence_number_origin
    from fervis.lookup.plan_execution.errors import VerificationError
    for operation in program.operations:
        spec = operation.spec
        if isinstance(spec,ReferenceGuardSpec) and spec.entity_key is None:
            for field in spec.occurrence_fields:
                if occurrence_number_origin(program, spec.input_relation, field) is None:
                    raise VerificationError('Reference occurrence key requires a carried row number')
            verify_record_projection(program, SimpleNamespace(relation_id=operation.output_relation,
                record_fields={field:field for field in spec.fields if field not in spec.occurrence_fields}),
                preserve_occurrences=True, preserve_property_names=not bool(spec.occurrence_fields))
