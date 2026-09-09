"""SQL computation inside the canonical verified relation-operation kernel."""
from sqlglot import exp
from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

from fervis.lookup.answer_program.operations import SqlQuerySpec
from fervis.lookup.answer_program.expressions import expression_references
from fervis.lookup.plan_execution.declared_values import parse_declared_value
from fervis.lookup.plan_execution.errors import VerificationError
from fervis.lookup.plan_execution.relations import CompletenessStatus
from fervis.lookup.outcomes.errors import IncompleteEvidenceError
from fervis.lookup.plan_execution.operation_engine.expression_evaluator import evaluate_expression
from fervis.lookup.plan_execution.operation_engine.shared import _operation_relation
from fervis.lookup.plan_execution.verification.contract_types import ProofLineage, RelationContract
from fervis.lookup.answer_program.relations import FieldBindingRole
from fervis.lookup.relational_sql.acquisition import _sql_type
from fervis.lookup.relational_sql.execution import SqlTable, _validate, execute_query, QueryValidationError
from fervis.lookup.relational_sql.identity_lineage import sql_identity_keys


def validate_sql_operation(spec: SqlQuerySpec):
    from .execution import query_timezone
    query_timezone(spec.timezone)
    tables = {item.name: SqlTable({column.name: 'TEXT' for column in item.columns}, ()) for item in spec.inputs}
    statement, sources = _validate(spec.query, tables)
    if set(sources) != set(tables):
        raise VerificationError('SQL inputs must exactly match referenced views')
    if {node.name for node in statement.find_all(exp.Placeholder)} != {item.name for item in spec.parameters}:
        raise VerificationError('SQL parameters must exactly match declared bindings')
    if any(expression_references(item.expression).fields for item in spec.parameters):
        raise VerificationError('SQL parameters must be scalar program inputs')
    from .column_usage import project_query
    project_query(spec.query,{name:table.columns for name,table in tables.items()},
                  output_columns=tuple(item.id for item in spec.outputs))


def sql_relation_contract(operation, contracts, proof_context):
    spec = operation.spec
    validate_sql_operation(spec)
    proof = ProofLineage.value(frozenset(proof_context.operation_refs.get(operation.id, ())))
    for item in spec.inputs:
        source = contracts[item.relation_id]
        proof = proof.merge(source.row_proof)
        for column in item.columns:
            if column.field_id not in source.fields:
                raise VerificationError('SQL input column is not in its relation contract')
            proof = proof.merge(source.field_proofs[column.field_id])
    return RelationContract(
        fields={item.id: frozenset({FieldBindingRole.OUTPUT}) for item in spec.outputs},
        grain_keys=(), entity_keys=sql_identity_keys(spec,contracts), field_proofs={item.id:proof for item in spec.outputs},
        field_types={item.id:item.value_type for item in spec.outputs}, row_proof=proof,
    )


def execute_sql_operation(operation, relations, *, environment, operation_refs=()):
    spec = operation.spec
    tables = {}
    inputs = tuple(relations[item.relation_id] for item in spec.inputs)
    for item, relation in zip(spec.inputs, inputs):
        if relation.completeness.status is not CompletenessStatus.COMPLETE:
            raise IncompleteEvidenceError(relation_id=relation.id, proof_refs=relation.completeness.proof_refs)
        rows = tuple({column.name:row[column.field_id] for column in item.columns} for row in relation.rows)
        types = {column.name:_sql_type((relation.field_types or {})[column.field_id],
                    values=tuple(row[column.name] for row in rows)) for column in item.columns}
        tables[item.name] = SqlTable(types, rows)
    parameters = {item.name:evaluate_expression(item.expression, environment=environment).value for item in spec.parameters}
    result = execute_query(spec.query, tables=tables, parameters=parameters, timezone=spec.timezone)
    from .column_usage import require_output_columns
    try:
        require_output_columns(result.columns, (item.id for item in spec.outputs))
    except QueryValidationError as exc:
        raise VerificationError(str(exc)) from exc
    positions = {name:index for index,name in enumerate(result.columns)}
    rows = tuple({item.id:_sql_output_value(row[positions[item.id]], item.value_type)
                  for item in spec.outputs} for row in result.rows)
    if spec.reference_input_ref:
        from .reference_results import require_unique_reference
        rows=require_unique_reference(spec,rows,relation_id=operation.output_relation,
            proof_refs=tuple(dict.fromkeys((*operation_refs,*(ref for relation in inputs for ref in relation.evidence.proof_refs)))))
    return _operation_relation(operation, rows, grain_keys=(), inputs=inputs,
        field_types={item.id:item.value_type for item in spec.outputs}, scalar_refs=operation_refs)


def _sql_output_value(value, value_type):
    # SQL has already assigned a scalar type. API wire-format coercions must
    # not reinterpret an observed string, Boolean, or numeric value here.
    kinds = {'integer': (int,), 'number': (int, Decimal), 'string': (str,),
             'boolean': (bool,), 'date': (date,), 'datetime': (datetime,), 'uuid': (str, UUID)}
    if value is not None and type(value) not in kinds[value_type]:
        raise VerificationError('SQL result type disagrees with its declared output type')
    return parse_declared_value(value, value_type)
