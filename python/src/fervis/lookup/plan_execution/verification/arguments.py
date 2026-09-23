"""Canonical API argument verification, including persisted scalar and row bindings."""
from fervis.lookup.api_arguments import compatible_argument, relation_argument_description
from fervis.lookup.relational_sql.parameter_usage import validate_sql_parameter_uses
from fervis.lookup.relational_sql.execution import QueryValidationError


def validate_bound_program_parameters(program, bindings, row_sources, contracts):
    """Validate all API bindings, then SQL-specific parameter uses where present."""
    from dataclasses import asdict
    from fervis.lookup.answer_program.operations import SqlQuerySpec
    from fervis.lookup.answer_program.values import ParameterRef, ConstantRef
    from fervis.lookup.answer_program.relations import SourceKind
    from fervis.lookup.relation_catalog.row_sources.lookup import (
        row_source_for_relation,
    )
    from fervis.lookup.answer_program.value_projections import projection_description

    queries = [
        operation.spec
        for operation in program.operations
        if isinstance(operation.spec, SqlQuerySpec)
    ]
    from fervis.lookup.question_contract.model import InputDenotationKind
    denotations = {item.input_ref:item for item in getattr(program, 'input_denotations', ())}
    parameter_inputs = {parameter.id:parameter.input_ref for parameter in program.parameters}
    def describe(expression, target=None, relation_id=None, literal_lookup=False):
        from fervis.lookup.answer_program.expressions import FieldRef
        if isinstance(expression, FieldRef):
            return relation_argument_description(contracts[relation_id], expression.field_id, target)
        if isinstance(expression, ParameterRef):
            binding = bindings.get(expression.parameter_id)
            value = binding.value if binding is not None else None
        elif isinstance(expression, ConstantRef):
            value = expression.value
        else:
            return {}
        description = projection_description(value, expression.component, expression.item_index) if value is not None else {}
        input_ref = parameter_inputs.get(expression.parameter_id) if isinstance(expression, ParameterRef) else None
        denotation = denotations.get(input_ref)
        if denotation is not None and denotation.kind is InputDenotationKind.IDENTITY_REFERENCE and not description.get('identity'):
            description = {**description, 'input_ref':input_ref, 'kind':'reference_literal' if literal_lookup or not denotation.reference_descriptions else 'definition'}
        return description

    parameter_inputs = {parameter.id: parameter.input_ref for parameter in program.parameters}
    lookup_relations = {}
    for spec in queries:
        if not spec.lookup_input_ref:
            continue
        if not any(isinstance(item.expression, ParameterRef) and
                   parameter_inputs.get(item.expression.parameter_id) == spec.lookup_input_ref
                   for item in spec.parameters):
            raise QueryValidationError('Literal lookup must consume its original declared input')
        denotation = denotations.get(spec.lookup_input_ref)
        if denotation is not None and any(
            describe(item.expression, literal_lookup=True).get('value') in denotation.reference_descriptions
            for item in spec.parameters if isinstance(item.expression, ParameterRef)
            and parameter_inputs.get(item.expression.parameter_id) == spec.lookup_input_ref
        ):
            raise QueryValidationError('A descriptive reference cannot be rebound as a literal lookup')
        for item in spec.inputs:
            lookup_relations.setdefault(item.relation_id, set()).add(spec.lookup_input_ref)

    for relation in program.relations:
        if relation.source.kind is not SourceKind.API_READ:
            continue
        source = row_source_for_relation(relation, row_sources=row_sources)
        for binding in relation.source.param_bindings:
            parameter = next(
                item for item in source.params if item.id == binding.param_id
            )
            literal_lookup = (isinstance(binding.value_expr, ParameterRef) and
                parameter_inputs.get(binding.value_expr.parameter_id) in lookup_relations.get(relation.id, set()))
            if not compatible_argument(asdict(parameter), describe(binding.value_expr, parameter.entity_target, relation.source.argument_relation_id, literal_lookup=literal_lookup), literal_lookup=literal_lookup):
                raise QueryValidationError(
                    "Request argument type or identity authority does not match the bound parameter"
                )
    for spec in queries:
        tables = {}
        for item in spec.inputs:
            contract = contracts[item.relation_id]
            keys = [
                {
                    "entity_kind": key.entity_kind,
                    "key_id": key.key_id,
                    "components": {component.component_id: column.name},
                }
                for key in contract.entity_keys
                for component in key.components
                for column in item.columns
                if column.field_id == component.field_id
            ]
            tables[item.name] = {
                "columns": {
                    column.name: {
                        "type": contract.field_types.get(column.field_id, "unknown")
                    }
                    for column in item.columns
                },
                "candidate_keys": keys,
            }
        validate_sql_parameter_uses(spec.query, tables,
            {item.name: describe(item.expression, literal_lookup=(isinstance(item.expression, ParameterRef) and
                parameter_inputs.get(item.expression.parameter_id) == spec.lookup_input_ref)) for item in spec.parameters},
            lookup_input_ref=spec.lookup_input_ref)
