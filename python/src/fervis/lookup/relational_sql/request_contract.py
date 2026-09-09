"""Verify one declared SQL request's complete computation and input footprint."""
from fervis.lookup.answer_program.operations import SqlQuerySpec, operation_expression_references
from fervis.lookup.answer_program.expressions import expression_references
from fervis.lookup.contract_codec import canonical_contract_fingerprint
from fervis.lookup.plan_execution.errors import VerificationError


def query_request_scope(fact, program):
    operations = {item.id:item for item in program.operations}
    relations = {item.id:item for item in program.relations}
    relation_producers = {item.output_relation:item.id for item in program.operations if item.output_relation}
    scalar_producers = {item.output_scalar:item.id for item in program.operations if item.output_scalar}
    used_operations, used_relations, used_parameters = set(),set(),set()
    visiting = set()

    def references(refs):
        used_parameters.update(item.parameter_id for item in refs.parameters)
        for item in refs.outputs:
            operation(item.node_id)

    def relation(ref):
        if ref in used_relations:
            return
        if ref in relation_producers:
            operation(relation_producers[ref])
            return
        if ref not in relations:
            raise VerificationError('SQL request depends on an unknown relation')
        used_relations.add(ref)
        source = relations[ref].source
        if source.argument_relation_id:
            relation(source.argument_relation_id)
        for binding in source.param_bindings:
            references(expression_references(binding.value_expr))

    def operation(ref):
        if ref in used_operations:
            return
        if ref in visiting or ref not in operations:
            raise VerificationError('SQL request has an invalid operation dependency')
        visiting.add(ref)
        item = operations[ref]
        for source_ref in item.input_relation_ids:
            relation(source_ref)
        for refs in operation_expression_references(item.spec):
            references(refs)
        visiting.remove(ref)
        used_operations.add(ref)

    from fervis.lookup.answer_program.result_projection import RelationResultOutput
    projections = {item.id:item for item in (*program.result_projection.relation_outputs,
                                             *program.result_projection.scalar_outputs)}
    for output in fact.outputs:
        projection = projections.get(output.result_output_id)
        if projection is None:
            raise VerificationError('SQL request references an unavailable projection')
        if isinstance(projection, RelationResultOutput):
            relation(projection.relation_id)
        elif projection.scalar_id in scalar_producers:
            operation(scalar_producers[projection.scalar_id])
        else:
            raise VerificationError('SQL output has no scalar producer')
    return used_operations, used_relations, used_parameters


def verify_query_request(fact, program):
    operations = {item.id:item for item in program.operations}
    if not fact.outputs or len({output.id for output in fact.outputs}) != len(fact.outputs):
        raise VerificationError('SQL request outputs must be nonempty and unique')
    actual_operations, actual_relations, actual_parameters = query_request_scope(fact,program)
    declarations = {item.operation_id:item for item in fact.operations}
    if len(declarations) != len(fact.operations) or set(declarations) != actual_operations:
        raise VerificationError('SQL request must declare its complete computation')
    if not any(isinstance(operations[ref].spec,SqlQuerySpec) for ref in actual_operations):
        raise VerificationError('SQL request requires a SQL producer')
    for ref in actual_operations:
        if declarations[ref].fingerprint != canonical_contract_fingerprint(operations[ref]):
            raise VerificationError('SQL computation differs from the declared request')
    sources = {item.relation_id:item for item in fact.sources}
    actual_sources = {item.id:item for item in program.relations}
    if len(sources) != len(fact.sources) or set(sources) != actual_relations:
        raise VerificationError('SQL request must declare its complete source graph')
    if any(sources[ref].fingerprint != canonical_contract_fingerprint(actual_sources[ref]) for ref in actual_relations):
        raise VerificationError('SQL source differs from the declared request')
    parameters = {item.id:item for item in program.parameters}
    for ref in actual_operations:
        spec = operations[ref].spec
        if isinstance(spec, SqlQuerySpec):
            for meaning in spec.meaning_inputs:
                declaration=parameters.get(meaning.parameter_id)
                lexical = declaration is not None and meaning.component == 'value' and (
                    declaration.value_type.value in {'string','named'} and meaning.item_index is None
                    or declaration.value_type.value == 'string_set' and meaning.item_index is not None
                )
                if declaration is None or not declaration.fixed_value_fingerprint or not lexical:
                    raise VerificationError('Catalog-interpreted meanings must be fixed lexical inputs')
    parameter_pins = {item.parameter_id:item for item in fact.parameters}
    if len(parameter_pins) != len(fact.parameters) or set(parameter_pins) != actual_parameters:
        raise VerificationError('SQL request must declare its complete parameter contract')
    if any(parameter_pins[ref].fingerprint != canonical_contract_fingerprint(parameters[ref]) for ref in actual_parameters):
        raise VerificationError('SQL parameter differs from the declared request')
    if not actual_parameters <= parameters.keys():
        raise VerificationError('SQL request references an undeclared program parameter')
    actual_inputs = {parameters[ref].input_ref for ref in actual_parameters if parameters[ref].input_ref}
    if set(fact.input_refs) != actual_inputs or len(set(fact.input_refs)) != len(fact.input_refs):
        raise VerificationError('SQL request input signature differs from its computation')


def verify_query_output(fact, fulfillment, program, relation_contracts):
    requested = next(output for output in fact.outputs if output.id == fulfillment.answer_output_id)
    projection = next((output for output in (*program.result_projection.relation_outputs,
                                             *program.result_projection.scalar_outputs)
                       if output.id == fulfillment.result_output_id), None)
    if (projection is None or projection.id != requested.result_output_id
            or canonical_contract_fingerprint(projection) != requested.projection_fingerprint
            or projection.role != 'answer_value'):
        raise VerificationError('SQL projection differs from the declared requested output')
    actual_type = projected_query_output_type(projection,program,relation_contracts)
    from fervis.lookup.plan_execution.declared_values import declared_comparison_types_compatible, declared_kind, DeclaredValueKind
    if requested.value_type == 'identity' or actual_type == 'identity':
        compatible = requested.value_type == actual_type
    else:
        compatible = (declared_kind(actual_type) is not DeclaredValueKind.RUNTIME
                      and declared_kind(requested.value_type) is not DeclaredValueKind.RUNTIME
                      and declared_comparison_types_compatible(actual_type,requested.value_type))
    if not compatible:
        raise VerificationError('SQL output type differs from the requested value type')



def projected_query_output_type(projection, program, relation_contracts):
    from fervis.lookup.answer_program.operations import ComputeSpec
    from fervis.lookup.answer_program.inputs import parameter_runtime_type
    from fervis.lookup.answer_program.result_projection import RelationResultOutput
    from fervis.lookup.plan_execution.expression_schema import expression_value_type
    if isinstance(projection, RelationResultOutput):
        if projection.entity_key is not None:
            return 'identity'
        return relation_contracts[projection.relation_id].field_types[projection.field_id]
    scalar_types = {f'parameter:{item.id}':parameter_runtime_type(item.value_type) for item in program.parameters}
    node_types = {}
    for operation in program.operations:
        if isinstance(operation.spec,ComputeSpec):
            kind = expression_value_type(operation.spec.expression, scalar_types=scalar_types,
                                         node_output_types=node_types)
            scalar_types[operation.spec.output_scalar] = kind
            node_types[operation.id] = {operation.spec.output_scalar:kind}
        elif operation.output_relation in relation_contracts:
            node_types[operation.id] = dict(relation_contracts[operation.output_relation].field_types)
    return scalar_types.get(projection.scalar_id, '')
