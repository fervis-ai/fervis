"""Bind one authored query to its declared input expressions and API views."""

from dataclasses import dataclass, replace
from fervis.lookup.answer_program.operations import SqlNamedInput, Operation, ProjectSpec, NamedExpression, CrossJoinSpec
from fervis.lookup.answer_program.expressions import expression_references, FieldRef
from fervis.lookup.answer_program.values import BindingSet
from fervis.lookup.contract_codec import canonical_contract_fingerprint


@dataclass(frozen=True)
class BoundQuery:
    views: tuple
    query_parameters: tuple
    parameters: tuple
    bindings: BindingSet
    meaning_inputs: tuple
    argument_operations: tuple = ()


def bind_query_answer(authored, menu, views, *, selection_boundary=None):
    arguments = {}
    parents = {}
    operations = []
    for view in views:
        owned = tuple(item for item in authored.request_arguments if item.view == view.name)
        relation_inputs = {}
        for item in owned:
            description = menu.descriptions[item.binding]
            if description.get('kind') == 'reference_argument':
                relation_inputs.setdefault(description['relation_id'], {})[item.binding] = menu.expressions[item.binding]
        for index, (relation_id, fields) in enumerate(relation_inputs.items()):
            operation_id = f'{view.name}.arguments_{index}'
            operation = Operation(operation_id, ProjectSpec(relation_id,
                tuple(NamedExpression(name, expression) for name, expression in fields.items())),
                output_relation=operation_id+'.rows')
            operations.append(operation)
            if view.name in parents:
                joined = Operation(operation_id+'.cross', CrossJoinSpec(parents[view.name], operation.output_relation),
                                   output_relation=operation_id+'.cross.rows')
                operations.append(joined)
                parents[view.name] = joined.output_relation
            else:
                parents[view.name] = operation.output_relation
        arguments[view.name] = {item.parameter_ref:
            FieldRef(item.binding) if menu.descriptions[item.binding].get('kind') == 'reference_argument'
            else menu.expressions[item.binding] for item in owned}
    views = tuple(replace(view, arguments=arguments.get(view.name, {}),
                          argument_relation_id=parents.get(view.name, view.argument_relation_id)) for view in views)
    query_parameters = tuple(
        SqlNamedInput(name, menu.expressions[name]) for name in authored.parameter_names
    )
    meaning_inputs = tuple(
        menu.expressions[name]
        for name in dict.fromkeys((*tuple(item.input for item in authored.interpretations), *authored.definition_inputs))
    )
    fixed = {
        ref.parameter_id
        for expression in meaning_inputs
        for ref in expression_references(expression).parameters
    }
    expressions = (
        *tuple(item.expression for item in query_parameters),
        *(
            expression
            for values in arguments.values()
            for expression in values.values()
        ),
        *((selection_boundary,) if selection_boundary is not None else ()),
    )
    used = fixed | {
        ref.parameter_id
        for expression in expressions
        for ref in expression_references(expression).parameters
    }
    parameters = tuple(
        replace(
            item,
            fixed_value_fingerprint=canonical_contract_fingerprint(
                menu.program_inputs.bindings.get(item.id).value.payload
            ),
        )
        if item.id in fixed
        else item
        for item in menu.program_inputs.parameters
        if item.id in used
    )
    bindings = BindingSet.from_bindings(
        tuple(
            item
            for item in menu.program_inputs.bindings.bindings
            if item.parameter_id in used
        )
    )
    return BoundQuery(views, query_parameters, parameters, bindings, meaning_inputs, tuple(operations))


def reads_requiring_access_discovery(authored, views, *, catalog):
    """Discover physical prerequisites only for selected, still-unbound reads."""
    from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
    from fervis.lookup.relation_catalog.model import requires_caller_supplied_input

    sources = {source.id:source for source in build_api_row_source_catalog(catalog).sources}
    result = []
    for view in views:
        if view.name not in authored.referenced_views:
            continue
        source = sources[view.row_source_id]
        supplied = {item.parameter_ref for item in authored.request_arguments if item.view == view.name}
        if any(requires_caller_supplied_input(param) and param.param_ref not in supplied for param in source.params):
            result.append(source.read_id)
    return tuple(dict.fromkeys(result))
