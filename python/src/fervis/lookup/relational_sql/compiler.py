"""Compile a declared SQL answer into the single canonical answer program."""
from dataclasses import dataclass, replace
from typing import Mapping

from fervis.lookup.answer_program.model import AnswerProgram, FactFulfillment, RelationProgram
from fervis.lookup.answer_program.operations import (
    Operation, SqlQuerySpec, SqlRelationInput, SqlColumnBinding, SqlNamedInput, SqlOutputField,
    OrderSpec, SortKey, SortDirection, OrderSelection, KeepAll, Take, AtPosition,
    AggregateSpec, AggregationSpec, AggregationFunction, ProjectSpec, NamedExpression,
)
from fervis.lookup.answer_program.expressions import BinaryExpression, ExpressionBinaryOperator, FieldRef, Expression
from fervis.lookup.answer_program.values import ConstantRef, FactValue, LiteralType, BindingSet, ParameterDeclaration, ParameterBinding, ParameterRef
from fervis.lookup.answer_program.result_projection import ResultProjection, RelationResultOutput
from fervis.lookup.answer_program.compilation import compile_answer_program, close_catalog_defaults
from fervis.lookup.answer_program.request_projection import project_request_arguments
from fervis.lookup.answer_program.inputs import compile_relation_program_inputs
from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
from fervis.lookup.question_contract import (
    QueryQuestionContract, QueryRequestedFact, QueryRequestedOutput, QueryOperationDeclaration,
    QuerySourceDeclaration, QueryParameterDeclaration, InputTerm, InputDenotation,
)
from fervis.lookup.semantic_types import SourceOrigin, SourceOriginKind
from fervis.lookup.contract_codec import canonical_contract_fingerprint
from fervis.lookup.source_reads.access_model import ReadAccessCatalog
from .acquisition import ApiView, RelationView, compile_api_views
from .execution import SqlTable, QueryValidationError, _validate
from .results import ResultContract
from .outputs import QueryOutput
from .column_usage import project_query
from .request_contract import query_request_scope


@dataclass(frozen=True)
class CompiledQueryAnswer:
    program: AnswerProgram
    bindings: BindingSet
    question_contract: QueryQuestionContract


def _constant(name, value):
    return ConstantRef(name,'sql_result_contract@1',FactValue.literal(
        id=name,literal_type=LiteralType.NUMBER,value=str(value),proof_refs=('result_contract',)))


def compile_query_answer(*, question: str, query: str, views: tuple[ApiView, ...],
                         output_types: Mapping[str,str], result_contract: ResultContract, catalog,
                         query_parameters: tuple[SqlNamedInput, ...] = (),
                         parameters: tuple[ParameterDeclaration, ...] = (), bindings: BindingSet = BindingSet(),
                         inputs: tuple[InputTerm, ...] = (), input_denotations: tuple[InputDenotation, ...] = (),
                         access: ReadAccessCatalog = ReadAccessCatalog(),
                         output_labels: Mapping[str,str] | None = None,
                         fact_id: str = 'fact_1', namespace: str = '', timezone: str = 'UTC',
                         selection_boundary: Expression | None = None,
                         meaning_inputs: tuple[ParameterRef, ...] = (),
                         public_outputs: tuple[QueryOutput, ...] | None = None,
                         output_origins: tuple[SourceOrigin, ...] | None = None,
                         expected_input_refs: tuple[str, ...] | None = None,
                         prerequisites: RelationProgram | None = None,
                         relation_views: tuple[RelationView, ...] = ()) -> CompiledQueryAnswer:
    def identifier(value):
        return namespace + value
    all_views: tuple[ApiView | RelationView, ...]=(*views,*relation_views)
    if len({view.name for view in all_views})!=len(all_views):
        raise QueryValidationError('SQL view names must be unique')
    statement, names = _validate(query,{view.name:SqlTable({},()) for view in all_views})
    if result_contract.mode == 'rows' and (statement.args.get('limit') or statement.args.get('offset')):
        raise QueryValidationError('Result selection must remain in its declared owner')
    query,used=project_query(query,{view.name:view.columns for view in all_views})
    selected = tuple(replace(view,columns={name:field for name,field in view.columns.items() if name in used[view.name]})
        for view in views if view.name in names)
    physical = compile_api_views(tuple(replace(view,name=identifier(view.name)) for view in selected),catalog=catalog,access=access)
    if prerequisites is not None:
        parameters=_merge_parameter_declarations((*prerequisites.parameters,*parameters))
        physical=replace(physical,relations=(*prerequisites.relations,*physical.relations),
            operations=(*prerequisites.operations,*physical.operations))
    sql_inputs = tuple(SqlRelationInput(view.name,identifier(view.name),
        tuple(SqlColumnBinding(name,field_id) for name,field_id in view.columns.items())) for view in selected)
    sql_inputs+=tuple(SqlRelationInput(view.name,view.relation_id,
        tuple(SqlColumnBinding(name,field_id) for name,field_id in view.columns.items() if name in used[view.name]))
        for view in relation_views if view.name in names)
    sql = Operation(identifier('answer_query'),SqlQuerySpec(query,sql_inputs,
        tuple(SqlOutputField(name,kind) for name,kind in output_types.items()),query_parameters,
        scalar=result_contract.mode=='scalar',meaning_inputs=meaning_inputs,timezone=timezone,
        entity_keys=tuple(output.identity for output in (public_outputs or ()) if output.identity is not None)),output_relation=identifier('query_result'))
    operations = [*physical.operations,sql]
    result_relation = sql.output_relation
    public_columns = result_contract.output_columns
    final_types = dict(output_types)
    if result_contract.mode == 'scalar':
        if (public_outputs is None and len(output_types)!=1) or (public_outputs is not None and len(public_outputs)!=1):
            raise QueryValidationError('Scalar result requires one public value')
        public_columns = tuple(output_types)
    elif result_contract.mode == 'existence':
        operations.append(Operation(identifier('existence_count'),AggregateSpec(result_relation,(),
            (AggregationSpec(AggregationFunction.COUNT,'witness_count'),)),output_relation=identifier('count_result')))
        operations.append(Operation(identifier('existence_value'),ProjectSpec(identifier('count_result'),(
            NamedExpression('exists',BinaryExpression(ExpressionBinaryOperator.GT,
                FieldRef('witness_count'),_constant('zero',0))),)),output_relation=identifier('exists_result')))
        result_relation,public_columns,final_types=identifier('exists_result'),('exists',),{'exists':'boolean'}
    elif result_contract.ordering:
        selection: OrderSelection
        if result_contract.selection=='all':
            selection=KeepAll()
        elif result_contract.selection=='position_with_ties':
            selection=AtPosition(selection_boundary or _constant('result_position',result_contract.limit))
        else:
            selection=Take(selection_boundary or _constant('result_limit',1 if result_contract.selection=='first_with_ties' else result_contract.limit))
        operations.append(Operation(identifier('result_order'),OrderSpec(result_relation,
            tuple(SortKey(order.column,SortDirection.DESC if order.descending else SortDirection.ASC)
                  for order in result_contract.ordering),selection),output_relation=identifier('selected_result')))
        result_relation=identifier('selected_result')
    projections=tuple(RelationResultOutput(identifier(f'result_{index}'),result_relation,field_id=name,
        label=(output_labels or {}).get(name,''),role='answer_value') for index,name in enumerate(public_columns,start=1))
    if public_outputs is not None and result_contract.mode!='existence':
        projections=tuple(RelationResultOutput(identifier(f'result_{index}'),result_relation,
            field_id=output.column,entity_key=output.identity,label=output.label,role='answer_value',
            display_field_id=output.display_column)
            for index,output in enumerate(public_outputs,start=1))
    origin=SourceOrigin(SourceOriginKind.QUESTION_CONTEXT,question)
    if output_origins is not None and len(output_origins)!=len(projections):
        raise QueryValidationError('Compiled output inventory differs from the requested meanings')
    if any(projection.entity_key is None and projection.field_id not in final_types for projection in projections):
        raise QueryValidationError('Result projection references an undeclared query column')
    outputs=tuple(QueryRequestedOutput.from_projection(f'r{index}',
        output_origins[index-1] if output_origins is not None else SourceOrigin(SourceOriginKind.QUESTION_CONTEXT,projection.label or projection.field_id or 'identity'),
        projection,'identity' if projection.entity_key is not None else final_types[projection.field_id])
        for index,projection in enumerate(projections,start=1))
    program=AnswerProgram(parameters=parameters,relations=physical.relations,operations=tuple(operations),
        result_projection=ResultProjection(relation_outputs=projections),
        fulfillment=tuple(FactFulfillment(fact_id,output.id,output.result_output_id) for output in outputs))
    program=close_catalog_defaults(project_request_arguments(program),row_sources=build_api_row_source_catalog(catalog))
    compiled_inputs=compile_relation_program_inputs(program,bindings=bindings)
    program=replace(program,parameters=compiled_inputs.parameters)
    fact=QueryRequestedFact(fact_id,origin,(),outputs)
    operation_ids,relation_ids,parameter_ids=query_request_scope(fact,program)
    declarations={item.id:item for item in program.parameters}
    actual_input_refs=tuple(sorted({declarations[ref].input_ref for ref in parameter_ids if declarations[ref].input_ref}))
    if expected_input_refs is not None and set(actual_input_refs)!=set(expected_input_refs):
        raise QueryValidationError('Query input operands differ from this requested fact ownership')
    fact=replace(fact,
        operations=tuple(QueryOperationDeclaration(item.id,canonical_contract_fingerprint(item))
                         for item in program.operations if item.id in operation_ids),
        sources=tuple(QuerySourceDeclaration(item.id,canonical_contract_fingerprint(item))
                      for item in program.relations if item.id in relation_ids),
        parameters=tuple(QueryParameterDeclaration(ref,canonical_contract_fingerprint(declarations[ref]))
                         for ref in sorted(parameter_ids)),
        input_refs=actual_input_refs)
    contract=QueryQuestionContract(inputs,(fact,),input_denotations)
    program,bound=compile_answer_program(program,question_contract=contract,catalog=catalog,bindings=compiled_inputs.bindings)
    return CompiledQueryAnswer(program,bound,contract)


def combine_query_answers(answers: tuple[CompiledQueryAnswer, ...], *, catalog) -> CompiledQueryAnswer:
    """Merge fact-local programs and repin their shared parameter declarations."""
    if not answers:
        raise QueryValidationError('At least one requested answer is required')
    from fervis.lookup.answer_program.model import ProgramCompatibility
    declarations = {item.id:item for item in _merge_parameter_declarations(tuple(
        item for answer in answers for item in answer.program.parameters))}
    bindings: dict[str, ParameterBinding] = {}
    inputs, denotations = {}, {}
    for answer in answers:
        for binding in answer.bindings.bindings:
            if binding.parameter_id in bindings and bindings[binding.parameter_id] != binding:
                raise QueryValidationError('Shared query bindings conflict')
            bindings[binding.parameter_id] = binding
        inputs.update({item.id:item for item in answer.question_contract.inputs})
        denotations.update({item.input_ref:item for item in answer.question_contract.input_denotations})
    facts = tuple(replace(fact, parameters=tuple(QueryParameterDeclaration(pin.parameter_id,
        canonical_contract_fingerprint(declarations[pin.parameter_id])) for pin in fact.parameters))
        for answer in answers for fact in answer.question_contract.requested_facts)
    def unique(items):
        result = {}
        for item in items:
            if item.id in result and result[item.id] != item:
                raise QueryValidationError('Fact-local query identifiers collide')
            result[item.id] = item
        return tuple(result.values())
    program = AnswerProgram(
        parameters=tuple(declarations.values()),
        relations=unique(item for answer in answers for item in answer.program.relations),
        operations=unique(item for answer in answers for item in answer.program.operations),
        result_projection=ResultProjection(relation_outputs=tuple(item for answer in answers
            for item in answer.program.result_projection.relation_outputs)),
        fulfillment=tuple(item for answer in answers for item in answer.program.fulfillment),
        compatibility=ProgramCompatibility(),
    )
    contract=QueryQuestionContract(tuple(inputs.values()),facts,tuple(denotations.values()))
    program,bound=compile_answer_program(program,question_contract=contract,catalog=catalog,
        bindings=BindingSet.from_bindings(tuple(bindings.values())))
    return CompiledQueryAnswer(program,bound,contract)


def _merge_parameter_declarations(items):
    declarations = {}
    for item in items:
        prior = declarations.get(item.id)
        if prior is not None:
            fixed_values = {value for value in (prior.fixed_value_fingerprint,item.fixed_value_fingerprint) if value}
            if len(fixed_values)>1:
                raise QueryValidationError('Shared query parameter fixed values conflict')
            fixed_value = next(iter(fixed_values),'')
            item = replace(item,fixed_value_fingerprint=fixed_value)
            if replace(prior, input_use_refs=item.input_use_refs,fixed_value_fingerprint=fixed_value) != item:
                raise QueryValidationError('Shared query parameter declarations conflict')
            item = replace(item, input_use_refs=tuple(sorted(set(prior.input_use_refs) | set(item.input_use_refs))))
        declarations[item.id] = item
    return tuple(declarations.values())
