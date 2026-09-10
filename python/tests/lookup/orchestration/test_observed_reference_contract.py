"""A declared row carrier supports reference selection without nominal keys."""
from dataclasses import replace
import pytest
from jsonschema import validate
from tests.lookup.orchestration.test_reference_authoring import context
from fervis.lookup.orchestration.reference_authoring import ReferenceContractPrompt, parse_reference_contracts


def test_unannotated_reference_has_an_observed_record_contract():
    original, menu, _ = context()
    tables = {name:{**table,'candidate_keys':[],'entity_references':[]} for name,table in original.reference_tables.items()}
    prompt = ReferenceContractPrompt(question=original.question,meaning=original.meaning,inputs=original.inputs,
        denotations=original.denotations,tables=original.tables,parameters=menu.descriptions,reference_tables=tables)
    view = next(iter(tables))
    body = {'reference_contracts':{'i1':{'kind':'record','view':view,'fields':list(tables[view]['columns'])}}}
    validate(body,prompt._schema())
    selected = parse_reference_contracts(body,prompt=prompt)
    assert selected.contracts['i1'].kind == 'record'
    assert len(selected.slots)==1
    assert selected.slots[0].key is None
    assert selected.slots[0].record_source == view
    assert set(selected.slots[0].table['columns']) == set(tables[view]['columns'])


@pytest.mark.parametrize('substitution', ['', 'authored', 'persisted'])
def test_observed_record_reference_replays_and_guards_dependent_reads(substitution):
    from fervis.lookup.relation_catalog import RelationCatalog, CatalogField, CatalogParam
    from fervis.lookup.relational_sql.catalog import build_query_view_catalog
    from fervis.lookup.relational_sql.reference_planning import ReferenceMeaning
    from fervis.lookup.orchestration.reference_slots import ReferenceContract, reference_slots
    from fervis.lookup.orchestration.reference_queries import plan_fact_references, reference_input_values, reference_prerequisites
    from fervis.lookup.relational_sql.parameters import query_parameter_menu, with_reference_arguments
    from fervis.lookup.relational_sql.authoring import parse_query_answer
    from fervis.lookup.relational_sql.binding import bind_query_answer
    from fervis.lookup.relational_sql.compiler import compile_query_answer
    from fervis.lookup.relational_sql.results import ResultContract
    from fervis.lookup.source_reads.access_model import ReadAccessCatalog
    from fervis.lookup.grounding.semantic import GroundingPartition
    from fervis.lookup.question_contract import InputTerm, InputDenotation
    from fervis.lookup.question_contract.model import InputDenotationKind
    from fervis.lookup.semantic_types import TextType, SourceOrigin, SourceOriginKind
    from fervis.lookup.contract_codec import canonical_answer_program_json, decode_answer_program
    from fervis.lookup.answer_program.invocation import invoke_answer_program, RuntimePorts
    from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
    from fervis.lookup.memory.projection import LookupMemory
    from tests.lookup.relational_engine.test_dependent_reads import _read
    from tests.lookup.relational_sql.test_authoring import payload

    channels = replace(_read('channels'),candidate_keys=(),fields=(*_read('channels').fields,
        CatalogField('other_id','integer',path='other_id',row_path_id='root'),
        CatalogField('label','string',path='label',row_path_id='root'),
        CatalogField('unused_flag','boolean',path='unused_flag',row_path_id='root')))
    events = _read('events',params=(CatalogParam('channel_id','channel_id','path','integer',required=True),))
    catalog = RelationCatalog(reads=(channels,events)); views = build_query_view_catalog(catalog)
    carrier = next(name for name,table in views.tables.items() if table['read_id']=='channels')
    origin = SourceOrigin(SourceOriginKind.QUESTION_CONTEXT,'Alpha')
    term = InputTerm('i1',origin,'Alpha',TextType())
    denotation = InputDenotation('d1','i1','the named channel','The question refers to one channel.',
        'channel',InputDenotationKind.IDENTITY_REFERENCE)
    meaning = ReferenceMeaning('fact_1','i1','channel','Count channel events.',(origin,),('i1',),reference_text='Alpha')
    slots = reference_slots(fact=meaning,inputs={'i1':term},denotations={'i1':denotation},tables=views.tables,
        selected_contracts={'i1':ReferenceContract('record',view=carrier,fields=('id','label','other_id'))})
    values = reference_input_values((GroundingPartition('i1',('fact_1:sql_input:i1',),TextType(),None,
        'the named channel',is_identity_reference=True),),inputs={'i1':term})
    def turn(purpose,prompt,parse):
        assert prompt.record_source==carrier
        body=payload(query=f'SELECT * FROM "{carrier}"',mode='rows',
            reference_binding={'kind':'literal','match_column':'label'})
        if substitution == 'authored':
            body['query'] = f'SELECT c.other_id AS id, c.label, c.other_id FROM "{carrier}" c'
        body.pop('outputs')
        validate(body,prompt._schema())
        return parse(body)
    def plan():
        return plan_fact_references(fact=meaning,inputs={'i1':term},denotations={'i1':denotation},values=values,
            catalog=catalog,access=ReadAccessCatalog(),selected_slots={'i1':slots[0]},responses=(),turn=turn)
    if substitution == 'authored':
        from fervis.lookup.relational_sql.execution import QueryValidationError
        with pytest.raises(QueryValidationError, match='selected carrier property'):
            plan()
        return
    references = plan()
    reference, = references
    assert reference.table['candidate_keys']==[]
    assert set(reference.table['columns'])=={'id','label','other_id'}
    menu=with_reference_arguments(query_parameter_menu(()),references)
    assert all('identity' not in item for item in menu.descriptions.values())
    symbol=next(name for name,item in menu.descriptions.items() if item['column']=='id')
    assert '(SELECT' in menu.descriptions[symbol]['sql_expression']
    view=next(view for view in views.views if views.tables[view.name]['read_id']=='events')
    tables={view.name:views.tables[view.name],reference.view.name:reference.table}
    answer=parse_query_answer(payload(query=f'SELECT COUNT(*) AS total FROM "{view.name}"',api_bindings=[
        {'view':view.name,'parameter_ref':'channel_id','binding':symbol}]),table_names=set(tables),tables=tables,
        parameter_names=set(menu.expressions),parameter_descriptions=menu.descriptions,
        meaning=replace(meaning,result_kind='scalar',output_kinds=('value',)),expected_input_refs=('i1',))
    bound=bind_query_answer(answer,menu,(view,))
    prerequisites,bindings=reference_prerequisites(references,bound.bindings,argument_operations=bound.argument_operations)
    compiled=compile_query_answer(question='How many events are in the channel named Alpha?',query=answer.query,
        views=bound.views,prerequisites=prerequisites,bindings=bindings,catalog=catalog,inputs=(term,),
        input_denotations=(denotation,),expected_input_refs=('i1',),output_types=answer.output_types,result_contract=ResultContract('scalar'))
    program=decode_answer_program(canonical_answer_program_json(compiled.program))
    if substitution == 'persisted':
        from fervis.lookup.answer_program.operations import SqlQuerySpec
        from fervis.lookup.relational_sql.request_contract import verify_observed_reference_guards
        from fervis.lookup.plan_execution.errors import VerificationError
        from sqlglot import exp, parse_one
        operation = next(op for op in program.operations if isinstance(op.spec, SqlQuerySpec) and op.spec.lookup_input_ref)
        statement = parse_one(operation.spec.query, read='duckdb')
        # Swap already observed properties in the saved query, preserving its
        # public column inventory and every other part of the program.
        for select in statement.find_all(exp.Select):
            for item in select.expressions:
                if item.alias_or_name == 'id':
                    item.replace(exp.alias_(exp.column('other_id'), 'id'))
        changed = replace(operation, spec=replace(operation.spec, query=statement.sql(dialect='duckdb')))
        program = replace(program, operations=tuple(changed if op.id == changed.id else op for op in program.operations))
        with pytest.raises(VerificationError, match='selected carrier property'):
            verify_observed_reference_guards(program)
        return
    for rows,expected in [([{'id':1,'label':'Alpha'}],1),([{'id':2,'label':'Alpha'}],2),([],None),
        ([{'id':1,'label':'Alpha'},{'id':2,'label':'Alpha'}],None),
        ([{'id':1,'label':'Alpha'},{'id':1,'label':'Alpha'}],None)]:
        calls=[]
        class Port:
            def read(self,*,endpoint_name,args):
                calls.append((endpoint_name,args))
                data=[{**row,'other_id':row['id']+10,'unused_flag':'not a boolean'} for row in rows] if endpoint_name=='channels' else [{'id':i} for i in range(args['channel_id'])]
                return {'responseStatus':200,'responseBody':data}
        result=invoke_answer_program(program=program,bindings=compiled.bindings,environment=ExecutionEnvironment(catalog=catalog),
            ports=RuntimePorts(Port(),LookupMemory()))
        if expected is not None:
            assert result.issue is None
            assert next(iter(result.fact_result.outcome.projected_rows[0].values.values()))==expected
            assert calls==[('channels',{}),('events',{'channel_id':expected})]
        else:
            assert result.issue.reference.reason.value==('NOT_FOUND' if not rows else 'AMBIGUOUS_RESULT')
            assert result.issue.reference.candidates==()
            from fervis.lookup.orchestration.terminal_results import reference_clarification_fact_result
            clarification=reference_clarification_fact_result(result.issue,contract=compiled.question_contract).outcome.clarifications[0]
            assert clarification.continuation.accepts_free_text
            assert calls==[('channels',{})]


def test_observed_candidate_cannot_hide_duplicates_behind_a_carrier_named_cte():
    import pytest
    from types import SimpleNamespace
    from fervis.lookup.relational_sql.authoring import ApiInvocation
    from fervis.lookup.relational_sql.reference_matching import require_record_candidate_source
    from fervis.lookup.relational_sql.record_lineage import record_field_origins
    from fervis.lookup.relational_sql.execution import QueryValidationError
    from fervis.lookup.plan_execution.errors import VerificationError
    query='WITH carrier AS (SELECT id FROM carrier UNION SELECT id FROM carrier) SELECT id FROM carrier'
    answer=SimpleNamespace(query=query,api_invocations=(ApiInvocation('declared_carrier','carrier'),))
    with pytest.raises(QueryValidationError,match='carrier|CTE'):
        require_record_candidate_source(answer,record_source='declared_carrier')
    with pytest.raises(VerificationError,match='observed'):
        record_field_origins(query,{'carrier':{'id'}},('id',),preserve_occurrences=True)
    assert record_field_origins(query,{'carrier':{'id'}},('id',))['id']=={('carrier','id')}
