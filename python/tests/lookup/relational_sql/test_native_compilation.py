from tests.lookup.orchestration.test_reference_authoring import reference_contract_payload
from tests.lookup.relational_sql.test_authoring import payload as query_payload
import pytest
from types import SimpleNamespace
from dataclasses import replace

from fervis.lookup.orchestration import semantic_compilation as compilation
from fervis.lookup.question_contract import QuestionContractRequest
from fervis.lookup.turn_prompts import HostPromptContext
from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.query_enrichment.semantic import SemanticQueryEnrichmentResult, RecallBucketMatch
from fervis.lookup.read_eligibility import SemanticReadEligibilityResult, ReadRequirementAssessment, SemanticReadDecision
from fervis.lookup.answer_program.operations import SqlQuerySpec
from fervis.lookup.answer_program.invocation import invoke_answer_program, RuntimePorts
from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
from fervis.lookup.memory.projection import LookupMemory
from fervis.lookup.contract_codec import canonical_answer_program_json, decode_answer_program
from tests.lookup.question_contract.test_question_frame import _frame_payload
from tests.lookup.relational_engine.test_dependent_reads import _read
from tests.lookup.relational_sql.test_authoring import payload


@pytest.mark.parametrize("extra_retained_read", [False, True])
def test_normal_compilation_authors_and_persists_sql_without_graph_turn(monkeypatch,extra_retained_read):
    question='How many stores?'
    purposes=[]
    access_targets=[]
    discover=compilation._discover_read_access
    def tracked_access(read_ids,**kwargs):
        access_targets.append(tuple(read_ids))
        return discover(read_ids,**kwargs)
    monkeypatch.setattr(compilation,'_discover_read_access',tracked_access)
    def scripted_turn(purpose, *, prompt, parse, **kwargs):
        purposes.append(purpose.value)
        if purpose.value=='question_contract':
            assert prompt.__class__.__name__=='SemanticQuestionFrameTurnPrompt'
            result=parse(_frame_payload())
        elif purpose.value=='query_enrichment':
            result=SemanticQueryEnrichmentResult(tuple(RecallBucketMatch(bucket.bucket_ref,('stores',),('stores',))
                for bucket in prompt.request.recall_buckets),())
        elif purpose.value=='source_realization':
            table=next(iter(prompt.tables))
            result=parse(payload(query=f'SELECT COUNT(*) AS total FROM "{table}"'))
        else:
            raise AssertionError(f'Unexpected model stage: {purpose}')
        return SimpleNamespace(result=result)
    def eligibility(eligibility_request,**kwargs):
        return SemanticReadEligibilityResult(tuple(ReadRequirementAssessment('fact_1',source.read_id,
            (source.id,),source.read_id,tuple(field.field_ref for field in source.fields),
            'Each row represents a store.',SemanticReadDecision.RETAIN)
            for source in eligibility_request.source_catalog.sources),())
    monkeypatch.setattr(compilation,'_turn',scripted_turn)
    monkeypatch.setattr(compilation,'_read_eligibility_turn',eligibility)
    class Port:
        def __init__(self): self.calls=0
        def read(self,**kwargs):
            self.calls+=1
            return {'responseStatus':200,'responseBody':[{'id':1},{'id':2},{'id':3}]}
    port=Port()
    catalog=RelationCatalog(reads=(replace(_read('stores'),resource_names=('stores',)),
        *((replace(_read('other_stores'),resource_names=('stores',)),) if extra_retained_read else ())))
    request=compilation.SemanticCompilationRequest('native-test',question,QuestionContractRequest(
        current_question=question,conversation_context={}),catalog,(),port,None,'openai',1,10,None,{},HostPromptContext())
    result=compilation.compile_semantic_question(request)
    assert isinstance(result,compilation.SemanticCompilationSuccess)
    assert purposes==['question_contract','query_enrichment','source_realization']
    assert port.calls==0
    assert access_targets==[]
    program=result.compilation.answer_program
    assert any(isinstance(operation.spec,SqlQuerySpec) for operation in program.operations)
    persisted=decode_answer_program(canonical_answer_program_json(program))
    assert persisted==program
    executed=invoke_answer_program(program=persisted,bindings=result.compilation.initial_bindings,
        environment=ExecutionEnvironment(catalog=catalog),ports=RuntimePorts(port,LookupMemory()))
    assert executed.issue is None
    assert [next(iter(row.values.values())) for row in executed.fact_result.outcome.projected_rows]==[3]
    assert port.calls==1


@pytest.mark.parametrize('scenario',['empty_catalog','all_dropped','declared_unavailable','unrelated_discovery_failure'])
def test_exhausted_sources_return_impossible_without_sql_authoring(monkeypatch,scenario):
    question='How many stores?'
    if scenario=='empty_catalog':
        catalog=RelationCatalog(reads=())
    else:
        read=replace(_read('stores'),
            resource_names=('stores',))
        catalog=RelationCatalog(reads=(read,))
        if scenario == 'unrelated_discovery_failure':
            from fervis.lookup.relation_catalog import EndpointRead
            catalog = replace(catalog, reads=(*catalog.reads, EndpointRead('unrelated', 'unrelated', resource_names=('stores',))))
    calls=[]
    def turn(purpose,*,prompt,parse,**kwargs):
        calls.append(purpose.value)
        if purpose.value=='question_contract':return SimpleNamespace(result=parse(_frame_payload()))
        if purpose.value=='query_enrichment':
            names=tuple(prompt.request.resource_names)
            return SimpleNamespace(result=SemanticQueryEnrichmentResult(tuple(RecallBucketMatch(bucket.bucket_ref,names,names)
                for bucket in prompt.request.recall_buckets),()))
        if scenario=='declared_unavailable' and purpose.value=='source_realization':
            return SimpleNamespace(result=parse({'unavailable':True,'reason':'These rows do not establish the requested population.'}))
        pytest.fail('Exhausted sources reached SQL authoring')
    def eligibility(eligibility_request,**kwargs):
        return SemanticReadEligibilityResult(tuple(ReadRequirementAssessment('fact_1',source.read_id,(source.id,),
            source.read_id,() if scenario in {'all_dropped', 'unrelated_discovery_failure'} else tuple(field.field_ref for field in source.fields),
            'The declared fixture source assessment.',SemanticReadDecision.DROP if scenario in {'all_dropped', 'unrelated_discovery_failure'} else SemanticReadDecision.RETAIN)
            for source in eligibility_request.source_catalog.sources),())
    # The compilation request is a keyword to this seam, distinct from its first argument.
    def assessed(eligibility_request,**kwargs):return eligibility(eligibility_request)
    monkeypatch.setattr(compilation,'_turn',turn)
    monkeypatch.setattr(compilation,'_read_eligibility_turn',assessed)
    class NoRead:
        def read(self,**kwargs):
            assert scenario == 'unrelated_discovery_failure' and kwargs['endpoint_name'] == 'unrelated'
            return {'responseStatus': 400, 'responseFormat': 'json', 'responseBody': {}}
    request=compilation.SemanticCompilationRequest('unavailable',question,QuestionContractRequest(
        current_question=question,conversation_context={}),catalog,(),NoRead(),None,'openai',1,10,None,{},HostPromptContext())
    result=compilation.compile_semantic_question(request)
    assert isinstance(result,compilation.SemanticCompilationImpossible)
    assert result.blocked_fact_ids==('fact_1',)
    assert result.reviewed_read_ids==tuple(read.id for read in catalog.reads)
    assert calls==['question_contract','query_enrichment']+(['source_realization'] if scenario=='declared_unavailable' else [])


def test_sql_views_follow_retained_row_grain_instead_of_every_nested_candidate(monkeypatch):
    from tests.lookup.test_available_sources import _nested_sales_read
    question='How many sales?'
    catalog=RelationCatalog(reads=(replace(_nested_sales_read(),resource_names=('sales',)),))
    frame=_frame_payload(returned_meanings=[{'meaning_ref':'r1','meaning':'sale count','origin':{'kind':'question'}}])
    frame['outcome']['answer_requests'][0]['request']['result']['population_rows']['instance_kind']='sale'
    def turn(purpose,*,prompt,parse,**kwargs):
        if purpose.value=='question_contract':return SimpleNamespace(result=parse(frame))
        if purpose.value=='query_enrichment':
            return SimpleNamespace(result=SemanticQueryEnrichmentResult(tuple(RecallBucketMatch(bucket.bucket_ref,('sales',),('sales',))
                for bucket in prompt.request.recall_buckets),()))
        assert purpose.value=='source_realization'
        assert len(prompt.tables)==1
        table,definition=next(iter(prompt.tables.items()))
        assert definition['row_path']=='data'
        return SimpleNamespace(result=parse(payload(query=f'SELECT COUNT(*) AS total FROM "{table}"')))
    def assessed(eligibility_request,**kwargs):
        return SemanticReadEligibilityResult((ReadRequirementAssessment('fact_1','list_sale_list',
            tuple(source.id for source in eligibility_request.source_catalog.sources),'list_sale_list',
            ('field.data.sale_id',),'The sale rows establish the requested count.',SemanticReadDecision.RETAIN),),())
    monkeypatch.setattr(compilation,'_turn',turn)
    monkeypatch.setattr(compilation,'_read_eligibility_turn',assessed)
    class Port:
        def read(self,**kwargs):
            return {'responseStatus':200,'responseBody':{'data':[
                {'sale_id':'00000000-0000-0000-0000-000000000001','amount':'1.00','items':[]},
                {'sale_id':'00000000-0000-0000-0000-000000000002','amount':'2.00','items':[]}]}}
    request=compilation.SemanticCompilationRequest('grain',question,QuestionContractRequest(
        current_question=question,conversation_context={}),catalog,(),Port(),None,'openai',1,10,None,{},HostPromptContext())
    result=compilation.compile_semantic_question(request)
    assert isinstance(result,compilation.SemanticCompilationSuccess)
    executed=invoke_answer_program(program=result.compilation.answer_program,bindings=result.compilation.initial_bindings,
        environment=ExecutionEnvironment(catalog=catalog),ports=RuntimePorts(Port(),LookupMemory()))
    assert executed.issue is None
    assert [next(iter(row.values.values())) for row in executed.fact_result.outcome.projected_rows]==[2]


@pytest.mark.parametrize('case',['shared','shared_category','shared_mixed','source_population','omitted','independent','foreign'])
def test_native_multi_answer_preserves_per_request_operand_ownership(monkeypatch,case):
    from copy import deepcopy
    question='Count stores above the first threshold and sum IDs of stores above the applicable threshold.'
    independent=case in {'independent','foreign'}
    values=[{'meaning':'first threshold','denotation_basis':'The numeric threshold qualifies the requested stores.',
        'non_entity_value':{'kind':'number','value':{'operands':['2' if independent else '1'],'origin':{'kind':'question'}}}}]
    if independent:
        values.append({'meaning':'second threshold','denotation_basis':'A separate threshold belongs to the second request.',
            'non_entity_value':{'kind':'number','value':{'operands':['1'],'origin':{'kind':'question'}}}})
    if case in {'shared_category','shared_mixed','source_population'}:
        values[0]['non_entity_value']={'kind':'categorical_value','value':{'operands':['accepted'],'origin':{'kind':'question'}}}
    frame=_frame_payload(supplied_values=values)
    frame['outcome']['answer_requests'].append(deepcopy(frame['outcome']['answer_requests'][0]))
    for index,item in enumerate(frame['outcome']['supplied_values']['operands'],start=1):
        item['answer_request_numbers']=[index] if independent else [1,2]
    def turn(purpose,*,prompt,parse,**kwargs):
        if purpose.value=='question_contract':result=parse(frame)
        elif purpose.value=='query_enrichment':
            result=SemanticQueryEnrichmentResult(tuple(RecallBucketMatch(b.bucket_ref,('stores',),('stores',))
                for b in prompt.request.recall_buckets),())
        elif purpose.value=='source_realization':
            table=next(iter(prompt.tables));first=prompt.meaning.requested_fact_id=='fact_1'
            measure='COUNT(*)' if first else 'SUM(id)'
            # Symbols are looked up by their independently supplied input identity.
            symbols={p['input_ref']:name for name,p in prompt.parameters.items() if p.get('kind')!='catalog_choice'}
            assert set(symbols)==({'i1' if first else 'i2'} if independent else {'i1'})
            ref='i2' if independent and not first else 'i1'
            predicate='' if case=='omitted' and not first else f' WHERE id > ${symbols[ref]}'
            if case=='foreign' and not first:
                predicate+=' AND id > $foreign'
            interpretations=[]
            if case in {'shared_category','shared_mixed','source_population'}:
                choice=next(name for name,item in prompt.parameters.items() if item.get('kind')=='catalog_choice' and item['value']=='true')
                predicate=f' WHERE active=${choice}'
                interpretations=[{'input':symbols['i1'],'choice':choice,'basis':'The declared acceptance flag is true for accepted rows.'}]
                if case=='shared_mixed' and first:
                    predicate=f' WHERE status=${symbols["i1"]}'
                    interpretations=[]
            body=payload(query=f'SELECT {measure} AS total FROM "{table}"{predicate}',interpretations=interpretations)
            if case=='source_population':
                body=payload(query=f'SELECT {measure} AS total FROM "{table}"')
                body['api_invocations'][0]['population_bindings']=[{'input':symbols['i1'],'basis':'The API contract restricts returned stores to accepted stores.'}]
            result=parse(body)
        else:raise AssertionError(purpose)
        return SimpleNamespace(result=result)
    def eligibility(eligibility_request,**kwargs):
        return SemanticReadEligibilityResult(tuple(ReadRequirementAssessment(ctx.requested_fact_id,source.read_id,
            (source.id,),source.read_id,tuple(f.field_ref for f in source.fields),'Store rows.',SemanticReadDecision.RETAIN)
            for ctx in eligibility_request.fact_contexts for source in eligibility_request.source_catalog.sources),())
    class Port:
        def read(self,**kwargs):
            rows=[{'id':1,'active':True,'status':'accepted'},{'id':2,'active':False,'status':'failed'},{'id':3,'active':True,'status':'accepted'}]
            return {'responseStatus':200,'responseBody':[row for row in rows if row['active']] if case=='source_population' else rows}
    monkeypatch.setattr(compilation,'_turn',turn);monkeypatch.setattr(compilation,'_read_eligibility_turn',eligibility)
    from fervis.lookup.relation_catalog.model import CatalogField
    read=replace(_read('stores'),resource_names=('stores',))
    if case in {'shared_category','shared_mixed','source_population'}:
        read=replace(read,fields=(*read.fields,CatalogField('active','boolean',path='active',row_path_id='root',metadata={'description':'True means accepted; false means failed.'}),CatalogField('status','string',path='status',row_path_id='root')))
    if case=='source_population':
        read=replace(read,source_metadata={'description':'Returns all accepted stores, excluding unaccepted stores.'})
    catalog=RelationCatalog(reads=(read,))
    request=compilation.SemanticCompilationRequest('input-scope',question,
        QuestionContractRequest(current_question=question,conversation_context={}),catalog,(),Port(),None,'openai',1,10,None,{},HostPromptContext())
    if case in {'omitted','foreign'}:
        with pytest.raises(ValueError,match='operand|parameter|input'):
            compilation.compile_semantic_question(request)
        return
    result=compilation.compile_semantic_question(request)
    assert [(f.id,f.input_refs) for f in result.question_contract.requested_facts]==[
        ('fact_1',('i1',)),('fact_2',('i2' if independent else 'i1',))]
    if case in {'shared_category','shared_mixed','source_population'}:
        assert all(p.fixed_value_fingerprint for p in result.compilation.answer_program.parameters)
    executed=invoke_answer_program(program=result.compilation.answer_program,bindings=result.compilation.initial_bindings,
        environment=ExecutionEnvironment(catalog=catalog),ports=RuntimePorts(Port(),LookupMemory()))
    assert executed.issue is None
    assert [next(iter(row.values.values())) for row in executed.fact_result.outcome.projected_rows]==[1 if independent else 2,4 if case in {'shared_category','shared_mixed','source_population'} else 5]


@pytest.mark.parametrize('failure_stage',['none','reference_once','reference_twice','consumer','provider_reference','provider_consumer'])
@pytest.mark.parametrize('key_type', ['integer', 'uuid'])
@pytest.mark.parametrize('two_facts', [False, True])
@pytest.mark.parametrize('opaque_consumer', [False, True])
def test_normal_reference_compilation_replays_guard_before_required_rest_read(monkeypatch, key_type, two_facts, opaque_consumer, failure_stage):
    dependency_failures={'reference_once':1,'reference_twice':2}.get(failure_stage,0)
    consumer_unavailable=failure_stage=='consumer'
    from uuid import UUID
    def key_value(number):
        return str(UUID(int=number)) if key_type == 'uuid' else number
    from fervis.lookup.relation_catalog import CatalogField,CatalogParam,ParamSource,EntityKeyComponentTarget,CandidateKey,CandidateKeyComponent
    from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
    from fervis.lookup.source_reads.access_model import ReadAccessCatalog
    from fervis.lookup.query_enrichment.semantic import InputResourceSearchTerms
    areas=replace(_read('areas'),resource_names=('areas',),fields=(replace(_read('areas').fields[0], type=key_type),
        CatalogField('primary','boolean',path='is_primary',row_path_id='root')))
    if consumer_unavailable:
        areas=replace(areas,fields=(*areas.fields,CatalogField('areas.code','string',path='code',row_path_id='root')),
            candidate_keys=(*areas.candidate_keys,CandidateKey('code','areas',(CandidateKeyComponent('code','areas.code'),),primary=False)))
    stores=replace(_read('stores',params=(CatalogParam('area_id','area_id',ParamSource.PATH,key_type,
        required=True,entity_target=None if opaque_consumer else EntityKeyComponentTarget('areas','primary','id')),)),resource_names=('stores',))
    foreign=replace(_read('foreign_areas', value_type=key_type), resource_names=('areas',))
    catalog=RelationCatalog(reads=(areas,stores,foreign))
    sources=build_api_row_source_catalog(catalog)
    monkeypatch.setattr(compilation,'_discover_read_access',lambda *args,**kwargs:ReadAccessCatalog(sources.sources))
    seen = []
    factual_prompts = []
    authority_prompts = []
    reference_attempts = []
    def turn(purpose,*,prompt,parse,**kwargs):
        seen.append(purpose.value)
        if purpose.value=='question_contract':
            frame=_frame_payload(supplied_values=[{'meaning':'the primary area',
                'denotation_basis':'The configured primary area identifies which stores to count.',
                'entity_reference':{'instance_kind':'area','value':{'operands':['the primary area'],'reference_kind':'description','origin':{'kind':'question'}}}}])
            if two_facts:
                from copy import deepcopy
                frame['outcome']['answer_requests'].append(deepcopy(frame['outcome']['answer_requests'][0]))
                frame['outcome']['supplied_values']['operands'][0]['answer_request_numbers'] = [1, 2]
            result=parse(frame)
        elif purpose.value=='query_enrichment':
            result=SemanticQueryEnrichmentResult(tuple(RecallBucketMatch(bucket.bucket_ref,('stores',),('stores',))
                for bucket in prompt.request.recall_buckets),
                tuple(InputResourceSearchTerms(task.input_use_ref,('areas',)) for task in prompt.request.reference_tasks))
        elif purpose.value=='grounding' and prompt.turn_name=='reference contract selection':
            authority_prompts.append(prompt)
            authority='areas/code(code)' if consumer_unavailable and not prompt.failed_reference_plans else 'areas/primary(id)'
            result=parse(reference_contract_payload({'i1':authority}))
        elif purpose.value=='grounding':
            reference_attempts.append(prompt)
            if failure_stage=='provider_reference':
                raise RuntimeError('Provider transport failed')
            if len(reference_attempts) <= dependency_failures:
                return SimpleNamespace(result=parse({'unavailable':True,'reason':'The selected reference route is unavailable.'}))
            assert authority_prompts
            assert prompt.expected_key['entity_kind'] == 'areas'
            assert {authority['entity_kind'] for authority in prompt.output_identity_authorities().values()} == {'areas'}
            foreign_view = next(name for name, table in prompt.tables.items() if table.get('read_id') == 'foreign_areas')
            from fervis.lookup.relational_sql.execution import QueryValidationError
            with pytest.raises(QueryValidationError, match='consuming identity demand'):
                parse(query_payload(**{'query':f'SELECT id FROM "{foreign_view}"', 'mode':'rows',

                    'outputs':[{'kind':'identity','authority':'foreign_areas/primary(id)','components':{'id':'id'},'label':'area','display_column':None}],
                    'ordering':[], 'api_bindings':[], 'interpretations':[],
                    'reference_binding':{'kind':'description','basis':'The fixture tries the other namespace.'}}))
            assert prompt.meaning.requested_fact_id in {'fact_1', 'fact_2'}
            assert any(parameter.get('entity_target', {}).get('entity_kind') == 'areas'
                       for view in prompt.consumer_view_refs for parameter in prompt.tables[view]['request_parameters'] if parameter.get('entity_target')) is (not opaque_consumer)
            assert any(table.get('read_id') == 'stores' for table in prompt.tables.values())
            view=next(name for name, table in prompt.tables.items() if table.get('read_id') == 'areas')
            choice=next(name for name,desc in prompt.parameters.items() if desc.get('kind')=='catalog_choice' and desc['value']=='true')
            assert {desc['value'] for desc in prompt.parameters.values() if desc.get('kind')=='catalog_choice'}=={'false','true'}
            field='code' if prompt.expected_key['key_id']=='code' else 'id'
            result=parse(query_payload(**{'query':f'SELECT {field} FROM "{view}" WHERE is_primary=${choice}','mode':'rows',

                'outputs':[{'kind':'identity','authority':f"areas/{prompt.expected_key['key_id']}({field})",'components':{field:field},'label':'area','display_column':None}],
                'ordering':[],'api_bindings':[],
                'interpretations':[],'reference_binding':{'kind':'description','basis':'The primary flag defines the configured primary area.'}}))
        elif purpose.value=='source_realization':
            factual_prompts.append(prompt)
            if failure_stage=='provider_consumer':
                raise RuntimeError('Provider transport failed')
            view=next(name for name,table in prompt.tables.items() if table.get('read_id')=='stores')
            assert prompt.tables['i1']['kind']=='resolved_reference'
            assert not any(table.get('kind')=='reference_slot' for table in prompt.tables.values())
            assert not any(description.get('kind')=='reference_literal' for description in prompt.parameters.values())
            reference_symbol=next(name for name,description in prompt.parameters.items() if description.get('kind')=='reference_argument')
            submitted=payload(query='SELECT COUNT(*) AS total FROM selected_items',
                api_bindings=[{'view':view,'name':'selected_items','parameter_ref':prompt.tables[view]['request_parameters'][0]['param_ref'],'binding':reference_symbol}])
            from jsonschema import validate,ValidationError
            if consumer_unavailable and prompt.parameters[reference_symbol]['identity']['key_id']=='code':
                if not opaque_consumer or key_type=='integer':
                    with pytest.raises(ValidationError):validate(submitted,prompt._schema())
                result=parse({'unavailable':True,'reason':'The consumer requires the primary identifier, not the resolved code.'})
            else:
                validate(submitted,prompt._schema())
                result=parse(submitted)
        else:raise AssertionError(purpose)
        return SimpleNamespace(result=result)
    monkeypatch.setattr(compilation,'_turn',turn)
    monkeypatch.setattr(compilation,'_read_eligibility_turn',lambda eligibility_request,**kwargs:SemanticReadEligibilityResult(
        tuple(ReadRequirementAssessment(fact_id,source.read_id,(source.id,),source.read_id,
            tuple(f.field_ref for f in source.fields),'Rows provide the requested store population.',SemanticReadDecision.RETAIN)
            for fact_id in (('fact_1','fact_2') if two_facts else ('fact_1',))
            for source in eligibility_request.source_catalog.sources),()))
    class Port:
        def __init__(self,primary):self.primary=primary;self.calls=[]
        def read(self,*,endpoint_name,args):
            assert endpoint_name in {'areas', 'stores'}
            self.calls.append((endpoint_name,args))
            rows=([{'id':key_value(i),'is_primary':i in self.primary,**({'code':'area-'+str(i)} if consumer_unavailable else {})} for i in (1,2)] if endpoint_name=='areas'
                  else [{'id':i} for i in range(next(i for i in (1,2) if str(key_value(i)) == str(args['area_id'])))])
            return {'responseStatus':200,'responseBody':rows}
    question='How many stores are in the primary area?'
    port=Port({1})
    request=compilation.SemanticCompilationRequest('reference-test',question,QuestionContractRequest(
        current_question=question,conversation_context={}),catalog,(),port,None,'openai',1,10,None,{},HostPromptContext())
    if failure_stage.startswith('provider_'):
        with pytest.raises(RuntimeError,match='Provider transport failed'):
            compilation.compile_semantic_question(request)
        assert len(authority_prompts)==1
        assert len(reference_attempts)==1
        assert port.calls==[]
        return
    compiled=compilation.compile_semantic_question(request)
    if dependency_failures:
        assert len(authority_prompts) >= 2
        assert authority_prompts[1].failed_reference_plans[0]['reason']=='The selected reference route is unavailable.'
        assert authority_prompts[1].failed_reference_plans[0]['reference_contracts']==reference_contract_payload({'i1':'areas/primary(id)'})['reference_contracts']
    if dependency_failures == 2:
        assert isinstance(compiled,compilation.SemanticCompilationImpossible)
        assert len(reference_attempts)==2
        assert port.calls==[]
        return
    if consumer_unavailable:
        assert len(authority_prompts)==(4 if two_facts else 2)
        assert authority_prompts[1].failed_reference_plans[0]['stage']=='factual_query'
        assert authority_prompts[1].failed_reference_plans[0]['reference_contracts']==reference_contract_payload({'i1':'areas/code(code)'})['reference_contracts']
    assert isinstance(compiled,compilation.SemanticCompilationSuccess)
    assert port.calls==[]
    persisted=decode_answer_program(canonical_answer_program_json(compiled.compilation.answer_program))
    assert persisted.input_denotations[0].reference_descriptions == ('the primary area',)
    for primary in ({1},{2},set(),{1,2}):
        port=Port(primary)
        result=invoke_answer_program(program=persisted,bindings=compiled.compilation.initial_bindings,
            environment=ExecutionEnvironment(catalog=catalog),ports=RuntimePorts(port,LookupMemory()))
        if len(primary)==1:
            expected=next(iter(primary))
            assert result.issue is None
            assert [next(iter(row.values.values())) for row in result.fact_result.outcome.projected_rows]==[expected]*(2 if two_facts else 1)
            assert port.calls==[('areas',{}),('stores',{'area_id':key_value(expected)})]
        else:
            assert result.issue.reference.input_ref=='i1'
            assert port.calls==[('areas',{})]


def test_selection_controls_do_not_become_sql_predicates_and_can_be_rebound(monkeypatch):
    from fervis.lookup.answer_program.values import BindingSet,FactValue,LiteralType
    frame=_frame_payload(result_kind='one_per_candidate',ordering=[{
        'ownership_basis':'Order stores by identifier.','kind':'unreturned_ordering_meaning',
        'meaning':'store identifier ascending','origin':{'kind':'question'}}])
    frame['outcome']['answer_requests'][0]['request']['result']['result_order']['selection']={'kind':'take_with_boundary_ties'}
    frame['outcome']['supplied_values']['selection_limits']=[{'answer_request_number':1,
        'meaning':'requested number of results','denotation_basis':'Two sets the requested result limit.',
        'non_entity_value':{'value':{'operands':['2'],'value_type':{'kind':'integer'},'origin':{'kind':'question'}}}}]
    def turn(purpose,*,prompt,parse,**kwargs):
        if purpose.value=='question_contract':result=parse(frame)
        elif purpose.value=='query_enrichment':
            result=SemanticQueryEnrichmentResult(tuple(RecallBucketMatch(bucket.bucket_ref,('stores',),('stores',))
                for bucket in prompt.request.recall_buckets),())
        elif purpose.value=='source_realization':
            assert not any(item.get('input_ref')==prompt.meaning.selection_limit_input_ref for item in prompt.parameters.values())
            view=next(iter(prompt.tables))
            result=parse(payload(query=f'SELECT id FROM "{view}"',mode='rows',
                outputs=[{'kind':'identity','authority':'stores/primary(id)','components':{'id':'id'},'label':'store','display_column':None}],
                ordering=[{'column':'id','descending':False}]))
        else:raise AssertionError(purpose)
        return SimpleNamespace(result=result)
    monkeypatch.setattr(compilation,'_turn',turn)
    monkeypatch.setattr(compilation,'_read_eligibility_turn',lambda eligibility_request,**kwargs:SemanticReadEligibilityResult(tuple(
        ReadRequirementAssessment('fact_1',source.read_id,(source.id,),source.read_id,tuple(f.field_ref for f in source.fields),
            'Store records expose their identifiers.',SemanticReadDecision.RETAIN)
        for source in eligibility_request.source_catalog.sources),()))
    catalog=RelationCatalog(reads=(replace(_read('stores'),resource_names=('stores',)),))
    class Port:
        def read(self,**kwargs):return {'responseStatus':200,'responseBody':[{'id':3},{'id':1},{'id':2}]}
    question='Show the first two stores ordered by identifier ascending.'
    request=compilation.SemanticCompilationRequest('selection-test',question,QuestionContractRequest(
        current_question=question,conversation_context={}),catalog,(),Port(),None,'openai',1,10,None,{},HostPromptContext())
    compiled=compilation.compile_semantic_question(request).compilation
    program=decode_answer_program(canonical_answer_program_json(compiled.answer_program))
    for limit in (2,3):
        bindings=BindingSet.from_bindings(tuple(replace(binding,value=FactValue.literal(
            id=binding.value.id,known_input_id=binding.value.known_input_id,literal_type=LiteralType.NUMBER,
            value=str(limit),proof_refs=('current_argument:limit',))) for binding in compiled.initial_bindings.bindings))
        result=invoke_answer_program(program=program,bindings=bindings,environment=ExecutionEnvironment(catalog=catalog),
            ports=RuntimePorts(Port(),LookupMemory()))
        assert result.issue is None
        keys=[next(iter(row.values.values())) for row in result.fact_result.outcome.projected_rows]
        assert [key.component_value('id') for key in keys]==list(range(1,limit+1))


@pytest.mark.parametrize('zone,instant,expected', [
    ('Africa/Nairobi','2026-09-08T22:00:00+00:00','2026-09-09'),
    ('America/New_York','2026-09-10T02:00:00+00:00','2026-09-09'),
])
def test_sql_calendar_uses_and_persists_the_question_timezone(monkeypatch,zone,instant,expected):
    from datetime import date
    from fervis.lookup.runtime_values import RuntimeValueContext
    from fervis.lookup.relation_catalog import CatalogField
    frame = _frame_payload()
    frame['outcome']['answer_requests'][0]['request']['result']['returned_meanings'][0]['meaning'] = 'local calendar date of the observation'
    def turn(purpose,*,prompt,parse,**kwargs):
        if purpose.value == 'question_contract': result = parse(frame)
        elif purpose.value == 'query_enrichment':
            result = SemanticQueryEnrichmentResult(tuple(RecallBucketMatch(bucket.bucket_ref,('events',),('events',)) for bucket in prompt.request.recall_buckets),())
        elif purpose.value == 'source_realization':
            view = next(iter(prompt.tables))
            result = parse(payload(query=f'SELECT CAST(recorded_at AS DATE) AS day FROM "{view}"',
                outputs=[{'kind':'value','column':'day','label':'date'}]))
        else: raise AssertionError(purpose)
        return SimpleNamespace(result=result)
    monkeypatch.setattr(compilation,'_turn',turn)
    monkeypatch.setattr(compilation,'_read_eligibility_turn',lambda eligibility_request,**kwargs: SemanticReadEligibilityResult(tuple(
        ReadRequirementAssessment('fact_1',source.read_id,(source.id,),source.read_id,tuple(field.field_ref for field in source.fields),'Observed event timestamp.',SemanticReadDecision.RETAIN)
        for source in eligibility_request.source_catalog.sources),()))
    read = replace(_read('events'),resource_names=('events',),fields=(*_read('events').fields,CatalogField('recorded_at','datetime',path='recorded_at',row_path_id='root')))
    catalog = RelationCatalog(reads=(read,))
    class Port:
        def read(self,**kwargs): return {'responseStatus':200,'responseBody':[{'id':1,'recorded_at':instant}]}
    question = 'What local calendar date contains the observation timestamp?'
    request = compilation.SemanticCompilationRequest('calendar',question,QuestionContractRequest(current_question=question,conversation_context={}),catalog,(),Port(),None,'openai',1,10,RuntimeValueContext('2026-09-09',zone),{},HostPromptContext())
    compiled = compilation.compile_semantic_question(request).compilation
    persisted = decode_answer_program(canonical_answer_program_json(compiled.answer_program))
    result = invoke_answer_program(program=persisted,bindings=compiled.initial_bindings,
        environment=ExecutionEnvironment(catalog=catalog),ports=RuntimePorts(Port(),LookupMemory()))
    assert result.issue is None
    assert next(iter(result.fact_result.outcome.projected_rows[0].values.values())) == date.fromisoformat(expected)


@pytest.mark.parametrize('first_retained', [False, True])
def test_normal_compilation_inspects_and_assesses_later_recalled_reads(monkeypatch, first_retained):
    from fervis.lookup.relation_catalog import EndpointRead
    question = 'How many stores?'
    observed = []
    assessed_reads = []
    authored_reads = []
    catalog = RelationCatalog(reads=tuple(EndpointRead(name, name, resource_names=('stores',)) for name in ('a', 'b')))
    from fervis.lookup.relation_catalog.selection.model import CatalogSelectionResult, RequestedFactCatalogSelection
    monkeypatch.setattr(compilation, 'select_semantic_relation_catalog', lambda request: CatalogSelectionResult(
        RelationCatalog(reads=(catalog.read('a'),)),
        (RequestedFactCatalogSelection('fact_1', ('stores',), (), ('a',), ('b',)),), ('a',)))
    class Port:
        def read(self, *, endpoint_name, args):
            observed.append(endpoint_name)
            return {'responseStatus': 200, 'responseFormat': 'json', 'responseBody': [{'name': endpoint_name}]}
    def turn(purpose, *, prompt, parse, **kwargs):
        if purpose.value == 'question_contract':
            return SimpleNamespace(result=parse(_frame_payload()))
        if purpose.value == 'query_enrichment':
            return SimpleNamespace(result=SemanticQueryEnrichmentResult(tuple(
                RecallBucketMatch(bucket.bucket_ref, ('stores',), ('stores',))
                for bucket in prompt.request.recall_buckets), ()))
        assert purpose.value == 'source_realization'
        authored_reads.extend(definition["read_id"] for definition in prompt.tables.values())
        table = next(name for name, definition in prompt.tables.items() if definition['read_id'] == 'b')
        return SimpleNamespace(result=parse(payload(query=f'SELECT COUNT(*) AS total FROM "{table}"')))
    def eligibility(eligibility_request, **kwargs):
        (source,) = eligibility_request.source_catalog.sources
        assessed_reads.append(source.read_id)
        assert any(field.label == 'name' for field in source.fields)
        keep = first_retained or source.read_id == 'b'
        return SemanticReadEligibilityResult((ReadRequirementAssessment('fact_1', source.read_id,
            (source.id,), source.read_id, tuple(f.field_ref for f in source.fields) if keep else (),
            'The fixture explicitly assesses the candidate population.',
            SemanticReadDecision.RETAIN if keep else SemanticReadDecision.DROP),), ())
    monkeypatch.setattr(compilation, '_turn', turn)
    monkeypatch.setattr(compilation, '_read_eligibility_turn', eligibility)
    request = compilation.SemanticCompilationRequest('later-batch', question, QuestionContractRequest(
        current_question=question, conversation_context={}), catalog, (), Port(), None, 'openai', 1, 1,
        None, {}, HostPromptContext())
    result = compilation.compile_semantic_question(request)
    assert isinstance(result, compilation.SemanticCompilationSuccess)
    assert observed == ['a', 'b']
    assert assessed_reads == ['a', 'b']
    assert set(authored_reads) == ({'a', 'b'} if first_retained else {'b'})


@pytest.mark.parametrize('persisted_mutation', [False, True])
@pytest.mark.parametrize(('identifier', 'address_type'), [
    ('00000000-0000-0000-0000-000000000001', 'uuid'), ('42', 'integer'), ('12.5', 'number')])
def test_literal_resource_address_does_not_require_an_invented_identity_namespace(monkeypatch, persisted_mutation, identifier, address_type):
    from fervis.lookup.relation_catalog import CatalogParam, ParamSource
    from fervis.lookup.query_enrichment.semantic import InputResourceSearchTerms
    read = replace(_read('entries', params=(CatalogParam('channel_id', 'channel_id', ParamSource.PATH,
        address_type, required=True),)), resource_names=('entries',))
    catalog = RelationCatalog(reads=(read,))
    question = f'How many entries belong to channel {identifier}?'
    stages = []
    def turn(purpose, *, prompt, parse, **kwargs):
        stages.append(purpose.value)
        if purpose.value == 'question_contract':
            return SimpleNamespace(result=parse(_frame_payload(supplied_values=[{
                'meaning':'the specified channel', 'denotation_basis':'The question supplies its literal identifier.',
                'entity_reference':{'instance_kind':'channel','value':{'operands':[identifier],'origin':{'kind':'question'}}}}])))
        if purpose.value == 'query_enrichment':
            return SimpleNamespace(result=SemanticQueryEnrichmentResult(tuple(
                RecallBucketMatch(bucket.bucket_ref, ('entries',), ('entries',)) for bucket in prompt.request.recall_buckets),
                tuple(InputResourceSearchTerms(task.input_use_ref, ('entries',)) for task in prompt.request.reference_tasks)))
        if purpose.value=='grounding':
            assert prompt.turn_name=='reference contract selection'
            return SimpleNamespace(result=parse(reference_contract_payload({'i1':None})))
        assert purpose.value == 'source_realization'
        view = next(name for name, table in prompt.tables.items() if table.get('read_id') == 'entries')
        symbol = next(name for name, description in prompt.parameters.items() if description.get('kind') == 'reference_literal')
        return SimpleNamespace(result=parse(payload(query=f'SELECT COUNT(*) AS total FROM "{view}"',
            api_bindings=[{'view':view,'parameter_ref':'channel_id','binding':symbol}])))
    monkeypatch.setattr(compilation, '_turn', turn)
    monkeypatch.setattr(compilation, '_read_eligibility_turn', lambda eligibility_request, **kwargs: SemanticReadEligibilityResult(
        tuple(ReadRequirementAssessment('fact_1',source.read_id,(source.id,),source.read_id,
            tuple(field.field_ref for field in source.fields),'Each returned row is one entry.',SemanticReadDecision.RETAIN)
            for source in eligibility_request.source_catalog.sources), ()))
    calls = []
    class Port:
        def read(self, *, endpoint_name, args):
            calls.append(endpoint_name)
            assert endpoint_name == 'entries' and str(args['channel_id']) == identifier
            return {'responseStatus':200,'responseBody':[{'id':1},{'id':2}]}
    request = compilation.SemanticCompilationRequest('opaque-address', question, QuestionContractRequest(
        current_question=question, conversation_context={}), catalog, (), Port(), None, 'openai', 1, 10,
        None, {}, HostPromptContext())
    result = compilation.compile_semantic_question(request)
    assert stages == ['question_contract', 'query_enrichment', 'grounding', 'source_realization']
    program = decode_answer_program(canonical_answer_program_json(result.compilation.answer_program))
    if persisted_mutation:
        from fervis.lookup.answer_program.operations import SqlNamedInput
        from fervis.lookup.answer_program.values import ParameterRef
        from fervis.lookup.contract_codec import canonical_contract_fingerprint
        from fervis.lookup.relational_sql.execution import QueryValidationError
        operation = next(item for item in program.operations if isinstance(item.spec, SqlQuerySpec))
        modified = replace(operation, spec=replace(operation.spec, query=operation.spec.query+' WHERE $raw IS NULL',
            parameters=(SqlNamedInput('raw', ParameterRef(program.parameters[0].id)),)))
        program = replace(program, operations=tuple(modified if item.id == operation.id else item for item in program.operations),
            fact_template=tuple(replace(fact, operations=tuple(
                replace(pin, fingerprint=canonical_contract_fingerprint(modified)) if pin.operation_id == modified.id else pin
                for pin in fact.operations)) for fact in program.fact_template))
        program = decode_answer_program(canonical_answer_program_json(program))
        with pytest.raises(QueryValidationError, match='Reference literals'):
            invoke_answer_program(program=program, bindings=result.compilation.initial_bindings,
                environment=ExecutionEnvironment(catalog=catalog), ports=RuntimePorts(Port(), LookupMemory()))
        assert calls == []
        return
    executed = invoke_answer_program(program=program, bindings=result.compilation.initial_bindings,
        environment=ExecutionEnvironment(catalog=catalog), ports=RuntimePorts(Port(), LookupMemory()))
    assert executed.issue is None
    assert next(iter(executed.fact_result.outcome.projected_rows[0].values.values())) == 2
