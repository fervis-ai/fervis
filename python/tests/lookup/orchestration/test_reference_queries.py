from dataclasses import replace
import pytest
from jsonschema import validate
from fervis.lookup.relational_sql.reference_planning import ReferenceMeaning,ReferenceQueryPrompt,parse_reference_query
from fervis.lookup.relational_sql.parameters import query_parameter_menu,with_catalog_choices
from fervis.lookup.relational_sql.catalog import build_query_view_catalog
from fervis.lookup.relational_sql.compiler import compile_query_answer
from fervis.lookup.relational_sql.results import ResultContract
from fervis.lookup.available_sources import build_available_source_catalog
from fervis.lookup.read_eligibility import SemanticReadEligibilityResult,ReadRequirementAssessment,SemanticReadDecision
from fervis.lookup.relation_catalog import RelationCatalog,CatalogField
from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
from fervis.lookup.source_reads.access_model import ReadAccessCatalog
from fervis.lookup.answer_program.values import FactValue,LiteralType
from fervis.lookup.grounding.semantic import CanonicalInputValue
from fervis.lookup.question_contract import InputTerm,InputDenotation
from fervis.lookup.question_contract.model import InputDenotationKind
from fervis.lookup.semantic_types import SourceOrigin,SourceOriginKind,TextType
from fervis.lookup.answer_program.invocation import invoke_answer_program,RuntimePorts
from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
from fervis.lookup.memory.projection import LookupMemory
from tests.lookup.relational_engine.test_dependent_reads import _read


@pytest.mark.parametrize(('case','text','expected'),[
    ('name','Default',1),('role','the primary record',2),('split','Ada Lovelace',2),
])
@pytest.mark.parametrize("dependent_api", [False, True])
def test_normal_reference_planner_compiles_live_reference_relations(case,text,expected,dependent_api):
    read=replace(_read('records'),fields=(*_read('records').fields,
        CatalogField('name','string',path='name',row_path_id='root'),
        CatalogField('given','string',path='given',row_path_id='root'),
        CatalogField('family','string',path='family',row_path_id='root'),
        CatalogField('primary','boolean',path='is_primary',row_path_id='root',metadata={'description':'True marks the configured primary record.'})))
    from fervis.lookup.relation_catalog import CatalogParam,ParamSource,EntityKeyComponentTarget
    observations=_read('observations',params=(CatalogParam('record_id','record_id',ParamSource.PATH,'integer',
        required=True,entity_target=EntityKeyComponentTarget('records','primary','id')),))
    catalog=RelationCatalog(reads=(read,observations));sources=build_api_row_source_catalog(catalog);source=sources.sources[0]
    eligibility=SemanticReadEligibilityResult((ReadRequirementAssessment('fact_1',read.id,(source.id,),read.id,
        tuple(f.field_ref for f in source.fields),'These records expose the reference attributes.',SemanticReadDecision.RETAIN),),())
    available=build_available_source_catalog(sources,read_eligibility=eligibility)
    view_catalog=build_query_view_catalog(catalog);view=view_catalog.views[0]
    value=FactValue.literal(id='reference_text',known_input_id='i1',literal_type=LiteralType.STRING,value=text,proof_refs=('question_input:i1',))
    menu=with_catalog_choices(query_parameter_menu((CanonicalInputValue(value.id,'i1',('fact_1:sql_input:i1',),value,value.proof_refs),)),
        source_catalog=available,source_refs={source.id})
    origin=SourceOrigin(SourceOriginKind.QUESTION_CONTEXT,text)
    meaning=ReferenceMeaning('fact_1','i1','record',f'Identify {text}.',(origin,),('i1',),reference_text=text,reference_kind="description" if case == "role" else "literal")
    prompt=ReferenceQueryPrompt(question=f'Return the ID for {text}.',meaning=meaning,tables=view_catalog.tables,parameters=menu.descriptions)
    interpretations=[]
    if case=='role':
        choice=next(name for name,desc in menu.descriptions.items() if desc.get('kind')=='catalog_choice' and desc['value']=='true')
        predicate=f'is_primary=${choice}'
        interpretations=[{'input':'p1_1','choice':choice,'basis':'The API identifies primary records using this flag.'}]
    else:predicate=("CONCAT(given, ' ', family)" if case=='split' else 'name')+'=$p1_1'
    matching = "CONCAT(given, ' ', family)" if case == 'split' else 'name'
    extra = f', {matching} AS matched_name' if case != 'role' else ''
    payload={'query':f'SELECT id AS record_id{extra} FROM "{view.name}" WHERE {predicate}',
        'mode':'rows','columns':[{'name':'record_id','value_type':'integer'}, *([{'name':'matched_name','value_type':'string'}] if case != 'role' else [])],
        'outputs':[{'kind':'identity','authority':view.name+':key:0','components':{'id':'record_id'},'label':'record','display_column':None}],
        'ordering':[],'request_arguments':[],'interpretations':[],
        'reference_binding':{'kind':'description','basis':'The primary property defines the role.'} if case=='role' else {'kind':'literal','match_column':'matched_name'}}
    if case != "role": payload["query"] = payload["query"].split(" WHERE ")[0]
    validate(payload,prompt._schema())
    authored=parse_reference_query(payload,prompt=prompt,menu=menu)
    inputs=(InputTerm('i1',origin,text,TextType()),)
    denotations=(InputDenotation('d1','i1','record reference','The input denotes a record.','records',InputDenotationKind.IDENTITY_REFERENCE,reference_descriptions=(text,) if case == 'role' else ()),)
    from fervis.lookup.orchestration.reference_queries import plan_fact_references, reference_input_values
    from fervis.lookup.grounding.semantic import GroundingPartition
    partitions=(GroundingPartition('i1',('fact_1:sql_input:i1',),TextType(),None,'record reference',is_identity_reference=True),)
    values=reference_input_values(partitions,inputs={'i1':inputs[0]})
    turns=[]
    def turn(purpose,prompt,parse):
        turns.append(purpose)
        submitted=payload
        if case=='role':
            symbol=next(name for name,desc in prompt.parameters.items() if desc.get('kind')=='catalog_choice' and desc['value']=='true')
            submitted={**payload,'query':payload['query'].replace('$'+choice,'$'+symbol)}
        return parse(submitted)
    reference,=plan_fact_references(fact=meaning,inputs={'i1':inputs[0]},
        denotations={'i1':denotations[0]},values=values,catalog=catalog,access=ReadAccessCatalog(),
        question=prompt.question,responses=(),turn=turn)
    assert len(turns)==1
    if dependent_api:
        from fervis.lookup.relational_sql.parameters import with_reference_arguments
        from fervis.lookup.relational_sql.binding import bind_query_answer
        from fervis.lookup.relational_sql.authoring import parse_query_answer
        from fervis.lookup.orchestration.reference_queries import reference_prerequisites
        final_view=next(view for view in view_catalog.views if view_catalog.tables[view.name]['read_id']=='observations')
        final_menu=with_reference_arguments(query_parameter_menu(()),(reference,))
        final_meaning=replace(meaning,result_kind='scalar',output_kinds=('value',))
        final_tables={final_view.name:view_catalog.tables[final_view.name],reference.view.name:reference.table}
        authored=parse_query_answer({'query':f'SELECT COUNT(*) AS value FROM "{final_view.name}"',
            'mode':'scalar','columns':[{'name':'value','value_type':'integer'}],
            'outputs':[{'kind':'value','column':'value','label':'count'}], 'ordering':[],
            'request_arguments':[{'view':final_view.name,'parameter_ref':view_catalog.tables[final_view.name]['request_parameters'][0]['param_ref'],'binding':'r1_1'}],
            'interpretations':[]},table_names=set(final_tables),parameter_names=set(final_menu.expressions),
            meaning=final_meaning,parameter_descriptions=final_menu.descriptions,tables=final_tables,expected_input_refs=('i1',))
        bound=bind_query_answer(authored,final_menu,(final_view,))
        prerequisites,bindings=reference_prerequisites((reference,),bound.bindings,argument_operations=bound.argument_operations)
        final=compile_query_answer(question='Count the referenced observations.',query=authored.query,
            views=bound.views,prerequisites=prerequisites,bindings=bindings,catalog=catalog,
            inputs=inputs,input_denotations=denotations,expected_input_refs=('i1',),
            output_types={'value':'integer'},result_contract=ResultContract('scalar'))
    else:
        final=compile_query_answer(question='Return the referenced ID.',query=f'SELECT id AS value FROM "{reference.view.name}"',
            views=(),relation_views=(reference.view,),prerequisites=reference.program,catalog=catalog,
            bindings=reference.bindings,inputs=inputs,input_denotations=denotations,expected_input_refs=('i1',),
            output_types={'value':'integer'},result_contract=ResultContract('scalar'))
    class Port:
        calls=[]
        def read(self,**kwargs):
            self.calls.append(kwargs)
            if kwargs['endpoint_name']=='observations':
                return {'responseStatus':200,'responseBody':[{'id':i} for i in range(kwargs['args']['record_id'])]}
            return {'responseStatus':200,'responseBody':[
            {'id':1,'name':'Default','given':'Ada','family':'Byron','is_primary':False},
            {'id':2,'name':'Cedar','given':'Ada','family':'Lovelace','is_primary':True}]}
    executed=invoke_answer_program(program=final.program,bindings=final.bindings,
        environment=ExecutionEnvironment(catalog=catalog),ports=RuntimePorts(Port(),LookupMemory()))
    assert executed.issue is None
    assert next(iter(executed.fact_result.outcome.projected_rows[0].values.values()))==expected

    if dependent_api:
        assert Port.calls[-1]=={'endpoint_name':'observations','args':{'record_id':expected}}


def test_interpreted_collection_member_keeps_the_original_collection_binding():
    from fervis.lookup.orchestration.reference_queries import plan_fact_references,reference_input_values
    from fervis.lookup.grounding.semantic import GroundingPartition
    from fervis.lookup.semantic_types import CollectionType
    read=replace(_read('records'),fields=(*_read('records').fields,
        CatalogField('name','string',path='name',row_path_id='root'),
        CatalogField('primary','boolean',path='is_primary',row_path_id='root')))
    catalog=RelationCatalog(reads=(read,))
    origin=SourceOrigin(SourceOriginKind.QUESTION_CONTEXT,'Alpha and the primary record')
    term=InputTerm('i1',origin,('the primary record','Alpha'),CollectionType(TextType()))
    denotation=InputDenotation('d1','i1','named and configured records','Identify each supplied record.','records',InputDenotationKind.IDENTITY_REFERENCE,reference_descriptions=('the primary record',))
    partitions=(GroundingPartition('i1',('fact_1:sql_input:i1',),term.value_type,None,'references',is_identity_reference=True),)
    values=reference_input_values(partitions,inputs={'i1':term})
    meaning=ReferenceMeaning('fact_1','i1','records','Count the selected records.',(origin,),('i1',),reference_text=origin.meaning)
    def turn(purpose,prompt,parse):
        view=next(iter(prompt.tables))
        interpretations=[]
        assert prompt.meaning.reference_is_collection_member
        assert prompt.meaning.reference_text==prompt.parameters['p1_1']['label']
        assert prompt.parameters['p1_1']['kind']==('definition' if prompt.meaning.reference_kind == 'description' else 'input')
        if prompt.parameters['p1_1']['label']=='the primary record':
            choice=next(name for name,desc in prompt.parameters.items() if desc.get('kind')=='catalog_choice' and desc['value']=='true')
            condition=f'is_primary=${choice}'
            interpretations=[{'input':'p1_1','choice':choice,'basis':'The configured primary record has its primary flag set.'}]
        else:
            condition='name=$p1_1'
        suffix = f' WHERE {condition}' if interpretations else ''
        return parse({'query':f'SELECT id, name AS matched_name FROM "{view}"{suffix}','mode':'rows',
            'columns':[{'name':'id','value_type':'integer'},{'name':'matched_name','value_type':'string'}],
            'outputs':[{'kind':'identity','authority':view+':key:0','components':{'id':'id'},'label':'record','display_column':None}],
            'ordering':[],'request_arguments':[],'interpretations':[],
            'reference_binding':{'kind':'description','basis':'The primary flag defines this member.'} if interpretations else {'kind':'literal','match_column':'matched_name'}})
    reference,=plan_fact_references(fact=meaning,inputs={'i1':term},denotations={'i1':denotation},
        values=values,catalog=catalog,access=ReadAccessCatalog(),question="Count Alpha and the primary record",responses=(),turn=turn)
    final=compile_query_answer(question='Count selected records.',query=f'SELECT COUNT(*) AS total FROM "{reference.view.name}"',
        views=(),relation_views=(reference.view,),prerequisites=reference.program,bindings=reference.bindings,catalog=catalog,
        inputs=(term,),input_denotations=(denotation,),expected_input_refs=('i1',),
        output_types={'total':'integer'},result_contract=ResultContract('scalar'))
    class Port:
        calls=0
        def read(self,**kwargs):
            self.calls+=1
            return {'responseStatus':200,'responseBody':[{'id':1,'name':'Alpha','is_primary':False},{'id':2,'name':'Cedar','is_primary':True}]}
    port=Port()
    result=invoke_answer_program(program=final.program,bindings=final.bindings,
        environment=ExecutionEnvironment(catalog=catalog),ports=RuntimePorts(port,LookupMemory()))
    assert result.issue is None
    assert next(iter(result.fact_result.outcome.projected_rows[0].values.values()))==2
    from fervis.lookup.answer_program.values import BindingSet
    changed=FactValue.string_set(id=values[0].typed_value.id,known_input_id='i1',values=('Alpha','the secondary record'),proof_refs=('question_input:i1',))
    rebound=BindingSet.from_bindings(tuple(replace(binding,value=changed) for binding in final.bindings.bindings))
    calls=port.calls
    with pytest.raises(ValueError,match='fixed|certified'):
        invoke_answer_program(program=final.program,bindings=rebound,
            environment=ExecutionEnvironment(catalog=catalog),ports=RuntimePorts(port,LookupMemory()))
    assert port.calls==calls


def test_reference_compilation_retains_prerequisites_outside_its_recalled_views():
    from fervis.lookup.relation_catalog import CatalogParam,ParamSource,EntityKeyComponentTarget
    from fervis.lookup.source_reads.access_model import ReadDependency,AccessArgument
    from fervis.lookup.orchestration.reference_queries import reference_input_values,plan_fact_references
    from fervis.lookup.grounding.semantic import GroundingPartition
    parent=_read('organizations')
    child=replace(_read('records',params=(CatalogParam('organization_id','organization_id',ParamSource.PATH,'integer',
        required=True,entity_target=EntityKeyComponentTarget('organizations','primary','id')),)),
        fields=(*_read('records').fields,CatalogField('name','string',path='name',row_path_id='root')))
    catalog=RelationCatalog(reads=(parent,child))
    parent_source,child_source=build_api_row_source_catalog(catalog).sources
    parent_key=next(field for field in parent_source.fields if field.path=='id')
    access=ReadAccessCatalog((parent_source,child_source),(ReadDependency(child_source.id,parent_source.id,
        (AccessArgument(child_source.params[0].param_ref,parent_key.field_ref),),'Each organization exposes its records.'),))
    origin=SourceOrigin(SourceOriginKind.QUESTION_CONTEXT,'Alpha')
    term=InputTerm('i1',origin,'Alpha',TextType())
    denotation=InputDenotation('d1','i1','record named Alpha','The supplied name identifies a record.','records',InputDenotationKind.IDENTITY_REFERENCE)
    values=reference_input_values((GroundingPartition('i1',('fact_1:sql_input:i1',),TextType(),None,'record name',is_identity_reference=True),),inputs={'i1':term})
    discovery=[]
    def discover(read_ids):
        discovery.append(read_ids)
        return access
    def turn(purpose,prompt,parse):
        view,=prompt.tables
        return parse({'query':f'SELECT id, name AS matched_name FROM "{view}"','mode':'rows',
            'columns':[{'name':'id','value_type':'integer'},{'name':'matched_name','value_type':'string'}],
            'outputs':[{'kind':'identity','authority':view+':key:0','components':{'id':'id'},'label':'record','display_column':None}],
            'ordering':[],'request_arguments':[],'interpretations':[],'reference_binding':{'kind':'literal','match_column':'matched_name'}})
    reference,=plan_fact_references(fact=ReferenceMeaning('fact_1','i1','records','Identify Alpha.',(origin,),('i1',),reference_text='Alpha'),
        inputs={'i1':term},denotations={'i1':denotation},values=values,catalog=catalog,
        reference_catalog=RelationCatalog(reads=(child,)),access=ReadAccessCatalog(),
        question='Identify Alpha.',responses=(),turn=turn,discover_access=discover)
    assert discovery==[('records',)]
    final=compile_query_answer(question='Return the reference key.',query=f'SELECT id AS value FROM "{reference.view.name}"',
        views=(),relation_views=(reference.view,),prerequisites=reference.program,bindings=reference.bindings,catalog=catalog,
        inputs=(term,),input_denotations=(denotation,),expected_input_refs=('i1',),output_types={'value':'integer'},result_contract=ResultContract('scalar'))
    calls=[]
    class Port:
        def read(self,*,endpoint_name,args):
            calls.append((endpoint_name,args))
            rows=[{'id':1},{'id':2}] if endpoint_name=='organizations' else [{'id':args['organization_id'],'name':'Alpha' if args['organization_id']==2 else 'Beta'}]
            return {'responseStatus':200,'responseBody':rows}
    result=invoke_answer_program(program=final.program,bindings=final.bindings,
        environment=ExecutionEnvironment(catalog=catalog),ports=RuntimePorts(Port(),LookupMemory()))
    assert result.issue is None
    assert next(iter(result.fact_result.outcome.projected_rows[0].values.values()))==2
    assert calls==[('organizations',{}),('records',{'organization_id':1}),('records',{'organization_id':2})]


def test_configured_reference_can_use_a_keyless_settings_relation(monkeypatch):
    from fervis.lookup.relation_catalog import EntityReference,EntityReferenceComponent
    from fervis.lookup.relation_catalog.selection import select_resolver_reads
    from fervis.lookup.orchestration.reference_queries import reference_input_values,plan_fact_references
    from fervis.lookup.grounding.semantic import GroundingPartition
    from fervis.lookup.relational_sql.execution import QueryValidationError
    settings=replace(_read('settings'),resource_names=('settings',),candidate_keys=(),
        fields=(CatalogField('default_record_id','integer',path='default_record_id',row_path_id='root'),),
        entity_references=(EntityReference('default_record','records','primary',
            (EntityReferenceComponent('id','default_record_id'),)),))
    catalog=RelationCatalog(reads=(_read('records'),settings))
    recalled=select_resolver_reads(catalog,catalog_search_terms=('settings',),limit=3)
    assert [read.id for read in recalled]==['settings']
    origin=SourceOrigin(SourceOriginKind.QUESTION_CONTEXT,'the configured default record')
    term=InputTerm('i1',origin,'the configured default record',TextType())
    denotation=InputDenotation('d1','i1','configured default record','The reference denotes the configured default.','records',InputDenotationKind.IDENTITY_REFERENCE,reference_descriptions=(term.operand,))
    values=reference_input_values((GroundingPartition('i1',('fact_1:sql_input:i1',),TextType(),None,'default record',is_identity_reference=True),),inputs={'i1':term})
    def turn(purpose,prompt,parse):
        view,=prompt.tables
        assert not any(item.get('kind')=='catalog_choice' for item in prompt.parameters.values())
        payload={'query':f'SELECT default_record_id AS id FROM "{view}"','mode':'rows',
            'columns':[{'name':'id','value_type':'integer'}],
            'outputs':[{'kind':'identity','authority':view+':reference:0','components':{'id':'id'},'label':'default record','display_column':None}],
            'ordering':[],'request_arguments':[],'interpretations':[],
            'reference_binding':{'kind':'description','basis':'The settings relation exposes the configured default record identity.'}}
        validate(payload,prompt._schema())
        with pytest.raises(QueryValidationError,match='syntax'):
            parse({**payload,'reference_binding':{'kind':'literal','match_column':'matched_name'}})
        return parse(payload)
    reference,=plan_fact_references(fact=ReferenceMeaning('fact_1','i1','records','Identify the default record.',(origin,),('i1',),reference_text=term.operand),
        inputs={'i1':term},denotations={'i1':denotation},values=values,catalog=catalog,
        reference_catalog=RelationCatalog(reads=recalled),access=ReadAccessCatalog(),question=origin.meaning,responses=(),turn=turn)
    final=compile_query_answer(question='Return the configured identity.',query=f'SELECT id AS value FROM "{reference.view.name}"',
        views=(),relation_views=(reference.view,),prerequisites=reference.program,bindings=reference.bindings,catalog=catalog,
        inputs=(term,),input_denotations=(denotation,),expected_input_refs=('i1',),output_types={'value':'integer'},result_contract=ResultContract('scalar'))
    for configured in (1,2):
        class Port:
            def read(self,*,endpoint_name,args):
                assert endpoint_name=='settings' and args=={}
                return {'responseStatus':200,'responseBody':[{'default_record_id':configured}]}
        result=invoke_answer_program(program=final.program,bindings=final.bindings,environment=ExecutionEnvironment(catalog=catalog),
            ports=RuntimePorts(Port(),LookupMemory()))
        assert result.issue is None
        assert next(iter(result.fact_result.outcome.projected_rows[0].values.values()))==configured
