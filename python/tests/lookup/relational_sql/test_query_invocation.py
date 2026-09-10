from decimal import Decimal
import pytest
from fervis.lookup.answer_program.invocation import invoke_answer_program, RuntimePorts
from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
from fervis.lookup.answer_program.operations import SqlNamedInput
from fervis.lookup.answer_program.values import FactValue, LiteralType, ConstantRef
from fervis.lookup.memory.projection import LookupMemory
from fervis.lookup.plan_execution.errors import VerificationError
from fervis.lookup.relational_sql.acquisition import ApiView
from fervis.lookup.relational_sql.execution import QueryValidationError
from fervis.lookup.relational_sql.compiler import compile_query_answer
from fervis.lookup.relational_sql.results import ResultContract
from fervis.lookup.relation_catalog.model import RelationCatalog
from fervis.lookup.relation_catalog.row_sources import build_api_row_source_catalog
from tests.lookup.relational_engine.test_dependent_reads import _read


def test_only_query_views_are_acquired_and_grounded_input_evidence_survives():
    catalog = RelationCatalog(reads=(_read('measurements'), _read('unrelated')))
    sources = build_api_row_source_catalog(catalog).sources
    views = tuple(ApiView(source.read_id, source.id, {'value': source.fields[0].id}, {}) for source in sources)
    calls = []
    class Port:
        def read(self, *, endpoint_name, args):
            calls.append(endpoint_name)
            return {'responseStatus': 200, 'responseBody': [{'id': 1}, {'id': 3}]}
    threshold = FactValue.literal(id='threshold', literal_type=LiteralType.NUMBER,
                                  value='2', proof_refs=('question:threshold',))
    compiled=compile_query_answer(question='Sum measurements above the supplied threshold.',
        query='SELECT SUM(value) AS total FROM measurements WHERE value > $threshold',
        views=views,catalog=catalog,output_types={'total':'number'},result_contract=ResultContract('scalar'),
        query_parameters=(SqlNamedInput('threshold',ConstantRef(threshold.id,'grounded_question',threshold)),))
    answer=invoke_answer_program(program=compiled.program,bindings=compiled.bindings,
        environment=ExecutionEnvironment(catalog=catalog),ports=RuntimePorts(Port(),LookupMemory()))
    assert answer.issue is None
    assert next(iter(answer.fact_result.outcome.projected_rows[0].values.values())) == Decimal('3')
    assert calls == ['measurements']
    reachable=set(answer.proof_node_refs_by_result_output_id['result_1'])
    while True:
        prior=set(reachable)
        reachable.update(edge.source for edge in answer.proof_graph.edges if edge.target in reachable)
        if reachable==prior:break
    proofs={ref for node in answer.proof_graph.nodes if node.id in reachable for ref in node.proof_refs}
    assert {'read:measurements','question:threshold'} <= proofs


def test_missing_grounding_and_early_selection_are_rejected_before_api_calls():
    catalog = RelationCatalog(reads=(_read('items'),))
    source = build_api_row_source_catalog(catalog).sources[0]
    views = (ApiView('items', source.id, {'id': source.fields[0].id}, {}),)
    for query in ('SELECT id FROM items WHERE id=$missing', 'SELECT id FROM items LIMIT 1'):
        with pytest.raises((QueryValidationError,VerificationError),match='parameters|selection'):
            compile_query_answer(question='List the requested items.',query=query,views=views,catalog=catalog,
                output_types={'id':'integer'},result_contract=ResultContract('rows',('id',)))
