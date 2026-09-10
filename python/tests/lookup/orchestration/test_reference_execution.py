from types import SimpleNamespace
import pytest

from fervis.lookup.orchestration import program_execution
from fervis.lookup.orchestration.request import LookupRequest
from fervis.lookup.orchestration.result import RunStatus
from fervis.lookup.answer_program.instantiation import ExecutionEnvironment
from fervis.lookup.identity_types import IdentityExecutionFailureReason,ReferenceResolutionFailure
from fervis.lookup.outcomes.errors import ExecutionIssue,ExecutionIssueKind
from fervis.lookup.outcomes.model import NeedsClarification
from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.canonical_data import EntityKeyValue,EntityKeyComponentValue
from tests.lookup.read_eligibility.test_semantic_read_eligibility import _semantic_contract


@pytest.mark.parametrize('reason',list(IdentityExecutionFailureReason))
def test_reference_issue_is_synthesized_as_clarification_without_runtime_error(monkeypatch,reason):
    from fervis.lookup.orchestration import execution_sources,result_synthesis
    contract=_semantic_contract().contract
    input_ref=contract.inputs[0].id
    keys=tuple(EntityKeyValue('staff','primary_key',(EntityKeyComponentValue('staff_id',value),)) for value in ('one','two')) if reason is IdentityExecutionFailureReason.AMBIGUOUS_RESULT else ()
    issue=ExecutionIssue(ExecutionIssueKind.REFERENCE_RESOLUTION,'Reference unresolved',proof_refs=('read:staff',),
        reference=ReferenceResolutionFailure(input_ref,reason,keys))
    program=SimpleNamespace(question_contract=contract)
    execution=SimpleNamespace(issue=issue,fact_result=None,relations=(),proof_refs=issue.proof_refs,
        program_id='program',invocation_id='invocation',program=program,proof_graph=None,proof_node_refs_by_result_output_id={})
    monkeypatch.setattr(execution_sources,'prepare_execution_catalog',lambda **kw:kw['catalog'])
    monkeypatch.setattr(program_execution,'invoke_answer_program',lambda **kw:execution)
    recorded=[];persisted=[]
    monkeypatch.setattr(program_execution,'record_execution_step',lambda *args,**kw:recorded.append(kw))
    monkeypatch.setattr(result_synthesis,'record_lookup_result_lineage',lambda **kw:persisted.append(kw))
    monkeypatch.setattr(program_execution,'record_runtime_error_lineage',lambda **kw:pytest.fail('Reference issue became a runtime error'))
    result=program_execution.run_answer_program_execution(request=LookupRequest('Which Ada?',run_id='run'),
        ports=program_execution.ProgramExecutionPorts(None),program=program,bindings=None,
        environment=ExecutionEnvironment(catalog=RelationCatalog()),invocation_binding=None,
        question_contract_step_id='frame',usage={'costUsd':0.02})
    assert result.status==RunStatus.NEEDS_CLARIFICATION
    assert isinstance(result.fact_result.outcome,NeedsClarification)
    assert result.usage=={'costUsd':0.02}
    assert len(recorded)==len(persisted)==1
    assert not recorded[0]['error_json']
    assert persisted[0]['fact_result']==result.fact_result
