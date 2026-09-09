"""Input grounding context shared by graph and SQL question declarations."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from .analysis import InputUseSite, RequestedFactSemanticIndex, analyze_requested_fact
from fervis.lookup.semantic_types import SourceOrigin
from .model import InputTerm, InputDenotation, InputDenotationKind, RequestedFact, QueryRequestedFact, IntentFact, IntentOutput


@dataclass(frozen=True)
class GroundingFactContext:
    requested_fact: RequestedFact | QueryRequestedFact | IntentFact
    input_by_ref: Mapping[str, InputTerm]
    input_denotation_by_ref: Mapping[str, InputDenotation]
    input_use_sites: tuple[InputUseSite, ...]
    graph_index: RequestedFactSemanticIndex | None = None

    @property
    def requested_fact_id(self):
        return self.requested_fact.id

    @property
    def term_by_ref(self):
        return self.graph_index.term_by_ref if self.graph_index is not None else {}


def grounding_fact_context(fact, *, inputs, input_denotations) -> GroundingFactContext:
    if isinstance(fact, (QueryRequestedFact, IntentFact)):
        uses = tuple(InputUseSite(
            use_ref=fact.input_use_ref(ref), input_ref=ref,
            operand_meaning=input_denotations[ref].operand_meaning,
            requested_fact_id=fact.id, structural_location=f'sql_input.{ref}',
            expression_ref=None, expected_value_type=inputs[ref].value_type,
            identity_set_ref=None, reference_fact_ref=None,
            is_identity_reference=input_denotations[ref].kind is InputDenotationKind.IDENTITY_REFERENCE,
        ) for ref in fact.input_refs)
        return GroundingFactContext(fact,inputs,input_denotations,uses)
    index = analyze_requested_fact(fact,inputs=inputs,input_denotations=input_denotations)
    return grounding_context_from_index(index)


def grounding_context_from_index(index: RequestedFactSemanticIndex) -> GroundingFactContext:
    uses = tuple(replace(use,is_identity_reference=(use.identity_set_ref is not None
                      or use.reference_fact_ref is not None)) for use in index.input_use_sites)
    return GroundingFactContext(index.requested_fact,index.input_by_ref,index.input_denotation_by_ref,uses,index)


def intent_grounding_contexts(meaning, *, question: str) -> tuple[GroundingFactContext, ...]:
    from fervis.lookup.semantic_types import SourceOriginKind
    inputs={item.id:item for item in meaning.inputs}
    denotations={item.input_ref:item for item in meaning.input_denotations}
    contexts=[]
    for request in meaning.answer_requests:
        fact=IntentFact(request.requested_fact_id,SourceOrigin(SourceOriginKind.QUESTION_CONTEXT,question),
            tuple(IntentOutput(f'r{index}',origin) for index,origin in enumerate(request.output_origins,start=1)),
            request.input_refs,request.result_kind)
        uses=tuple(InputUseSite(f'{fact.id}:sql_input:{ref}',ref,denotations[ref].operand_meaning,fact.id,
            f'input.{ref}',None,term.value_type,None,None,
            denotations[ref].kind is InputDenotationKind.IDENTITY_REFERENCE) for ref,term in inputs.items() if ref in request.input_refs)
        contexts.append(GroundingFactContext(fact,inputs,denotations,uses))
    return tuple(contexts)
