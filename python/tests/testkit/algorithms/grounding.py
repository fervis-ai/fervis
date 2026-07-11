from __future__ import annotations

from typing import Any

from fervis.lookup.conversation_resolution import (
    CompiledConversationResolution,
    ResolvedCanonicalIdentity,
    ResolvedLiteralQuestionInput,
)
from fervis.lookup.conversation_resolution.compilation import CompiledResolvedClause
from fervis.lookup.answer_program.values import FactValue, IdentityValuePayload
from fervis.lookup.fact_planning.request import RuntimeValueContext
from fervis.lookup.grounding.resolution import ground_question_inputs
from fervis.lookup.question_contract import (
    KnownInputSource,
    LiteralInputRole,
    QuestionContract,
    RequestedFact,
    RequestedFactAnswerOutput,
    RequestedFactLiteralInput,
)
from fervis.lookup.relation_catalog import RelationCatalog

from tests.testkit.assertions import exact_mismatches, subset_mismatches


def run_grounding_contract_case(payload: dict[str, Any]) -> list[str]:
    if "known_input" in payload["input"]:
        return _run_grounding_runtime_case(payload)
    actual = {
        "certifications": payload["input"].get("certifications") or (),
    }
    return subset_mismatches(
        actual=actual,
        expected_subset=payload["expect"]["result_contains"],
    )


def _run_grounding_runtime_case(payload: dict[str, Any]) -> list[str]:
    input_payload = payload["input"]
    output = ground_question_inputs(
        question=str(input_payload["question"]),
        question_contract=_question_contract_from_input(input_payload),
        full_catalog=RelationCatalog(),
        resolver_catalog=RelationCatalog(),
        data_access_port=_NoDataAccess(),
        runtime_values=RuntimeValueContext(
            runtime_date="2026-07-04",
            timezone="Africa/Nairobi",
        ),
        model_port=_NoGroundingModel(),
        provider="test",
        model_key="test",
        max_thinking_tokens=0,
        conversation_resolution=_compiled_resolution_from_input(input_payload),
    )
    actual = {
        "values": [_value_payload(value) for value in output.ledger.values],
        "certifications": [
            certification.to_payload()
            for certification in output.ledger.certifications
        ],
    }
    if "result_equals" in payload["expect"]:
        return exact_mismatches(
            actual=actual,
            expected=payload["expect"]["result_equals"],
        )
    return subset_mismatches(
        actual=actual,
        expected_subset=payload["expect"]["result_contains"],
    )


def _question_contract_from_input(payload: dict[str, Any]) -> QuestionContract:
    known = _known_input(payload["known_input"])
    fact_payload = payload["requested_fact"]
    return QuestionContract(
        requested_facts=(
            RequestedFact(
                id=str(fact_payload["id"]),
                description=str(fact_payload["description"]),
                answer_outputs=(
                    RequestedFactAnswerOutput(
                        id=str(fact_payload["answer_output_id"]),
                    ),
                ),
                known_inputs=(known,),
            ),
        )
    )


def _known_input(payload: dict[str, Any]) -> RequestedFactLiteralInput:
    return RequestedFactLiteralInput(
        id=str(payload["id"]),
        source=KnownInputSource(str(payload["source"])),
        text=str(payload["text"]),
        resolved_input_ref=str(payload.get("resolved_input_ref") or ""),
        resolved_value_text=str(payload["resolved_value_text"]),
        value_meaning_hint=str(payload.get("value_meaning_hint") or ""),
        field_label_text=str(payload.get("field_label_text") or ""),
        role=LiteralInputRole(str(payload["role"])),
    )


def _compiled_resolution_from_input(
    payload: dict[str, Any],
) -> CompiledConversationResolution:
    question = str(payload["question"])
    return CompiledConversationResolution(
        current_question_text=question,
        contextualized_question=question,
        clauses=(
            CompiledResolvedClause(
                current_clause_text=question,
                resolved_text=question,
                retained_frame_parts=(),
                values=(),
            ),
        ),
        inputs=(
            _resolved_question_input(payload["resolved_question_input"]),
        ),
        frame_call=None,
        used_source_card_ids=(),
        used_memory_ids=(),
    )


def _resolved_question_input(payload: dict[str, Any]) -> ResolvedLiteralQuestionInput:
    return ResolvedLiteralQuestionInput(
        input_ref=str(payload["resolved_input_ref"]),
        value_source_text=str(payload["source_text"]),
        occurrence=int(payload.get("occurrence") or 1),
        resolved_value_text=str(payload["resolved_value_text"]),
        value_meaning_hint=str(payload.get("value_meaning_hint") or ""),
        field_label_text=str(payload.get("field_label_text") or ""),
        role=LiteralInputRole(str(payload["role"])),
        canonical_identity=_resolved_canonical_identity(
            payload["resolved_canonical_identity"]
        ),
    )


def _resolved_canonical_identity(
    payload: dict[str, Any],
) -> ResolvedCanonicalIdentity:
    return ResolvedCanonicalIdentity(
        identity_type=str(payload["identity_type"]),
        identity_field=str(payload["identity_field"]),
        value=str(payload["value"]),
        authority_refs=tuple(str(ref) for ref in payload["authority_refs"]),
        lineage_refs=tuple(str(ref) for ref in payload["lineage_refs"]),
    )


def _value_payload(value: FactValue) -> dict[str, object]:
    if isinstance(value.payload, IdentityValuePayload):
        return {
            "id": value.id,
            "kind": value.kind.value,
            "identity_type": value.payload.identity_type,
            "identity_field": value.payload.identity_field,
            "value": value.payload.value,
            "display_value": value.payload.display_value,
            "proof_refs": list(value.proof_refs),
            "applies_to_requested_fact_ids": list(
                value.applies_to_requested_fact_ids
            ),
        }
    return {
        "id": value.id,
        "kind": value.kind.value,
        "proof_refs": list(value.proof_refs),
        "applies_to_requested_fact_ids": list(value.applies_to_requested_fact_ids),
    }


class _NoDataAccess:
    def read(self, **kwargs: object) -> object:
        raise AssertionError("canonical identity import must not execute source reads")


class _NoGroundingModel:
    def generate(self, **kwargs: object) -> object:
        raise AssertionError("canonical identity import must not call grounding model")
