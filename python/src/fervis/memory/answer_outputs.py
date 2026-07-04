"""Prior answer-output memory projection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fervis.memory.addresses import FactAddressKind


@dataclass(frozen=True)
class PriorAnswerOutputFrame:
    output_id: str
    description: str
    requested_value_frame: str
    source_lineage: tuple[str, ...] = ()

    def to_request_shape(self) -> dict[str, str]:
        return {
            "output_id": self.output_id,
            "description": self.description,
            "requested_value_frame": self.requested_value_frame,
        }


@dataclass(frozen=True)
class PriorAnswerKnownInput:
    id: str
    kind: str
    text: str
    role: str = ""
    description: str = ""


@dataclass(frozen=True)
class PriorAnswerRequestArtifact:
    id: str
    answer_fact: str
    output_frames: tuple[PriorAnswerOutputFrame, ...]
    known_inputs: tuple[PriorAnswerKnownInput, ...] = ()


def prior_answer_request_artifacts(
    artifact: Any,
) -> tuple[PriorAnswerRequestArtifact, ...]:
    provenance = getattr(artifact, "provenance", {}) or {}
    question_contract = provenance.get("question_contract")
    if not isinstance(question_contract, dict):
        return ()
    question_inputs_by_ref = {
        str(item.get("id") or "").strip(): item
        for item in question_contract.get("question_inputs") or ()
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    }
    output: list[PriorAnswerRequestArtifact] = []
    for index, item in enumerate(
        question_contract.get("answer_requests") or (), start=1
    ):
        if not isinstance(item, dict):
            continue
        answer_request = _answer_request_with_question_inputs(
            item,
            question_inputs_by_ref=question_inputs_by_ref,
        )
        output.append(
            PriorAnswerRequestArtifact(
                id=str(answer_request.get("id") or f"fact_{index}").strip(),
                answer_fact=str(answer_request.get("answer_fact") or "").strip(),
                output_frames=_prior_answer_output_frames(
                    answer_request=answer_request,
                    artifact=artifact,
                ),
                known_inputs=_prior_answer_known_inputs(answer_request),
            )
        )
    return tuple(item for item in output if item.id and item.answer_fact)


def _prior_answer_output_frames(
    *,
    answer_request: dict[str, Any],
    artifact: Any | None = None,
) -> tuple[PriorAnswerOutputFrame, ...]:
    lineage_by_output_id = _answer_output_source_lineage_by_id(artifact)
    output: list[PriorAnswerOutputFrame] = []
    for index, item in enumerate(answer_request.get("answer_outputs") or (), start=1):
        if not isinstance(item, dict):
            continue
        output_id = str(item.get("id") or f"answer_output_{index}").strip()
        description = str(item.get("description") or "").strip()
        if not description:
            continue
        output.append(
            PriorAnswerOutputFrame(
                output_id=output_id,
                description=description,
                requested_value_frame=str(
                    item.get("requested_value_frame") or description
                ).strip(),
                source_lineage=lineage_by_output_id.get(output_id, ()),
            )
        )
    return tuple(output)


def _answer_request_with_question_inputs(
    item: dict[str, Any],
    *,
    question_inputs_by_ref: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    output = dict(item)
    if not question_inputs_by_ref:
        return output
    output["known_inputs"] = [
        question_inputs_by_ref[input_ref]
        for input_ref in _answer_request_used_input_refs(item)
        if input_ref in question_inputs_by_ref
    ]
    return output


def _answer_request_used_input_refs(item: dict[str, Any]) -> tuple[str, ...]:
    raw = item.get("input_decisions")
    if not isinstance(raw, list):
        return ()
    output: list[str] = []
    for decision in raw:
        if not isinstance(decision, dict) or decision.get("use_input") is not True:
            continue
        input_ref = str(decision.get("input_ref") or "").strip()
        if input_ref:
            output.append(input_ref)
    return tuple(output)


def _prior_answer_known_inputs(
    answer_request: dict[str, Any],
) -> tuple[PriorAnswerKnownInput, ...]:
    output: list[PriorAnswerKnownInput] = []
    for item in answer_request.get("known_inputs") or ():
        if not isinstance(item, dict):
            continue
        input_id = str(item.get("id") or "").strip()
        kind = str(item.get("kind") or "").strip()
        text = str(item.get("text") or item.get("reference_text") or "").strip()
        if not input_id or not kind or not text:
            continue
        output.append(
            PriorAnswerKnownInput(
                id=input_id,
                kind=kind,
                text=text,
                role=str(item.get("role") or "").strip(),
                description=str(
                    item.get("value_meaning_hint") or item.get("description") or ""
                ).strip(),
            )
        )
    return tuple(output)


def _answer_output_source_lineage_by_id(
    artifact: Any | None,
) -> dict[str, tuple[str, ...]]:
    if artifact is None:
        return {}
    artifact_id = str(getattr(artifact, "artifact_id", "") or "").strip()
    if not artifact_id:
        return {}
    relation_addresses = {
        str(getattr(address, "address", "") or "").strip()
        for address in getattr(artifact, "addresses", ()) or ()
        if getattr(address, "kind", None) == FactAddressKind.RELATION
    }
    output: dict[str, list[str]] = {}
    for address in getattr(artifact, "addresses", ()) or ():
        kind = getattr(address, "kind", None)
        if kind == FactAddressKind.VALUE:
            _append_output_lineage(
                output,
                output_ids=_answer_output_ids(getattr(address, "scalar_value", {})),
                memory_id=f"{artifact_id}.{address.address}",
            )
        elif kind == FactAddressKind.ROW:
            relation = str(getattr(address, "source_relation", "") or "").strip()
            if relation not in relation_addresses:
                continue
            for value in (getattr(address, "values", {}) or {}).values():
                if not isinstance(value, dict):
                    continue
                _append_output_lineage(
                    output,
                    output_ids=_answer_output_ids(value),
                    memory_id=f"{artifact_id}.{relation}",
                )
    return {key: tuple(dict.fromkeys(values)) for key, values in output.items()}


def _answer_output_ids(value: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        text
        for item in value.get("answer_output_ids") or ()
        if (text := str(item or "").strip())
    )


def _append_output_lineage(
    output: dict[str, list[str]],
    *,
    output_ids: tuple[str, ...],
    memory_id: str,
) -> None:
    if not memory_id:
        return
    for output_id in output_ids:
        output.setdefault(output_id, []).append(memory_id)
