"""Typed prior-request memory projected from persisted fact artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias

from fervis.lookup.question_inputs import (
    KnownInputKind,
    LiteralInputRole,
    literal_role_part_kind,
)
from fervis.memory.addresses import FactAddress, FactAddressKind
from fervis.memory.artifacts import FactArtifact, FactOutcome


@dataclass(frozen=True)
class PriorRequestOutput:
    output_id: str
    description: str
    role: str
    source_lineage: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.output_id or not self.description or not self.role:
            raise ValueError("prior request output requires identity and meaning")

    def to_request_shape(self) -> dict[str, str]:
        return {
            "output_id": self.output_id,
            "description": self.description,
            "role": self.role,
        }


@dataclass(frozen=True)
class PriorRequestAnswerShape:
    expression_family: str
    output_roles: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.expression_family or not self.output_roles:
            raise ValueError("prior request requires typed answer shape")

    def to_request_shape(self) -> dict[str, object]:
        return {
            "expression_family": self.expression_family,
            "output_roles": self.output_roles,
        }


class PriorRequestSlotKind(StrEnum):
    ENTITY_IDENTITY = "entity_identity"
    TIME_SCOPE = "time_scope"
    LIMIT = "limit"


class PriorRequestSemanticPartKind(StrEnum):
    POPULATION_CONSTRAINT = "population_constraint"
    GROUPING = "grouping"


@dataclass(frozen=True)
class PriorRequestSemanticPart:
    kind: PriorRequestSemanticPartKind
    role: str
    text: str

    def __post_init__(self) -> None:
        if not self.role or not self.text:
            raise ValueError("prior request semantic part requires role and text")

    def to_request_shape(self) -> dict[str, str]:
        return {
            "part_kind": self.kind.value,
            "role": self.role,
            "text": self.text,
        }


@dataclass(frozen=True)
class PriorRequestSlot:
    slot_id: str
    kind: PriorRequestSlotKind
    text: str
    resolved_value_text: str
    field_label_text: str = ""
    value_meaning_hint: str = ""

    def __post_init__(self) -> None:
        if not self.slot_id or not self.text or not self.resolved_value_text:
            raise ValueError("prior request slot requires identity and resolved text")

    def to_request_shape(self) -> dict[str, str]:
        payload = {
            "slot_id": self.slot_id,
            "slot_kind": self.kind.value,
            "text": self.text,
            "resolved_value_text": self.resolved_value_text,
        }
        if self.field_label_text:
            payload["field_label_text"] = self.field_label_text
        if self.value_meaning_hint:
            payload["value_meaning_hint"] = self.value_meaning_hint
        return payload


@dataclass(frozen=True)
class PriorEntityIdentityBinding:
    identity_type: str
    display: str
    canonical_values: dict[str, str]
    source_lineage: tuple[str, ...]
    kind: PriorRequestSlotKind = field(
        default=PriorRequestSlotKind.ENTITY_IDENTITY,
        init=False,
    )

    def __post_init__(self) -> None:
        if not self.identity_type or not self.canonical_values:
            raise ValueError("entity slot binding requires canonical identity")
        _require_source_lineage(self.source_lineage)

    def to_payload(self) -> dict[str, object]:
        return {
            "value_kind": self.kind.value,
            "source_lineage": list(self.source_lineage),
            "display": self.display,
            "identity_type": self.identity_type,
            "canonical_values": dict(self.canonical_values),
        }


@dataclass(frozen=True)
class PriorTimeScopeBinding:
    value: str
    display: str
    resolved_start: str
    resolved_end: str
    granularity: str
    source_lineage: tuple[str, ...]
    kind: PriorRequestSlotKind = field(
        default=PriorRequestSlotKind.TIME_SCOPE,
        init=False,
    )

    def __post_init__(self) -> None:
        if not all(
            (
                self.value,
                self.display,
                self.resolved_start,
                self.resolved_end,
                self.granularity,
            )
        ):
            raise ValueError("time slot binding requires a resolved interval")
        _require_source_lineage(self.source_lineage)

    def to_payload(self) -> dict[str, object]:
        return {
            "value_kind": self.kind.value,
            "source_lineage": list(self.source_lineage),
            "value": self.value,
            "display": self.display,
            "resolved_start": self.resolved_start,
            "resolved_end": self.resolved_end,
            "granularity": self.granularity,
        }


@dataclass(frozen=True)
class PriorLimitBinding:
    value: int
    display: str
    source_lineage: tuple[str, ...]
    kind: PriorRequestSlotKind = field(
        default=PriorRequestSlotKind.LIMIT,
        init=False,
    )

    def __post_init__(self) -> None:
        if type(self.value) is not int or self.value < 1:
            raise ValueError("limit slot binding requires a positive integer")
        _require_source_lineage(self.source_lineage)

    def to_payload(self) -> dict[str, object]:
        return {
            "value_kind": self.kind.value,
            "source_lineage": list(self.source_lineage),
            "value": self.value,
            "display": self.display,
        }


PriorRequestSlotBinding: TypeAlias = (
    PriorEntityIdentityBinding | PriorTimeScopeBinding | PriorLimitBinding
)


@dataclass(frozen=True)
class PriorRequestBoundSlot:
    slot_id: str
    binding: PriorRequestSlotBinding

    def __post_init__(self) -> None:
        if not self.slot_id:
            raise ValueError("bound prior request slot requires identity")


@dataclass(frozen=True)
class PriorRequestMemory:
    memory_id: str
    artifact_id: str
    request_id: str
    answer_fact: str
    answer_shape: PriorRequestAnswerShape | None
    output_frames: tuple[PriorRequestOutput, ...]
    run_id: str = ""
    program_request_ids: tuple[str, ...] = ()
    answer_subject_text: str = ""
    slots: tuple[PriorRequestSlot, ...] = ()
    semantic_parts: tuple[PriorRequestSemanticPart, ...] = ()
    bound_slots: tuple[PriorRequestBoundSlot, ...] = ()

    def __post_init__(self) -> None:
        if not all(
            (
                self.memory_id,
                self.artifact_id,
                self.request_id,
                self.answer_fact,
            )
        ):
            raise ValueError("prior request memory requires stable identity")
        if not self.output_frames:
            raise ValueError("prior request memory requires answer outputs")
        if len(set(self.program_request_ids)) != len(self.program_request_ids):
            raise ValueError("prior request memory contains duplicate program requests")
        if self.program_request_ids and self.request_id not in self.program_request_ids:
            raise ValueError("prior request is not part of its declared program")
        slots_by_id = {slot.slot_id: slot for slot in self.slots}
        if len(slots_by_id) != len(self.slots):
            raise ValueError("prior request memory contains duplicate slots")
        bound_ids = tuple(bound.slot_id for bound in self.bound_slots)
        if len(set(bound_ids)) != len(bound_ids):
            raise ValueError("prior request memory contains duplicate bindings")
        for bound in self.bound_slots:
            slot = slots_by_id.get(bound.slot_id)
            if slot is None or slot.kind is not bound.binding.kind:
                raise ValueError("prior request binding does not match its slot")

    def slot(self, slot_id: str) -> PriorRequestSlot | None:
        return next((slot for slot in self.slots if slot.slot_id == slot_id), None)

    def binding(self, slot_id: str) -> PriorRequestSlotBinding | None:
        return next(
            (
                bound.binding
                for bound in self.bound_slots
                if bound.slot_id == slot_id
            ),
            None,
        )

    def request_shape_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "answer_fact_template": self.answer_fact,
            "answer_outputs": tuple(
                frame.to_request_shape() for frame in self.output_frames
            ),
            "slots": tuple(slot.to_request_shape() for slot in self.slots),
            "semantic_parts": tuple(
                part.to_request_shape() for part in self.semantic_parts
            ),
        }
        if self.answer_subject_text:
            payload["answer_subject"] = self.answer_subject_text
        if self.answer_shape is not None:
            payload["answer_shape"] = self.answer_shape.to_request_shape()
        return payload

    def slot_bindings_payload(self) -> dict[str, dict[str, object]]:
        return {
            bound.slot_id: bound.binding.to_payload()
            for bound in self.bound_slots
        }

    @property
    def source_lineage(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                memory_id
                for source_lineage in (
                    *(frame.source_lineage for frame in self.output_frames),
                    *(bound.binding.source_lineage for bound in self.bound_slots),
                )
                for memory_id in source_lineage
            )
        )


def prior_requests_from_artifact(
    artifact: FactArtifact,
) -> tuple[PriorRequestMemory, ...]:
    """Parse the persisted question contract and bind it to artifact addresses."""

    if artifact.outcome is not FactOutcome.ANSWERED:
        return ()
    raw_contract = artifact.provenance.get("question_contract")
    if raw_contract is None:
        return ()
    contract = _mapping(raw_contract, path="question_contract")
    slots_by_id = _question_input_slots(
        _list_field(contract, "question_inputs", path="question_contract")
    )
    raw_requests = _list_field(
        contract,
        "answer_requests",
        path="question_contract",
    )
    if not raw_requests:
        raise ValueError("question_contract.answer_requests must not be empty")
    lineage_by_output_id = _output_source_lineage_by_id(artifact)
    requests: list[PriorRequestMemory] = []
    request_ids: set[str] = set()
    for index, raw_request in enumerate(raw_requests):
        path = f"question_contract.answer_requests[{index}]"
        request = _mapping(raw_request, path=path)
        request_id = _text_field(request, "id", path=path)
        if request_id in request_ids:
            raise ValueError("question_contract contains duplicate answer request ids")
        request_ids.add(request_id)
        slots = _request_slots(
            request,
            slots_by_id=slots_by_id,
            path=path,
        )
        requests.append(
            PriorRequestMemory(
                memory_id=f"{artifact.artifact_id}.prior_request.{request_id}",
                artifact_id=artifact.artifact_id,
                request_id=request_id,
                answer_fact=_text_field(request, "answer_fact", path=path),
                answer_shape=_request_answer_shape(request, path=path),
                run_id=str(artifact.provenance.get("runId") or "").strip(),
                program_request_ids=tuple(
                    str(item).strip()
                    for item in artifact.provenance.get("programRequestedFactIds") or ()
                    if str(item).strip()
                ),
                answer_subject_text=_answer_subject_text(request, path=path),
                output_frames=_request_outputs(
                    request,
                    lineage_by_output_id=lineage_by_output_id,
                    path=path,
                ),
                slots=slots,
                semantic_parts=_request_semantic_parts(request, path=path),
                bound_slots=_prior_slot_bindings(
                    artifact,
                    slots_by_id={slot.slot_id: slot for slot in slots},
                ),
            )
        )
    return tuple(requests)


def _question_input_slots(
    raw_inputs: list[object],
) -> dict[str, PriorRequestSlot | None]:
    slots: dict[str, PriorRequestSlot | None] = {}
    for index, raw_input in enumerate(raw_inputs):
        path = f"question_contract.question_inputs[{index}]"
        item = _mapping(raw_input, path=path)
        input_id = _text_field(item, "id", path=path)
        if input_id in slots:
            raise ValueError("question_contract contains duplicate question input ids")
        kind = KnownInputKind(_text_field(item, "kind", path=path))
        if kind is KnownInputKind.ROW_SET_REFERENCE:
            _text_field(item, "text", path=path)
            _text_field(item, "resolved_input_ref", path=path)
            slots[input_id] = None
            continue
        role = LiteralInputRole(_text_field(item, "role", path=path))
        slots[input_id] = PriorRequestSlot(
            slot_id=input_id,
            kind=PriorRequestSlotKind(literal_role_part_kind(role)),
            text=_text_field(item, "text", path=path),
            resolved_value_text=_text_field(
                item,
                "resolved_value_text",
                path=path,
            ),
            field_label_text=_optional_text_field(item, "field_label_text", path=path),
            value_meaning_hint=_optional_text_field(
                item,
                "value_meaning_hint",
                path=path,
            ),
        )
    return slots


def _request_slots(
    request: dict[str, object],
    *,
    slots_by_id: dict[str, PriorRequestSlot | None],
    path: str,
) -> tuple[PriorRequestSlot, ...]:
    raw_refs = _list_field(request, "used_question_inputs", path=path)
    refs: list[str] = []
    for index, raw_ref in enumerate(raw_refs):
        ref = _text(raw_ref, path=f"{path}.used_question_inputs[{index}]")
        if ref in refs:
            raise ValueError(f"{path}.used_question_inputs contains duplicate refs")
        if ref not in slots_by_id:
            raise ValueError(f"{path}.used_question_inputs references unknown input")
        refs.append(ref)
    return tuple(
        slot
        for ref in refs
        if (slot := slots_by_id[ref]) is not None
    )


def _request_outputs(
    request: dict[str, object],
    *,
    lineage_by_output_id: dict[str, tuple[str, ...]],
    path: str,
) -> tuple[PriorRequestOutput, ...]:
    raw_outputs = _list_field(request, "answer_outputs", path=path)
    if not raw_outputs:
        raise ValueError(f"{path}.answer_outputs must not be empty")
    outputs: list[PriorRequestOutput] = []
    output_ids: set[str] = set()
    for index, raw_output in enumerate(raw_outputs):
        output_path = f"{path}.answer_outputs[{index}]"
        item = _mapping(raw_output, path=output_path)
        output_id = _text_field(item, "id", path=output_path)
        if output_id in output_ids:
            raise ValueError(f"{path}.answer_outputs contains duplicate ids")
        output_ids.add(output_id)
        description = _text_field(item, "description", path=output_path)
        outputs.append(
            PriorRequestOutput(
                output_id=output_id,
                description=description,
                role=_optional_text_field(item, "role", path=output_path)
                or "ANSWER_VALUE",
                source_lineage=lineage_by_output_id.get(output_id, ()),
            )
        )
    return tuple(outputs)


def _answer_subject_text(request: dict[str, object], *, path: str) -> str:
    raw_subject = request.get("answer_subject")
    if raw_subject is None:
        return ""
    subject = _mapping(raw_subject, path=f"{path}.answer_subject")
    return _text_field(subject, "subject_text", path=f"{path}.answer_subject")


def _request_answer_shape(
    request: dict[str, object],
    *,
    path: str,
) -> PriorRequestAnswerShape | None:
    raw_expression = request.get("answer_expression")
    if raw_expression is None:
        return None
    expression = _mapping(raw_expression, path=f"{path}.answer_expression")
    outputs = _list_field(request, "answer_outputs", path=path)
    return PriorRequestAnswerShape(
        expression_family=_text_field(
            expression,
            "family",
            path=f"{path}.answer_expression",
        ),
        output_roles=tuple(
            _optional_text_field(
                _mapping(output, path=f"{path}.answer_outputs[{index}]"),
                "role",
                path=f"{path}.answer_outputs[{index}]",
            )
            or "ANSWER_VALUE"
            for index, output in enumerate(outputs)
        ),
    )


def _request_semantic_parts(
    request: dict[str, object],
    *,
    path: str,
) -> tuple[PriorRequestSemanticPart, ...]:
    return (
        *_population_constraint_parts(request, path=path),
        *_grouping_parts(request, path=path),
    )


def _population_constraint_parts(
    request: dict[str, object],
    *,
    path: str,
) -> tuple[PriorRequestSemanticPart, ...]:
    raw_population = request.get("answer_population")
    if raw_population is None:
        return ()
    population = _mapping(raw_population, path=f"{path}.answer_population")
    tests = _list_field(
        population,
        "membership_tests",
        path=f"{path}.answer_population",
    )
    parts: list[PriorRequestSemanticPart] = []
    for index, raw_test in enumerate(tests):
        test_path = f"{path}.answer_population.membership_tests[{index}]"
        test = _mapping(raw_test, path=test_path)
        test_kind = _text_field(test, "kind", path=test_path)
        owned_input_refs = _list_field(
            test,
            "owned_question_input_refs",
            path=test_path,
        )
        if test_kind == "SUBJECT_IDENTITY" or owned_input_refs:
            continue
        parts.append(
            PriorRequestSemanticPart(
                kind=PriorRequestSemanticPartKind.POPULATION_CONSTRAINT,
                role=test_kind.casefold(),
                text=_text_field(test, "test_question", path=test_path),
            )
        )
    return tuple(parts)


def _grouping_parts(
    request: dict[str, object],
    *,
    path: str,
) -> tuple[PriorRequestSemanticPart, ...]:
    raw_expression = request.get("answer_expression")
    if raw_expression is None:
        return ()
    expression = _mapping(raw_expression, path=f"{path}.answer_expression")
    raw_group_key = expression.get("group_key")
    if raw_group_key is None:
        return ()
    group_key = _mapping(raw_group_key, path=f"{path}.answer_expression.group_key")
    return (
        PriorRequestSemanticPart(
            kind=PriorRequestSemanticPartKind.GROUPING,
            role=_text_field(
                group_key,
                "domain",
                path=f"{path}.answer_expression.group_key",
            ).casefold(),
            text=_text_field(
                group_key,
                "description",
                path=f"{path}.answer_expression.group_key",
            ),
        ),
    )


def _output_source_lineage_by_id(
    artifact: FactArtifact,
) -> dict[str, tuple[str, ...]]:
    relation_addresses = {
        address.address
        for address in artifact.addresses
        if address.kind is FactAddressKind.RELATION
    }
    output: dict[str, list[str]] = {}
    for address in artifact.addresses:
        if address.kind is FactAddressKind.VALUE:
            _append_output_lineage(
                output,
                output_ids=_answer_output_ids(address.scalar_value),
                memory_id=f"{artifact.artifact_id}.{address.address}",
            )
            continue
        if address.kind is not FactAddressKind.ROW:
            continue
        if address.source_relation not in relation_addresses:
            continue
        for field_id, raw_value in address.values.items():
            value = _mapping(
                raw_value,
                path=f"address {address.address}.values.{field_id}",
            )
            _append_output_lineage(
                output,
                output_ids=_answer_output_ids(value),
                memory_id=f"{artifact.artifact_id}.{address.source_relation}",
            )
    return {key: tuple(dict.fromkeys(values)) for key, values in output.items()}


def _answer_output_ids(value: dict[str, object]) -> tuple[str, ...]:
    raw_ids = value.get("answer_output_ids")
    if raw_ids is None:
        return ()
    if not isinstance(raw_ids, list):
        raise ValueError("answer_output_ids must be a list")
    return tuple(
        _text(raw_id, path=f"answer_output_ids[{index}]")
        for index, raw_id in enumerate(raw_ids)
    )


def _append_output_lineage(
    output: dict[str, list[str]],
    *,
    output_ids: tuple[str, ...],
    memory_id: str,
) -> None:
    for output_id in output_ids:
        output.setdefault(output_id, []).append(memory_id)


def _prior_slot_bindings(
    artifact: FactArtifact,
    *,
    slots_by_id: dict[str, PriorRequestSlot],
) -> tuple[PriorRequestBoundSlot, ...]:
    output: dict[str, PriorRequestBoundSlot] = {}
    for address in artifact.addresses:
        slot_ids = tuple(
            input_id
            for input_id in _address_known_input_ids(address)
            if input_id in slots_by_id
        )
        if address.kind is FactAddressKind.ENTITY:
            for slot_id in slot_ids:
                slot = slots_by_id[slot_id]
                if slot.kind is not PriorRequestSlotKind.ENTITY_IDENTITY:
                    raise ValueError("prior request slot binding kind does not match")
                _add_prior_slot_binding(
                    output,
                    PriorRequestBoundSlot(
                        slot_id=slot_id,
                        binding=PriorEntityIdentityBinding(
                            identity_type=address.resource,
                            display=address.reference_text,
                            canonical_values=dict(address.identity),
                            source_lineage=(
                                f"{artifact.artifact_id}.{address.address}",
                            ),
                        ),
                    ),
                )
            continue
        if address.kind is not FactAddressKind.VALUE:
            continue
        for slot_id in slot_ids:
            _add_prior_slot_binding(
                output,
                PriorRequestBoundSlot(
                    slot_id=slot_id,
                    binding=_prior_value_slot_binding(
                        artifact=artifact,
                        address=address,
                        slot=slots_by_id[slot_id],
                    ),
                ),
            )
    return tuple(output.values())


def _prior_value_slot_binding(
    *,
    artifact: FactArtifact,
    address: FactAddress,
    slot: PriorRequestSlot,
) -> PriorTimeScopeBinding | PriorLimitBinding:
    value = address.scalar_value
    source_lineage = (f"{artifact.artifact_id}.{address.address}",)
    if slot.kind is PriorRequestSlotKind.TIME_SCOPE:
        if value.get("type") != "time_scope":
            raise ValueError("time slot requires a time-scope binding")
        expression = _text(
            value.get("expression") or value.get("value"),
            path=f"address {address.address}.expression",
        )
        return PriorTimeScopeBinding(
            value=_text(
                value.get("value") or expression,
                path=f"address {address.address}.value",
            ),
            display=expression,
            resolved_start=_text(
                value.get("resolvedStart"),
                path=f"address {address.address}.resolvedStart",
            ),
            resolved_end=_text(
                value.get("resolvedEnd"),
                path=f"address {address.address}.resolvedEnd",
            ),
            granularity=_text(
                value.get("granularity"),
                path=f"address {address.address}.granularity",
            ),
            source_lineage=source_lineage,
        )
    if slot.kind is PriorRequestSlotKind.LIMIT:
        raw_limit = value.get("value")
        if type(raw_limit) is not int or raw_limit < 1:
            raise ValueError("limit slot requires a positive integer binding")
        return PriorLimitBinding(
            value=raw_limit,
            display=address.display or str(raw_limit),
            source_lineage=source_lineage,
        )
    raise ValueError("entity slot requires an entity-identity binding")


def _add_prior_slot_binding(
    output: dict[str, PriorRequestBoundSlot],
    bound_slot: PriorRequestBoundSlot,
) -> None:
    if bound_slot.slot_id in output:
        raise ValueError("prior request slot has multiple persisted bindings")
    output[bound_slot.slot_id] = bound_slot


def _address_known_input_ids(address: FactAddress) -> tuple[str, ...]:
    if address.evidence is None:
        return ()
    prefix = "known_input:"
    return tuple(
        proof_ref.removeprefix(prefix).strip()
        for proof_ref in address.evidence.step_ids
        if proof_ref.startswith(prefix) and proof_ref.removeprefix(prefix).strip()
    )


def _require_source_lineage(source_lineage: tuple[str, ...]) -> None:
    if not source_lineage or any(not ref for ref in source_lineage):
        raise ValueError("prior request binding requires source lineage")


def _mapping(value: object, *, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _list_field(
    payload: dict[str, object],
    field_name: str,
    *,
    path: str,
) -> list[object]:
    value = payload.get(field_name)
    if not isinstance(value, list):
        raise ValueError(f"{path}.{field_name} must be a list")
    return value


def _text_field(
    payload: dict[str, object],
    field_name: str,
    *,
    path: str,
) -> str:
    return _text(payload.get(field_name), path=f"{path}.{field_name}")


def _optional_text_field(
    payload: dict[str, object],
    field_name: str,
    *,
    path: str,
) -> str:
    value = payload.get(field_name)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{path}.{field_name} must be text")
    return value.strip()


def _text(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be non-empty text")
    return value.strip()
