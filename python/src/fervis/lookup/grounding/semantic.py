"""Typed grounding ledger and identity-task partition for semantic inputs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from typing import Mapping

from fervis.lookup.answer_program.values import FactValue, LiteralType
from fervis.lookup.canonical_data import EntityKeyValue
from fervis.lookup.grounding.identity import IdentifierKind, InputBindingOption, LookupTextResolutionDecision
from fervis.lookup.source_reads.access_model import ReadAccessCatalog
from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.question_contract import FactLocalRef, InputTerm, InputUseSite
from fervis.lookup.semantic_types import (
    BooleanType,
    CollectionType,
    DecimalType,
    IntegerType,
    NumericType,
    input_operand_matches_value_type,
    PercentageMeasure,
    TemporalScopeType,
    TextType,
    ValueType,
    SourceOrigin,
)
from fervis.types.enums import StrEnum


@dataclass(frozen=True)
class CanonicalIdentityOption:
    canonical_option_id: str
    identity_ref: str
    resolver_route_refs: tuple[str, ...]


@dataclass(frozen=True)
class IdentityGroundingTask:
    task_ref: str
    input_ref: str
    use_refs: tuple[str, ...]
    expected_set_ref: FactLocalRef | None
    options: tuple[InputBindingOption, ...]


@dataclass(frozen=True)
class ReferenceGroundingTask:
    task_ref: str
    input_ref: str
    use_refs: tuple[str, ...]
    operand_meaning: str
    denoted_instance_kind: str
    reference_fact_ref: FactLocalRef | None
    expected_set_ref: FactLocalRef | None
    options: tuple[InputBindingOption, ...]


@dataclass(frozen=True)
class IdentityResolverRoute:
    route_ref: str
    option: InputBindingOption
    compatibility: CompatibleIdentityRoute


@dataclass(frozen=True)
class CompatibleIdentityRoute:
    option_id: str
    identifier_kind: IdentifierKind
    lookup_request_param_refs: tuple[str, ...]
    returned_identity_verification_field_paths: tuple[str, ...]
    resolution_method: LookupTextResolutionDecision = LookupTextResolutionDecision.CAN_RESOLVE_LOOKUP_TEXT

    def __post_init__(self) -> None:
        if self.resolution_method not in {LookupTextResolutionDecision.CAN_RESOLVE_LOOKUP_TEXT,LookupTextResolutionDecision.ENUMERATE_COMPLETE_SOURCE}:
            raise ValueError('identity route has an invalid resolution method')
        if bool(self.lookup_request_param_refs) != (self.resolution_method is LookupTextResolutionDecision.CAN_RESOLVE_LOOKUP_TEXT):
            raise ValueError('identity route request parameters disagree with its method')
        if not self.option_id:
            raise ValueError("compatible identity route is incomplete")
        if not self.returned_identity_verification_field_paths:
            raise ValueError("compatible identity route lacks identity verification")
        if len(self.lookup_request_param_refs) != len(
            set(self.lookup_request_param_refs)
        ):
            raise ValueError("compatible identity route repeats a request parameter")
        if len(self.returned_identity_verification_field_paths) != len(
            set(self.returned_identity_verification_field_paths)
        ):
            raise ValueError("compatible identity route repeats a verification field")


@dataclass(frozen=True)
class IdentityResolutionTask:
    task_ref: str
    input_ref: str
    use_refs: tuple[str, ...]
    expected_set_ref: FactLocalRef | None
    canonical_options: tuple[CanonicalIdentityOption, ...]
    resolver_routes: tuple[IdentityResolverRoute, ...]


@dataclass(frozen=True)
class CanonicalInputValue:
    canonical_value_id: str
    input_ref: str
    use_refs: tuple[str, ...]
    typed_value: FactValue
    certification_refs: tuple[str, ...]


@dataclass(frozen=True)
class SemanticTimeGroundingTask:
    task_ref: str
    input_ref: str
    use_refs: tuple[str, ...]
    expression: str
    operand_meaning: str


class IdentityExecutionFailureReason(StrEnum):
    NOT_FOUND = "NOT_FOUND"
    AMBIGUOUS_RESULT = "AMBIGUOUS_RESULT"
    INVALID_RESOLVER_RESULT = "INVALID_RESOLVER_RESULT"


@dataclass(frozen=True)
class ResolvedIdentity:
    task_ref: str
    input_ref: str
    use_refs: tuple[str, ...]
    canonical_option_id: str
    resolver_route_id: str
    canonical_value: CanonicalInputValue


@dataclass(frozen=True)
class IdentityExecutionCandidate:
    key: EntityKeyValue
    display_value: str
    matched_field_ref: str
    matched_field_path: str
    resolver_read_id: str


@dataclass(frozen=True)
class IdentityExecutionClarification:
    task_ref: str
    input_ref: str
    use_refs: tuple[str, ...]
    reason: IdentityExecutionFailureReason
    evidence_refs: tuple[str, ...]
    candidates: tuple[IdentityExecutionCandidate, ...] = ()


@dataclass(frozen=True)
class SemanticGroundingResult:
    identity_tasks: tuple[IdentityResolutionTask, ...]
    canonical_values: tuple[CanonicalInputValue, ...]


@dataclass(frozen=True)
class GroundingPartition:
    input_ref: str
    use_refs: tuple[str, ...]
    expected_value_type: ValueType
    expected_set_ref: FactLocalRef | None
    operand_meaning: str
    reference_fact_ref: FactLocalRef | None = None


@dataclass(frozen=True)
class SemanticGroundingRequest:
    question: str
    inputs: tuple[InputTerm, ...]
    tasks: tuple[ReferenceGroundingTask, ...]
    set_origins: Mapping[FactLocalRef, SourceOrigin]
    resolver_catalog: RelationCatalog
    time_tasks: tuple[SemanticTimeGroundingTask, ...] = ()
    runtime_date: str = ""
    timezone: str = "timezone.utc"
    read_access: ReadAccessCatalog = ReadAccessCatalog()

    def __post_init__(self) -> None:
        input_ids = tuple(item.id for item in self.inputs)
        if len(input_ids) != len(set(input_ids)):
            raise ValueError("semantic grounding repeats an input")
        task_refs = tuple(item.task_ref for item in self.tasks)
        if len(task_refs) != len(set(task_refs)):
            raise ValueError("semantic grounding repeats an identity task")
        if any(task.input_ref not in input_ids for task in self.tasks):
            raise ValueError("semantic grounding task references an unknown input")
        if any(
            task.expected_set_ref is not None
            and task.expected_set_ref not in self.set_origins
            for task in self.tasks
        ):
            raise ValueError("semantic grounding task lacks expected-set context")
        time_task_refs = tuple(item.task_ref for item in self.time_tasks)
        if len(time_task_refs) != len(set(time_task_refs)):
            raise ValueError("semantic grounding repeats a time task")
        if any(task.input_ref not in input_ids for task in self.time_tasks):
            raise ValueError("semantic time task references an unknown input")
        if self.time_tasks and not self.runtime_date:
            raise ValueError("semantic time grounding requires a runtime date")

    def input(self, input_ref: str) -> InputTerm:
        return next(item for item in self.inputs if item.id == input_ref)


def grounding_partitions(
    input_use_sites: tuple[InputUseSite, ...],
) -> tuple[GroundingPartition, ...]:
    grouped: dict[
        tuple[str, ValueType, FactLocalRef | None, FactLocalRef | None, str], list[str]
    ] = {}
    for use in input_use_sites:
        key = (
            use.input_ref,
            use.expected_value_type,
            use.identity_set_ref,
            use.reference_fact_ref,
            use.operand_meaning,
        )
        grouped.setdefault(key, []).append(use.use_ref)
    return tuple(
        GroundingPartition(
            input_ref=input_ref,
            use_refs=tuple(use_refs),
            expected_value_type=value_type,
            expected_set_ref=set_ref,
            operand_meaning=operand_meaning,
            reference_fact_ref=reference_fact_ref,
        )
        for (
            input_ref,
            value_type,
            set_ref,
            reference_fact_ref,
            operand_meaning,
        ), use_refs in grouped.items()
    )


def reference_grounding_tasks(
    partitions: tuple[GroundingPartition, ...],
    *,
    resolver_options_by_use_ref: Mapping[str, tuple[InputBindingOption, ...]],
    denoted_instance_kinds_by_input_ref: Mapping[str, str],
) -> tuple[ReferenceGroundingTask, ...]:
    tasks: list[ReferenceGroundingTask] = []
    for partition in partitions:
        if partition.reference_fact_ref is None and partition.expected_set_ref is None:
            continue
        options = tuple(
            {
                option.id: option
                for use_ref in partition.use_refs
                for option in resolver_options_by_use_ref.get(use_ref, ())
            }.values()
        )
        tasks.append(
            ReferenceGroundingTask(
                task_ref=f"{partition.use_refs[0]}:reference_grounding",
                input_ref=partition.input_ref,
                use_refs=partition.use_refs,
                operand_meaning=partition.operand_meaning,
                denoted_instance_kind=denoted_instance_kinds_by_input_ref[
                    partition.input_ref
                ],
                reference_fact_ref=partition.reference_fact_ref,
                expected_set_ref=partition.expected_set_ref,
                options=options,
            )
        )
    return tuple(tasks)


def time_grounding_tasks(
    partitions: tuple[GroundingPartition, ...],
    *,
    inputs: Mapping[str, InputTerm],
) -> tuple[SemanticTimeGroundingTask, ...]:
    return tuple(
        SemanticTimeGroundingTask(
            task_ref=f"{partition.use_refs[0]}:time_resolution",
            input_ref=partition.input_ref,
            use_refs=partition.use_refs,
            expression=_scalar_operand(inputs[partition.input_ref]),
            operand_meaning=partition.operand_meaning,
        )
        for partition in partitions
        if partition.expected_set_ref is None
        and isinstance(partition.expected_value_type, TemporalScopeType)
    )


def _scalar_operand(input_term: InputTerm) -> str:
    if not isinstance(input_term.operand, str):
        raise ValueError("scalar grounding task received a collection input")
    return input_term.operand


def identity_resolution_tasks(
    tasks: tuple[IdentityGroundingTask, ...],
    *,
    compatible_bindings_by_task_ref: Mapping[str, tuple[CompatibleIdentityRoute, ...]],
) -> tuple[IdentityResolutionTask, ...]:
    output: list[IdentityResolutionTask] = []
    for task in tasks:
        options = {option.id: option for option in task.options}
        bindings = compatible_bindings_by_task_ref.get(task.task_ref, ())
        routes: list[IdentityResolverRoute] = []
        option_routes: dict[tuple[str, str, tuple[str, ...]], list[str]] = {}
        for binding in bindings:
            option = options.get(binding.option_id)
            if option is None:
                raise ValueError("identity compatibility references an unknown route")
            candidate = option.candidate
            identity = (
                candidate.entity_kind,
                candidate.key_id,
                tuple(item.component_id for item in candidate.key_components),
            )
            option_routes.setdefault(identity, []).append(option.id)
            routes.append(
                IdentityResolverRoute(
                    route_ref=option.id,
                    option=option,
                    compatibility=binding,
                )
            )
        canonical_options = tuple(
            CanonicalIdentityOption(
                canonical_option_id=f"{task.task_ref}:canonical_option:{index}",
                identity_ref=f"{entity_kind}:{key_id}",
                resolver_route_refs=tuple(route_refs),
            )
            for index, ((entity_kind, key_id, _), route_refs) in enumerate(
                option_routes.items(), start=1
            )
        )
        output.append(
            IdentityResolutionTask(
                task_ref=task.task_ref,
                input_ref=task.input_ref,
                use_refs=task.use_refs,
                expected_set_ref=task.expected_set_ref,
                canonical_options=canonical_options,
                resolver_routes=tuple(routes),
            )
        )
    return tuple(output)


def deterministic_scalar_values(
    partitions: tuple[GroundingPartition, ...],
    *,
    inputs: Mapping[str, InputTerm],
) -> tuple[CanonicalInputValue, ...]:
    output: list[CanonicalInputValue] = []
    for index, partition in enumerate(partitions, start=1):
        if partition.expected_set_ref is not None:
            continue
        if partition.reference_fact_ref is not None:
            continue
        input_term = inputs[partition.input_ref]
        fact_value = _deterministic_scalar_value(
            input_term,
            value_type=partition.expected_value_type,
            value_id=f"canonical_input_value_{index}",
        )
        if fact_value is None:
            continue
        certification_refs = (f"question_input:{input_term.id}",)
        output.append(
            CanonicalInputValue(
                canonical_value_id=fact_value.id,
                input_ref=input_term.id,
                use_refs=partition.use_refs,
                typed_value=fact_value,
                certification_refs=certification_refs,
            )
        )
    return tuple(output)


def build_canonical_input_ledger(
    values: tuple[CanonicalInputValue, ...],
    *,
    required_use_refs: tuple[str, ...],
) -> tuple[CanonicalInputValue, ...]:
    actual = tuple(use_ref for value in values for use_ref in value.use_refs)
    if len(actual) != len(set(actual)):
        raise ValueError("canonical input ledger repeats an input use")
    if set(actual) != set(required_use_refs):
        raise ValueError("canonical input ledger does not cover every input use")
    if any(not value.certification_refs for value in values):
        raise ValueError("canonical input value lacks certification")

    ledger: dict[str, CanonicalInputValue] = {}
    for value in values:
        previous = ledger.get(value.canonical_value_id)
        if previous is None:
            ledger[value.canonical_value_id] = value
            continue
        if (previous.input_ref != value.input_ref
                or previous.typed_value.known_input_id != value.typed_value.known_input_id
                or not previous.typed_value.has_same_value_as(value.typed_value)):
            raise ValueError("canonical value ID has conflicting input values")
        typed = previous.typed_value
        other = value.typed_value
        ledger[value.canonical_value_id] = replace(
            previous,
            use_refs=(*previous.use_refs, *value.use_refs),
            certification_refs=tuple(dict.fromkeys((*previous.certification_refs, *value.certification_refs))),
            typed_value=replace(
                typed,
                identity_evidence=tuple(dict.fromkeys((*typed.identity_evidence, *other.identity_evidence))),
                proof_refs=tuple(dict.fromkeys((*typed.proof_refs, *other.proof_refs))),
                source_refs=tuple(dict.fromkeys((*typed.source_refs, *other.source_refs))),
                dependencies=tuple(dict.fromkeys((*typed.dependencies, *other.dependencies))),
                applies_to_requested_fact_ids=tuple(dict.fromkeys((*typed.applies_to_requested_fact_ids, *other.applies_to_requested_fact_ids))),
            ),
        )
    return tuple(ledger.values())


def _deterministic_scalar_value(
    input_term: InputTerm,
    *,
    value_type: ValueType,
    value_id: str,
) -> FactValue | None:
    if isinstance(value_type, (CollectionType, TemporalScopeType)):
        return None
    if not isinstance(input_term.operand, str):
        raise ValueError("scalar grounding received a collection operand")
    text = input_term.operand.strip()
    literal_type: LiteralType
    value: str
    if isinstance(value_type, BooleanType):
        normalized = text.casefold()
        if normalized not in {"true", "false"}:
            return None
        literal_type = LiteralType.BOOLEAN
        value = normalized
    elif isinstance(value_type, IntegerType):
        try:
            value = str(int(text))
        except ValueError:
            return None
        literal_type = LiteralType.NUMBER
    elif isinstance(value_type, (DecimalType, NumericType)):
        if not input_operand_matches_value_type(text, value_type):
            return None
        normalized = text.removesuffix("%").strip()
        try:
            number = Decimal(normalized)
        except InvalidOperation:
            return None
        if text.endswith("%") or (isinstance(value_type, DecimalType) and isinstance(value_type.measure, PercentageMeasure)):
            number /= Decimal(100)
        literal_type = LiteralType.NUMBER
        value = format(number, "f")
    elif isinstance(value_type, TextType):
        literal_type = LiteralType.STRING
        value = text
    else:
        return None
    proof_ref = f"question_input:{input_term.id}"
    return FactValue.literal(
        id=value_id,
        known_input_id=input_term.id,
        literal_type=literal_type,
        value=value,
        label=input_term.operand,
        proof_refs=(proof_ref,),
    )


__all__ = tuple(name for name in globals() if not name.startswith("_"))
