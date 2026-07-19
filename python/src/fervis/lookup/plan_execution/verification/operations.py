"""Operation reference checks for fact-plan verification."""

from ._shared import (
    AggregateSpec,
    AggregationFunction,
    AnswerProgram,
    AntiJoinSpec,
    ComputeSpec,
    FieldBindingRole,
    FilterSpec,
    JoinSpec,
    Operation,
    ProjectSpec,
    ProjectToKeySpec,
    RankSpec,
    UniversalConditionSpec,
    VerificationError,
)
from .contract_types import RelationContract, _contract, _field_roles
from .scalars import _operation_scalar_inputs
from fervis.lookup.answer_program.operations import (
    Predicate,
    RelationRoleRef,
)
from fervis.lookup.answer_program.expressions import expression_references


def _verify_answer_uses_evidence_input(answer: AnswerProgram) -> None:
    if any(_operation_input_refs(operation) for operation in answer.operations):
        return
    if any(_operation_scalar_inputs(operation) for operation in answer.operations):
        return
    if any(_compute_uses_direct_value(operation) for operation in answer.operations):
        return
    raise VerificationError("answer plan requires evidence input")


def _compute_uses_direct_value(operation: Operation) -> bool:
    spec = operation.spec
    if not isinstance(spec, ComputeSpec):
        return False
    references = expression_references(spec.expression)
    return bool(references.parameters or references.constants)


def _verify_operation_references(answer: AnswerProgram) -> None:
    available = {item.id for item in answer.relations}
    operation_ids: set[str] = set()
    scalar_outputs: set[str] = _operation_field_outputs(answer.operations)
    for operation in answer.operations:
        if not operation.id:
            raise VerificationError("operation requires id")
        if operation.id in operation_ids:
            raise VerificationError(f"duplicate operation {operation.id}")
        operation_ids.add(operation.id)
        for ref in _operation_input_refs(operation):
            if ref not in available:
                raise VerificationError(
                    f"operation {operation.id} references unknown input"
                )
        if operation.output_relation:
            if operation.output_relation in available:
                raise VerificationError(
                    f"duplicate relation {operation.output_relation}"
                )
            available.add(operation.output_relation)
        if isinstance(operation.spec, ComputeSpec):
            output_scalar = operation.spec.output_scalar
            if (
                output_scalar in scalar_outputs
                or output_scalar in _operation_scalar_inputs(operation)
            ):
                raise VerificationError(f"duplicate scalar {output_scalar}")
            scalar_outputs.add(output_scalar)


def _operation_field_outputs(operations: tuple[Operation, ...]) -> set[str]:
    outputs: set[str] = set()
    for operation in operations:
        spec = operation.spec
        if isinstance(spec, AggregateSpec):
            outputs.update(spec.group_by)
            for aggregation in spec.aggregations:
                if aggregation.function == AggregationFunction.COUNT:
                    outputs.add(aggregation.output_field)
                elif aggregation.output_field:
                    outputs.add(aggregation.output_field)
        elif isinstance(spec, ProjectSpec):
            outputs.update(field.output or field.source for field in spec.fields)
        elif isinstance(spec, ProjectToKeySpec):
            outputs.update(spec.key_fields)
    return outputs


def _verify_compute_scalar_availability(answer: AnswerProgram) -> None:
    available_outputs = {
        f"parameter:{parameter.id}": parameter.id for parameter in answer.parameters
    }
    for operation in answer.operations:
        spec = operation.spec
        if not isinstance(spec, ComputeSpec):
            _require_available_scalar_inputs(operation, available_outputs)
            continue
        _require_available_compute_outputs(operation.id, spec, available_outputs)
        available_outputs[operation.id] = spec.output_scalar


def _require_available_compute_outputs(
    operation_id: str,
    spec: ComputeSpec,
    available_outputs: dict[str, str],
) -> None:
    references = expression_references(spec.expression).outputs
    if any(
        available_outputs.get(reference.node_id) != reference.output_id
        for reference in references
    ):
        raise VerificationError(
            f"operation {operation_id} references unbound scalar input"
        )


def _require_available_scalar_inputs(
    operation: Operation,
    available_outputs: dict[str, str],
) -> None:
    available_scalar_ids = set(available_outputs.values())
    if any(
        scalar_input not in available_scalar_ids
        for scalar_input in _operation_scalar_inputs(operation)
    ):
        raise VerificationError(
            f"operation {operation.id} references unbound scalar input"
        )


def _operation_input_refs(operation: Operation) -> tuple[str, ...]:
    return operation.input_relation_ids


def _verify_coverage_operation_relation_contracts(
    answer: AnswerProgram,
    *,
    relation_contracts: dict[str, RelationContract],
) -> None:
    for operation in answer.operations:
        spec = operation.spec
        if isinstance(spec, AntiJoinSpec):
            candidate = relation_contracts.get(spec.candidate.relation_id)
            observed = relation_contracts.get(spec.observed.relation_id)
            if candidate is not None:
                _verify_role_relation_fields(
                    contract=candidate,
                    fields=tuple(key.left for key in spec.join_keys),
                    expected_role=FieldBindingRole.IDENTITY,
                    role="anti_join.candidate",
                )
                _verify_role_relation_fields(
                    contract=candidate,
                    fields=tuple(field.source for field in spec.output_fields),
                    expected_role=FieldBindingRole.OUTPUT,
                    role="anti_join.candidate",
                )
                _verify_role_grain(
                    ref=spec.candidate,
                    contract=candidate,
                    fields=tuple(key.left for key in spec.join_keys),
                    role="anti_join.candidate",
                )
            if observed is not None:
                _verify_role_relation_fields(
                    contract=observed,
                    fields=tuple(key.right for key in spec.join_keys),
                    expected_role=FieldBindingRole.IDENTITY,
                    role="anti_join.observed",
                )
                _verify_role_grain(
                    ref=spec.observed,
                    contract=observed,
                    fields=tuple(key.right for key in spec.join_keys),
                    role="anti_join.observed",
                )
        elif isinstance(spec, UniversalConditionSpec):
            candidate = relation_contracts.get(spec.candidate_subject.relation_id)
            dimension = relation_contracts.get(spec.required_dimension.relation_id)
            observation = relation_contracts.get(spec.observation.relation_id)
            subject_fields = tuple(key.left for key in spec.subject_keys)
            observation_subject_fields = tuple(key.right for key in spec.subject_keys)
            dimension_fields = tuple(key.left for key in spec.dimension_keys)
            observation_dimension_fields = tuple(
                key.right for key in spec.dimension_keys
            )
            if candidate is not None:
                _verify_role_relation_fields(
                    contract=candidate,
                    fields=subject_fields,
                    expected_role=FieldBindingRole.IDENTITY,
                    role="universal_condition.candidate_subject",
                )
                _verify_role_relation_fields(
                    contract=candidate,
                    fields=tuple(field.source for field in spec.output_fields),
                    expected_role=FieldBindingRole.OUTPUT,
                    role="universal_condition.candidate_subject",
                )
                _verify_role_grain(
                    ref=spec.candidate_subject,
                    contract=candidate,
                    fields=subject_fields,
                    role="universal_condition.candidate_subject",
                )
            if dimension is not None:
                _verify_role_relation_fields(
                    contract=dimension,
                    fields=dimension_fields,
                    expected_role=FieldBindingRole.IDENTITY,
                    role="universal_condition.required_dimension",
                )
                _verify_role_grain(
                    ref=spec.required_dimension,
                    contract=dimension,
                    fields=dimension_fields,
                    role="universal_condition.required_dimension",
                )
            if observation is not None:
                _verify_role_relation_fields(
                    contract=observation,
                    fields=(
                        *observation_subject_fields,
                        *observation_dimension_fields,
                    ),
                    expected_role=FieldBindingRole.IDENTITY,
                    role="universal_condition.observation",
                )
                _verify_role_relation_fields(
                    contract=observation,
                    fields=tuple(
                        item.field_id
                        for item in expression_references(
                            spec.predicate.left,
                            *(
                                (spec.predicate.right,)
                                if spec.predicate.right is not None
                                else ()
                            ),
                        ).fields
                    ),
                    expected_role=FieldBindingRole.PREDICATE,
                    role="universal_condition.observation",
                )
                _verify_role_grain(
                    ref=spec.observation,
                    contract=observation,
                    fields=(
                        *observation_subject_fields,
                        *observation_dimension_fields,
                    ),
                    role="universal_condition.observation",
                )


def _verify_operation_field_references(
    answer: AnswerProgram,
    *,
    relation_contracts: dict[str, RelationContract],
) -> None:
    for operation in answer.operations:
        spec = operation.spec
        if isinstance(spec, FilterSpec):
            _verify_predicate_fields(
                contract=_contract(relation_contracts, spec.input_relation),
                predicate=spec.predicate,
                label="filter",
            )
        elif isinstance(spec, JoinSpec):
            left = _contract(relation_contracts, spec.left)
            right = _contract(relation_contracts, spec.right)
            for key in spec.join_keys:
                _field_roles(left, key.left, "join")
                _field_roles(right, key.right, "join")
        elif isinstance(spec, AggregateSpec):
            source = _contract(relation_contracts, spec.input_relation)
            for field in spec.group_by:
                _field_roles(source, field, "aggregate")
            for aggregation in spec.aggregations:
                if aggregation.function != AggregationFunction.COUNT:
                    _field_roles(source, aggregation.input_field, "aggregate")
        elif isinstance(spec, RankSpec):
            source = _contract(relation_contracts, spec.input_relation)
            for sort_key in (*spec.order_by, *spec.tie_breakers):
                _field_roles(source, sort_key.field, "rank")
        elif isinstance(spec, UniversalConditionSpec):
            _verify_predicate_fields(
                contract=_contract(
                    relation_contracts,
                    spec.observation.relation_id,
                ),
                predicate=spec.predicate,
                label="universal_condition",
            )


def _verify_predicate_fields(
    *,
    contract: RelationContract,
    predicate: Predicate,
    label: str,
) -> None:
    references = expression_references(
        predicate.left,
        *((predicate.right,) if predicate.right is not None else ()),
    )
    for field in references.fields:
        _field_roles(contract, field.field_id, label)


def _verify_role_relation_fields(
    *,
    contract: RelationContract,
    fields: tuple[str, ...],
    expected_role: FieldBindingRole,
    role: str,
) -> None:
    if not contract.fields:
        raise VerificationError(f"{role} requires relation field bindings")
    for field in fields:
        roles = contract.fields.get(field)
        if roles is None:
            raise VerificationError(f"{role} references unknown field")
        if expected_role not in roles:
            raise VerificationError(f"{role} field has wrong binding role")


def _verify_role_grain(
    *,
    ref: RelationRoleRef,
    contract: RelationContract,
    fields: tuple[str, ...],
    role: str,
) -> None:
    required_identity_fields = ref.required_identity_fields
    if contract.grain_keys != required_identity_fields:
        raise VerificationError(f"{role} requires exact relation grain")
    for field in fields:
        if field not in required_identity_fields:
            raise VerificationError(f"{role} requires grain obligation")
