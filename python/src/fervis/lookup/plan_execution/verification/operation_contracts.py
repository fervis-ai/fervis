"""Operation-specific relation contract projection."""

from typing_extensions import assert_never

from ._shared import (
    AggregateSpec,
    AggregationFunction,
    AntiJoinSpec,
    CrossJoinSpec,
    ComputeSpec,
    FieldBindingRole,
    FilterSpec,
    JoinSpec,
    Operation,
    ProjectToKeySpec,
    ProjectSpec,
    RankSpec,
    RoleExpandSpec,
    UnionSpec,
    UniversalConditionSpec,
)
from .contract_types import (
    PopulationCoverage,
    ProofLineage,
    RelationContract,
    _combined_entity_keys,
    _common_entity_keys,
    _contract,
    _copy_contract,
    _field_proof,
    _field_roles,
    _join_contract_grain,
    _project_contract_grain,
    _project_entity_keys,
    _union_field_proof,
    _union_field_roles,
)
from .execution_proof import ExecutionProofContext
from fervis.lookup.answer_program.operations import JoinKey, Predicate
from fervis.lookup.plan_execution.errors import VerificationError
from fervis.lookup.plan_execution.declared_values import declared_types_compatible


def _operation_relation_contract(
    operation: Operation,
    contracts: dict[str, RelationContract],
    *,
    proof_context: ExecutionProofContext,
) -> RelationContract:
    spec = operation.spec
    if isinstance(spec, FilterSpec):
        source = _contract(contracts, spec.input_relation)
        return _with_dependency_proof(
            _copy_contract(contracts, spec.input_relation),
            _predicate_dependency_proof(source, spec.predicate).merge(
                _operation_value_proof(proof_context, operation.id)
            ),
        )
    if isinstance(spec, ProjectSpec):
        return _project_contract(spec, contracts)
    if isinstance(spec, ProjectToKeySpec):
        return _project_to_key_contract(spec, contracts)
    if isinstance(spec, JoinSpec):
        return _join_contract(spec, contracts)
    if isinstance(spec, UnionSpec):
        return _union_contract(spec, contracts)
    if isinstance(spec, RoleExpandSpec):
        return _role_expand_contract(spec, contracts)
    if isinstance(spec, CrossJoinSpec):
        return _cross_join_contract(spec, contracts)
    if isinstance(spec, AntiJoinSpec):
        return _anti_join_contract(spec, contracts)
    if isinstance(spec, UniversalConditionSpec):
        return _with_operation_proof_refs(
            _universal_condition_contract(spec, contracts),
            proof_context.operation_refs.get(operation.id, frozenset()),
        )
    if isinstance(spec, AggregateSpec):
        return _aggregate_contract(spec, contracts)
    if isinstance(spec, RankSpec):
        source = _contract(contracts, spec.input_relation)
        return _with_dependency_proof(
            _copy_contract(contracts, spec.input_relation),
            _rank_dependency_proof(source, spec).merge(
                _operation_value_proof(proof_context, operation.id)
            ),
        )
    if isinstance(spec, ComputeSpec):
        raise VerificationError("compute operation cannot produce a relation")
    assert_never(spec)


def _with_operation_proof_refs(
    contract: RelationContract,
    refs: frozenset[str],
) -> RelationContract:
    if not refs:
        return contract
    return _with_dependency_proof(contract, ProofLineage.value(refs))


def _operation_value_proof(
    proof_context: ExecutionProofContext,
    operation_id: str,
) -> ProofLineage:
    return ProofLineage.value(
        frozenset(proof_context.operation_refs.get(operation_id, frozenset()))
    )


def _with_dependency_proof(
    contract: RelationContract,
    proof: ProofLineage,
) -> RelationContract:
    if not proof.fulfillment_refs():
        return contract
    return RelationContract(
        fields=dict(contract.fields),
        grain_keys=contract.grain_keys,
        field_proofs={
            field: field_proof.merge(proof)
            for field, field_proof in contract.field_proofs.items()
        },
        field_types=dict(contract.field_types),
        entity_keys=contract.entity_keys,
        population_proof=contract.population_proof.merge(proof),
    )


def _fields_dependency_proof(
    contract: RelationContract,
    fields: tuple[str, ...],
    label: str,
) -> ProofLineage:
    proof = ProofLineage()
    for field in fields:
        proof = proof.merge(_field_proof(contract, field, label))
    return proof


def _predicate_dependency_proof(
    contract: RelationContract,
    predicate: Predicate,
) -> ProofLineage:
    fields = [predicate.left]
    if predicate.right:
        fields.append(predicate.right)
    return _fields_dependency_proof(
        contract,
        tuple(field for field in fields if field),
        "predicate",
    )


def _rank_dependency_proof(
    contract: RelationContract,
    spec: RankSpec,
) -> ProofLineage:
    return _fields_dependency_proof(
        contract,
        tuple(sort.field for sort in (*spec.order_by, *spec.tie_breakers)),
        "rank",
    )


def _project_contract(
    spec: ProjectSpec,
    contracts: dict[str, RelationContract],
) -> RelationContract:
    source = _contract(contracts, spec.input_relation)
    fields: dict[str, frozenset[FieldBindingRole]] = {}
    field_proofs: dict[str, ProofLineage] = {}
    for field in spec.fields:
        output = field.output or field.source
        fields[output] = _field_roles(source, field.source, "project")
        field_proofs[output] = _field_proof(source, field.source, "project")
    projections = {
        field.source: field.output or field.source for field in spec.fields
    }
    return RelationContract(
        fields=fields,
        grain_keys=_project_contract_grain(source, spec.fields),
        field_proofs=field_proofs,
        field_types={
            field.output or field.source: source.field_types.get(field.source, "")
            for field in spec.fields
        },
        entity_keys=_project_entity_keys(source, projections),
        population_proof=source.population_proof,
    )


def _project_to_key_contract(
    spec: ProjectToKeySpec,
    contracts: dict[str, RelationContract],
) -> RelationContract:
    source = _contract(contracts, spec.input_relation)
    fields: dict[str, frozenset[FieldBindingRole]] = {}
    field_proofs: dict[str, ProofLineage] = {}
    for key_field in spec.key_fields:
        fields[key_field] = _field_roles(source, key_field, "project_to_key")
        field_proofs[key_field] = _field_proof(source, key_field, "project_to_key")
    projections = {field: field for field in spec.key_fields}
    return RelationContract(
        fields=fields,
        grain_keys=spec.key_fields,
        field_proofs=field_proofs,
        field_types={
            field: source.field_types.get(field, "") for field in spec.key_fields
        },
        entity_keys=_project_entity_keys(source, projections),
        population_proof=source.population_proof,
    )


def _join_contract(
    spec: JoinSpec,
    contracts: dict[str, RelationContract],
) -> RelationContract:
    left = _contract(contracts, spec.left)
    right = _contract(contracts, spec.right)
    dependency_proof = _join_dependency_proof(
        left,
        right,
        spec.join_keys,
        require_identity_authority=bool(
            left.population_proof.population_coverage.all_tests
            or right.population_proof.population_coverage.all_tests
        ),
    )
    fields = {**left.fields}
    joined_coverage = left.population_proof.population_coverage.additive(
        right.population_proof.population_coverage
    )
    field_proofs = {
        field: proof.with_population_coverage(joined_coverage).merge(dependency_proof)
        for field, proof in left.field_proofs.items()
    }
    for field, roles in right.fields.items():
        existing = fields.get(field)
        fields[field] = roles if existing is None else frozenset({*existing, *roles})
        proof = (
            right.field_proofs.get(field, ProofLineage())
            .with_population_coverage(joined_coverage)
            .merge(dependency_proof)
        )
        field_proofs[field] = field_proofs.get(field, ProofLineage()).merge(proof)
    return RelationContract(
        fields=fields,
        grain_keys=_join_contract_grain(left.grain_keys, right.grain_keys),
        field_proofs=field_proofs,
        field_types=_merge_contract_field_types(left, right),
        entity_keys=_combined_entity_keys(left, right),
        population_proof=ProofLineage(
            value_refs=frozenset(
                {
                    *left.population_proof.value_refs,
                    *right.population_proof.value_refs,
                    *dependency_proof.value_refs,
                }
            ),
            population_coverage=joined_coverage,
        ),
    )


def _join_dependency_proof(
    left: RelationContract,
    right: RelationContract,
    join_keys: tuple[JoinKey, ...],
    *,
    require_identity_authority: bool = False,
) -> ProofLineage:
    if (
        require_identity_authority
        and not _join_keys_include_shared_entity_key(left, right, join_keys)
    ):
        raise VerificationError("join keys lack declared identity authority")
    proof = ProofLineage()
    for key in join_keys:
        if not declared_types_compatible(
            left.field_types.get(key.left), right.field_types.get(key.right)
        ):
            raise VerificationError("join keys have incompatible declared types")
        proof = proof.merge(
            _field_proof(left, key.left, "join"),
            _field_proof(right, key.right, "join"),
        )
    return proof


def _join_keys_include_shared_entity_key(
    left: RelationContract,
    right: RelationContract,
    join_keys: tuple[JoinKey, ...],
) -> bool:
    declared_pairs = {(key.left, key.right) for key in join_keys}
    for left_key in left.entity_keys:
        for right_key in right.entity_keys:
            if (left_key.entity_kind, left_key.key_id) != (
                right_key.entity_kind,
                right_key.key_id,
            ):
                continue
            right_fields = {
                component.component_id: component.field_id
                for component in right_key.components
            }
            required_pairs = {
                (component.field_id, right_fields.get(component.component_id, ""))
                for component in left_key.components
            }
            if required_pairs and all(
                right_field and (left_field, right_field) in declared_pairs
                for left_field, right_field in required_pairs
            ):
                return True
    return False


def _union_contract(
    spec: UnionSpec,
    contracts: dict[str, RelationContract],
) -> RelationContract:
    population_proofs = tuple(
        _contract(contracts, relation_id).population_proof
        for relation_id in spec.inputs
    )
    population_proof = ProofLineage(
        value_refs=frozenset(
            ref for proof in population_proofs for ref in proof.value_refs
        ),
        population_coverage=PopulationCoverage.guaranteed_by_every(
            tuple(proof.population_coverage for proof in population_proofs)
        ),
    )
    field_types = {
        field: _union_field_type(contracts, spec.inputs, field)
        for field in spec.output_fields
    }
    input_contracts = tuple(_contract(contracts, item) for item in spec.inputs)
    return RelationContract(
        fields={
            field: _union_field_roles(contracts, spec.inputs, field)
            for field in spec.output_fields
        },
        grain_keys=spec.identity_fields,
        field_proofs={
            field: _union_field_proof(contracts, spec.inputs, field)
            for field in spec.output_fields
        },
        field_types=field_types,
        entity_keys=_common_entity_keys(input_contracts),
        population_proof=population_proof,
    )


def _role_expand_contract(
    spec: RoleExpandSpec,
    contracts: dict[str, RelationContract],
) -> RelationContract:
    source = _contract(contracts, spec.input_relation)
    fields: dict[str, frozenset[FieldBindingRole]] = {}
    field_proofs: dict[str, ProofLineage] = {}
    field_types: dict[str, str] = {}
    alternative_proofs: dict[str, list[ProofLineage]] = {}
    for field in spec.carry_fields:
        if field in spec.output_fields:
            fields[field] = _field_roles(source, field, "role_expand")
            field_proofs[field] = _field_proof(source, field, "role_expand")
            field_types[field] = source.field_types.get(field, "")
    if spec.role_field in spec.output_fields:
        fields[spec.role_field] = frozenset(
            {FieldBindingRole.IDENTITY, FieldBindingRole.OUTPUT}
        )
        field_proofs[spec.role_field] = _role_expand_role_proof(source, spec)
        field_types[spec.role_field] = "string"
    for mapping in spec.mappings:
        if mapping.output_field not in spec.output_fields:
            continue
        roles = _field_roles(source, mapping.source_field, "role_expand")
        existing = fields.get(mapping.output_field, frozenset())
        fields[mapping.output_field] = frozenset({*existing, *roles})
        alternative_proofs.setdefault(mapping.output_field, []).append(
            _field_proof(source, mapping.source_field, "role_expand")
        )
        field_types[mapping.output_field] = source.field_types.get(
            mapping.source_field, ""
        )
    for output_field, proofs in alternative_proofs.items():
        field_proofs[output_field] = ProofLineage(
            value_refs=frozenset(
                ref for proof in proofs for ref in proof.value_refs
            ),
            population_coverage=PopulationCoverage.guaranteed_by_every(
                tuple(proof.population_coverage for proof in proofs)
            ),
        )
    grain_keys: tuple[str, ...] = ()
    if source.grain_keys:
        grain_keys = (*source.grain_keys, spec.role_field)
        for grain_key in grain_keys:
            fields[grain_key] = fields.get(grain_key, frozenset())
    return RelationContract(
        fields=fields,
        grain_keys=grain_keys,
        field_proofs=field_proofs,
        field_types=field_types,
        entity_keys=_project_entity_keys(
            source,
            {field: field for field in spec.carry_fields if field in fields},
        ),
        population_proof=source.population_proof,
    )


def _role_expand_role_proof(
    source: RelationContract,
    spec: RoleExpandSpec,
) -> ProofLineage:
    proof = source.population_proof
    for mapping in spec.mappings:
        proof = proof.merge(_field_proof(source, mapping.source_field, "role_expand"))
    return proof


def _cross_join_contract(
    spec: CrossJoinSpec,
    contracts: dict[str, RelationContract],
) -> RelationContract:
    left = _contract(contracts, spec.left)
    right = _contract(contracts, spec.right)
    fields = {**left.fields}
    field_proofs = dict(left.field_proofs)
    for field, roles in right.fields.items():
        existing = fields.get(field)
        fields[field] = roles if existing is None else frozenset({*existing, *roles})
        right_proof = right.field_proofs.get(field, ProofLineage())
        if field not in field_proofs:
            field_proofs[field] = right_proof
            continue
        left_proof = field_proofs[field]
        field_proofs[field] = ProofLineage(
            value_refs=frozenset({*left_proof.value_refs, *right_proof.value_refs}),
            population_coverage=PopulationCoverage.guaranteed_by_every(
                (
                    left_proof.population_coverage,
                    right_proof.population_coverage,
                )
            ),
        )
    return RelationContract(
        fields=fields,
        grain_keys=(*left.grain_keys, *right.grain_keys),
        field_proofs=field_proofs,
        field_types=_merge_contract_field_types(left, right),
        entity_keys=_combined_entity_keys(left, right),
        population_proof=ProofLineage(
            value_refs=frozenset(
                {
                    *left.population_proof.value_refs,
                    *right.population_proof.value_refs,
                }
            ),
            population_coverage=PopulationCoverage.guaranteed_by_every(
                (
                    left.population_proof.population_coverage,
                    right.population_proof.population_coverage,
                )
            ),
        ),
    )


def _anti_join_contract(
    spec: AntiJoinSpec,
    contracts: dict[str, RelationContract],
) -> RelationContract:
    candidate = _contract(contracts, spec.candidate.relation_id)
    observed = _contract(contracts, spec.observed.relation_id)
    consumed_conditions = frozenset(
        {
            *candidate.population_proof.population_coverage.condition_tests,
            *observed.population_proof.population_coverage.condition_tests,
        }
    )
    output_coverage = PopulationCoverage(
        row_tests=frozenset(
            {
                *candidate.population_proof.population_coverage.row_tests,
                *consumed_conditions,
            }
        ),
        condition_tests=consumed_conditions,
    )
    dependency_proof = _join_dependency_proof(
        candidate,
        observed,
        spec.join_keys,
        require_identity_authority=bool(consumed_conditions),
    )
    fields: dict[str, frozenset[FieldBindingRole]] = {}
    field_proofs: dict[str, ProofLineage] = {}
    field_types: dict[str, str] = {}
    for grain_key in spec.candidate.required_identity_fields:
        _field_roles(candidate, grain_key, "anti_join")
        fields[grain_key] = frozenset({FieldBindingRole.IDENTITY})
        field_proofs[grain_key] = (
            _field_proof(candidate, grain_key, "anti_join")
            .with_population_coverage(output_coverage)
            .merge(dependency_proof)
        )
        field_types[grain_key] = candidate.field_types.get(grain_key, "")
    for field in spec.output_fields:
        output = field.output or field.source
        fields[output] = _field_roles(candidate, field.source, "anti_join")
        field_proofs[output] = (
            _field_proof(candidate, field.source, "anti_join")
            .with_population_coverage(output_coverage)
            .merge(dependency_proof)
        )
        field_types[output] = candidate.field_types.get(field.source, "")
    projections = {
        **{
            field: field for field in spec.candidate.required_identity_fields
        },
        **{field.source: field.output or field.source for field in spec.output_fields},
    }
    return RelationContract(
        fields=fields,
        grain_keys=spec.candidate.required_identity_fields,
        field_proofs=field_proofs,
        field_types=field_types,
        entity_keys=_project_entity_keys(candidate, projections),
        population_proof=ProofLineage(
            value_refs=frozenset(
                {
                    *candidate.population_proof.value_refs,
                    *observed.population_proof.value_refs,
                    *dependency_proof.value_refs,
                }
            ),
            population_coverage=output_coverage,
        ),
    )


def _universal_condition_contract(
    spec: UniversalConditionSpec,
    contracts: dict[str, RelationContract],
) -> RelationContract:
    candidate = _contract(contracts, spec.candidate_subject.relation_id)
    required_dimension = _contract(contracts, spec.required_dimension.relation_id)
    observation = _contract(contracts, spec.observation.relation_id)
    consumed_conditions = frozenset(
        {
            *candidate.population_proof.population_coverage.condition_tests,
            *required_dimension.population_proof.population_coverage.condition_tests,
            *observation.population_proof.population_coverage.condition_tests,
        }
    )
    output_coverage = PopulationCoverage(
        row_tests=frozenset(
            {
                *candidate.population_proof.population_coverage.row_tests,
                *consumed_conditions,
            }
        ),
        condition_tests=consumed_conditions,
    )
    dependency_proof = _universal_dependency_proof(
        candidate,
        required_dimension,
        observation,
        spec,
    )
    fields: dict[str, frozenset[FieldBindingRole]] = {}
    field_proofs: dict[str, ProofLineage] = {}
    field_types: dict[str, str] = {}
    for grain_key in spec.candidate_subject.required_identity_fields:
        _field_roles(candidate, grain_key, "universal_condition")
        fields[grain_key] = frozenset({FieldBindingRole.IDENTITY})
        field_proofs[grain_key] = (
            _field_proof(candidate, grain_key, "universal_condition")
            .merge(dependency_proof)
            .with_population_coverage(output_coverage)
        )
        field_types[grain_key] = candidate.field_types.get(grain_key, "")
    for field in spec.output_fields:
        output = field.output or field.source
        fields[output] = _field_roles(candidate, field.source, "universal_condition")
        field_proofs[output] = (
            _field_proof(candidate, field.source, "universal_condition")
            .merge(dependency_proof)
            .with_population_coverage(output_coverage)
        )
        field_types[output] = candidate.field_types.get(field.source, "")
    projections = {
        **{
            field: field
            for field in spec.candidate_subject.required_identity_fields
        },
        **{field.source: field.output or field.source for field in spec.output_fields},
    }
    return RelationContract(
        fields=fields,
        grain_keys=spec.candidate_subject.required_identity_fields,
        field_proofs=field_proofs,
        field_types=field_types,
        entity_keys=_project_entity_keys(candidate, projections),
        population_proof=ProofLineage(
            value_refs=frozenset(
                {
                    *candidate.population_proof.value_refs,
                    *required_dimension.population_proof.value_refs,
                    *observation.population_proof.value_refs,
                    *dependency_proof.value_refs,
                }
            ),
            population_coverage=output_coverage,
        ),
    )


def _universal_dependency_proof(
    candidate: RelationContract,
    required_dimension: RelationContract,
    observation: RelationContract,
    spec: UniversalConditionSpec,
) -> ProofLineage:
    proof = _join_dependency_proof(
        candidate,
        observation,
        spec.subject_keys,
        require_identity_authority=bool(
            candidate.population_proof.population_coverage.condition_tests
            or observation.population_proof.population_coverage.condition_tests
        ),
    ).merge(
        _join_dependency_proof(
            required_dimension,
            observation,
            spec.dimension_keys,
            require_identity_authority=bool(
                required_dimension.population_proof.population_coverage.condition_tests
                or observation.population_proof.population_coverage.condition_tests
            ),
        )
    )
    return proof.merge(_predicate_dependency_proof(observation, spec.predicate))


def _aggregate_contract(
    spec: AggregateSpec,
    contracts: dict[str, RelationContract],
) -> RelationContract:
    source = _contract(contracts, spec.input_relation)
    fields: dict[str, frozenset[FieldBindingRole]] = {}
    field_proofs: dict[str, ProofLineage] = {}
    field_types: dict[str, str] = {}
    group_proof = _fields_dependency_proof(source, tuple(spec.group_by), "aggregate")
    for field in spec.group_by:
        fields[field] = _field_roles(source, field, "aggregate")
        field_proofs[field] = _field_proof(
            source, field, "aggregate"
        ).with_population_coverage(source.population_proof.population_coverage)
        field_types[field] = source.field_types.get(field, "")
    for aggregation in spec.aggregations:
        fields[aggregation.output_field] = frozenset(
            {FieldBindingRole.OUTPUT, FieldBindingRole.PREDICATE}
        )
        field_proofs[aggregation.output_field] = (
            source.population_proof.merge(group_proof)
            if aggregation.function == AggregationFunction.COUNT
            else _field_proof(source, aggregation.input_field, "aggregate")
            .merge(group_proof)
            .with_population_coverage(source.population_proof.population_coverage)
        )
        field_types[aggregation.output_field] = (
            "integer"
            if aggregation.function == AggregationFunction.COUNT
            else "decimal"
            if aggregation.function
            in {AggregationFunction.SUM, AggregationFunction.AVG}
            else source.field_types.get(aggregation.input_field, "")
        )
    return RelationContract(
        fields=fields,
        grain_keys=spec.group_by,
        field_proofs=field_proofs,
        field_types=field_types,
        entity_keys=_project_entity_keys(
            source,
            {field: field for field in spec.group_by},
        ),
        population_proof=source.population_proof,
    )


def _merge_contract_field_types(
    left: RelationContract, right: RelationContract
) -> dict[str, str]:
    output = dict(left.field_types)
    for field, field_type in right.field_types.items():
        if field in output and not declared_types_compatible(output[field], field_type):
            raise VerificationError(f"field {field} has incompatible declared types")
        output.setdefault(field, field_type)
    return output


def _union_field_type(
    contracts: dict[str, RelationContract], relation_ids: tuple[str, ...], field: str
) -> str:
    field_type = ""
    for relation_id in relation_ids:
        candidate = _contract(contracts, relation_id).field_types.get(field, "")
        if field_type and not declared_types_compatible(field_type, candidate):
            raise VerificationError("union fields have incompatible declared types")
        if candidate:
            field_type = candidate
    return field_type
