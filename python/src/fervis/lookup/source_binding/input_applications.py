"""One contract for applying resolved question inputs to a source invocation."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from fervis.types.enums import StrEnum
from fervis.lookup.answer_program.values import (
    FactValue,
    IdentitySetValuePayload,
    IdentityValuePayload,
    TimeComponent,
    ValueComponent,
)
from fervis.lookup.answer_program.relations import (
    PopulationCoverageClaim,
    PopulationCoverageRole,
)
from fervis.lookup.canonical_data import RuntimeValue
from fervis.lookup.source_binding import provider_contract as provider_output
from fervis.lookup.source_binding.candidates.contracts import (
    CandidateKeyEvidence,
    EntityEvidence,
    EntityReferenceEvidence,
)
from fervis.lookup.source_binding.candidates.model import (
    CandidateParameter,
    SourceCandidate,
)
from fervis.lookup.source_binding.compiler_ir import (
    RelationInputOrigin,
    SourceAppliedFilter,
)
from fervis.lookup.source_binding.closed_key_params import ClosedKeyParamBindingIndex
from fervis.lookup.source_binding.model import SourceBindingRequest
from fervis.lookup.source_binding.plan_targets import SourceBindingTarget
from fervis.lookup.source_binding.param_binding_sets import (
    ParamBindingSetAlternatives,
    combine_param_binding_sets,
    parameter_binding_sets,
)
from fervis.lookup.source_binding.param_values import (
    compatible_identity_parameter_component_ids,
    identity_parameter_component_value,
    identity_value_matches_entity_target,
)
from fervis.lookup.turn_prompts.projections import resolved_values_for_requested_fact
from fervis.lookup.question_contract import (
    AnswerPopulationMembershipTestKind,
    RequestedFactAnswerPopulationMembershipTest,
)
from fervis.lookup.source_binding.population_effects import (
    population_coverage_claims,
    population_test_basis_payload,
    population_test_results_schema,
)


class ResolvedInputApplicationTargetKind(StrEnum):
    REQUEST_PARAMETER = "request_parameter"
    RETURNED_IDENTITY = "returned_identity"


@dataclass(frozen=True)
class ResolvedInputApplicationOwner:
    known_input_id: str
    owner_surface_id: str
    proof_refs: tuple[str, ...]


@dataclass(frozen=True)
class ParsedResolvedInputApplications:
    param_binding_sets: ParamBindingSetAlternatives = ((),)
    applied_filters: tuple[SourceAppliedFilter, ...] = ()
    known_input_ids: tuple[str, ...] = ()
    population_coverage_claims: tuple[PopulationCoverageClaim, ...] = ()


@dataclass(frozen=True)
class _ParameterApplication:
    output: provider_output.ResolvedInputApplicationOutput
    param: CandidateParameter
    value: FactValue


@dataclass(frozen=True)
class ResolvedInputApplicationSurface:
    requested_fact_id: str
    values_by_id: dict[str, FactValue]
    parameter_targets_by_id: dict[str, CandidateParameter]
    identity_targets_by_id: dict[str, EntityEvidence]
    membership_tests_by_input_id: dict[
        str, tuple[RequestedFactAnswerPopulationMembershipTest, ...]
    ]
    alternative_input_test: RequestedFactAnswerPopulationMembershipTest | None
    coverage_role: PopulationCoverageRole | None
    role_text: str

    def prompt_payload(self) -> dict[str, object]:
        return {
            "resolved_values": list(self._resolved_value_payloads()),
            "targets_by_kind": self._targets_by_kind(),
        }

    def provider_schema(self) -> dict[str, object]:
        variants = [
            _application_schema(
                value_id=value_id,
                value_component=value_component,
                target_kind=target_kind,
                target_ids=self._application_target_ids(
                    value_id=value_id,
                    value_component=value_component,
                    target_kind=target_kind,
                ),
                population_test_ids=population_test_ids,
            )
            for value_id, target_kind, value_component, population_test_ids in (
                self._schema_value_variants()
            )
        ]
        if not variants:
            return empty_resolved_input_applications_schema()
        item_schema = variants[0] if len(variants) == 1 else {"oneOf": variants}
        return {
            "type": "array",
            "items": item_schema,
            "maxItems": len(self.parameter_targets_by_id)
            + len(self.identity_targets_by_id)
            + self._additional_parameter_alternatives(),
        }

    def _additional_parameter_alternatives(self) -> int:
        if self.alternative_input_test is None:
            return 0
        return (len(self.alternative_input_test.owned_question_input_refs) - 1) * len(
            self.parameter_targets_by_id
        )

    def _resolved_value_payloads(self) -> tuple[dict[str, object], ...]:
        payloads: list[dict[str, object]] = []
        for value_id, value in self.values_by_id.items():
            parameter_components = tuple(
                dict.fromkeys(
                    component
                    for param in self.parameter_targets_by_id.values()
                    for component in _parameter_value_components(param, value=value)
                )
            )
            has_identity_target = any(
                _identity_target_accepts_value(evidence, value=value)
                for evidence in self.identity_targets_by_id.values()
            )
            components_by_target_kind: dict[str, list[str]] = {}
            if parameter_components:
                components_by_target_kind[
                    ResolvedInputApplicationTargetKind.REQUEST_PARAMETER.value
                ] = list(parameter_components)
            if has_identity_target:
                components_by_target_kind[
                    ResolvedInputApplicationTargetKind.RETURNED_IDENTITY.value
                ] = ["canonical_key"]
            if not components_by_target_kind:
                continue
            tests = self.membership_tests_by_input_id.get(value.known_input_id, ())
            payloads.append(
                {
                    "value_id": value_id,
                    "components_by_target_kind": components_by_target_kind,
                    "population_test_basis": population_test_basis_payload(
                        tests,
                        role_text=self.role_text,
                    ),
                    **(
                        {
                            "request_parameter_alternative_group": (
                                self.alternative_input_test.id
                            )
                        }
                        if self.alternative_input_test is not None
                        and value.known_input_id
                        in self.alternative_input_test.owned_question_input_refs
                        else {}
                    ),
                }
            )
        return tuple(payloads)

    def _targets_by_kind(self) -> dict[str, list[str]]:
        return {
            ResolvedInputApplicationTargetKind.REQUEST_PARAMETER.value: list(
                self.parameter_targets_by_id
            ),
            ResolvedInputApplicationTargetKind.RETURNED_IDENTITY.value: list(
                self.identity_targets_by_id
            ),
        }

    def _application_target_ids(
        self,
        *,
        value_id: str,
        value_component: str,
        target_kind: str,
    ) -> list[str]:
        value = self.values_by_id[value_id]
        if target_kind == ResolvedInputApplicationTargetKind.REQUEST_PARAMETER.value:
            return [
                param_id
                for param_id, param in self.parameter_targets_by_id.items()
                if value_component in _parameter_value_components(param, value=value)
            ]
        if target_kind != ResolvedInputApplicationTargetKind.RETURNED_IDENTITY.value:
            return []
        return [
            evidence_id
            for evidence_id, evidence in self.identity_targets_by_id.items()
            if value_component == "canonical_key"
            and _identity_target_accepts_value(evidence, value=value)
        ]

    def accepts_application(
        self,
        *,
        value_id: str,
        value_component: str,
        target_kind: str,
        target_id: str,
    ) -> bool:
        if value_id not in self.values_by_id:
            return False
        return target_id in self._application_target_ids(
            value_id=value_id,
            value_component=value_component,
            target_kind=target_kind,
        )

    def _schema_value_variants(
        self,
    ) -> tuple[tuple[str, str, str, tuple[str, ...]], ...]:
        variants: list[tuple[str, str, str, tuple[str, ...]]] = []
        for value in self._resolved_value_payloads():
            test_basis = value["population_test_basis"]
            assert isinstance(test_basis, dict)
            components_by_kind = value["components_by_target_kind"]
            assert isinstance(components_by_kind, dict)
            for target_kind, components in components_by_kind.items():
                assert isinstance(target_kind, str)
                assert isinstance(components, list)
                variants.extend(
                    (
                        str(value["value_id"]),
                        target_kind,
                        str(component),
                        tuple(str(test_id) for test_id in test_basis),
                    )
                    for component in components
                )
        return tuple(variants)

    def owners(self) -> tuple[ResolvedInputApplicationOwner, ...]:
        owners: list[ResolvedInputApplicationOwner] = []
        for param in self.parameter_targets_by_id.values():
            for value in self.values_by_id.values():
                if not _parameter_value_components(param, value=value):
                    continue
                owners.append(
                    _application_owner(
                        value,
                        owner_surface_id=f"source_param:{param.id}",
                    )
                )
        for evidence in self.identity_targets_by_id.values():
            for value in self.values_by_id.values():
                if not _identity_target_accepts_value(evidence, value=value):
                    continue
                owners.append(
                    _application_owner(
                        value,
                        owner_surface_id=f"returned_identity:{evidence.evidence_id}",
                    )
                )
        return _dedupe_owners(tuple(owners))


def resolved_input_application_surface(
    *,
    candidate: SourceCandidate,
    requested_fact_id: str,
    resolved_values: tuple[FactValue, ...],
    membership_tests: tuple[RequestedFactAnswerPopulationMembershipTest, ...] = (),
    alternative_input_test: RequestedFactAnswerPopulationMembershipTest | None = None,
    coverage_role: PopulationCoverageRole | None = None,
    role_text: str = "source rows",
) -> ResolvedInputApplicationSurface:
    """Describe independently selectable values and invocation-local targets."""

    values_by_id = {value.id: value for value in resolved_values}
    if alternative_input_test is not None and (
        alternative_input_test not in membership_tests
        or len(alternative_input_test.owned_question_input_refs) < 2
    ):
        raise ValueError("alternative input test must own multiple fact inputs")
    parameter_targets = {
        param.id: param
        for param in candidate.params
        if any(
            _parameter_value_components(param, value=value)
            for value in resolved_values
        )
    }
    identity_targets = {
        evidence.evidence_id: evidence
        for evidence in candidate.evidence_items
        if isinstance(evidence, (CandidateKeyEvidence, EntityReferenceEvidence))
        if any(
            _identity_target_accepts_value(evidence, value=value)
            for value in resolved_values
        )
    }
    return ResolvedInputApplicationSurface(
        requested_fact_id=requested_fact_id,
        values_by_id=values_by_id,
        parameter_targets_by_id=parameter_targets,
        identity_targets_by_id=identity_targets,
        membership_tests_by_input_id=_membership_tests_by_input_id(
            membership_tests if coverage_role is not None else (),
            alternative_input_test=(
                alternative_input_test if coverage_role is not None else None
            ),
        ),
        alternative_input_test=alternative_input_test,
        coverage_role=coverage_role,
        role_text=role_text,
    )


def empty_resolved_input_applications_schema() -> dict[str, object]:
    """Return the strict schema for an invocation with no input applications."""

    return {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "properties": {},
            "required": [],
        },
        "maxItems": 0,
    }


def resolved_input_application_surfaces(
    request: SourceBindingRequest,
    *,
    targets: tuple[SourceBindingTarget, ...],
    candidates_by_id: dict[str, SourceCandidate],
    closed_key_bindings: ClosedKeyParamBindingIndex,
) -> dict[str, ResolvedInputApplicationSurface]:
    """Build each invocation surface from its fact, values, and source candidate."""

    facts_by_id = {fact.id: fact for fact in request.requested_facts}
    output: dict[str, ResolvedInputApplicationSurface] = {}
    for target in targets:
        fact = facts_by_id.get(target.requested_fact_id)
        candidate = candidates_by_id.get(target.source_candidate_id)
        if fact is None or candidate is None:
            continue
        specified_group_test = fact.specified_group_membership_test()
        parameter_alternative_test = (
            specified_group_test
            if specified_group_test is not None
            and len(specified_group_test.owned_question_input_refs) > 1
            else None
        )
        surface = resolved_input_application_surface(
            candidate=candidate,
            requested_fact_id=fact.id,
            resolved_values=resolved_values_for_requested_fact(
                fact,
                available_values=request.available_values,
            ),
            membership_tests=(
                fact.answer_population.membership_tests
                if fact.answer_population is not None
                else ()
            ),
            alternative_input_test=parameter_alternative_test,
            coverage_role=(
                PopulationCoverageRole.ROW_POPULATION
                if target.requires_answer_fulfillment
                else None
            ),
            role_text=target.requirement_id,
        )
        backend_owned_param_ids = closed_key_bindings.owned_param_ids(
            target.binding_target_id
        )
        backend_owned_value_ids = closed_key_bindings.owned_value_ids(
            target.binding_target_id
        )
        model_visible_values = {
            value_id: value
            for value_id, value in surface.values_by_id.items()
            if value_id not in backend_owned_value_ids
        }
        alternative_input_test = surface.alternative_input_test
        if alternative_input_test is not None and not set(
            alternative_input_test.owned_question_input_refs
        ).issubset(
            value.known_input_id for value in model_visible_values.values()
        ):
            alternative_input_test = None
        output[target.binding_target_id] = ResolvedInputApplicationSurface(
            requested_fact_id=surface.requested_fact_id,
            values_by_id=model_visible_values,
            parameter_targets_by_id={
                target_id: param
                for target_id, param in surface.parameter_targets_by_id.items()
                if target_id not in backend_owned_param_ids
            },
            identity_targets_by_id={
                evidence_id: evidence
                for evidence_id, evidence in surface.identity_targets_by_id.items()
                if any(
                    _identity_target_accepts_value(evidence, value=value)
                    for value in model_visible_values.values()
                )
            },
            membership_tests_by_input_id=surface.membership_tests_by_input_id,
            alternative_input_test=alternative_input_test,
            coverage_role=surface.coverage_role,
            role_text=surface.role_text,
        )
    return output


def parse_resolved_input_applications(
    applications: tuple[provider_output.ResolvedInputApplicationOutput, ...],
    *,
    surface: ResolvedInputApplicationSurface,
) -> ParsedResolvedInputApplications:
    param_binding_groups: list[ParamBindingSetAlternatives] = []
    parameter_applications_by_target: dict[str, list[_ParameterApplication]] = {}
    applied_filters: list[SourceAppliedFilter] = []
    known_input_ids: list[str] = []
    population_coverage_claims: list[PopulationCoverageClaim] = []
    used_identity_targets: set[str] = set()
    for application in applications:
        if not application.match_basis_explanation.strip():
            raise ValueError("resolved input application requires a match basis")
        value = surface.values_by_id.get(application.value_id)
        if value is None:
            raise ValueError("resolved input application references unknown value")
        if not surface.accepts_application(
            target_kind=application.target_kind,
            target_id=application.target_id,
            value_id=application.value_id,
            value_component=application.value_component,
        ):
            raise ValueError("resolved value is incompatible with application target")
        if value.known_input_id:
            known_input_ids.append(value.known_input_id)
        test_proof_refs: tuple[str, ...]
        if (
            application.target_kind
            == ResolvedInputApplicationTargetKind.REQUEST_PARAMETER.value
        ):
            param = surface.parameter_targets_by_id.get(application.target_id)
            if param is None:
                raise ValueError("resolved input application references unknown param")
            parameter_applications_by_target.setdefault(param.id, []).append(
                _ParameterApplication(
                    output=application,
                    param=param,
                    value=value,
                )
            )
            continue
        elif (
            application.target_kind
            == ResolvedInputApplicationTargetKind.RETURNED_IDENTITY.value
        ):
            if _is_parameter_alternative(value, surface=surface):
                raise ValueError(
                    "grouped input alternatives require a request parameter target"
                )
            if application.target_id in used_identity_targets:
                raise ValueError("resolved input application repeats a target")
            used_identity_targets.add(application.target_id)
            evidence = surface.identity_targets_by_id.get(application.target_id)
            if evidence is None:
                raise ValueError(
                    "resolved input application references unknown identity target"
                )
            applied_filters.extend(
                _identity_filters(application, evidence=evidence, value=value)
            )
            test_proof_refs = tuple(
                dict.fromkeys(
                    (
                        *value.proof_refs,
                        f"returned_identity:{application.target_id}",
                    )
                )
            )
        else:
            raise ValueError("resolved input application has unsupported target kind")
        population_coverage_claims.extend(
            _application_population_coverage_claims(
                application,
                value=value,
                surface=surface,
                proof_refs=test_proof_refs,
            )
        )
    for parameter_applications in parameter_applications_by_target.values():
        binding_sets, claims = _parameter_application_group(
            tuple(parameter_applications),
            surface=surface,
        )
        param_binding_groups.append(binding_sets)
        population_coverage_claims.extend(claims)
    return ParsedResolvedInputApplications(
        param_binding_sets=combine_param_binding_sets(param_binding_groups),
        applied_filters=tuple(applied_filters),
        known_input_ids=tuple(dict.fromkeys(known_input_ids)),
        population_coverage_claims=tuple(population_coverage_claims),
    )


def _parameter_application_group(
    applications: tuple[_ParameterApplication, ...],
    *,
    surface: ResolvedInputApplicationSurface,
) -> tuple[ParamBindingSetAlternatives, tuple[PopulationCoverageClaim, ...]]:
    if len(applications) == 1:
        application = applications[0]
        if _is_parameter_alternative(application.value, surface=surface):
            raise ValueError("parameter alternatives must apply together")
        proof_refs = _parameter_application_proof_refs(applications)
        return (
            _parameter_binding_sets(
                application.output,
                param=application.param,
                value=application.value,
            ),
            _application_population_coverage_claims(
                application.output,
                value=application.value,
                surface=surface,
                proof_refs=proof_refs,
            ),
        )
    _require_parameter_alternative_group(applications, surface=surface)
    proof_refs = _parameter_application_proof_refs(applications)
    coverage_claim_sets = tuple(
        _application_population_coverage_claims(
            application.output,
            value=application.value,
            surface=surface,
            proof_refs=proof_refs,
        )
        for application in applications
    )
    if any(claims != coverage_claim_sets[0] for claims in coverage_claim_sets[1:]):
        raise ValueError("parameter alternatives disagree on population effects")
    return (
        tuple(
            binding_set
            for application in applications
            for binding_set in _parameter_binding_sets(
                application.output,
                param=application.param,
                value=application.value,
            )
        ),
        coverage_claim_sets[0],
    )


def _require_parameter_alternative_group(
    applications: tuple[_ParameterApplication, ...],
    *,
    surface: ResolvedInputApplicationSurface,
) -> None:
    membership_test = surface.alternative_input_test
    if membership_test is None:
        raise ValueError("resolved input application repeats a target")
    values = tuple(application.value for application in applications)
    input_ids = tuple(value.known_input_id for value in values)
    if (
        any(not isinstance(value.payload, IdentityValuePayload) for value in values)
        or len(set(input_ids)) != len(input_ids)
        or frozenset(input_ids)
        != frozenset(membership_test.owned_question_input_refs)
        or len({value.id for value in values}) != len(values)
        or len({application.output.value_component for application in applications})
        != 1
    ):
        raise ValueError("repeated parameter target requires proven input alternatives")


def _is_parameter_alternative(
    value: FactValue,
    *,
    surface: ResolvedInputApplicationSurface,
) -> bool:
    membership_test = surface.alternative_input_test
    return (
        membership_test is not None
        and value.known_input_id in membership_test.owned_question_input_refs
    )


def _parameter_application_proof_refs(
    applications: tuple[_ParameterApplication, ...],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                *(
                    proof_ref
                    for application in applications
                    for proof_ref in application.value.proof_refs
                ),
                f"source_param:{applications[0].param.id}",
            )
        )
    )


def _application_schema(
    *,
    value_id: str,
    value_component: str,
    target_kind: str,
    target_ids: list[str],
    population_test_ids: tuple[str, ...],
) -> dict[str, object]:
    return provider_output.ResolvedInputApplicationOutput.schema(
        {
            "target_kind": {"enum": [target_kind]},
            "target_id": {"enum": target_ids},
            "value_id": {"enum": [value_id]},
            "value_component": {"enum": [value_component]},
            "match_basis_explanation": {"type": "string", "minLength": 1},
            "population_test_results": population_test_results_schema(
                population_test_ids
            ),
        }
    )


def _parameter_binding_sets(
    application: provider_output.ResolvedInputApplicationOutput,
    *,
    param: CandidateParameter,
    value: FactValue,
) -> ParamBindingSetAlternatives:
    selected_component = application.value_component
    compiler_component = selected_component
    if isinstance(value.payload, (IdentityValuePayload, IdentitySetValuePayload)):
        component_id = (
            param.entity_target.component_id
            if param.entity_target is not None
            else selected_component
        )
        compiler_component = f"key_component:{component_id}"
    concrete_value = _parameter_component_value(
        param,
        value=value,
        component=selected_component,
    )
    return parameter_binding_sets(
        param_id=param.id,
        value=concrete_value,
        param=param,
        origin_kind=RelationInputOrigin.QUESTION_INPUT,
        value_id=value.id,
        value_component=compiler_component,
        proof_refs=tuple(value.proof_refs),
    )


def _parameter_value_components(
    param: CandidateParameter,
    *,
    value: FactValue,
) -> tuple[str, ...]:
    payload = value.payload
    if isinstance(payload, (IdentityValuePayload, IdentitySetValuePayload)):
        if param.entity_target is not None:
            target = param.entity_target
            if not identity_value_matches_entity_target(
                value,
                entity_kind=target.entity_kind,
                key_id=target.key_id,
                component_id=target.component_id,
            ):
                return ()
            candidates = tuple(
                binding.value_component or "value"
                for binding in param.binding_values
                if binding.source == "available_value" and binding.value == value.id
            ) or ("value",)
        else:
            candidates = compatible_identity_parameter_component_ids(
                value,
                type_name=param.type,
                choices=param.choices,
            )
    else:
        binding_components = (
            binding.value_component or "value"
            for binding in param.binding_values
            if binding.source == "available_value" and binding.value == value.id
        )
        return tuple(dict.fromkeys(binding_components))
    return tuple(
        component
        for component in dict.fromkeys(candidates)
        if _parameter_accepts_component(param, value=value, component=component)
    )


def _parameter_accepts_component(
    param: CandidateParameter,
    *,
    value: FactValue,
    component: str,
) -> bool:
    try:
        _parameter_component_value(param, value=value, component=component)
    except ValueError:
        return False
    return True


def _parameter_component_value(
    param: CandidateParameter,
    *,
    value: FactValue,
    component: str,
) -> RuntimeValue:
    payload = value.payload
    if isinstance(payload, (IdentityValuePayload, IdentitySetValuePayload)):
        component_id = (
            param.entity_target.component_id
            if param.entity_target is not None
            else component
        )
        return identity_parameter_component_value(
            value,
            component_id=component_id,
            type_name=param.type,
            choices=param.choices,
        )
    ordinary_value = _ordinary_component_value(value, component=component)
    if isinstance(ordinary_value, Decimal):
        return str(ordinary_value)
    return ordinary_value


def _identity_filters(
    application: provider_output.ResolvedInputApplicationOutput,
    *,
    evidence: EntityEvidence,
    value: FactValue,
) -> tuple[SourceAppliedFilter, ...]:
    if application.value_component != "canonical_key":
        raise ValueError("returned identity requires the complete canonical key")
    payload = value.payload
    assert isinstance(payload, IdentityValuePayload)
    proof_refs = tuple(
        dict.fromkeys(
            (*value.proof_refs, f"returned_identity:{application.target_id}")
        )
    )
    return tuple(
        SourceAppliedFilter(
            known_input_id=value.known_input_id,
            predicate_field_ids=(component.field_id,),
            value_id=value.id,
            value_component=f"key_component:{component.component_id}",
            value_kind=value.kind.value,
            display_value=payload.display_value or value.label,
            matched_field_ref=payload.matched_field_ref,
            matched_field_path=payload.matched_field_path,
            proof_refs=proof_refs,
        )
        for component in evidence.components
    )


def _membership_tests_by_input_id(
    tests: tuple[RequestedFactAnswerPopulationMembershipTest, ...],
    *,
    alternative_input_test: RequestedFactAnswerPopulationMembershipTest | None,
) -> dict[str, tuple[RequestedFactAnswerPopulationMembershipTest, ...]]:
    output: dict[str, list[RequestedFactAnswerPopulationMembershipTest]] = {}
    for test in tests:
        if test.kind is not AnswerPopulationMembershipTestKind.EXPLICIT_USER_CONSTRAINT:
            continue
        if len(test.owned_question_input_refs) == 1:
            output.setdefault(test.owned_question_input_refs[0], []).append(test)
        elif test is alternative_input_test:
            for input_ref in test.owned_question_input_refs:
                output.setdefault(input_ref, []).append(test)
    return {input_id: tuple(items) for input_id, items in output.items()}


def _application_population_coverage_claims(
    application: provider_output.ResolvedInputApplicationOutput,
    *,
    value: FactValue,
    surface: ResolvedInputApplicationSurface,
    proof_refs: tuple[str, ...],
) -> tuple[PopulationCoverageClaim, ...]:
    tests = surface.membership_tests_by_input_id.get(value.known_input_id, ())
    if surface.coverage_role is None:
        if application.population_test_results:
            raise ValueError("resolved input application cannot claim this role")
        return ()
    return population_coverage_claims(
        application.population_test_results,
        tests=tests,
        requested_fact_id=surface.requested_fact_id,
        role_text=surface.role_text,
        coverage_role=surface.coverage_role,
        proof_refs=proof_refs,
    )


def _identity_target_accepts_value(
    evidence: EntityEvidence,
    *,
    value: FactValue,
) -> bool:
    payload = value.payload
    if not isinstance(payload, IdentityValuePayload):
        return False
    entity_kind, key_id = _identity_target_contract(evidence)
    target_components = {component.component_id for component in evidence.components}
    value_components = {component.component_id for component in payload.key.components}
    return (
        payload.entity_kind == entity_kind
        and payload.key_id == key_id
        and value_components == target_components
    )


def _identity_target_contract(evidence: EntityEvidence) -> tuple[str, str]:
    if isinstance(evidence, CandidateKeyEvidence):
        return evidence.entity_kind, evidence.key_id
    return evidence.target_entity_kind, evidence.target_key_id


def _ordinary_component_value(
    value: FactValue,
    *,
    component: str,
) -> RuntimeValue:
    typed_component: ValueComponent | TimeComponent
    if component in {item.value for item in TimeComponent}:
        typed_component = TimeComponent(component)
    else:
        typed_component = ValueComponent(component)
    return value.payload.component_value(typed_component)


def _application_owner(
    value: FactValue,
    *,
    owner_surface_id: str,
) -> ResolvedInputApplicationOwner:
    return ResolvedInputApplicationOwner(
        known_input_id=value.known_input_id,
        owner_surface_id=owner_surface_id,
        proof_refs=tuple(
            dict.fromkeys(
                (
                    *(value.proof_refs),
                    *(
                        (f"known_input:{value.known_input_id}",)
                        if value.known_input_id
                        else ()
                    ),
                )
            )
        ),
    )


def _dedupe_owners(
    owners: tuple[ResolvedInputApplicationOwner, ...],
) -> tuple[ResolvedInputApplicationOwner, ...]:
    output: dict[tuple[str, str], ResolvedInputApplicationOwner] = {}
    for owner in owners:
        if owner.known_input_id:
            output[(owner.known_input_id, owner.owner_surface_id)] = owner
    return tuple(output.values())


__all__ = [
    "ParsedResolvedInputApplications",
    "ResolvedInputApplicationOwner",
    "ResolvedInputApplicationSurface",
    "ResolvedInputApplicationTargetKind",
    "empty_resolved_input_applications_schema",
    "parse_resolved_input_applications",
    "resolved_input_application_surface",
    "resolved_input_application_surfaces",
]
