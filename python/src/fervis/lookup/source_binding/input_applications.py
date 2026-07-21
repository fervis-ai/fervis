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
    LiteralType,
    LiteralValuePayload,
)
from fervis.lookup.answer_program.operations import PredicateOperator
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
    FieldEvidence,
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
    GroupKeyDomainKind,
    RequestedFactAnswerPopulationMembershipTest,
    RequestedFactGroupKey,
)
from fervis.lookup.source_binding.population_effects import (
    population_coverage_claims,
    population_test_basis_payload,
    population_test_results_schema,
)
from fervis.lookup.source_binding.membership_tests import membership_test_key


class ResolvedInputApplicationTargetKind(StrEnum):
    REQUEST_PARAMETER = "request_parameter"
    RETURNED_IDENTITY = "returned_identity"
    RETURNED_FIELD = "returned_field"


@dataclass(frozen=True)
class ResolvedInputApplicationOwner:
    known_input_id: str
    owner_surface_id: str
    proof_refs: tuple[str, ...]


@dataclass(frozen=True)
class _ResolvedValueSurface:
    value_id: str
    value: FactValue
    components_by_target_kind: dict[str, tuple[str, ...]]

    def prompt_payload(
        self,
        *,
        membership_tests: tuple[RequestedFactAnswerPopulationMembershipTest, ...],
        role_text: str,
        parameter_alternative_group: RequestedFactGroupKey | None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "value_id": self.value_id,
            "kind": self.value.kind.value,
            "value_meaning": self.value.label or self.value.id,
            "components_by_target_kind": {
                kind: list(components)
                for kind, components in self.components_by_target_kind.items()
            },
            "population_test_basis": population_test_basis_payload(
                membership_tests,
                role_text=role_text,
            ),
        }
        if isinstance(self.value.payload, LiteralValuePayload):
            payload.update(
                literal_type=self.value.payload.literal_type.value,
                literal_value=self.value.payload.value,
            )
        if (
            parameter_alternative_group is not None
            and self.value.known_input_id
            in parameter_alternative_group.question_input_refs
        ):
            payload["request_parameter_alternative_group"] = (
                parameter_alternative_group.id
            )
        return payload


@dataclass(frozen=True)
class _ApplicationTarget:
    id: str
    kind: str
    target_id: str
    application_value: str = ""


@dataclass(frozen=True)
class ParsedResolvedInputApplications:
    param_binding_sets: ParamBindingSetAlternatives = ((),)
    applied_filters: tuple[SourceAppliedFilter, ...] = ()
    known_input_ids: tuple[str, ...] = ()
    population_coverage_claims: tuple[PopulationCoverageClaim, ...] = ()


@dataclass(frozen=True)
class _ParameterApplication:
    output: provider_output.ResolvedInputTargetApplicationOutput
    param: CandidateParameter
    value: FactValue
    application_value: str = ""


@dataclass(frozen=True)
class _ReturnedFieldApplication:
    output: provider_output.ResolvedInputTargetApplicationOutput
    field: FieldEvidence
    value: FactValue
    application_value: str = ""


@dataclass(frozen=True)
class ResolvedInputApplicationSurface:
    requested_fact_id: str
    values_by_id: dict[str, FactValue]
    parameter_targets_by_id: dict[str, CandidateParameter]
    identity_targets_by_id: dict[str, EntityEvidence]
    returned_field_targets_by_id: dict[str, FieldEvidence]
    application_values_by_id: dict[str, tuple[str, str, str]]
    membership_tests_by_input_id: dict[
        str, tuple[RequestedFactAnswerPopulationMembershipTest, ...]
    ]
    parameter_alternative_group: RequestedFactGroupKey | None
    coverage_role: PopulationCoverageRole | None
    role_text: str

    def prompt_payload(self) -> dict[str, object]:
        return {
            "resolved_values": list(self._resolved_value_payloads()),
            "predicate_requirements": list(self._predicate_requirement_payloads()),
            "targets_by_kind": self._targets_by_kind(),
        }

    def provider_schema(self) -> dict[str, object]:
        value_surfaces = self._resolved_value_surfaces()
        if not value_surfaces or not self._application_targets():
            return empty_resolved_input_applications_schema()
        return {
            "type": "array",
            "items": {
                "oneOf": [
                    provider_output.ResolvedInputApplicationOutput.schema(
                        {
                            "value_id": {"enum": [surface.value_id]},
                            "applications": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": self._max_value_application_count(
                                    surface
                                ),
                                "items": self._value_application_schema(surface),
                            },
                            "population_test_results": (
                                _bounded_population_test_results_schema(
                                    tuple(
                                        membership_test_key(test)
                                        for test in self.membership_tests_by_input_id.get(
                                            surface.value.known_input_id,
                                            (),
                                        )
                                    )
                                )
                            ),
                        }
                    )
                    for surface in value_surfaces
                ]
            },
            "maxItems": len(value_surfaces),
        }

    def _value_application_schema(
        self,
        surface: _ResolvedValueSurface,
    ) -> dict[str, object]:
        components = tuple(
            component
            for values in surface.components_by_target_kind.values()
            for component in values
        )
        compatible_target_ids = {
            (target_kind, target_id)
            for target_kind, value_components in (
                surface.components_by_target_kind.items()
            )
            for value_component in value_components
            for target_id in self._application_target_ids(
                value_id=surface.value_id,
                value_component=value_component,
                target_kind=target_kind,
            )
        }
        application_target_ids = tuple(
            target.id
            for target in self._application_targets()
            if (target.kind, target.target_id) in compatible_target_ids
        )
        return provider_output.ResolvedInputTargetApplicationOutput.schema(
            {
                "application_target_id": {
                    "enum": list(application_target_ids)
                },
                "value_component": {"enum": list(dict.fromkeys(components))},
                "match_basis_explanation": {"type": "string", "minLength": 1},
            }
        )

    def _max_value_application_count(
        self,
        surface: _ResolvedValueSurface,
    ) -> int:
        return sum(
            len(
                self._application_target_ids(
                    value_id=surface.value_id,
                    value_component=value_component,
                    target_kind=target_kind,
                )
            )
            for target_kind, components in surface.components_by_target_kind.items()
            for value_component in components
        )

    def _resolved_value_payloads(self) -> tuple[dict[str, object], ...]:
        return tuple(
            surface.prompt_payload(
                membership_tests=self.membership_tests_by_input_id.get(
                    surface.value.known_input_id, ()
                ),
                role_text=self.role_text,
                parameter_alternative_group=self.parameter_alternative_group,
            )
            for surface in self._resolved_value_surfaces()
        )

    def _resolved_value_surfaces(self) -> tuple[_ResolvedValueSurface, ...]:
        surfaces: list[_ResolvedValueSurface] = []
        for value_id, value in self.values_by_id.items():
            parameter_components = tuple(
                dict.fromkeys(
                    component
                    for param in self.parameter_targets_by_id.values()
                    for component in self._parameter_value_components(
                        param,
                        value=value,
                    )
                )
            )
            has_identity_target = any(
                _identity_target_accepts_value(evidence, value=value)
                for evidence in self.identity_targets_by_id.values()
            )
            components_by_target_kind: dict[str, tuple[str, ...]] = {}
            if parameter_components:
                components_by_target_kind[
                    ResolvedInputApplicationTargetKind.REQUEST_PARAMETER.value
                ] = parameter_components
            if has_identity_target:
                components_by_target_kind[
                    ResolvedInputApplicationTargetKind.RETURNED_IDENTITY.value
                ] = ("canonical_key",)
            if any(
                _returned_field_accepts_value(field, value=value)
                for field in self.returned_field_targets_by_id.values()
            ):
                components_by_target_kind[
                    ResolvedInputApplicationTargetKind.RETURNED_FIELD.value
                ] = ("value",)
            if not components_by_target_kind:
                continue
            surfaces.append(
                _ResolvedValueSurface(
                    value_id=value_id,
                    value=value,
                    components_by_target_kind=components_by_target_kind,
                )
            )
        return tuple(surfaces)

    def _predicate_requirement_payloads(self) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "test_id": test.id,
                "test_question": test.test_question,
                "value_id": value.id,
                "value_meaning": value.label,
                "operator": _predicate_operator(test).value,
            }
            for value in self.values_by_id.values()
            for test in self.membership_tests_by_input_id.get(
                value.known_input_id, ()
            )
            if isinstance(value.payload, LiteralValuePayload)
        )

    def _targets_by_kind(self) -> dict[str, list[dict[str, object]]]:
        return {
            ResolvedInputApplicationTargetKind.REQUEST_PARAMETER.value: [
                {
                    "target_id": param.id,
                    **self._application_target_payload(
                        target_kind=(
                            ResolvedInputApplicationTargetKind.REQUEST_PARAMETER.value
                        ),
                        target_id=param.id,
                    ),
                    "type": param.type,
                    **({"description": param.description} if param.description else {}),
                    **({"semantics": param.semantics} if param.semantics else {}),
                    **({"choices": list(param.choices)} if param.choices else {}),
                }
                for param in self._visible_parameter_targets().values()
            ],
            ResolvedInputApplicationTargetKind.RETURNED_IDENTITY.value: [
                {
                    "target_id": target_id,
                    **self._application_target_payload(
                        target_kind=(
                            ResolvedInputApplicationTargetKind.RETURNED_IDENTITY.value
                        ),
                        target_id=target_id,
                    ),
                }
                for target_id in self.identity_targets_by_id
            ],
            ResolvedInputApplicationTargetKind.RETURNED_FIELD.value: [
                {
                    "target_id": target_id,
                    **self._application_target_payload(
                        target_kind=(
                            ResolvedInputApplicationTargetKind.RETURNED_FIELD.value
                        ),
                        target_id=target_id,
                    ),
                    "field_id": field.field_id,
                    "type": field.type,
                    **({"label": field.label} if field.label else {}),
                    **(
                        {"description": field.description}
                        if field.description
                        else {}
                    ),
                }
                for target_id, field in self.returned_field_targets_by_id.items()
            ],
        }

    def _application_target_payload(
        self,
        *,
        target_kind: str,
        target_id: str,
    ) -> dict[str, object]:
        targets = tuple(
            target
            for target in self._application_targets()
            if target.kind == target_kind and target.target_id == target_id
        )
        if len(targets) == 1 and not targets[0].application_value:
            return {"application_target_id": targets[0].id}
        return {
            "application_targets": [
                {
                    "application_target_id": target.id,
                    "value": target.application_value,
                }
                for target in targets
            ]
        }

    def _target_ids_for_kind(self, target_kind: str) -> tuple[str, ...]:
        if target_kind == ResolvedInputApplicationTargetKind.REQUEST_PARAMETER.value:
            return tuple(self._visible_parameter_targets())
        if target_kind == ResolvedInputApplicationTargetKind.RETURNED_IDENTITY.value:
            return tuple(self.identity_targets_by_id)
        if target_kind == ResolvedInputApplicationTargetKind.RETURNED_FIELD.value:
            return tuple(self.returned_field_targets_by_id)
        return ()

    def _application_targets(self) -> tuple[_ApplicationTarget, ...]:
        output: list[_ApplicationTarget] = []
        for target_kind in ResolvedInputApplicationTargetKind:
            for target_id in self._target_ids_for_kind(target_kind.value):
                choices = tuple(
                    _ApplicationTarget(
                        id=application_value_id,
                        kind=target_kind.value,
                        target_id=target_id,
                        application_value=value,
                    )
                    for application_value_id, (kind, choice_target_id, value) in (
                        self.application_values_by_id.items()
                    )
                    if (kind, choice_target_id) == (target_kind.value, target_id)
                )
                if choices:
                    output.extend(choices)
                else:
                    output.append(
                        _ApplicationTarget(
                            id=f"{target_kind.value}.{target_id}",
                            kind=target_kind.value,
                            target_id=target_id,
                        )
                    )
        return tuple(output)

    def application_target(
        self,
        application_target_id: str,
    ) -> _ApplicationTarget | None:
        return next(
            (
                target
                for target in self._application_targets()
                if target.id == application_target_id
            ),
            None,
        )

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
                if value_component
                in self._parameter_value_components(param, value=value)
            ]
        if target_kind != ResolvedInputApplicationTargetKind.RETURNED_IDENTITY.value:
            if target_kind != ResolvedInputApplicationTargetKind.RETURNED_FIELD.value:
                return []
            return [
                target_id
                for target_id, field in self.returned_field_targets_by_id.items()
                if value_component == "value"
                and _returned_field_accepts_value(field, value=value)
            ]
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

    def owners(self) -> tuple[ResolvedInputApplicationOwner, ...]:
        owners: list[ResolvedInputApplicationOwner] = []
        for param in self._visible_parameter_targets().values():
            for value in self.values_by_id.values():
                if not self._parameter_value_components(param, value=value):
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

    def _visible_parameter_targets(self) -> dict[str, CandidateParameter]:
        return {
            param_id: param
            for param_id, param in self.parameter_targets_by_id.items()
            if any(
                self._parameter_value_components(param, value=value)
                for value in self.values_by_id.values()
            )
        }

    def _parameter_value_components(
        self,
        param: CandidateParameter,
        *,
        value: FactValue,
    ) -> tuple[str, ...]:
        components = _parameter_value_components(param, value=value)
        if (
            not components
            or param.required
            or param.entity_target is not None
            or not isinstance(
                value.payload,
                (IdentityValuePayload, IdentitySetValuePayload),
            )
        ):
            return components
        if any(
            _identity_target_accepts_value(evidence, value=value)
            for evidence in self.identity_targets_by_id.values()
        ):
            return ()
        return components


def resolved_input_application_surface(
    *,
    candidate: SourceCandidate,
    requested_fact_id: str,
    resolved_values: tuple[FactValue, ...],
    membership_tests: tuple[RequestedFactAnswerPopulationMembershipTest, ...] = (),
    parameter_alternative_group: RequestedFactGroupKey | None = None,
    coverage_role: PopulationCoverageRole | None = None,
    role_text: str = "source rows",
) -> ResolvedInputApplicationSurface:
    """Describe independently selectable values and invocation-local targets."""

    values_by_id = {value.id: value for value in resolved_values}
    if parameter_alternative_group is not None and (
        parameter_alternative_group.domain
        is not GroupKeyDomainKind.SPECIFIED_QUESTION_INPUTS
        or len(parameter_alternative_group.question_input_refs) < 2
    ):
        raise ValueError("parameter alternative group requires multiple fact inputs")
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
    returned_field_targets = {
        evidence.evidence_id: evidence
        for evidence in candidate.evidence_items
        if isinstance(evidence, FieldEvidence)
        if any(
            _returned_field_accepts_value(evidence, value=value)
            and _membership_tests_by_input_id(membership_tests).get(
                value.known_input_id
            )
            for value in resolved_values
        )
    }
    application_values = _application_values(
        candidate,
        parameter_targets=parameter_targets,
        returned_field_targets=returned_field_targets,
    )
    return ResolvedInputApplicationSurface(
        requested_fact_id=requested_fact_id,
        values_by_id=values_by_id,
        parameter_targets_by_id=parameter_targets,
        identity_targets_by_id=identity_targets,
        returned_field_targets_by_id=returned_field_targets,
        application_values_by_id=application_values,
        membership_tests_by_input_id=_membership_tests_by_input_id(
            membership_tests if coverage_role is not None else (),
        ),
        parameter_alternative_group=parameter_alternative_group,
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


def _bounded_population_test_results_schema(
    test_ids: tuple[str, ...],
) -> dict[str, object]:
    schema = population_test_results_schema(test_ids)
    schema["required"] = []
    return schema


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
        specified_group_key = fact.specified_group_key()
        parameter_alternative_group = (
            specified_group_key
            if specified_group_key is not None
            and len(specified_group_key.question_input_refs) > 1
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
            parameter_alternative_group=parameter_alternative_group,
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
        parameter_alternative_group = surface.parameter_alternative_group
        if parameter_alternative_group is not None and not set(
            parameter_alternative_group.question_input_refs
        ).issubset(
            value.known_input_id for value in model_visible_values.values()
        ):
            parameter_alternative_group = None
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
            returned_field_targets_by_id={
                evidence_id: evidence
                for evidence_id, evidence in (
                    surface.returned_field_targets_by_id.items()
                )
                if any(
                    _returned_field_accepts_value(evidence, value=value)
                    for value in model_visible_values.values()
                )
            },
            application_values_by_id={
                application_value_id: item
                for application_value_id, item in (
                    surface.application_values_by_id.items()
                )
                if (
                    item[0]
                    != ResolvedInputApplicationTargetKind.REQUEST_PARAMETER.value
                    or item[1] not in backend_owned_param_ids
                )
            },
            membership_tests_by_input_id=surface.membership_tests_by_input_id,
            parameter_alternative_group=parameter_alternative_group,
            coverage_role=surface.coverage_role,
            role_text=surface.role_text,
        )
    return output


def _application_values(
    candidate: SourceCandidate,
    *,
    parameter_targets: dict[str, CandidateParameter],
    returned_field_targets: dict[str, FieldEvidence],
) -> dict[str, tuple[str, str, str]]:
    choices_by_field_id = {
        predicate.field_id: predicate.allowed_values
        for predicate in candidate.row_predicates
        if predicate.allowed_values
    }
    output: dict[str, tuple[str, str, str]] = {}
    for target_kind, targets in (
        (
            ResolvedInputApplicationTargetKind.REQUEST_PARAMETER.value,
            ((target_id, param.choices) for target_id, param in parameter_targets.items()),
        ),
        (
            ResolvedInputApplicationTargetKind.RETURNED_FIELD.value,
            (
                (target_id, choices_by_field_id.get(field.field_id, ()))
                for target_id, field in returned_field_targets.items()
            ),
        ),
    ):
        for target_id, choices in targets:
            for index, value in enumerate(choices, start=1):
                application_value_id = f"{target_kind}.{target_id}.choice_{index}"
                output[application_value_id] = (target_kind, target_id, value)
    return output


def parse_resolved_input_applications(
    applications: tuple[provider_output.ResolvedInputApplicationOutput, ...],
    *,
    surface: ResolvedInputApplicationSurface,
) -> ParsedResolvedInputApplications:
    param_binding_groups: list[ParamBindingSetAlternatives] = []
    parameter_applications_by_target: dict[str, list[_ParameterApplication]] = {}
    field_applications_by_target: dict[str, list[_ReturnedFieldApplication]] = {}
    applied_filters: list[SourceAppliedFilter] = []
    known_input_ids: list[str] = []
    population_coverage_claims: list[PopulationCoverageClaim] = []
    used_identity_targets: set[str] = set()
    used_value_ids: set[str] = set()
    for application_group in applications:
        if application_group.value_id in used_value_ids:
            raise ValueError("resolved input application repeats a value")
        used_value_ids.add(application_group.value_id)
        value = surface.values_by_id.get(application_group.value_id)
        if value is None:
            raise ValueError("resolved input application references unknown value")
        if not application_group.applications:
            raise ValueError("resolved input application requires a target")
        if value.known_input_id:
            known_input_ids.append(value.known_input_id)
        application_proof_refs: list[str] = list(value.proof_refs)
        for application in application_group.applications:
            if not application.match_basis_explanation.strip():
                raise ValueError("resolved input application requires a match basis")
            application_target = surface.application_target(
                application.application_target_id
            )
            if application_target is None:
                raise ValueError(
                    "resolved input application references unknown application target"
                )
            target_kind = application_target.kind
            target_id = application_target.target_id
            if not surface.accepts_application(
                target_kind=target_kind,
                target_id=target_id,
                value_id=application_group.value_id,
                value_component=application.value_component,
            ):
                raise ValueError(
                    "resolved value is incompatible with application target"
                )
            if target_kind == ResolvedInputApplicationTargetKind.REQUEST_PARAMETER.value:
                param = surface.parameter_targets_by_id.get(target_id)
                if param is None:
                    raise ValueError(
                        "resolved input application references unknown param"
                    )
                parameter_applications_by_target.setdefault(param.id, []).append(
                    _ParameterApplication(
                        output=application,
                        param=param,
                        value=value,
                        application_value=application_target.application_value,
                    )
                )
                application_proof_refs.append(f"source_param:{param.id}")
                continue
            if target_kind == ResolvedInputApplicationTargetKind.RETURNED_IDENTITY.value:
                if _is_specified_group_alternative(value, surface=surface):
                    raise ValueError(
                        "grouped input alternatives require a request parameter target"
                    )
                if target_id in used_identity_targets:
                    raise ValueError("resolved input application repeats a target")
                used_identity_targets.add(target_id)
                evidence = surface.identity_targets_by_id.get(target_id)
                if evidence is None:
                    raise ValueError(
                        "resolved input application references unknown identity target"
                    )
                applied_filters.extend(
                    _identity_filters(
                        application,
                        target_id=target_id,
                        evidence=evidence,
                        value=value,
                    )
                )
                application_proof_refs.append(f"returned_identity:{target_id}")
                continue
            if target_kind == ResolvedInputApplicationTargetKind.RETURNED_FIELD.value:
                field = surface.returned_field_targets_by_id.get(target_id)
                if field is None:
                    raise ValueError(
                        "resolved input application references unknown returned field"
                    )
                field_applications_by_target.setdefault(
                    field.evidence_id, []
                ).append(
                    _ReturnedFieldApplication(
                        output=application,
                        field=field,
                        value=value,
                        application_value=application_target.application_value,
                    )
                )
                application_proof_refs.append(
                    f"returned_field:{field.evidence_id}"
                )
                continue
            raise ValueError("resolved input application has unsupported target kind")
        population_coverage_claims.extend(
            _application_population_coverage_claims(
                application_group.population_test_results,
                value=value,
                surface=surface,
                proof_refs=tuple(dict.fromkeys(application_proof_refs)),
            )
        )
    for parameter_applications in parameter_applications_by_target.values():
        binding_sets = _parameter_application_group(
            tuple(parameter_applications),
            surface=surface,
        )
        param_binding_groups.append(binding_sets)
    for field_applications in field_applications_by_target.values():
        filters = _returned_field_application_group(
            tuple(field_applications),
            surface=surface,
        )
        applied_filters.extend(filters)
    return ParsedResolvedInputApplications(
        param_binding_sets=combine_param_binding_sets(param_binding_groups),
        applied_filters=tuple(applied_filters),
        known_input_ids=tuple(dict.fromkeys(known_input_ids)),
        population_coverage_claims=tuple(population_coverage_claims),
    )


def _returned_field_application_group(
    applications: tuple[_ReturnedFieldApplication, ...],
    *,
    surface: ResolvedInputApplicationSurface,
) -> tuple[SourceAppliedFilter, ...]:
    tests = _shared_application_tests(
        tuple(application.value for application in applications),
        surface=surface,
    )
    if len(tests) != 1:
        raise ValueError(
            "returned-field predicate application requires one fact predicate"
        )
    test = tests[0]
    operator = _predicate_operator(test)
    proof_refs = tuple(
        dict.fromkeys(
            (
                *(ref for application in applications for ref in application.value.proof_refs),
                f"returned_field:{applications[0].field.evidence_id}",
            )
        )
    )
    concrete_values = tuple(
        application.application_value
        or str(application.value.payload.component_value(ValueComponent.VALUE))
        for application in applications
    )
    if len(applications) > 1 and operator is PredicateOperator.EQUALS:
        return (
            SourceAppliedFilter(
                    known_input_id="",
                    predicate_field_ids=(applications[0].field.field_id,),
                    value_id=f"predicate_set.{test.id}",
                    value_kind="string_set",
                    display_value=test.test_question,
                    operator=PredicateOperator.IN.value,
                    application_values=tuple(dict.fromkeys(concrete_values)),
                    proof_refs=proof_refs,
            ),
        )
    if len(applications) > 1 and operator is not PredicateOperator.NOT_EQUALS:
        raise ValueError("ordered predicate cannot have alternative boundaries")
    return tuple(
            SourceAppliedFilter(
                known_input_id=application.value.known_input_id,
                predicate_field_ids=(application.field.field_id,),
                value_id=application.value.id,
                value_kind=application.value.kind.value,
                value_component="value",
                display_value=application.value.label,
                literal_type=(
                    application.value.payload.literal_type.value
                    if isinstance(application.value.payload, LiteralValuePayload)
                    else ""
                ),
                operator=operator.value,
                application_value=application.application_value,
                proof_refs=proof_refs,
            )
            for application in applications
    )


def _parameter_application_group(
    applications: tuple[_ParameterApplication, ...],
    *,
    surface: ResolvedInputApplicationSurface,
) -> ParamBindingSetAlternatives:
    if len(applications) == 1:
        application = applications[0]
        if _is_specified_group_alternative(application.value, surface=surface):
            raise ValueError("parameter alternatives must apply together")
        return _parameter_binding_sets(
            application.output,
            param=application.param,
            value=application.value,
            application_value=application.application_value,
        )
    _require_parameter_alternative_group(applications, surface=surface)
    return tuple(
            binding_set
            for application in applications
            for binding_set in _parameter_binding_sets(
                application.output,
                param=application.param,
                value=application.value,
                application_value=application.application_value,
            )
    )


def _require_parameter_alternative_group(
    applications: tuple[_ParameterApplication, ...],
    *,
    surface: ResolvedInputApplicationSurface,
) -> None:
    alternative_group = surface.parameter_alternative_group
    values = tuple(application.value for application in applications)
    input_ids = tuple(value.known_input_id for value in values)
    if alternative_group is None:
        tests = _shared_application_tests(values, surface=surface)
        if (
            len(tests) == 1
            and len(values) > 1
            and _predicate_operator(tests[0]) is PredicateOperator.EQUALS
            and frozenset(input_ids)
            == frozenset(tests[0].owned_question_input_refs)
            and len(set(input_ids)) == len(input_ids)
        ):
            return
        raise ValueError("resolved input application repeats a target")
    if (
        any(not isinstance(value.payload, IdentityValuePayload) for value in values)
        or len(set(input_ids)) != len(input_ids)
        or frozenset(input_ids)
        != frozenset(alternative_group.question_input_refs)
        or len({value.id for value in values}) != len(values)
        or len({application.output.value_component for application in applications})
        != 1
    ):
        raise ValueError("repeated parameter target requires proven input alternatives")


def _is_specified_group_alternative(
    value: FactValue,
    *,
    surface: ResolvedInputApplicationSurface,
) -> bool:
    alternative_group = surface.parameter_alternative_group
    return bool(
        alternative_group is not None
        and value.known_input_id in alternative_group.question_input_refs
    )


def _shared_application_tests(
    values: tuple[FactValue, ...],
    *,
    surface: ResolvedInputApplicationSurface,
) -> tuple[RequestedFactAnswerPopulationMembershipTest, ...]:
    if not values:
        return ()
    shared = set(
        surface.membership_tests_by_input_id.get(values[0].known_input_id, ())
    )
    for value in values[1:]:
        shared &= set(
            surface.membership_tests_by_input_id.get(value.known_input_id, ())
        )
    return tuple(
        test
        for test in surface.membership_tests_by_input_id.get(
            values[0].known_input_id, ()
        )
        if test in shared
    )


def _parameter_application_proof_refs(
    applications: tuple[_ParameterApplication, ...],
    *,
    surface: ResolvedInputApplicationSurface,
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


def _parameter_binding_sets(
    application: provider_output.ResolvedInputTargetApplicationOutput,
    *,
    param: CandidateParameter,
    value: FactValue,
    application_value: str = "",
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
    concrete_value = application_value or _parameter_component_value(
            param,
            value=value,
            component=selected_component,
        )
    return parameter_binding_sets(
        param_id=param.id,
        value=concrete_value,
        param=param,
        origin_kind=(
            RelationInputOrigin.CONTEXT_CONSTANT
            if application_value
            else RelationInputOrigin.QUESTION_INPUT
        ),
        value_id="" if application_value else value.id,
        value_component="value" if application_value else compiler_component,
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
        components = tuple(dict.fromkeys(binding_components))
        if components:
            return components
        if isinstance(payload, LiteralValuePayload) and _literal_accepts_catalog_type(
            payload,
            type_name=param.type,
            has_choices=bool(param.choices),
        ):
            return ("value",)
        return ()
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


def _literal_accepts_catalog_type(
    payload: LiteralValuePayload,
    *,
    type_name: str,
    has_choices: bool = False,
) -> bool:
    normalized = type_name.strip().casefold()
    if has_choices or normalized == "choice":
        return payload.literal_type is LiteralType.STRING
    if normalized in {"integer", "number", "decimal", "float", "double"}:
        if payload.literal_type is not LiteralType.NUMBER:
            return False
        return normalized != "integer" or Decimal(payload.value) == Decimal(
            payload.value
        ).to_integral_value()
    if normalized in {"string", "path", "pk", ""}:
        return payload.literal_type is LiteralType.STRING
    if normalized == "boolean":
        return payload.literal_type is LiteralType.BOOLEAN
    return False


def _returned_field_accepts_value(
    field: FieldEvidence,
    *,
    value: FactValue,
) -> bool:
    return (
        isinstance(value.payload, LiteralValuePayload)
        and not field.presentation_only
        and bool(field.field_id)
        and _literal_accepts_catalog_type(
            value.payload,
            type_name=field.type,
            has_choices=field.type.strip().casefold() == "choice",
        )
    )


def _predicate_operator(
    test: RequestedFactAnswerPopulationMembershipTest,
) -> PredicateOperator:
    if test.comparison_operator is not None:
        return test.comparison_operator
    return (
        PredicateOperator.NOT_EQUALS
        if test.polarity.value == "MUST_FAIL"
        else PredicateOperator.EQUALS
    )


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
    application: provider_output.ResolvedInputTargetApplicationOutput,
    *,
    target_id: str,
    evidence: EntityEvidence,
    value: FactValue,
) -> tuple[SourceAppliedFilter, ...]:
    if application.value_component != "canonical_key":
        raise ValueError("returned identity requires the complete canonical key")
    payload = value.payload
    assert isinstance(payload, IdentityValuePayload)
    proof_refs = tuple(
        dict.fromkeys(
            (*value.proof_refs, f"returned_identity:{target_id}")
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
) -> dict[str, tuple[RequestedFactAnswerPopulationMembershipTest, ...]]:
    output: dict[str, list[RequestedFactAnswerPopulationMembershipTest]] = {}
    for test in tests:
        if test.kind is not AnswerPopulationMembershipTestKind.EXPLICIT_USER_CONSTRAINT:
            continue
        for input_id in test.owned_question_input_refs:
            output.setdefault(input_id, []).append(test)
    return {input_id: tuple(items) for input_id, items in output.items()}


def _application_population_coverage_claims(
    population_test_results: dict[
        str, provider_output.RowPredicatePopulationTestResultOutput
    ],
    *,
    value: FactValue,
    surface: ResolvedInputApplicationSurface,
    proof_refs: tuple[str, ...],
) -> tuple[PopulationCoverageClaim, ...]:
    tests = surface.membership_tests_by_input_id.get(value.known_input_id, ())
    if surface.coverage_role is None:
        if population_test_results:
            raise ValueError("resolved input application cannot claim this role")
        return ()
    return population_coverage_claims(
        population_test_results,
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
