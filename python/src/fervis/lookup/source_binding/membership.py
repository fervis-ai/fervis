"""Ordinary source membership, authored without question qualifications."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from fervis.lookup.question_contract import RawDataRecord
from fervis.host_api.contracts import ParameterSemantics
from fervis.lookup.available_sources import SourceChoiceSurfaceKind
from fervis.lookup.source_binding.model import SourceRealization
from fervis.lookup.source_binding.subject_obligations import (
    NORMAL_INSTANCE_EXCLUDED_STATE_ROLES,
)
from fervis.lookup.turn_prompts import (
    ProviderResponseContract,
    ProviderToolContract,
    TurnPromptBase,
)
from fervis.model_io.structured_output.specs import required_tool_spec

TOOL_NAME = "submit_source_membership"


@dataclass(frozen=True)
class MembershipScope:
    owner_set_ref: str
    source_ref: str
    subject_kind: str
    surfaces: tuple[Any, ...]


def membership_scopes(realization: SourceRealization):
    from fervis.lookup.question_contract import FactLocalRef
    from fervis.lookup.relation_catalog.row_sources.model import RowSourceIdentityKind
    from fervis.lookup.source_binding.occurrences import occurrence_scope

    request = realization.request
    result = {}
    for branch in request.strategy.branches:
        scopes = []
        for occurrence in occurrence_scope(
            request, realization, branch.branch_id
        ).occurrences:
            source_ref = occurrence.source_ref
            bindings = tuple(
                (ref, value)
                for ref, values in realization.set_bindings.items()
                for value in values
                if ref in occurrence.set_refs and value.branch_id == branch.branch_id
            )
            row_owners = tuple(
                (ref, value)
                for ref, value in bindings
                if value.identity_ref is None
                or request.source_catalog.identity(value.identity_ref).kind
                is RowSourceIdentityKind.ENTITY_ROW
            )
            # Co-resident references do not own the carrier's row-state fields.
            owners = row_owners or bindings[:1]
            surfaces = tuple(
                surface
                for surface in request.source_catalog.choice_surfaces
                if surface.source_ref == source_ref and not surface.declared_entity_kind
                and not (surface.kind is SourceChoiceSurfaceKind.REQUEST_PARAMETER
                         and any(param.param_ref == surface.target_ref and param.semantics is ParameterSemantics.RESPONSE_SHAPE
                                 for param in request.source_catalog.source(source_ref).params))
            )
            for ref, value in owners:
                if not surfaces or (
                    isinstance(request.index.subject_obligation, RawDataRecord)
                    and ref == request.index.subject_obligation.subject_set_ref.token
                ):
                    continue
                kind = request.index.term_by_ref[
                    FactLocalRef.from_token(ref)
                ].origin.meaning
                if not row_owners:
                    source = request.source_catalog.source(source_ref)
                    kinds = tuple(
                        dict.fromkeys(
                            key.entity_kind
                            for key in source.candidate_keys
                            if key.primary
                        )
                    )
                    kind = ", ".join(kinds) or source.label
                scopes.append(MembershipScope(ref, source_ref, kind, surfaces))
        result[branch.branch_id] = tuple(scopes)
    return result


def membership_surfaces(realization: SourceRealization):
    return {
        branch: tuple(
            {
                surface.surface_ref: surface
                for scope in scopes
                for surface in scope.surfaces
            }.values()
        )
        for branch, scopes in membership_scopes(realization).items()
    }


def requirement_choice_surfaces(request, branch_id):
    branch = next(b for b in request.strategy.branches if b.branch_id == branch_id)
    return tuple(
        surface
        for surface in request.source_catalog.choice_surfaces
        if surface.source_ref in branch.source_refs
        and (branch_id, surface.surface_ref)
        not in request.unrestricted_parameter_surfaces
        and any(
            request.explicit_subject_requirement_refs(choice, branch_id=branch_id)
            for choice in surface.values
        )
    )


def _object(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def membership_schema(realization: SourceRealization):
    text = {"type": "string", "minLength": 1}
    return _object(
        {
            branch: _object(
                {
                    scope.owner_set_ref: _object(
                        {
                            surface.surface_ref: _object(
                                {
                                    "surface_mapping_basis": text,
                                    "choice_reviews": _object(
                                        {
                                            choice.value: _object(
                                                {
                                                    "choice_domain_meaning": text,
                                                    "decision_basis": text,
                                                    "baseline_decision": {
                                                        "enum": ["INCLUDE", "EXCLUDE"]
                                                    },
                                                }
                                            )
                                            for choice in surface.values
                                        }
                                    ),
                                }
                            )
                            for surface in scope.surfaces
                        }
                    )
                    for scope in scopes
                }
            )
            for branch, scopes in membership_scopes(realization).items()
        }
    )


@dataclass(frozen=True)
class SourceMembership:
    realization: SourceRealization
    reviews: dict[str, Any]


def parse_source_membership(
    payload: dict[str, Any], *, realization: SourceRealization
) -> SourceMembership:
    from jsonschema import Draft7Validator

    errors = tuple(Draft7Validator(membership_schema(realization)).iter_errors(payload))
    if errors:
        raise ValueError(
            "source membership violates its declared schema: " + errors[0].message
        )
    from fervis.lookup.available_sources import SourceChoiceSurfaceKind

    unrestricted = []
    scopes_by_branch = membership_scopes(realization)
    for branch in realization.request.strategy.branches:
        for surface in realization.request.source_catalog.choice_surfaces:
            if (
                surface.source_ref not in branch.source_refs
                or surface.kind is not SourceChoiceSurfaceKind.REQUEST_PARAMETER
            ):
                continue
            owners = tuple(
                scope
                for scope in scopes_by_branch[branch.branch_id]
                if surface in scope.surfaces
            )
            if all(
                choice["baseline_decision"] == "INCLUDE"
                for scope in owners
                for choice in payload[branch.branch_id][scope.owner_set_ref][
                    surface.surface_ref
                ]["choice_reviews"].values()
            ):
                unrestricted.append((branch.branch_id, surface.surface_ref))
    realization = replace(
        realization,
        request=replace(
            realization.request, unrestricted_parameter_surfaces=tuple(unrestricted)
        ),
    )
    return SourceMembership(realization, payload)


class SourceMembershipTurnPrompt(TurnPromptBase):
    turn_name = "source membership"
    turn_task = "classify ordinary instances in each logical row scope"
    include_current_question = False

    def __init__(self, realization: SourceRealization):
        self.realization = realization

    def system_prompt(self, context):
        return "Classify source-contract choices using only each scope's declared subject kind, source descriptions, and ordinary-instance definitions. Do not invent business rules."

    def data_sections(self, builder):
        request = self.realization.request
        return (
            builder.json_section(
                "Ordinary membership authority:",
                {
                    "excluded_states": [
                        {"role": role.role.value, "definition": role.definition}
                        for role in NORMAL_INSTANCE_EXCLUDED_STATE_ROLES
                    ],
                    "branches": {
                        branch: [
                            {
                                "owner_set_ref": scope.owner_set_ref,
                                "subject_kind": scope.subject_kind,
                                "surfaces": [
                                    {
                                        "surface_ref": surface.surface_ref,
                                        "source_description": request.source_catalog.source(
                                            surface.source_ref
                                        ).description,
                                        "surface_description": surface.description,
                                        "surface_label": surface.label,
                                        "choices": [
                                            {
                                                "value": choice.value,
                                                "label": choice.label,
                                            }
                                            for choice in surface.values
                                        ],
                                    }
                                    for surface in scope.surfaces
                                ],
                            }
                            for scope in scopes
                        ]
                        for branch, scopes in membership_scopes(
                            self.realization
                        ).items()
                    },
                },
                indent=2,
            ),
        )

    def instruction_sections(self, builder):
        return (
            builder.instruction_block(
                "Membership decisions",
                (
                    "Review every shown choice against its own scope’s subject kind and the excluded-state definitions. A related row is not excluded merely because its kind differs from the main answer subject. Decisions in one scope do not restrict another scope.",
                    "Write what the choice means, explain membership, then select INCLUDE or EXCLUDE.",
                    "Classify the rows admitted by the choice, not the option itself. A retrieval scope admitting deleted, superseded, or other excluded artifacts is EXCLUDE even when it also admits ordinary rows.",
                    "INCLUDE admits ordinary instances of this subject. EXCLUDE admits excluded states or artifacts, a mixed scope containing excluded artifacts, or a different subject kind.",
                    "Source descriptions determine whether a state is an effective subject instance. A persisted record does not establish realization when the source explicitly says it is not realized. Do not exclude an ordinary instance merely for lacking optional validation or presentation properties.",
                    "Request parameters and returned fields expressing the same population property have the same membership decisions.",
                    "Return exactly one submit_source_membership tool call.",
                ),
            ),
        )

    def response_contract(self):
        return ProviderResponseContract(
            provider_schema={TOOL_NAME: membership_schema(self.realization)}
        )

    def tool_contract(self):
        return ProviderToolContract(
            tool_specs=(
                required_tool_spec(
                    tool_name=TOOL_NAME,
                    tool_description="Submit ordinary source membership.",
                    input_schema=membership_schema(self.realization),
                ),
            )
        )
