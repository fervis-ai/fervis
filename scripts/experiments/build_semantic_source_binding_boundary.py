#!/usr/bin/env python3
"""Build a reusable semantic Source Binding stability boundary."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_SRC = REPO_ROOT / "python" / "src"
EXPERIMENTS = Path(__file__).parent
for path in (PYTHON_SRC, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_semantic_grounding_boundary import identity_catalog  # noqa: E402
from fervis.lookup.answer_program.values import FactValue, LiteralType  # noqa: E402
from fervis.lookup.expression_operators import ExpressionBinaryOperator  # noqa: E402
from fervis.lookup.available_sources import (  # noqa: E402
    build_available_source_catalog,
)
from fervis.lookup.canonical_data import (  # noqa: E402
    EntityKeyComponentValue,
    EntityKeyValue,
)
from fervis.lookup.relation_catalog.row_sources import (  # noqa: E402
    RowSource,
    RowSourceCandidateKey,
    RowSourceEntityReference,
    RowSourceEntityReferenceComponent,
    RowSourceField,
    RowSourceKeyComponent,
    RowSourceKind,
    RowSourceParam,
    RowSourceCatalog,
    RowSourceValueType,
    build_row_source_catalog,
)
from fervis.lookup.grounding import CanonicalInputValue  # noqa: E402
from fervis.lookup.plan_selection.semantic import (  # noqa: E402
    CandidateSourceStrategy,
    SourceAlignment,
    SourceAlignmentAssessment,
    SourceStrategyBranch,
)
from fervis.lookup.question_contract.semantic_parser import (  # noqa: E402
    ParsedSemanticQuestionContract,
)
from fervis.lookup.question_contract.semantic_analysis import (  # noqa: E402
    RequestedFactSemanticIndex,
    analyze_requested_fact,
)
from fervis.lookup.question_contract.semantic_model import (  # noqa: E402
    Aggregate,
    AggregateFunction,
    AllResults,
    AssociationTerm,
    Comparison,
    FactTerm,
    InputDenotation,
    InputDenotationKind,
    InputTerm,
    InstanceInterpretation,
    RequestedFact,
    RequestedOutput,
    SetTerm,
    Subject,
)
from fervis.lookup.read_eligibility.semantic import (  # noqa: E402
    ReadRequirementAssessment,
    SemanticReadDecision,
    SemanticReadEligibilityResult,
)
from fervis.lookup.relation_catalog import EntityKeyComponentTarget  # noqa: E402
from fervis.lookup.source_binding import (  # noqa: E402
    SemanticSourceBindingRequest,
    SemanticSourceBindingTurnPrompt,
)
from fervis.lookup.semantic_types import (  # noqa: E402
    BooleanType,
    SourceOrigin,
    SourceOriginKind,
    TextType,
)
from fervis.lookup.turn_prompts import build_turn_prompt_context  # noqa: E402
from semantic_contract_fixtures import count_contract  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model-key", default="openai:gpt-5.4-mini")
    args = parser.parse_args()
    payload = json.loads(args.request_file.read_text())
    request = build_request(payload)
    invocation = SemanticSourceBindingTurnPrompt(request).to_model_invocation(
        build_turn_prompt_context(
            current_question=str(payload["question"]), conversation_context={}
        )
    )
    boundary = {
        "source_run_id": "semantic-source-binding-experiment",
        "sequence": 1,
        "purpose": "source_binding",
        "provider": args.provider,
        "model_key": args.model_key,
        "system_prompt": invocation.system_prompt,
        "prompt": invocation.prompt_text,
        "tool_specs": [asdict(item) for item in invocation.tool_specs],
        "assertion_context": payload,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(boundary, indent=2) + "\n")
    return 0


def build_request(payload: dict[str, Any]) -> SemanticSourceBindingRequest:
    if payload.get("kind") == "time_binding":
        return _time_binding_request(payload)
    if payload.get("kind") == "required_choice_binding":
        return _time_binding_request(payload, required_choices=True)
    if payload.get("kind") == "scalar_binding":
        return _scalar_binding_request(payload)
    if payload.get("kind") == "finite_choice_binding":
        return _finite_choice_binding_request(payload)
    if payload.get("kind") == "explicit_choice_override":
        return _explicit_choice_override_request(payload)
    if payload.get("kind") == "association_binding":
        return _association_binding_request(
            co_resident=payload.get("expected_association_kind") == "CO_RESIDENT"
        )
    if payload.get("kind") == "identity_set_binding":
        return _identity_set_binding_request(payload)
    resource_type = str(payload["resource_type"])
    key_component = str(payload["key_component"])
    catalog = identity_catalog(
        resource_type=resource_type,
        key_component=key_component,
    )
    all_sources = build_row_source_catalog(catalog)
    row_sources = RowSourceCatalog(
        sources=tuple(
            source
            for source in all_sources.sources
            if source.read_id == str(payload["expected_read"])
        )
    )
    if surface := payload.get("subject_surface"):
        [source] = row_sources.sources
        description = (
            f"Classification of each returned {resource_type.replace('_', ' ')}."
        )
        if surface.get("kind") == "returned_field":
            row_sources = RowSourceCatalog(
                sources=(
                    replace(
                        source,
                        fields=(
                            *source.fields,
                            RowSourceField(
                                id=str(surface["id"]),
                                field_ref=f"field.{surface['id']}",
                                label=str(surface["id"]),
                                type=RowSourceValueType.CHOICE,
                                allowed_roles=(),
                                choices=tuple(str(item) for item in surface["choices"]),
                                description=description,
                            ),
                        ),
                    ),
                )
            )
        else:
            row_sources = RowSourceCatalog(
                sources=(
                    replace(
                        source,
                        params=(
                            *source.params,
                            RowSourceParam(
                                id=str(surface["id"]),
                                param_ref=f"{source.id}.{surface['id']}",
                                name=str(surface["id"]),
                                type=RowSourceValueType.CHOICE,
                                source="query",
                                choices=tuple(str(item) for item in surface["choices"]),
                                description=description,
                            ),
                        ),
                    ),
                )
            )
    parsed = _semantic_contract(payload)
    [index] = parsed.semantic_indexes
    [input_term] = parsed.contract.inputs
    [source] = row_sources.sources
    read_result = SemanticReadEligibilityResult(
        read_assessments=(_retained_assessment(index, source),),
        identity_outcomes=(),
    )
    available = build_available_source_catalog(
        row_sources, read_eligibility=read_result
    )
    [clause] = index.qualification.clauses
    strategy = CandidateSourceStrategy(
        requested_fact_id=index.requested_fact_id,
        source_assessments=(_source_assessment(source, SourceAlignment.DIRECT),),
        strategy_basis="The retained source is direct.",
        branches=(
            SourceStrategyBranch(
                branch_id=f"{index.requested_fact_id}:source_branch:1",
                source_refs=(source.id,),
                relation_evidence_refs=(),
                qualification_clause_refs=(clause.clause_ref,),
            ),
        ),
    )
    [use] = index.input_use_sites
    identity = FactValue.identity(
        id="canonical_identity_1",
        known_input_id=input_term.id,
        key=EntityKeyValue(
            entity_kind=resource_type,
            key_id="primary_key",
            components=(EntityKeyComponentValue(key_component, "member-canonical-1"),),
        ),
        display_value=str(payload["operand"]),
        proof_refs=(f"resolver:{payload['expected_read']}",),
    )
    return SemanticSourceBindingRequest(
        index=index,
        strategy=strategy,
        source_catalog=available,
        canonical_values=(
            CanonicalInputValue(
                canonical_value_id=identity.id,
                input_ref=input_term.id,
                use_refs=(use.use_ref,),
                typed_value=identity,
                certification_refs=(f"resolver:{payload['expected_read']}",),
            ),
        ),
    )


def _time_binding_request(
    payload: dict[str, Any],
    *,
    required_choices: bool = False,
) -> SemanticSourceBindingRequest:
    parsed = _time_semantic_contract(payload)
    source = RowSource(
        id="source_events",
        kind=RowSourceKind.API_READ,
        label="events",
        read_id="list_events",
        endpoint_name="list_events",
        fields=(
            RowSourceField(
                id="event_id",
                field_ref="field.event_id",
                label="event identifier",
                type=RowSourceValueType.STRING,
                allowed_roles=(),
            ),
            RowSourceField(
                id="occurred_on",
                field_ref="field.occurred_on",
                label="event date",
                type=RowSourceValueType.DATE,
                allowed_roles=(),
            ),
        ),
        candidate_keys=(
            RowSourceCandidateKey(
                id="primary_key",
                entity_kind="event",
                components=(RowSourceKeyComponent("event_id", "event_id"),),
                primary=True,
            ),
        ),
        params=(
            RowSourceParam(
                id="start_date",
                param_ref="source_events.start_date",
                name="start_date",
                type=RowSourceValueType.DATE,
                source="query",
                required=True,
                description="Inclusive start date of the requested event interval.",
            ),
            RowSourceParam(
                id="end_date",
                param_ref="source_events.end_date",
                name="end_date",
                type=RowSourceValueType.DATE,
                source="query",
                required=True,
                description="Inclusive end date of the requested event interval.",
            ),
            *(
                (
                    RowSourceParam(
                        id="group_by",
                        param_ref="source_events.group_by",
                        name="group_by",
                        type=RowSourceValueType.CHOICE,
                        source="query",
                        required=True,
                        choices=("date", "location"),
                        description="Dimension used to group the event summary.",
                    ),
                    RowSourceParam(
                        id="granularity",
                        param_ref="source_events.granularity",
                        name="granularity",
                        type=RowSourceValueType.CHOICE,
                        source="query",
                        required=True,
                        choices=("day", "month"),
                        description="Calendar grain of the event summary.",
                    ),
                )
                if required_choices
                else ()
            ),
            *(
                (
                    RowSourceParam(
                        id=str(payload["unrestricted_choice_surface"]),
                        param_ref=(
                            "source_events."
                            f"{payload['unrestricted_choice_surface']}"
                        ),
                        name=str(payload["unrestricted_choice_surface"]),
                        type=RowSourceValueType.CHOICE,
                        source="query",
                        choices=("true", "false"),
                        description=(
                            "Optional classification that the question does not "
                            "restrict."
                        ),
                    ),
                )
                if payload.get("unrestricted_choice_surface")
                else ()
            ),
        ),
    )
    [input_term] = parsed.contract.inputs
    period = FactValue.time(
        id="canonical_time_1",
        known_input_id=input_term.id,
        expression=input_term.operand,
        resolved_start=str(payload["expected_start"]),
        resolved_end=str(payload["expected_end"]),
        granularity="month",
        proof_refs=("time_resolution:input_1",),
    )
    return _direct_binding_request(parsed, source=source, value=period)


def _scalar_binding_request(payload: dict[str, Any]) -> SemanticSourceBindingRequest:
    parsed = _scalar_semantic_contract(payload)
    [input_term] = parsed.contract.inputs
    source = RowSource(
        id="source_measurements",
        kind=RowSourceKind.API_READ,
        label="measurements",
        read_id="list_measurements",
        endpoint_name="list_measurements",
        fields=(
            RowSourceField(
                id="measurement_id",
                field_ref="field.measurement_id",
                label="measurement identifier",
                type=RowSourceValueType.STRING,
                allowed_roles=(),
            ),
            RowSourceField(
                id="amount",
                field_ref="field.amount",
                label="measured amount",
                type=RowSourceValueType.DECIMAL,
                allowed_roles=(),
            ),
        ),
        candidate_keys=(
            RowSourceCandidateKey(
                id="primary_key",
                entity_kind="measurement",
                components=(RowSourceKeyComponent("measurement_id", "measurement_id"),),
                primary=True,
            ),
        ),
        params=(
            RowSourceParam(
                id="minimum_amount",
                param_ref="source_measurements.minimum_amount",
                name="minimum_amount",
                type=RowSourceValueType.DECIMAL,
                source="query",
                required=True,
            ),
        ),
    )
    threshold = FactValue.literal(
        id="canonical_scalar_1",
        known_input_id=input_term.id,
        literal_type=LiteralType.NUMBER,
        value=str(payload["operand"]),
        proof_refs=("question_input:input_1",),
    )
    return _direct_binding_request(parsed, source=source, value=threshold)


def _finite_choice_binding_request(
    payload: dict[str, Any],
) -> SemanticSourceBindingRequest:
    parsed = count_contract(
        question=str(payload["question"]),
        operand=str(payload["operand"]),
        candidate_meaning="observations",
        fact_meaning="priority category",
        fact_value_type={"kind": "text"},
        input_value_type={"kind": "text"},
        operator="equals",
    )
    [input_term] = parsed.contract.inputs
    source = RowSource(
        id="source_observations",
        kind=RowSourceKind.API_READ,
        label="observations",
        read_id="list_observations",
        endpoint_name="list_observations",
        fields=(
            RowSourceField(
                id="observation_id",
                field_ref="field.observation_id",
                label="observation identifier",
                type=RowSourceValueType.STRING,
                allowed_roles=(),
            ),
            RowSourceField(
                id="priority",
                field_ref="field.priority",
                label="priority category",
                type=RowSourceValueType.CHOICE,
                allowed_roles=(),
            ),
        ),
        candidate_keys=(
            RowSourceCandidateKey(
                id="primary_key",
                entity_kind="observation",
                components=(RowSourceKeyComponent("observation_id", "observation_id"),),
                primary=True,
            ),
        ),
        params=(
            RowSourceParam(
                id="priority",
                param_ref="source_observations.priority",
                name="priority",
                type=RowSourceValueType.CHOICE,
                source="query",
                choices=("HIGH", "LOW"),
                choice_labels={"HIGH": "High priority", "LOW": "Low priority"},
            ),
        ),
    )
    value = FactValue.literal(
        id="canonical_scalar_1",
        known_input_id=input_term.id,
        literal_type=LiteralType.STRING,
        value=str(payload["operand"]),
        proof_refs=("question_input:input_1",),
    )
    return _direct_binding_request(parsed, source=source, value=value)


def _explicit_choice_override_request(
    payload: dict[str, Any],
) -> SemanticSourceBindingRequest:
    question_origin = SourceOrigin(
        SourceOriginKind.QUESTION_CONTEXT,
        str(payload["question"]),
    )
    requested_choice = str(payload["requested_choice"]).lower()
    state_origin = SourceOrigin(
        SourceOriginKind.QUESTION_CONTEXT,
        requested_choice,
    )
    input_term = InputTerm(
        id="i1",
        origin=state_origin,
        operand="true",
        value_type=BooleanType(),
    )
    input_denotation = InputDenotation(
        id="denotation_1",
        input_ref=input_term.id,
        operand_meaning=requested_choice,
        denotation_basis=(
            f"The question requires sales whose {requested_choice} fact is true."
        ),
        denoted_instance_kind=None,
        kind=InputDenotationKind.NON_IDENTITY_SCALAR,
    )
    requested_fact = RequestedFact(
        id="fact_1",
        origin=question_origin,
        sets=(SetTerm("s1", SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, "sale")),),
        associations=(),
        facts=(
            FactTerm("f1", "s1", BooleanType(), state_origin),
        ),
        expressions=(
            Comparison(
                "e1",
                ExpressionBinaryOperator.EQUALS,
                "f1",
                input_term.id,
                state_origin,
            ),
            Aggregate(
                "e2",
                AggregateFunction.COUNT,
                "s1",
                None,
                False,
                question_origin,
            ),
        ),
        subject=Subject("s1", InstanceInterpretation.NORMAL_BUSINESS_INSTANCE),
        qualification_ref="e1",
        grouping_refs=(),
        outputs=(RequestedOutput("output_1", "e2", question_origin),),
        ordering=(),
        selection=AllResults(),
        distinct_by=(),
    )
    index = analyze_requested_fact(
        requested_fact,
        inputs={input_term.id: input_term},
        input_denotations={input_term.id: input_denotation},
    )
    source = RowSource(
        id="source_sales",
        kind=RowSourceKind.API_READ,
        label="sales",
        read_id="list_sales",
        endpoint_name="list_sales",
        fields=(
            RowSourceField(
                id="sale_id",
                field_ref="field.sale_id",
                label="sale identifier",
                type=RowSourceValueType.STRING,
                allowed_roles=(),
            ),
        ),
        candidate_keys=(
            RowSourceCandidateKey(
                id="primary_key",
                entity_kind="sale",
                components=(RowSourceKeyComponent("sale_id", "sale_id"),),
                primary=True,
            ),
        ),
        params=(
            RowSourceParam(
                id="include_items",
                param_ref="source_sales.include_items",
                name="include_items",
                type=RowSourceValueType.BOOLEAN,
                source="query",
                description="Whether item details are included in each sale row.",
            ),
            RowSourceParam(
                id="status",
                param_ref="source_sales.status",
                name="status",
                type=RowSourceValueType.CHOICE,
                source="query",
                choices=("DRAFT", "PLACED", "COMPLETED", "CANCELED"),
                description="Lifecycle status of returned sales.",
            ),
        ),
    )
    row_sources = RowSourceCatalog(sources=(source,))
    read_result = SemanticReadEligibilityResult(
        read_assessments=(_retained_assessment(index, source),),
        identity_outcomes=(),
    )
    available = build_available_source_catalog(
        row_sources,
        read_eligibility=read_result,
    )
    [clause] = index.qualification.clauses
    strategy = CandidateSourceStrategy(
        requested_fact_id=index.requested_fact_id,
        source_assessments=(_source_assessment(source, SourceAlignment.DIRECT),),
        strategy_basis="The sales source is direct.",
        branches=(
            SourceStrategyBranch(
                branch_id="fact_1:source_branch:1",
                source_refs=(source.id,),
                relation_evidence_refs=(),
                qualification_clause_refs=(clause.clause_ref,),
            ),
        ),
    )
    [use] = index.input_use_sites
    canonical_value = FactValue.literal(
        id="canonical_input_value_1",
        known_input_id=input_term.id,
        literal_type=LiteralType.BOOLEAN,
        value="true",
        proof_refs=(f"question_input:{input_term.id}",),
    )
    return SemanticSourceBindingRequest(
        index=index,
        strategy=strategy,
        source_catalog=available,
        canonical_values=(
            CanonicalInputValue(
                canonical_value_id=canonical_value.id,
                input_ref=input_term.id,
                use_refs=(use.use_ref,),
                typed_value=canonical_value,
                certification_refs=(f"question_input:{input_term.id}",),
            ),
        ),
    )


def _direct_binding_request(
    parsed: ParsedSemanticQuestionContract,
    *,
    source: RowSource,
    value: FactValue,
) -> SemanticSourceBindingRequest:
    [index] = parsed.semantic_indexes
    [input_term] = parsed.contract.inputs
    row_sources = RowSourceCatalog(sources=(source,))
    read_result = SemanticReadEligibilityResult(
        read_assessments=(_retained_assessment(index, source),),
        identity_outcomes=(),
    )
    available = build_available_source_catalog(
        row_sources, read_eligibility=read_result
    )
    [clause] = index.qualification.clauses
    strategy = CandidateSourceStrategy(
        requested_fact_id=index.requested_fact_id,
        source_assessments=(_source_assessment(source, SourceAlignment.DIRECT),),
        strategy_basis="The source is direct.",
        branches=(
            SourceStrategyBranch(
                branch_id=f"{index.requested_fact_id}:source_branch:1",
                source_refs=(source.id,),
                relation_evidence_refs=(),
                qualification_clause_refs=(clause.clause_ref,),
            ),
        ),
    )
    [use] = index.input_use_sites
    return SemanticSourceBindingRequest(
        index=index,
        strategy=strategy,
        source_catalog=available,
        canonical_values=(
            CanonicalInputValue(
                canonical_value_id=value.id,
                input_ref=input_term.id,
                use_refs=(use.use_ref,),
                typed_value=value,
                certification_refs=value.proof_refs,
            ),
        ),
    )


def _association_binding_request(
    *, co_resident: bool = False
) -> SemanticSourceBindingRequest:
    index, sources, _read_result, available = association_scenario()
    projects, assignments = sources.sources
    if co_resident:
        read_result = SemanticReadEligibilityResult(
            read_assessments=(_retained_assessment(index, assignments),),
            identity_outcomes=(),
        )
        available = build_available_source_catalog(
            RowSourceCatalog(sources=(assignments,)),
            read_eligibility=read_result,
        )
        return SemanticSourceBindingRequest(
            index=index,
            strategy=CandidateSourceStrategy(
                requested_fact_id=index.requested_fact_id,
                source_assessments=(
                    _source_assessment(assignments, SourceAlignment.DIRECT),
                ),
                strategy_basis="One source contains the related instances.",
                branches=(
                    SourceStrategyBranch(
                        branch_id="fact_1:source_branch:1",
                        source_refs=(assignments.id,),
                        relation_evidence_refs=(),
                        qualification_clause_refs=(),
                    ),
                ),
            ),
            source_catalog=available,
            canonical_values=(),
        )
    [relation] = available.relation_evidence
    return SemanticSourceBindingRequest(
        index=index,
        strategy=CandidateSourceStrategy(
            requested_fact_id=index.requested_fact_id,
            source_assessments=(
                _source_assessment(projects, SourceAlignment.PARTIAL),
                _source_assessment(assignments, SourceAlignment.PARTIAL),
            ),
            strategy_basis="The declared relation connects the two sources.",
            branches=(
                SourceStrategyBranch(
                    branch_id="fact_1:source_branch:1",
                    source_refs=(projects.id, assignments.id),
                    relation_evidence_refs=(relation.evidence_ref,),
                    qualification_clause_refs=(),
                ),
            ),
        ),
        source_catalog=available,
        canonical_values=(),
    )


def _identity_set_binding_request(
    payload: dict[str, Any],
) -> SemanticSourceBindingRequest:
    raw_operand = payload["operand"]
    parsed = count_contract(
        question=str(payload["question"]),
        operand=[str(item) for item in raw_operand],
        candidate_meaning="observations",
        fact_meaning="associated member identity",
        fact_value_type={"kind": "text"},
        input_value_type={
            "kind": "collection",
            "element_type": {"kind": "text"},
        },
        operator="in",
        identity_set_meaning="members",
    )
    [input_term] = parsed.contract.inputs
    source = RowSource(
        id="source_observations_by_member",
        kind=RowSourceKind.API_READ,
        label="observations by member",
        read_id="get_member_observations",
        endpoint_name="get_member_observations",
        fields=(
            RowSourceField(
                id="observation_id",
                field_ref="observations.observation_id",
                label="observation identifier",
                type=RowSourceValueType.STRING,
                allowed_roles=(),
            ),
            RowSourceField(
                id="member_id",
                field_ref="observations.member_id",
                label="member identifier",
                type=RowSourceValueType.STRING,
                allowed_roles=(),
            ),
        ),
        candidate_keys=(
            RowSourceCandidateKey(
                id="primary_key",
                entity_kind="observation",
                components=(RowSourceKeyComponent("observation_id", "observation_id"),),
                primary=True,
            ),
        ),
        entity_references=(
            RowSourceEntityReference(
                id="observation_member",
                target_entity_kind="member",
                target_key_id="primary_key",
                components=(
                    RowSourceEntityReferenceComponent(
                        target_component_id="member_id",
                        local_field_id="member_id",
                    ),
                ),
            ),
        ),
        params=(
            RowSourceParam(
                id="member_id",
                param_ref="source_observations_by_member.member_id",
                name="member_id",
                type=RowSourceValueType.STRING,
                source="path",
                required=True,
                entity_target=EntityKeyComponentTarget(
                    "member", "primary_key", "member_id"
                ),
            ),
        ),
    )
    keys = tuple(
        EntityKeyValue(
            entity_kind="member",
            key_id="primary_key",
            components=(EntityKeyComponentValue("member_id", f"member-{index}"),),
        )
        for index, _operand in enumerate(input_term.operand, start=1)
    )
    identities = FactValue.identity_set(
        id="canonical_identity_set_1",
        known_input_id=input_term.id,
        keys=keys,
        display_value=", ".join(input_term.operand),
        proof_refs=("resolver:get_member",),
    )
    return _direct_binding_request(parsed, source=source, value=identities)


def association_scenario():
    index = _association_semantic_index()
    projects = RowSource(
        id="source_projects",
        kind=RowSourceKind.API_READ,
        label="projects",
        read_id="list_projects",
        endpoint_name="list_projects",
        fields=(
            RowSourceField(
                id="project_id",
                field_ref="projects.project_id",
                label="project identifier",
                type=RowSourceValueType.STRING,
                allowed_roles=(),
            ),
        ),
        candidate_keys=(
            RowSourceCandidateKey(
                id="primary_key",
                entity_kind="project",
                components=(RowSourceKeyComponent("project_id", "project_id"),),
                primary=True,
            ),
        ),
    )
    assignments = RowSource(
        id="source_assignments",
        kind=RowSourceKind.API_READ,
        label="assignments",
        read_id="list_assignments",
        endpoint_name="list_assignments",
        fields=(
            RowSourceField(
                id="assignment_id",
                field_ref="assignments.assignment_id",
                label="assignment identifier",
                type=RowSourceValueType.STRING,
                allowed_roles=(),
            ),
            RowSourceField(
                id="project_id",
                field_ref="assignments.project_id",
                label="assigned project identifier",
                type=RowSourceValueType.STRING,
                allowed_roles=(),
            ),
            RowSourceField(
                id="assignment_label",
                field_ref="assignments.assignment_label",
                label="assignment label",
                type=RowSourceValueType.STRING,
                allowed_roles=(),
            ),
        ),
        candidate_keys=(
            RowSourceCandidateKey(
                id="primary_key",
                entity_kind="assignment",
                components=(RowSourceKeyComponent("assignment_id", "assignment_id"),),
                primary=True,
            ),
        ),
        entity_references=(
            RowSourceEntityReference(
                id="assignment_project",
                target_entity_kind="project",
                target_key_id="primary_key",
                components=(
                    RowSourceEntityReferenceComponent(
                        target_component_id="project_id",
                        local_field_id="project_id",
                    ),
                ),
            ),
        ),
    )
    sources = RowSourceCatalog(sources=(projects, assignments))
    read_result = SemanticReadEligibilityResult(
        read_assessments=(
            _retained_assessment(index, projects),
            _retained_assessment(index, assignments),
        ),
        identity_outcomes=(),
    )
    available = build_available_source_catalog(sources, read_eligibility=read_result)
    return index, sources, read_result, available


def _retained_assessment(
    index: RequestedFactSemanticIndex,
    source: RowSource,
) -> ReadRequirementAssessment:
    return ReadRequirementAssessment(
        requested_fact_id=index.requested_fact_id,
        candidate_ref=source.read_id or source.id,
        source_refs=(source.id,),
        read_id=source.read_id,
        relevant_field_refs=tuple(field.field_ref for field in source.fields),
        assessment_basis="The source is retained for the binding experiment.",
        decision=SemanticReadDecision.RETAIN,
    )


def _source_assessment(
    source: RowSource,
    alignment: SourceAlignment,
) -> SourceAlignmentAssessment:
    return SourceAlignmentAssessment(
        source_ref=source.id,
        basis="The experiment fixture declares this source's strategy role.",
        alignment=alignment,
    )


def _association_semantic_index() -> RequestedFactSemanticIndex:
    def origin(meaning: str) -> SourceOrigin:
        return SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, meaning)

    requested = RequestedFact(
        id="fact_1",
        origin=origin("assignment labels for projects"),
        sets=(
            SetTerm("s1", origin("projects")),
            SetTerm("s2", origin("assignments")),
        ),
        associations=(
            AssociationTerm(
                "a1",
                from_set_ref="s1",
                to_set_ref="s2",
                origin=origin("assignments belonging to projects"),
            ),
        ),
        facts=(
            FactTerm(
                "f1",
                owner_ref="a1",
                value_type=TextType(),
                origin=origin("assignment label"),
            ),
        ),
        expressions=(),
        subject=Subject("s1", InstanceInterpretation.NORMAL_BUSINESS_INSTANCE),
        qualification_ref=None,
        grouping_refs=(),
        outputs=(RequestedOutput("o1", "f1", origin("assignment label")),),
        ordering=(),
        selection=AllResults(),
        distinct_by=(),
    )
    return analyze_requested_fact(requested, inputs={}, input_denotations={})


def _semantic_contract(payload: dict[str, Any]) -> ParsedSemanticQuestionContract:
    return count_contract(
        question=str(payload["question"]),
        operand=str(payload["operand"]),
        candidate_meaning="members",
        fact_meaning="member identity",
        fact_value_type={"kind": "text"},
        input_value_type={"kind": "text"},
        operator="equals",
        identity_set_meaning="members",
    )


def _time_semantic_contract(payload: dict[str, Any]) -> ParsedSemanticQuestionContract:
    return count_contract(
        question=str(payload["question"]),
        operand=str(payload["operand"]),
        candidate_meaning="events",
        fact_meaning="event date",
        fact_value_type={"kind": "date"},
        input_value_type={"kind": "temporal_scope"},
        operator="within",
    )


def _scalar_semantic_contract(
    payload: dict[str, Any],
) -> ParsedSemanticQuestionContract:
    return count_contract(
        question=str(payload["question"]),
        operand=str(payload["operand"]),
        candidate_meaning="measurements",
        fact_meaning="measured amount",
        fact_value_type={
            "kind": "decimal",
            "measure": {"kind": "unitless"},
        },
        input_value_type={"kind": "integer"},
        operator="gt",
    )


if __name__ == "__main__":
    raise SystemExit(main())
