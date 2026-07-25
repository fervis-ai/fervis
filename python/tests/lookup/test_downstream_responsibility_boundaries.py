"""Outcome contracts for the semantic downstream responsibility cut."""

from dataclasses import replace

from jsonschema import ValidationError, validate
import pytest

from fervis.lookup.answer_program.values import FactValue
from fervis.lookup.available_sources import (
    SourceContractSnapshot,
    AvailableSourceCatalog,
)
from fervis.lookup.canonical_data import EntityKeyComponentValue, EntityKeyValue
from fervis.lookup.grounding import CanonicalInputValue
from fervis.lookup.plan_selection.semantic import SemanticPlanSelectionRequest
from fervis.lookup.plan_selection.semantic_parser import parse_semantic_plan_selection
from fervis.lookup.plan_selection.semantic_prompt import SemanticPlanSelectionTurnPrompt
from fervis.lookup.plan_selection.semantic_schema import (
    build_semantic_plan_selection_schema,
)
from fervis.lookup.question_contract.analysis import analyze_requested_fact
from fervis.lookup.question_contract.model import (
    Aggregate,
    AggregateFunction,
    AllResults,
    FactTerm,
    InstanceInterpretation,
    RequestedFact,
    RequestedOutput,
    SetTerm,
    Subject,
)
from fervis.lookup.read_eligibility.semantic import SemanticReadEligibilityRequest
from fervis.lookup.read_eligibility.semantic_parser import (
    parse_semantic_read_eligibility,
)
from fervis.lookup.read_eligibility.semantic_schema import (
    build_semantic_read_eligibility_schema,
)
from fervis.lookup.relation_catalog import RelationCatalog
from fervis.lookup.relation_catalog.row_sources import build_row_source_catalog
from fervis.lookup.turn_prompts import build_turn_prompt_context
from fervis.lookup.turn_prompts.projections.semantic_requirements import (
    plan_selection_fact_prompt_payload,
    semantic_requirements_prompt_payload,
)
from fervis.lookup.semantic_types import (
    BooleanType,
    SourceOrigin,
    SourceOriginKind,
)
from tests.lookup.grounding._fixtures import _staff_read
from tests.lookup.read_eligibility.test_semantic_read_eligibility import (
    _semantic_contract,
)


def test_read_eligibility_retains_evidence_without_authoring_realizations() -> None:
    parsed = _semantic_contract()
    [index] = parsed.semantic_indexes
    catalog = RelationCatalog(reads=(_staff_read(),))
    row_sources = build_row_source_catalog(catalog)
    request = SemanticReadEligibilityRequest(
        indexes=(index,),
        source_catalog=row_sources,
        answer_catalog=catalog,
        identity_tasks=(),
        resolver_catalog=RelationCatalog(),
    )
    [candidate] = tuple(item for item in request.read_candidates if item.read_id)
    field_ref = candidate.fields[0].field_ref
    payload = {
        "identity_outcomes": {},
        "read_assessments_by_requested_fact": {
            index.requested_fact_id: {
                item.candidate_ref: (
                    {
                        "assessment_basis": "The returned staff rows and identity can contribute to this fact.",
                        "relevant_field_refs": [field_ref],
                        "decision": "RETAIN",
                    }
                    if item == candidate
                    else {
                        "assessment_basis": "The generated calendar does not contribute to this fact.",
                        "relevant_field_refs": [],
                        "decision": "DROP",
                    }
                )
                for item in request.read_candidates
            }
        },
    }

    schema = build_semantic_read_eligibility_schema(request)
    validate(payload, schema)
    result = parse_semantic_read_eligibility(payload, request=request)

    assessment = next(
        item for item in result.read_assessments if item.candidate_ref == candidate.candidate_ref
    )
    assert assessment.requested_fact_id == index.requested_fact_id
    assert assessment.relevant_field_refs == (field_ref,)
    assert "set_reviews" not in str(schema)
    assert "identifier_reviews" not in str(schema)
    assert "supported_terms" not in str(schema)


def test_downstream_requirements_project_a_direct_boolean_fact_predicate() -> None:
    origin = SourceOrigin(SourceOriginKind.QUESTION_CONTEXT, "qualifying events")
    requested_fact = RequestedFact(
        id="fact_1",
        origin=origin,
        sets=(SetTerm("s1", origin),),
        associations=(),
        facts=(FactTerm("f1", "s1", BooleanType(), origin),),
        expressions=(
            Aggregate(
                "e1",
                AggregateFunction.COUNT,
                "s1",
                None,
                False,
                origin,
            ),
        ),
        subject=Subject("s1", InstanceInterpretation.NORMAL_BUSINESS_INSTANCE),
        qualification_ref="f1",
        grouping_refs=(),
        outputs=(RequestedOutput("output_1", "e1", origin),),
        ordering=(),
        selection=AllResults(),
        distinct_by=(),
    )
    index = analyze_requested_fact(
        requested_fact,
        inputs={},
        input_denotations={},
    )

    payload = semantic_requirements_prompt_payload(index)
    plan_payload = plan_selection_fact_prompt_payload(index)

    assert payload["boolean_requirements"] == [
        {
            "requirement_ref": (
                "fact_1:population:qualification:fact_1:fact:f1:positive"
            ),
            "expression_ref": "fact_1:fact:f1",
            "polarity": "positive",
            "use_site": "population",
            "owner_expression_ref": None,
            "condition": {
                "operator": "is_true",
                "operands": [
                    {
                        "ref": "fact_1:fact:f1",
                        "meaning": "qualifying events",
                    }
                ],
            },
        }
    ]
    assert plan_payload["qualification_clauses"] == [
        {
            "clause_ref": "fact_1:qualification_clause:1",
            "conditions": ["qualifying events"],
        }
    ]


def test_downstream_requirement_preserves_comparison_operator_and_operands() -> None:
    [index] = _semantic_contract().semantic_indexes

    [requirement] = semantic_requirements_prompt_payload(index)[
        "boolean_requirements"
    ]

    assert requirement["condition"] == {
        "operator": "equals",
        "operands": [
            {
                "ref": "fact_1:fact:f1",
                "meaning": "staff member identity",
            },
            {
                "ref": "i1",
                "meaning": "the staff member being listed",
            },
        ],
    }


def test_read_eligibility_schema_makes_dropped_field_retention_impossible() -> None:
    parsed = _semantic_contract()
    [index] = parsed.semantic_indexes
    catalog = RelationCatalog(reads=(_staff_read(),))
    request = SemanticReadEligibilityRequest(
        indexes=(index,),
        source_catalog=build_row_source_catalog(catalog),
        answer_catalog=catalog,
        identity_tasks=(),
        resolver_catalog=RelationCatalog(),
    )
    [candidate] = tuple(item for item in request.read_candidates if item.read_id)
    payload = {
        "identity_outcomes": {},
        "read_assessments_by_requested_fact": {
            index.requested_fact_id: {
                item.candidate_ref: {
                    "assessment_basis": "This read does not contribute to the fact.",
                    "relevant_field_refs": (
                        [candidate.fields[0].field_ref] if item == candidate else []
                    ),
                    "decision": "DROP",
                }
                for item in request.read_candidates
            }
        },
    }

    with pytest.raises(ValidationError):
        validate(payload, build_semantic_read_eligibility_schema(request))


def test_plan_selection_authors_alignment_and_backend_composes_strategy() -> None:
    parsed = _semantic_contract()
    [index] = parsed.semantic_indexes
    [source] = tuple(
        source
        for source in build_row_source_catalog(
            RelationCatalog(reads=(_staff_read(),))
        ).sources
        if source.read_id
    )
    catalog = AvailableSourceCatalog(
        contract_snapshot=SourceContractSnapshot.from_content("{}"),
        sources=(source,),
        relation_evidence=(),
    )
    request = SemanticPlanSelectionRequest(
        indexes=(index,),
        source_catalog=catalog,
    )
    payload = {
        "source_assessments_by_requested_fact": {
            index.requested_fact_id: {
                source.id: {
                    "basis": "This source contains the complete raw ingredients for the requested fact.",
                    "alignment": "DIRECT",
                }
            }
        }
    }

    schema = build_semantic_plan_selection_schema(request)
    validate(payload, schema)
    [strategy] = parse_semantic_plan_selection(payload, request=request)

    [assessment] = strategy.source_assessments
    assert assessment.source_ref == source.id
    assert assessment.alignment.value == "DIRECT"
    assert strategy.branches[0].source_refs == (source.id,)


def test_plan_selection_sees_certified_values_and_declared_source_identities() -> None:
    parsed = _semantic_contract()
    [index] = parsed.semantic_indexes
    [input_term] = parsed.contract.inputs
    [use] = index.input_use_sites
    [source] = tuple(
        source
        for source in build_row_source_catalog(
            RelationCatalog(reads=(_staff_read(),))
        ).sources
        if source.read_id
    )
    source = replace(source, resource_names=("staff member",))
    canonical_value = CanonicalInputValue(
        canonical_value_id="canonical_staff_1",
        input_ref=input_term.id,
        use_refs=(use.use_ref,),
        typed_value=FactValue.identity(
            id="canonical_staff_1",
            known_input_id=input_term.id,
            key=EntityKeyValue(
                entity_kind="staff",
                key_id="primary_key",
                components=(EntityKeyComponentValue("staff_id", "staff_1"),),
            ),
            display_value="Ada",
            proof_refs=("resolver:list_staff_list",),
        ),
        certification_refs=("resolver:list_staff_list",),
    )
    request = SemanticPlanSelectionRequest(
        indexes=(index,),
        source_catalog=AvailableSourceCatalog(
            contract_snapshot=SourceContractSnapshot.from_content("{}"),
            sources=(source,),
            relation_evidence=(),
        ),
        canonical_values=(canonical_value,),
    )

    invocation = SemanticPlanSelectionTurnPrompt(request).to_model_invocation(
        build_turn_prompt_context(
            current_question="Which staff member is named Ada?",
            conversation_context={},
        )
    )

    assert "Requested facts:" in invocation.prompt_text
    assert '"fact_text": "staff members named Ada"' in invocation.prompt_text
    assert '"candidate_instance_kind": "staff members"' in invocation.prompt_text
    assert '"answer_outputs"' in invocation.prompt_text
    assert '"value_kind": "canonical_identity"' in invocation.prompt_text
    assert '"identified_set_meaning": "staff members"' in invocation.prompt_text
    assert '"required_facts"' in invocation.prompt_text
    assert '"meaning": "staff member identity"' in invocation.prompt_text
    assert '"term_requirements"' not in invocation.prompt_text
    assert '"boolean_requirements"' not in invocation.prompt_text
    assert '"entity_kind": "staff"' in invocation.prompt_text
    assert '"key_id": "primary_key"' in invocation.prompt_text
    assert '"identity_ref": "source_identity:' in invocation.prompt_text
    assert '"resource_names": [' in invocation.prompt_text
    assert '"staff member"' in invocation.prompt_text
    assert '"param_ref": "list_staff_list.query.name"' in invocation.prompt_text
    assert (
        "Assess each source without reinterpreting any shown certified input. "
        "Applying those inputs belongs to Source Binding."
        in invocation.prompt_text
    )
    assert (
        "A source restricted to a specialized population not requested by the "
        "question is NOT_ALIGNED"
        in invocation.prompt_text
    )
    assert (
        "including every requested output at its declared value kind"
        in invocation.prompt_text
    )
    assert (
        "Producing a requested canonical identity requires matching declared "
        "identity evidence."
        in invocation.prompt_text
    )

    incompatible_source = replace(
        source,
        candidate_keys=tuple(
            replace(key, entity_kind="area") for key in source.candidate_keys
        ),
    )
    incompatible_request = replace(
        request,
        source_catalog=replace(request.source_catalog, sources=(incompatible_source,)),
    )
    alignment_schema = build_semantic_plan_selection_schema(incompatible_request)[
        "properties"
    ]["source_assessments_by_requested_fact"]["properties"][
        index.requested_fact_id
    ]["properties"][incompatible_source.id]["properties"]["alignment"]

    assert alignment_schema["enum"] == ["PARTIAL", "NOT_ALIGNED"]
