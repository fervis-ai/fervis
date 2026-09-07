"""Outcome contracts for the semantic downstream responsibility cut."""

from jsonschema import ValidationError, validate
import pytest

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
from fervis.lookup.turn_prompts.projections.semantic_requirements import (
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
        item
        for item in result.read_assessments
        if item.candidate_ref == candidate.candidate_ref
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
        subject=Subject("s1", InstanceInterpretation.RESOURCE_POPULATION),
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


def test_downstream_requirement_preserves_comparison_operator_and_operands() -> None:
    [index] = _semantic_contract().semantic_indexes

    [requirement] = semantic_requirements_prompt_payload(index)["boolean_requirements"]

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
