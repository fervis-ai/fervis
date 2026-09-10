from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fervis.lookup.question_contract import (  # noqa: E402
    SemanticQuestionFrameTurnPrompt,
)
from scripts.experiments.question_frame.boundary import (  # noqa: E402
    build_boundary_payload,
)
from scripts.experiments.query_enrichment.assertion import (  # noqa: E402
    validate as validate_query_enrichment,
)
from scripts.experiments.read_eligibility.assertion import (  # noqa: E402
    validate as validate_read_eligibility,
)
from fervis.model_io.structured_output.schema import (  # noqa: E402
    without_unreferenced_definitions,
)
from scripts.experiments.source_binding.assertion import (  # noqa: E402
    validate as validate_source_binding,
)


class RecordingQuestionFramePrompt(SemanticQuestionFrameTurnPrompt):
    called = False

    def to_model_invocation(self, context):
        type(self).called = True
        return super().to_model_invocation(context)


def test_entry_boundary_uses_the_injected_production_prompt_interface():
    boundary = build_boundary_payload(
        question="How many events happened today?",
        expected_request_count=1,
        prompt_type=RecordingQuestionFramePrompt,
    )

    assert RecordingQuestionFramePrompt.called
    assert boundary["purpose"] == "question_frame"
    assert boundary["prompt"]
    assert boundary["tool_specs"]


def test_schema_experiment_removes_only_unreachable_definitions():
    schema = {
        "type": "object",
        "properties": {"value": {"$ref": "#/$defs/reachable"}},
        "$defs": {
            "reachable": {"$ref": "#/$defs/transitive"},
            "transitive": {"type": "string"},
            "unused": {"type": "integer"},
        },
    }

    transformed = without_unreferenced_definitions(schema)

    assert tuple(transformed["$defs"]) == ("reachable", "transitive")
    assert tuple(schema["$defs"]) == ("reachable", "transitive", "unused")


def test_query_enrichment_assertion_checks_semantic_recall_outcomes():
    arguments = {
        "recall_bucket_matches": [
            {
                "bucket_ref": "fact_1:recall:output_1",
                "exhaustive_resource_names": ["compensation", "staff"],
                "matching_resource_names": ["compensation"],
            }
        ],
        "input_resource_search_terms": [],
    }
    context = {
        "recall_buckets": [
            {
                "bucket_ref": "fact_1:recall:output_1",
                "expected_any_resource_names": ["compensation"],
            }
        ]
    }

    assert validate_query_enrichment(arguments, context) == []
    assert validate_query_enrichment(arguments, {}) == [
        "query enrichment assertion requires an expected recall outcome"
    ]


def test_read_eligibility_assertion_checks_fact_local_read_decisions():
    arguments = {
        "read_assessments_by_requested_fact": {
            "fact_1": {
                "list_location_list": {"decision": "RETAIN"},
                "list_store_list": {"decision": "DROP"},
            }
        },
        "identity_outcomes": {},
    }
    context = {
        "requested_fact_ref": "fact_1",
        "expected_read_decisions": {
            "list_location_list": "RETAIN",
            "list_store_list": "DROP",
        },
    }

    assert validate_read_eligibility(arguments, context) == []


def test_source_binding_assertion_checks_declared_binding_outcomes():
    arguments = {
        "finite_choice_applications": {
            "branch_1": {
                "requirement_1": {
                    "application_basis": "The requested state is completed.",
                    "surface_ref": "sale.status",
                    "selected_choice_values": ["COMPLETED"],
                }
            }
        },
        "choice_requirement_applications": {
            "branch_1": {"sale.status": {"COMPLETED": {"selected_by_requirements": ["requirement_1"]}}},
        },
    }
    context = {
        "expected_choice_reviews": [
            {
                "surface_ref": "sale.status",
                "choice": "COMPLETED",
                "selected_by_requirements": ["requirement_1"],
            }
        ],
        "expected_finite_choice_applications": [
            {
                "branch_ref": "branch_1",
                "owner_ref": "requirement_1",
                "surface_ref": "sale.status",
                "selected_choice_values": ["COMPLETED"],
            }
        ],
    }

    assert validate_source_binding(arguments, context) == []


def test_population_assertion_checks_polarity_and_unknown_rows():
    from scripts.experiments.source_realization.population_assertion import validate
    context = {"sets": {"set": {"kind": "restricted_population", "truth_cases": [
        {"fields": {"flag": True}, "expected": False},
        {"fields": {"flag": False}, "expected": True},
        {"fields": {"flag": None}, "expected": False},
    ]}}}
    def output(condition):
        return {"populations": {"set": [{"population": {"kind": "restricted_population", "condition": condition}}]}}
    assert validate(output({"kind": "boolean_field", "field_ref": "flag", "expected_value": False}), context) == []
    assert validate(output({"kind": "boolean_field", "field_ref": "flag", "expected_value": True}), context)
    assert validate(output({"kind": "unary", "operator": "not", "operand": {"kind": "field", "field_ref": "flag"}}), context) == []
    assert validate({"populations": {"set": [{"population": {"kind": "exact_population"}}]}}, context)


def test_question_frame_assertion_preserves_typed_reference_values():
    from scripts.experiments.question_frame.assertion import validate
    from tests.lookup.question_contract.test_question_frame import _frame_payload
    for value,kind in [('Alpha','literal'),('the configured site','description')]:
        question=f'How many events are associated with {value}?'
        body=_frame_payload(supplied_values=[{'meaning':'the supplied site',
            'denotation_basis':'The question denotes one site.',
            'entity_reference':{'instance_kind':'site','value':{'operands':[value],
                'reference_kind':kind,'origin':{'kind':'question'}}}}])
        context={'question':question,'expected_request_count':1,
            'required_supplied_values':{value:{'denotation':'identity_reference'}},
            'accepted_supplied_value_inventories':[[value]]}
        assert validate(body,context)==[]
        changed={**context,'required_supplied_values':{value:{'denotation':'scalar'}}}
        assert any('denotation' in error for error in validate(body,changed))
