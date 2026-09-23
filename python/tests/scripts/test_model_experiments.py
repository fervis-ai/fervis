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
from scripts.experiments.source_access.inspection_input_assertion import (  # noqa: E402
    validate as validate_inspection_input,
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


def test_inspection_input_assertion_checks_bound_and_unsupported_controls():
    context = {"reads": {
        "readings": {"kind": "bound_arguments", "parameter_inputs": {"site_id": "i1"}},
        "reports": {"kind": "bound_arguments", "parameter_inputs": {"shape": "choice:summary"}},
        "private": {"kind": "unsupported"},
    }}
    correct = {"reads": {
        "readings": {"kind": "bound_arguments", "parameter_inputs": {"site_id": "i1"},
                     "mapping_basis": "Question-owned site ID."},
        "reports": {"kind": "bound_arguments", "parameter_inputs": {"shape": "choice:summary"},
                    "mapping_basis": "The declared summary shape supplies the requested total."},
        "private": {"kind": "unsupported", "reason": "No caller-owned address."},
    }}
    assert validate_inspection_input(correct, context) == []
    wrong = {"reads": {**correct["reads"], "reports": {
        "kind": "bound_arguments", "parameter_inputs": {"shape": "choice:detail"},
    }}}
    assert validate_inspection_input(wrong, context) == [
        "reports: incorrect typed parameter-to-input mapping"
    ]
    wrong_input = {"reads": {**correct["reads"], "readings": {
        "kind": "bound_arguments", "parameter_inputs": {"site_id": "i2"},
    }}}
    assert validate_inspection_input(wrong_input, context) == [
        "readings: incorrect typed parameter-to-input mapping"
    ]
    assert validate_inspection_input(correct, {}) == [
        "inspection input assertion requires expected read decisions"
    ]


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


def test_property_scope_assertion_distinguishes_related_from_candidate_properties():
    from dataclasses import replace
    from fervis.lookup.question_contract import QuestionContract
    from fervis.lookup.question_contract.analysis import analyze_requested_fact
    from fervis.lookup.question_contract.model import Quantifier
    from fervis.lookup.question_contract.parser import ParsedSemanticQuestionContract
    from scripts.experiments.question_contract.assertion import _input_property_scope_errors
    from tests.lookup.relational_engine.test_scoped_compilation import employee_query

    index = employee_query(Quantifier.EXISTS, manager_minimum=True).request.index
    context = {"input_property_scopes": {"minimum": "related"}}

    def check(current):
        parsed = ParsedSemanticQuestionContract(
            "Staff whose managers exceed a supplied salary threshold.",
            QuestionContract(tuple(current.input_by_ref.values()), (current.requested_fact,),
                             tuple(current.input_denotation_by_ref.values())),
            (current,),
        )
        return _input_property_scope_errors(parsed, context)

    # The threshold is outside the correlated comparison in this fixture.
    assert any("lacks a correlated quantifier" in error for error in check(index))
    fact = index.requested_fact
    quantified = next(node for node in fact.expressions if node.id == "quantified")
    fact = replace(fact, qualification_ref="quantified", expressions=(
        *(node for node in fact.expressions if node.id not in {"quantified", "all_conditions"}),
        replace(quantified, condition_ref="minimum_manager_salary"),
    ))
    correlated = analyze_requested_fact(
        fact, inputs=index.input_by_ref, input_denotations=index.input_denotation_by_ref
    )
    assert check(correlated) == []
    candidate_fact = replace(fact, qualification_ref="minimum_manager_salary",
        expressions=tuple(node for node in fact.expressions if node.id != "quantified"),
        facts=tuple(
        replace(term, owner_ref="employee") if term.id == "manager_salary" else term
        for term in fact.facts
    ))
    candidate = analyze_requested_fact(
        candidate_fact, inputs=index.input_by_ref, input_denotations=index.input_denotation_by_ref
    )
    assert any("belongs to candidate scope" in error for error in check(candidate))


def test_source_realization_assertion_accepts_equivalent_carriers_but_rejects_borrowed_fields():
    from scripts.experiments.source_realization.assertion import validate

    context = {"expected_carrier_options": {"set": {
        "rows_a": {"fact": ["source_field:rows_a:name"]},
        "rows_b": {"fact": ["source_field:rows_b:name"]},
    }}}
    for source in ("rows_a", "rows_b"):
        arguments = {
            "set_bindings": {"set": [{"branch_id": "branch", "rows_ref": source}]},
            "fact_bindings": {"fact": [{
                "branch_id": "branch", "field_ref": f"source_field:{source}:name"
            }]},
            "association_bindings": {},
        }
        assert validate(arguments, context) == []
        arguments["fact_bindings"]["fact"][0]["field_ref"] = "source_field:other:name"
        assert any("selected carrier" in error for error in validate(arguments, context))
