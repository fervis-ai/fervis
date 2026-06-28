from __future__ import annotations


def provider_native_test_arguments(
    *,
    tool_name: str,
    prompt: str,
    tool_specs,
) -> dict:
    del prompt, tool_specs
    if tool_name == "submit_answer_request_contract":
        return {
            "kind": "question_contract",
            "answer_requests_count": 1,
            "question_inputs": [],
            "answer_requests": [
                {
                    "answer_fact": "test adapter answer",
                    "answer_expression": {"family": "scalar_value"},
                    "answer_subject": {
                        "subject_text": "test adapter answer",
                        "instance_interpretation": {"kind": "NORMAL_BUSINESS_INSTANCE"},
                    },
                    "input_requirements": {"time_requirements": []},
                    "answer_population": {
                        "population_label": "test adapter answer",
                        "counted_unit": "test adapter answer",
                        "membership_tests": [
                            {
                                "test_id": "pop_test_1",
                                "kind": "SUBJECT_IDENTITY",
                                "polarity": "MUST_PASS",
                                "test_question": (
                                    "Does the row/value represent test adapter answer?"
                                ),
                            }
                        ],
                    },
                    "answer_outputs": [
                        {
                            "description": "test adapter answer",
                        }
                    ],
                    "input_decisions": [],
                }
            ],
            "question_input_inventory_check": {
                "all_input_like_phrases_declared": True,
            },
        }
    if tool_name == "submit_pattern_fact_plan":
        return {
            "outcome": {
                "kind": "impossible",
                "blocked_facts": [
                    {
                        "requested_fact_id": "fact_1",
                        "basis": "catalog_access",
                        "evidence_refs": ["row_source:test_adapter"],
                    }
                ],
            }
        }
    return {}
