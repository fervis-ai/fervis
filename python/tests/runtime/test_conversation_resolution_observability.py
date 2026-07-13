from __future__ import annotations

from tests.lookup.orchestrator._helpers import *  # noqa: F403

from fervis.memory.addresses import FactAddress
from fervis.memory.artifacts import (
    build_fact_artifact,
    FactOutcome,
)


def test_question_contract_failure_includes_conversation_resolution_usage():
    artifact = build_fact_artifact(
        artifact_id="turn_1",
        outcome=FactOutcome.ANSWERED,
        source_question="How much money did we make today?",
        source_answer="100",
        provenance={
            "question_contract": {
                "question_inputs": [],
                "answer_requests": [
                    {
                        "id": "fact_1",
                        "answer_fact": "money we made today",
                        "answer_outputs": [
                            {
                                "id": "answer_1",
                                "description": "total sales amount",
                                "role": "ANSWER_VALUE",
                            }
                        ],
                        "used_question_inputs": [],
                    }
                ],
            }
        },
        addresses=(
            FactAddress.value(
                address="value.total",
                value={"type": "decimal", "value": "100.00"},
            ),
        ),
    )
    planner = _ToolNamePlannerPort(
        responses={
            CONVERSATION_RESOLUTION_TOOL_NAME: (
                lambda *, prompt, tool_specs: (
                    _conversation_resolution_payload_from_prompt(prompt)
                )
            ),
            "submit_question_contract_outcome": {
                "kind": "question_contract",
                "answer_requests_count": 2,
                "answer_requests": [
                    {
                        "answer_fact": "money made yesterday",
                        "answer_outputs": [
                            {
                                "description": "total money",
                                "role": "ANSWER_VALUE",
                            }
                        ],
                        "used_question_inputs": [],
                    }
                ],
            },
        }
    )

    result = run_lookup_question(
        LookupRequest(
            question="How much money did we make yesterday?",
            run_id="run_question_contract_usage",
            conversation_context={"factArtifacts": [artifact.to_dict()]},
        ),
        LookupRuntimePorts(
            relation_catalog_port=_CatalogPort(_catalog(_metric_read("metric_read"))),
            data_access_port=_DataAccessPort({}),
            planner_model_port=planner,
        ),
    )

    assert (
        result.status,
        result.usage["inputTokens"],
        result.usage["outputTokens"],
    ) == ("FAILED", 2, 2)
