from __future__ import annotations

from fervis.lookup.source_binding.candidates import SourceCandidate
from fervis.lookup.source_binding.candidates.params import (
    _candidate_with_param_decision_options,
    _candidate_with_param_population_contracts,
    _param_bind_options,
    _param_omit_option,
)
from fervis.lookup.source_binding.model import AnswerPopulation
from fervis.lookup.source_binding.parser.params import parse_param_decision_binding_sets


def test_optional_static_boolean_param_exposes_omit_decision():
    param = {
        "param_id": "is_open",
        "source": "query",
        "type": "boolean",
        "required": False,
        "binding_values": [
            {"value": "true", "label": "true", "source": "static_choice"},
            {"value": "false", "label": "false", "source": "static_choice"},
        ],
    }
    bind_options = _param_bind_options(param)

    candidate = _candidate_with_param_decision_options(
        {
            "source_candidate_id": "source_1",
            "params": [
                {
                    **param,
                    "bind_options": bind_options,
                    "omit_option": _param_omit_option(
                        param,
                        bind_options=bind_options,
                    ),
                }
            ],
        }
    )

    decision_options = candidate["params"][0]["decision_options"]
    omit_options = [
        option for option in decision_options if option.get("decision") == "omit"
    ]

    assert len(omit_options) == 1
    assert omit_options[0]["param_decision_id"] == "param_decision.source_1.is_open.omit"
    assert "true and false" in omit_options[0]["meaning"]

    candidate = _candidate_with_param_population_contracts(candidate)

    assert "population_contract" not in candidate["params"][0]


def test_non_exhaustive_identity_param_does_not_expose_omit_decision():
    param = {
        "param_id": "customer_id",
        "source": "query",
        "type": "uuid",
        "required": False,
        "binding_values": [
            {"value": "customer_1", "label": "Alice", "source": "available_value"},
        ],
    }
    bind_options = _param_bind_options(param)

    candidate = _candidate_with_param_decision_options(
        {
            "source_candidate_id": "source_1",
            "params": [
                {
                    **param,
                    "bind_options": bind_options,
                    "omit_option": _param_omit_option(
                        param,
                        bind_options=bind_options,
                    ),
                }
            ],
        }
    )

    decisions = {
        option["decision"] for option in candidate["params"][0]["decision_options"]
    }

    assert decisions == {"bind"}


def test_boolean_finite_choice_param_keeps_population_contract_review_surface():
    param = {
        "param_id": "is_active",
        "source": "query",
        "type": "boolean",
        "required": False,
        "choices": ["true", "false"],
        "binding_values": [
            {"value": "true", "label": "true", "source": "static_choice"},
            {"value": "false", "label": "false", "source": "static_choice"},
        ],
    }
    bind_options = _param_bind_options(param)
    candidate = {
        "source_candidate_id": "source_1",
        "params": [
            {
                **param,
                "bind_options": bind_options,
                "omit_option": _param_omit_option(param, bind_options=bind_options),
            }
        ],
    }

    candidate = _candidate_with_param_decision_options(candidate)
    candidate = _candidate_with_param_population_contracts(candidate)

    reviewed_param = candidate["params"][0]
    decisions = {
        option["decision"] for option in reviewed_param.get("decision_options") or ()
    }

    assert decisions == {"bind"}
    assert isinstance(reviewed_param.get("population_contract"), dict)


def test_omit_param_decision_compiles_to_no_endpoint_binding():
    candidate = SourceCandidate(
        id="source_1",
        requested_fact_id="fact_1",
        kind="read",
        params=(
            {
                "param_id": "is_open",
                "source": "query",
                "type": "boolean",
                "decision_options": [
                    {
                        "decision": "omit",
                        "meaning": "Omitting is_open includes true and false.",
                        "param_decision_id": "param_decision.source_1.is_open.omit",
                    }
                ],
            },
        ),
    )

    parsed = parse_param_decision_binding_sets(
        {
            "is_open": {
                "match_basis_explanation": "No open-status filter was requested.",
                "param_decision_id": "param_decision.source_1.is_open.omit",
                "population_intent": "All open-status values.",
            }
        },
        candidate=candidate,
        available_values=(),
        answer_population=AnswerPopulation(
            population_binding_id="pop.source_1.candidate_population",
            intent_text="all stores",
            match_basis_explanation="Use the full store population.",
        ),
    )

    assert parsed.binding_sets == ((),)
