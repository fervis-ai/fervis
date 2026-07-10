from __future__ import annotations

import json

from jsonschema import Draft202012Validator
import pytest

from tests.testkit.case_loader import SCHEMA_PATH


@pytest.mark.parametrize(
    ("algorithm", "input_payload"),
    (
        ("answer_program.compile", {"fixture": "catalog_default"}),
        ("answer_program.invoke", {"scenario": "same_binding_fresh_evidence"}),
        (
            "questions.lifecycle",
            {"schema_revision": 1, "scenario": "rerun"},
        ),
    ),
)
def test_portable_cases_reject_python_owned_scenario_shorthand(
    algorithm: str,
    input_payload: dict[str, object],
) -> None:
    validator = Draft202012Validator(json.loads(SCHEMA_PATH.read_text()))

    errors = tuple(
        validator.iter_errors(
            {
                "id": "portable_answer_program_contract",
                "kind": "algorithm",
                "algorithm": algorithm,
                "input": input_payload,
                "expect": {"result_equals": {}},
            }
        )
    )

    assert errors
