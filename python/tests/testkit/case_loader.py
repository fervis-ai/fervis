from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
import yaml


ROOT = Path(__file__).resolve().parents[1]
CASE_ROOT = ROOT / "conformance" / "cases"
SCHEMA_PATH = ROOT / "conformance" / "schemas" / "case.schema.json"


@dataclass(frozen=True)
class ConformanceCase:
    path: Path
    payload: dict[str, Any]

    @property
    def id(self) -> str:
        return str(self.payload["id"])


def load_all_conformance_cases(root: Path = CASE_ROOT) -> tuple[ConformanceCase, ...]:
    schema = json.loads(SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)
    cases = []
    for path in sorted((*root.rglob("*.json"), *root.rglob("*.yaml"), *root.rglob("*.yml"))):
        payload = _load_case(path)
        validator.validate(payload)
        _lint_case(payload, path=path)
        cases.append(ConformanceCase(path=path, payload=payload))
    return tuple(cases)


def _load_case(path: Path) -> dict[str, Any]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
    else:
        payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain an object")
    return payload


def _lint_case(payload: dict[str, Any], *, path: Path) -> None:
    if payload.get("kind") != "algorithm":
        return
    forbidden = _forbidden_prompt_local_keys(payload.get("description"))
    if payload.get("algorithm") not in _TURN_CONTRACT_ALGORITHMS:
        forbidden |= _forbidden_prompt_local_keys(payload.get("input"))
        forbidden |= _forbidden_prompt_local_keys(payload.get("expect"))
    if forbidden:
        formatted = ", ".join(sorted(forbidden))
        raise ValueError(f"{path} leaks prompt-local identifiers: {formatted}")


_TURN_CONTRACT_ALGORITHMS = {
    "conversation_resolution.parse",
    "conversation_resolution.schema",
    "question_contract.parse",
    "question_contract.schema",
    "question_contract.schema_validate",
    "question_contract.prompt",
    "query_enrichment.parse",
    "query_enrichment.schema",
    "query_enrichment.prompt",
    "read_eligibility.parse",
    "read_eligibility.schema_validate",
    "read_eligibility.prompt",
    "read_eligibility.cards",
    "source_binding.fulfillment_support",
    "source_binding.metric_fit_surface",
    "source_binding.row_predicates",
    "planning.fact_plan_schema",
}


def _forbidden_prompt_local_keys(value: Any) -> set[str]:
    forbidden_names = {
        "source_candidate_id",
        "population_binding_id",
        "fulfillment_choice_id",
        "param_decision_id",
        "operation_support_set_id",
        "metric_option_id",
    }
    if isinstance(value, dict):
        found = set(value) & forbidden_names
        for item in value.values():
            found |= _forbidden_prompt_local_keys(item)
        return found
    if isinstance(value, list):
        found: set[str] = set()
        for item in value:
            found |= _forbidden_prompt_local_keys(item)
        return found
    return set()
