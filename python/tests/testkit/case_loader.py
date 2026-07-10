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
FIXTURE_ROOT = ROOT / "conformance" / "fixtures"
ANSWER_PROGRAM_FIXTURE_SCHEMA_PATH = (
    ROOT / "conformance" / "schemas" / "answer-program-fixture.schema.json"
)


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
    _validate_portable_fixtures()
    cases = []
    for path in sorted((*root.rglob("*.json"), *root.rglob("*.yaml"), *root.rglob("*.yml"))):
        payload = _resolve_local_refs(_load_case(path), source_path=path)
        validator.validate(payload)
        _lint_case(payload, path=path)
        cases.append(ConformanceCase(path=path, payload=payload))
    return tuple(cases)


def _validate_portable_fixtures() -> None:
    schema = json.loads(ANSWER_PROGRAM_FIXTURE_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)
    for path in sorted(FIXTURE_ROOT.glob("answer_program_v*.yaml")):
        validator.validate(_load_case(path))


def _resolve_local_refs(value: Any, *, source_path: Path) -> Any:
    if isinstance(value, list):
        return [
            _resolve_local_refs(item, source_path=source_path) for item in value
        ]
    if not isinstance(value, dict):
        return value
    if "$ref" not in value:
        return {
            key: _resolve_local_refs(item, source_path=source_path)
            for key, item in value.items()
        }
    if set(value) != {"$ref"}:
        raise ValueError("portable fixture references cannot have sibling fields")
    reference = str(value["$ref"])
    relative_path, separator, pointer = reference.partition("#")
    fixture_path = (source_path.parent / relative_path).resolve()
    if not fixture_path.is_relative_to(FIXTURE_ROOT.resolve()):
        raise ValueError("portable fixture reference must remain inside fixture root")
    referenced = _load_case(fixture_path)
    selected: Any = referenced
    if separator:
        if pointer and not pointer.startswith("/"):
            raise ValueError("portable fixture reference must use a JSON pointer")
        for raw_token in pointer.removeprefix("/").split("/") if pointer else ():
            token = raw_token.replace("~1", "/").replace("~0", "~")
            if not isinstance(selected, dict) or token not in selected:
                raise ValueError(f"portable fixture pointer does not exist: {reference}")
            selected = selected[token]
    return _resolve_local_refs(selected, source_path=fixture_path)


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
