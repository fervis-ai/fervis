from __future__ import annotations

import ast
from pathlib import Path

import fervis
from fervis.lookup.turn_prompts.builder import LOOKUP_SYSTEM_PROMPT


FERVIS_ROOT = Path(fervis.__file__).resolve().parent
LOOKUP_ROOT = FERVIS_ROOT / "lookup"
TURN_PROMPTS_ROOT = LOOKUP_ROOT / "turn_prompts"


PROMPT_SURFACE_CONTRACTS = {
    "conversation_resolution": {
        "path": LOOKUP_ROOT / "conversation_resolution" / "prompt.py",
        "instruction_headings": (
            "Task",
            "Clause Resolution",
            "Other Dependencies",
            "Resolved Clauses",
            "Needs Clarification",
            "Boundaries",
            "Output",
        ),
    },
    "grounding": {
        "path": LOOKUP_ROOT / "grounding" / "prompt.py",
        "instruction_headings": (
            "Grounding Objective",
            "Time Resolution",
            "Resolver Selection",
            "Copying And Validity",
            "Output",
        ),
    },
    "plan_selection": {
        "path": LOOKUP_ROOT / "plan_selection" / "prompt.py",
        "instruction_headings": (
            "Task Boundary",
            "Source Alignment",
            "Validity",
            "Output",
        ),
    },
    "fact_planning": {
        "path": LOOKUP_ROOT / "fact_planning" / "prompt_sections.py",
        "instruction_headings": (
            "Decision Scope",
            "Answer Identity",
            "Source Selection",
            "Field Selection",
            "List And Field Patterns",
            "Metric Patterns",
            "Grouped Metric Patterns",
            "Computed Scalar",
            "Set Difference",
            "Joined Rows",
            "Copying And Validity",
            "Output",
        ),
    },
    "query_enrichment": {
        "path": LOOKUP_ROOT / "query_enrichment" / "prompt.py",
        "instruction_headings": (
            "Vocabulary Scope",
            "Conversation Resolution Annotations",
            "Answer Output Resource Lineage",
            "Resource Lineage Matches",
            "Reference Value Resolver Search Terms",
            "Boundaries",
            "Copying And Validity",
            "Output",
        ),
    },
    "question_contract": {
        "path": LOOKUP_ROOT / "question_contract" / "prompt.py",
        "instruction_headings": (
            "Decision Scope",
            "Question Boundary",
            "Clarification Boundary",
            "Answer Requests",
            "Answer Outputs",
            "Question Inputs Overview",
            "Question Input Inventory",
            "Question Input Sources",
            "Conversation Resolution Inputs",
            "Literal Reference Inputs",
            "Literal Time Inputs",
            "Literal Limits",
            "Output",
        ),
    },
    "read_eligibility": {
        "path": LOOKUP_ROOT / "read_eligibility" / "prompt.py",
        "instruction_headings": (
            "Task Boundary",
            "Conversation Resolution",
            "Retention Rules",
            "Output Shape",
            "Validity",
            "Output",
        ),
    },
    "source_binding": {
        "path": LOOKUP_ROOT / "source_binding" / "prompt.py",
        "instruction_headings": (
            "Source Binding",
            "Source Population And Fulfillment",
            "Param Binding",
            "Row Predicates",
            "Finite Choice Review Shape",
            "Population Test Basis",
            "Choice Test Results",
            "Normal Instance Guard",
            "Test Effect Semantics",
            "Finite Choice Guardrails",
            "Terminal Outcomes",
            "Copying And Validity",
            "Output",
        ),
    },
}


PROMPT_HYGIENE_RULES = {
    "Retail Ops": "prompt rules must remain host-API neutral",
    "Ask Ozai": "prompt rules must remain product-name neutral",
    "submit_source_invocation_selection": "deleted source-binding turn contract",
    "submit_source_param_value_bindings": "deleted source-binding turn contract",
    "row_predicate_decisions": "row filters must be derived from reviews",
    "operation_support_set_id": "fact planning uses compact choice aliases",
    "metric_option_id": "fact planning uses compact choice aliases",
    "row_predicates are optional": "row predicate reviews are mandatory when shown",
    "optional response-row filters": (
        "row predicate reviews are mandatory when shown"
    ),
    "shown in finite_choice_param_reviews": (
        "finite-choice population params are shown in binding_params with population_contract"
    ),
    "cash deposited today": "prompt rules must not use business-specific examples",
}

PROMPT_HYGIENE_PATHS = (
    *(contract["path"] for contract in PROMPT_SURFACE_CONTRACTS.values()),
    TURN_PROMPTS_ROOT / "builder.py",
)

SYSTEM_PROMPT_REQUIRED_TEXT = (
    "framework-neutral Fervis runtime",
    "host API data",
    "only the available endpoint contracts",
    "domain-specific assumptions",
)


def prompt_instruction_heading_failures() -> list[str]:
    failures: list[str] = []
    for name, contract in PROMPT_SURFACE_CONTRACTS.items():
        actual = _instruction_block_headings(contract["path"])
        expected = list(contract["instruction_headings"])
        if actual != expected:
            failures.append(f"{name}: headings {actual!r} did not match {expected!r}")
    return failures


def prompt_hygiene_failures() -> list[str]:
    failures: list[str] = []
    for path in PROMPT_HYGIENE_PATHS:
        text = path.read_text(encoding="utf-8")
        for banned, reason in PROMPT_HYGIENE_RULES.items():
            if banned in text:
                failures.append(f"{path}: {banned!r}: {reason}")
    return failures


def shared_system_prompt_failures() -> list[str]:
    failures: list[str] = []
    for text in SYSTEM_PROMPT_REQUIRED_TEXT:
        if text not in LOOKUP_SYSTEM_PROMPT:
            failures.append(f"system prompt missing {text!r}")
    return failures


def _instruction_block_headings(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    headings: list[str] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "instruction_block"
        ):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        heading = node.args[0].value
        if isinstance(heading, str):
            headings.append(heading)
    return headings
