from __future__ import annotations

from tests.testkit.prompt_surfaces import (
    prompt_hygiene_failures,
    prompt_instruction_heading_failures,
    shared_system_prompt_failures,
)


def test_model_turn_instruction_blocks_keep_required_headings():
    assert prompt_instruction_heading_failures() == []


def test_prompt_rule_text_does_not_use_banned_or_stale_terms():
    assert prompt_hygiene_failures() == []


def test_shared_system_prompt_keeps_runtime_boundary_terms():
    assert shared_system_prompt_failures() == []
