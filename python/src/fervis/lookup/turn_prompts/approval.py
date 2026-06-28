"""Approved prompt-size metadata for Lookup model-turn baselines."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ApprovedPromptChars:
    turn_name: str
    fixture_name: str
    system_prompt_chars: int
    prompt_text_chars: int
    provider_payload_chars: int


@dataclass(frozen=True)
class PromptApprovalManifest:
    approved_chars: tuple[ApprovedPromptChars, ...] = ()

    def maximum_approved_chars_for(
        self,
        *,
        turn_name: str,
        fixture_name: str,
    ) -> ApprovedPromptChars:
        for item in self.approved_chars:
            if item.turn_name == turn_name and item.fixture_name == fixture_name:
                return item
        raise KeyError(f"missing approved prompt chars: {turn_name}/{fixture_name}")
