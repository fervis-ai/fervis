"""Shared Lookup prompt builder."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from fervis.lookup.turn_prompts.rendering import PromptRenderer
from fervis.lookup.turn_prompts.sections import (
    ModelPromptPayload,
    PromptSection,
    PromptSectionKind,
)
from fervis.lookup.turn_prompts.context import (
    HostPromptContext,
    TurnPromptContext,
)

if TYPE_CHECKING:
    from fervis.lookup.turn_prompts.turn_prompt import TurnPromptBase


LOOKUP_SYSTEM_PROMPT = (
    "You are a framework-neutral Fervis runtime for factual business and "
    "operational questions over host API data. Answer using only the available "
    "endpoint contracts, grounded inputs, conversation context, and returned "
    "data. Do not invent endpoints, fields, values, calculations, business "
    "rules, or domain-specific assumptions that are not supported by the "
    "prompt data."
)


def system_prompt_for(host: HostPromptContext) -> str:
    lines: list[str] = []
    organization_name = host.organization_name.strip()
    if organization_name:
        lines.append(f"You are an Fervis runtime for {organization_name}.")
    about_api = host.about_api.strip()
    if about_api:
        lines.append(f"About the API:\n{about_api}")
    lines.append(LOOKUP_SYSTEM_PROMPT)
    return "\n\n".join(lines)


class TurnPromptBuilder:
    def __init__(
        self,
        context: TurnPromptContext,
        *,
        renderer: PromptRenderer | None = None,
    ) -> None:
        self.context = context
        self.renderer = renderer or PromptRenderer()

    def build(self, turn: "TurnPromptBase") -> ModelPromptPayload:
        sections: list[PromptSection] = []
        if turn.include_current_question:
            sections.append(self.current_question_section())
            if turn.include_active_clarification:
                active_clarification_section = self.active_clarification_section()
                if active_clarification_section is not None:
                    sections.append(active_clarification_section)
        sections.append(self.turn_description_section(turn))
        sections.extend(turn.prompt_sections(self))
        prompt_text = "\n\n".join(section.render(self.renderer) for section in sections)
        return ModelPromptPayload(
            system_prompt=self.system_prompt(),
            prompt_text=prompt_text,
            sections=tuple(sections),
        )

    def system_prompt(self) -> str:
        return system_prompt_for(self.context.host)

    def current_question_section(self) -> PromptSection:
        return PromptSection(
            title="Current question:",
            content=self.context.current_question.strip(),
            kind=PromptSectionKind.TEXT,
        )

    def active_clarification_section(self) -> PromptSection | None:
        active = self.context.active_clarification
        if active is None:
            return None
        return PromptSection(
            title="Active clarification context:",
            content={
                "original_question": active.original_question,
                "exchanges": [
                    {"questions": list(exchange.questions), "answer": exchange.answer}
                    for exchange in active.exchanges
                ],
            },
            kind=PromptSectionKind.JSON,
            json_indent=2,
        )

    def turn_description_section(self, turn: "TurnPromptBase") -> PromptSection:
        return PromptSection(
            title="",
            content=(
                f"We are currently on the {turn.turn_name} step.\n"
                f"Your task is to {turn.turn_task}."
            ),
            kind=PromptSectionKind.TEXT,
        )

    def json_section(
        self,
        title: str,
        payload: object,
        *,
        indent: int | None = None,
    ) -> PromptSection:
        return PromptSection(
            title=title,
            content=payload,  # type: ignore[arg-type]
            kind=PromptSectionKind.JSON,
            json_indent=indent,
        )

    def text_section(
        self,
        title: str,
        text: str,
    ) -> PromptSection:
        return PromptSection(
            title=title,
            content=text,
            kind=PromptSectionKind.TEXT,
        )

    def instruction_block(
        self,
        title: str,
        lines: Iterable[str],
    ) -> PromptSection:
        content = "\n".join(lines)
        return PromptSection(
            title=title,
            content=content,
            kind=PromptSectionKind.TEXT,
        )
