"""Prompt section types for Lookup model turns."""

from __future__ import annotations

from dataclasses import dataclass
from fervis.types.enums import StrEnum
from typing import Mapping, Sequence

from fervis.lookup.turn_prompts.rendering import PromptRenderer


class PromptSectionKind(StrEnum):
    TEXT = "text"
    JSON = "json"


@dataclass(frozen=True)
class PromptSection:
    title: str
    content: str | Mapping[str, object] | Sequence[object]
    kind: PromptSectionKind
    json_indent: int | None = None

    def render(self, renderer: PromptRenderer) -> str:
        if self.kind == PromptSectionKind.JSON:
            rendered = renderer.json(self.content, indent=self.json_indent)
        else:
            rendered = renderer.text(self.content)
        if not self.title:
            return rendered
        return f"{self.title}\n{rendered}"


@dataclass(frozen=True)
class ModelPromptPayload:
    system_prompt: str
    prompt_text: str
    sections: tuple[PromptSection, ...]
