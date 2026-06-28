"""Base class for structured Lookup model-turn prompts."""

from __future__ import annotations

from abc import ABC

from fervis.lookup.turn_prompts.builder import TurnPromptBuilder
from fervis.lookup.turn_prompts.context import TurnPromptContext
from fervis.lookup.turn_prompts.invocation import (
    ModelTurnInvocation,
    ProviderInvocationMetadata,
    ProviderResponseContract,
    ProviderToolContract,
)
from fervis.lookup.turn_prompts.sections import (
    ModelPromptPayload,
    PromptSection,
)


class TurnPromptBase(ABC):
    turn_name: str
    turn_task: str
    include_current_question: bool = True
    include_active_clarification: bool = False

    def prompt_sections(
        self,
        builder: TurnPromptBuilder,
    ) -> tuple[PromptSection, ...]:
        return self.data_sections(builder) + self.instruction_sections(builder)

    def data_sections(
        self,
        builder: TurnPromptBuilder,
    ) -> tuple[PromptSection, ...]:
        del builder
        return ()

    def instruction_sections(
        self,
        builder: TurnPromptBuilder,
    ) -> tuple[PromptSection, ...]:
        del builder
        return ()

    def response_contract(self) -> ProviderResponseContract:
        raise NotImplementedError

    def tool_contract(self) -> ProviderToolContract:
        raise NotImplementedError

    def provider_metadata(self) -> ProviderInvocationMetadata:
        return ProviderInvocationMetadata()

    def to_model_payload(self, context: TurnPromptContext) -> ModelPromptPayload:
        return TurnPromptBuilder(context).build(self)

    def to_model_invocation(
        self,
        context: TurnPromptContext,
    ) -> ModelTurnInvocation:
        return ModelTurnInvocation(
            turn_name=self.turn_name,
            prompt=self.to_model_payload(context),
            response_contract=self.response_contract(),
            tool_contract=self.tool_contract(),
            metadata=self.provider_metadata(),
        )
