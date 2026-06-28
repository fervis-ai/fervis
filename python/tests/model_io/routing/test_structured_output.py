import pytest

from fervis.model_io.routing.router import (
    ModelOutputValidationError,
    ModelRouter,
)


class InvalidPayloadAdapter:
    def generate(
        self,
        *,
        prompt: str,
        max_thinking_tokens: int,
        system_prompt: str = "",
        output_mode=None,
        tool_specs=(),
    ):
        del system_prompt
        return {"answer": 123, "usage": {"inputTokens": 1}}


def test_model_router_structured_output_validation_rejects_invalid_payload(
    fervis_foundation_reset,
):
    router = ModelRouter(adapters={"broken": InvalidPayloadAdapter()})

    with pytest.raises(ModelOutputValidationError):
        router.generate(
            provider="broken",
            system_prompt="system",
            prompt="q",
            max_thinking_tokens=8,
        )


def test_structured_output_validation_enforced_after_retirement(
    fervis_foundation_reset,
):
    router = ModelRouter(adapters={"broken": InvalidPayloadAdapter()})

    with pytest.raises(ModelOutputValidationError):
        router.generate(
            provider="broken",
            system_prompt="system",
            prompt="retired",
            max_thinking_tokens=8,
        )


def test_strict_schema_rejects_invalid_model_payload(fervis_foundation_reset):
    router = ModelRouter(adapters={"broken": InvalidPayloadAdapter()})

    with pytest.raises(ModelOutputValidationError):
        router.generate(
            provider="broken",
            system_prompt="system",
            prompt="invalid",
            max_thinking_tokens=8,
        )
