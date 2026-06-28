import pytest

from fervis.model_io.routing.router import ModelRouter, ThinkingTokenLimitError


class HighThinkingAdapter:
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
        return {
            "answer": "x",
            "toolRequests": [],
            "usage": {
                "inputTokens": 1,
                "outputTokens": 1,
                "thinkingTokens": max_thinking_tokens + 10,
                "costUsd": 0.01,
                "inputCostUsd": 0.004,
                "outputCostUsd": 0.004,
                "thinkingCostUsd": 0.002,
            },
        }


def test_model_router_enforces_max_thinking_tokens_cap(fervis_foundation_reset):
    router = ModelRouter(adapters={"strict": HighThinkingAdapter()})

    with pytest.raises(ThinkingTokenLimitError):
        router.generate(
            provider="strict",
            system_prompt="system",
            prompt="hi",
            max_thinking_tokens=8,
        )
