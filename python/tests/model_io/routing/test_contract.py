import pytest

from fervis.observability.usage_types import CostSource
from fervis.model_io.routing.router import (
    ModelOutputValidationError,
    ModelRouter,
)


class ContractAdapter:
    def __init__(self):
        self.prompts = []
        self.system_prompts = []

    def generate(
        self,
        *,
        prompt: str,
        max_thinking_tokens: int,
        system_prompt: str = "",
        output_mode=None,
        tool_specs=(),
    ):
        self.system_prompts.append(system_prompt)
        self.prompts.append(prompt)
        return {
            "provider": "contract",
            "answer": "",
            "toolRequests": [],
            "usage": {
                "inputTokens": max(1, len(prompt.split())),
                "outputTokens": 1,
                "thinkingTokens": min(max_thinking_tokens, 4),
                "costUsd": 0.000003,
                "inputCostUsd": 0.000001,
                "outputCostUsd": 0.000001,
                "thinkingCostUsd": 0.000001,
                "costSource": CostSource.CONFIGURED_PROVIDER_PRICING,
                "pricingVersion": "test-provider-2026-05",
            },
        }


def test_model_router_normalizes_provider_response_contract(fervis_foundation_reset):
    router = ModelRouter(adapters={"contract": ContractAdapter()})
    payload = router.generate(
        provider="contract",
        system_prompt="system",
        prompt="find top products",
        max_thinking_tokens=128,
    )

    assert {
        "has_answer": "answer" in payload,
        "usage_keys": set(payload["usage"].keys()),
    } == {
        "has_answer": True,
        "usage_keys": {
            "inputTokens",
            "outputTokens",
            "thinkingTokens",
            "costUsd",
            "inputCostUsd",
            "outputCostUsd",
            "thinkingCostUsd",
            "costSource",
            "pricingVersion",
        },
    }


def test_model_router_returns_usage_tokens_consistently(fervis_foundation_reset):
    router = ModelRouter(adapters={"contract": ContractAdapter()})
    payload = router.generate(
        provider="contract",
        system_prompt="system",
        prompt="hello world",
        max_thinking_tokens=128,
    )

    usage = payload["usage"]
    assert {
        token_key: type(usage[token_key])
        for token_key in ("inputTokens", "outputTokens", "thinkingTokens")
    } == {
        "inputTokens": int,
        "outputTokens": int,
        "thinkingTokens": int,
    }


def test_model_router_sends_raw_prompt_to_provider(fervis_foundation_reset):
    adapter = ContractAdapter()
    router = ModelRouter(adapters={"contract": adapter})

    router.generate(
        provider="contract",
        system_prompt="system",
        prompt="Find sales for owner@example.com and phone +254700000000.",
        max_thinking_tokens=128,
    )

    assert adapter.prompts == [
        "Find sales for owner@example.com and phone +254700000000."
    ]


def test_model_router_sends_system_prompt_to_provider(fervis_foundation_reset):
    adapter = ContractAdapter()
    router = ModelRouter(adapters={"contract": adapter})

    router.generate(
        provider="contract",
        system_prompt="You are Ask Ozai.",
        prompt="Find sales for owner@example.com and phone +254700000000.",
        max_thinking_tokens=128,
    )

    assert adapter.system_prompts == ["You are Ask Ozai."]


class CostWithoutBreakdownAdapter:
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
            "provider": "contract",
            "answer": "ok",
            "toolRequests": [],
            "usage": {
                "inputTokens": 1,
                "outputTokens": 1,
                "thinkingTokens": 0,
                "costUsd": 0.01,
            },
        }


def test_model_router_requires_cost_breakdown_for_nonzero_cost(
    fervis_foundation_reset,
):
    router = ModelRouter(adapters={"contract": CostWithoutBreakdownAdapter()})

    with pytest.raises(ModelOutputValidationError, match="cost breakdown"):
        router.generate(
            provider="contract",
            system_prompt="system",
            prompt="hi",
            max_thinking_tokens=8,
        )


class ContradictingCostBreakdownAdapter:
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
            "provider": "contract",
            "answer": "ok",
            "toolRequests": [],
            "usage": {
                "inputTokens": 1,
                "outputTokens": 1,
                "thinkingTokens": 0,
                "costUsd": 0.001,
                "inputCostUsd": 1.00,
                "outputCostUsd": 0,
                "thinkingCostUsd": 0,
            },
        }


def test_model_router_reconciles_cost_breakdown_with_total(
    fervis_foundation_reset,
):
    router = ModelRouter(adapters={"contract": ContradictingCostBreakdownAdapter()})

    with pytest.raises(ModelOutputValidationError, match="cost breakdown total"):
        router.generate(
            provider="contract",
            system_prompt="system",
            prompt="hi",
            max_thinking_tokens=8,
        )


class UnlabeledZeroCostTokenUsageAdapter:
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
            "provider": "contract",
            "answer": "ok",
            "toolRequests": [],
            "usage": {
                "inputTokens": 1,
                "outputTokens": 1,
                "thinkingTokens": 0,
                "costUsd": 0,
            },
        }


def test_model_router_rejects_unlabeled_zero_cost_token_usage(
    fervis_foundation_reset,
):
    router = ModelRouter(adapters={"contract": UnlabeledZeroCostTokenUsageAdapter()})

    with pytest.raises(ModelOutputValidationError, match="unpriced"):
        router.generate(
            provider="contract",
            system_prompt="system",
            prompt="hi",
            max_thinking_tokens=8,
        )
