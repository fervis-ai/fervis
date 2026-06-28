"""Usage and cost source constants."""

from __future__ import annotations


class UsageKey:
    INPUT_TOKENS = "inputTokens"
    OUTPUT_TOKENS = "outputTokens"
    THINKING_TOKENS = "thinkingTokens"
    COST_USD = "costUsd"
    INPUT_COST_USD = "inputCostUsd"
    OUTPUT_COST_USD = "outputCostUsd"
    THINKING_COST_USD = "thinkingCostUsd"
    COST_SOURCE = "costSource"
    PRICING_VERSION = "pricingVersion"
    MODEL_SUBCALLS = "modelSubcalls"


class CostSource:
    CONFIGURED_PROVIDER_PRICING = "configured_provider_pricing"
    MODELS_DEV = "models_dev"
    PROVIDER_USAGE_UNPRICED = "provider_usage_unpriced"
    LINEAGE_MODEL_CALL_USAGE = "lineage_model_call_usage"
