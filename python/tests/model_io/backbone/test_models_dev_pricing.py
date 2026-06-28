from fervis.model_io.pricing import models_dev
from fervis.observability.usage_types import CostSource


def test_models_dev_pricing_resolves_provider_model_cost(monkeypatch):
    monkeypatch.setattr(
        models_dev,
        "_load_catalog",
        lambda: {
            "openai": {
                "models": {
                    "gpt-5.4-mini": {
                        "cost": {
                            "input": 0.4,
                            "output": 3,
                        }
                    }
                }
            }
        },
    )

    pricing = models_dev.resolve_model_pricing(
        provider="openai",
        model_key="gpt-5.4-mini",
    )

    assert (
        pricing.input_cost_per_million_tokens,
        pricing.output_cost_per_million_tokens,
        pricing.thinking_cost_per_million_tokens,
        pricing.pricing_version,
        pricing.cost_source,
    ) == (
        0.4,
        3,
        3,
        "models.dev:openai/gpt-5.4-mini",
        CostSource.MODELS_DEV,
    )


def test_models_dev_pricing_returns_unpriced_for_missing_cost(monkeypatch):
    monkeypatch.setattr(
        models_dev,
        "_load_catalog",
        lambda: {"openai": {"models": {"gpt-5.4-mini": {}}}},
    )

    pricing = models_dev.resolve_model_pricing(
        provider="openai",
        model_key="gpt-5.4-mini",
    )

    assert (
        pricing.priced,
        pricing.cost_source,
        pricing.pricing_version,
    ) == (
        False,
        CostSource.PROVIDER_USAGE_UNPRICED,
        "models.dev:openai/gpt-5.4-mini",
    )
