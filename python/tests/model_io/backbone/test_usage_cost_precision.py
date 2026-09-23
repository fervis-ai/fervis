"""Provider usage costs must not inherit the caller's Decimal precision."""

from decimal import Context, Decimal, localcontext

from fervis.model_io.pricing import ModelPricing
from fervis.model_io.providers.chat_runtime import _input_token_cost, _token_cost
from fervis.observability.usage_types import CostSource


def test_token_costs_are_stable_under_low_caller_precision():
    pricing = ModelPricing(
        input_cost_per_million_tokens=12.345678,
        output_cost_per_million_tokens=0,
        thinking_cost_per_million_tokens=0,
        cached_input_cost_per_million_tokens=1.234567,
        pricing_version="test@1", cost_source=CostSource.CONFIGURED_PROVIDER_PRICING,
    )
    with localcontext(Context(prec=50)):
        expected = (
            (Decimal(123_456_789) * Decimal("12.345678")) / Decimal(1_000_000)
        ).quantize(Decimal("0.000001"))
        expected_cached = (
            (Decimal(100_000_000) * Decimal("12.345678")
             + Decimal(23_456_789) * Decimal("1.234567")) / Decimal(1_000_000)
        ).quantize(Decimal("0.000001"))
    for precision in (6, 28, 50):
        with localcontext() as context:
            context.prec = precision
            assert _token_cost(123_456_789, 12.345678) == expected
            assert _input_token_cost(123_456_789, 23_456_789, pricing) == expected_cached
