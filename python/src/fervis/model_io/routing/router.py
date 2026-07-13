"""Normalized model-router boundary used by Fervis runtimes."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Protocol

from fervis.model_io.backbone.dto import ToolSpec, ProviderOutputMode
from fervis.model_io.backbone.model_routing import resolve_model_route
from fervis.observability.usage_types import CostSource
from fervis.observability.usage_types import UsageKey
from fervis import errors as api_errors


class ModelOutputValidationError(ValueError):
    pass


class ThinkingTokenLimitError(ValueError):
    pass


class ModelAdapter(Protocol):
    def generate(
        self,
        *,
        model_id: str | None = None,
        prompt: str,
        max_thinking_tokens: int,
        system_prompt: str,
        output_mode: ProviderOutputMode = ProviderOutputMode.TEXT,
        tool_specs: tuple[ToolSpec, ...] = (),
    ) -> dict[str, Any]: ...


class ModelRouter:
    """Provider router with structured-output validation gate."""

    def __init__(self, adapters: Mapping[str, ModelAdapter] | None = None):
        self.adapters = dict(adapters or {})

    def _validate_output(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ModelOutputValidationError("Model output must be a JSON object")

        if not isinstance(payload.get("answer"), str):
            raise ModelOutputValidationError(
                "Model output must include string 'answer'"
            )

        usage = payload.get("usage")
        if not isinstance(usage, dict):
            raise ModelOutputValidationError("Model output must include 'usage' object")

        required_usage_keys = {
            UsageKey.INPUT_TOKENS,
            UsageKey.OUTPUT_TOKENS,
            UsageKey.THINKING_TOKENS,
            UsageKey.COST_USD,
        }
        missing = required_usage_keys - set(usage.keys())
        if missing:
            raise ModelOutputValidationError(
                f"Usage payload missing keys: {', '.join(sorted(missing))}"
            )
        _validate_cost_breakdown(usage)

        return payload

    def generate(
        self,
        *,
        provider: str,
        model_key: str | None = None,
        prompt: str,
        max_thinking_tokens: int,
        system_prompt: str,
        output_mode: ProviderOutputMode = ProviderOutputMode.TEXT,
        tool_specs: tuple[ToolSpec, ...] = (),
    ) -> dict[str, Any]:
        adapter = self.adapters.get(provider)
        if adapter is None:
            raise api_errors.APIError.configuration_error(
                "Fervis model provider is not configured."
            )
        route = resolve_model_route(model_key) if model_key else None
        if route is not None and route.provider != provider:
            raise api_errors.APIError.configuration_error(
                "Fervis model provider does not match requested model."
            )

        generate_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "max_thinking_tokens": max_thinking_tokens,
            "system_prompt": system_prompt,
            "output_mode": output_mode,
            "tool_specs": tool_specs,
        }
        if route is not None and ":" in route.model_key:
            generate_kwargs["model_id"] = route.model_id
        payload = adapter.generate(**generate_kwargs)

        validated = self._validate_output(payload)
        thinking_tokens = int(validated["usage"]["thinkingTokens"])
        if thinking_tokens > int(max_thinking_tokens):
            raise ThinkingTokenLimitError(
                f"Thinking token cap exceeded: {thinking_tokens} > {max_thinking_tokens}"
            )

        return validated


def _validate_cost_breakdown(usage: dict[str, Any]) -> None:
    cost = _decimal_usage(usage.get(UsageKey.COST_USD), UsageKey.COST_USD)
    required = {
        UsageKey.INPUT_COST_USD,
        UsageKey.OUTPUT_COST_USD,
        UsageKey.THINKING_COST_USD,
    }
    if cost == 0 and not (required & set(usage)):
        _validate_zero_cost_token_usage(usage)
        return
    missing = required - set(usage)
    if cost == 0:
        _validate_zero_cost_token_usage(usage)
    if missing:
        raise ModelOutputValidationError(
            "Usage payload missing cost breakdown keys: " + ", ".join(sorted(missing))
        )
    component_total = sum(_decimal_usage(usage.get(key), key) for key in required)
    if component_total != cost:
        raise ModelOutputValidationError(
            "Usage payload cost breakdown total must equal costUsd"
        )


def _validate_zero_cost_token_usage(usage: dict[str, Any]) -> None:
    if not _has_token_usage(usage):
        return
    cost_source = str(usage.get(UsageKey.COST_SOURCE) or "")
    if cost_source not in {
        CostSource.CONFIGURED_PROVIDER_PRICING,
        CostSource.PROVIDER_USAGE_UNPRICED,
    }:
        raise ModelOutputValidationError(
            "Usage payload with zero cost and token usage must be marked unpriced"
        )


def _has_token_usage(usage: dict[str, Any]) -> bool:
    return any(
        _decimal_usage(usage.get(key), key) > 0
        for key in (
            UsageKey.INPUT_TOKENS,
            UsageKey.OUTPUT_TOKENS,
            UsageKey.THINKING_TOKENS,
        )
    )


def _decimal_usage(value: Any, key: str) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ModelOutputValidationError(f"Usage payload has invalid {key}") from exc
