"""Models.dev-backed pricing lookup."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from fervis.observability.usage_types import CostSource


MODELS_DEV_API_URL = "https://models.dev/api.json"


@dataclass(frozen=True)
class ModelPricing:
    input_cost_per_million_tokens: float
    output_cost_per_million_tokens: float
    thinking_cost_per_million_tokens: float
    pricing_version: str
    cost_source: str

    @property
    def priced(self) -> bool:
        return self.cost_source != CostSource.PROVIDER_USAGE_UNPRICED

    @classmethod
    def unpriced(cls, *, pricing_version: str) -> "ModelPricing":
        return cls(
            input_cost_per_million_tokens=0,
            output_cost_per_million_tokens=0,
            thinking_cost_per_million_tokens=0,
            pricing_version=pricing_version,
            cost_source=CostSource.PROVIDER_USAGE_UNPRICED,
        )


_CATALOG_CACHE: dict[str, Any] | None = None


def resolve_model_pricing(*, provider: str, model_key: str) -> ModelPricing:
    provider_key = str(provider or "").strip()
    model_id = str(model_key or "").strip()
    version = f"models.dev:{provider_key}/{model_id}"
    if not provider_key or not model_id:
        return ModelPricing.unpriced(pricing_version=version)

    catalog = _load_catalog()
    provider_payload = _dict(catalog.get(provider_key))
    models = _dict(provider_payload.get("models"))
    model = _dict(models.get(model_id))
    cost = _dict(model.get("cost"))
    input_cost = _cost_rate(cost.get("input"))
    output_cost = _cost_rate(cost.get("output"))
    if input_cost is None or output_cost is None:
        return ModelPricing.unpriced(pricing_version=version)
    return ModelPricing(
        input_cost_per_million_tokens=float(input_cost),
        output_cost_per_million_tokens=float(output_cost),
        thinking_cost_per_million_tokens=float(output_cost),
        pricing_version=version,
        cost_source=CostSource.MODELS_DEV,
    )


def _load_catalog() -> dict[str, Any]:
    global _CATALOG_CACHE
    if _CATALOG_CACHE is not None:
        return _CATALOG_CACHE
    try:
        request = Request(
            MODELS_DEV_API_URL,
            headers={
                "Accept": "application/json",
                "User-Agent": "fervis-pricing-resolver",
            },
        )
        with urlopen(request, timeout=5) as response:
            payload = json.load(response)
    except (OSError, URLError, json.JSONDecodeError, TimeoutError):
        return {}
    _CATALOG_CACHE = _dict(payload)
    return _CATALOG_CACHE


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _cost_rate(value: Any) -> Decimal | None:
    try:
        rate = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    if rate < 0:
        return None
    return rate
