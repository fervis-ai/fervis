"""Model-key routing for Fervis provider selection."""

from __future__ import annotations

from dataclasses import dataclass

from fervis.model_io.models import ModelRef
from fervis.model_io.providers.specs import supported_provider_spec


class ModelKey:
    HAIKU = "HAIKU"
    GPT_5_4_MINI = "GPT_5_4_MINI"


DEFAULT_MODEL_KEY = ModelKey.HAIKU


class ProviderName:
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


@dataclass(frozen=True)
class ModelRoute:
    model_key: str
    provider: str
    model_id: str


MODEL_ROUTES: dict[str, ModelRoute] = {
    ModelKey.HAIKU: ModelRoute(
        model_key=ModelKey.HAIKU,
        provider=ProviderName.ANTHROPIC,
        model_id="claude-haiku-4-5-20251001",
    ),
    ModelKey.GPT_5_4_MINI: ModelRoute(
        model_key=ModelKey.GPT_5_4_MINI,
        provider=ProviderName.OPENAI,
        model_id="gpt-5.4-mini",
    ),
}


def resolve_model_route(model_key: str | None) -> ModelRoute:
    normalized = (model_key or DEFAULT_MODEL_KEY).strip().upper()
    if not normalized:
        normalized = DEFAULT_MODEL_KEY
    if ":" in str(model_key or ""):
        ref = ModelRef.parse(str(model_key))
        supported_provider_spec(ref.provider)
        return ModelRoute(
            model_key=str(ref),
            provider=ref.provider,
            model_id=ref.model_id,
        )
    try:
        return MODEL_ROUTES[normalized]
    except KeyError as exc:
        raise ValueError(f"Unknown Fervis model key: {model_key}") from exc
