"""Model references used by public Fervis configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelRef:
    provider: str
    model_id: str

    @classmethod
    def parse(cls, value: str) -> "ModelRef":
        provider, separator, model_id = str(value or "").strip().partition(":")
        provider = provider.strip()
        model_id = model_id.strip()
        if not provider or not separator or not model_id:
            raise ValueError("model ref must use provider:model syntax")
        return cls(provider=provider, model_id=model_id)

    def __str__(self) -> str:
        return f"{self.provider}:{self.model_id}"
