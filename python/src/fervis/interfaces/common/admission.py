"""Public Fervis operation admission contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fervis.model_io.models import ModelRef
from fervis.project.integration import ModelConfig


class ModelPolicyValidationError(ValueError):
    def __init__(self, *, field: str, message: str) -> None:
        super().__init__(message)
        self.field = field
        self.message = message


@dataclass(frozen=True)
class AdmittedModel:
    provider: str
    model_key: str


@dataclass(frozen=True)
class ConfiguredModelPolicy:
    default_provider: str = ""
    default_model_key: str = ""
    allowed_model_keys_by_provider: dict[str, frozenset[str]] = field(
        default_factory=dict
    )

    @classmethod
    def from_config(cls, config: ModelConfig) -> "ConfiguredModelPolicy":
        return cls(
            default_provider=config.default_provider,
            default_model_key=config.default_model_key,
            allowed_model_keys_by_provider={
                provider.name: frozenset(provider.allowed_model_keys)
                for provider in config.providers
            },
        )

    def admit(
        self,
        *,
        requested_provider: Any,
        requested_model_key: str,
    ) -> AdmittedModel:
        requested = str(requested_model_key or "").strip()
        ref = _model_ref(requested) if requested else self.default_ref()
        provider = str(requested_provider or "").strip()
        if provider and provider != ref.provider:
            raise ModelPolicyValidationError(
                field="provider",
                message="provider must match the configured Fervis model policy.",
            )
        allowed_model_keys = self.allowed_model_keys_by_provider.get(
            ref.provider,
            frozenset(),
        )
        if ref.model_id not in allowed_model_keys:
            raise ModelPolicyValidationError(
                field="modelKey",
                message=(
                    "modelKey is not allowed by the configured Fervis model policy."
                ),
            )
        return AdmittedModel(provider=ref.provider, model_key=str(ref))

    def default_ref(self) -> ModelRef:
        if not self.default_provider or not self.default_model_key:
            raise ModelPolicyValidationError(
                field="modelKey",
                message="Fervis model policy is not configured.",
            )
        return ModelRef(
            provider=self.default_provider,
            model_id=self.default_model_key,
        )


def _model_ref(value: str) -> ModelRef:
    return ModelRef.parse(value)
