from __future__ import annotations

from dataclasses import dataclass

from fervis.integrations.registry import register_framework_integration
from fervis.project.integration import FervisConfig, RuntimeRoutes


@dataclass(frozen=True)
class DjangoIntegration:
    config: FervisConfig
    framework: str = "django"

    @property
    def routes(self) -> RuntimeRoutes:
        return self.config.routes

    @property
    def urls(self) -> str:
        return "fervis.integrations.django.urls"


register_framework_integration("django", DjangoIntegration)
