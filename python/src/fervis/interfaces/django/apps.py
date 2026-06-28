from django.apps import AppConfig
from django.conf import settings


class FervisDjangoInterfaceConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "fervis.interfaces.django"
    label = "fervis_django_interface"

    def ready(self) -> None:
        from fervis.host_api.context import HostContext
        from fervis.host_api.context import (
            HostApiContext,
            configure_host_api_context,
        )
        from fervis.host_api.adapters.django.adapter import (
            DjangoHostApiAdapter,
        )

        host_context = HostContext(
            **dict(getattr(settings, "FERVIS_HOST_CONTEXT", {}) or {})
        )
        from fervis.project.source_scope import configured_django_source_scopes

        source_scopes = configured_django_source_scopes()
        adapter = DjangoHostApiAdapter(sources=source_scopes)
        configure_host_api_context(
            HostApiContext(
                adapter=adapter,
                host_context=host_context,
            )
        )
