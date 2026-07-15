from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fervis.integrations.registry import register_framework_integration
from fervis.project.integration import FervisConfig, RuntimeRoutes

if TYPE_CHECKING:
    from fervis.interfaces.common.questions import QuestionInterface


@dataclass(frozen=True)
class FastAPIIntegration:
    config: FervisConfig
    framework: str = "fastapi"
    question_interface_factory: Callable[[], QuestionInterface] | None = None
    read_context_capture: Callable[[object], object] | None = None
    delegated_credential_capture: Callable[[object], object] | None = None
    principal_dependency: Callable[..., object] | None = None
    principal_dependency_factory: Callable[[], Callable[..., object]] | None = None
    principal_id_attr: str = "id"
    require_read_context: bool = False

    @property
    def routes(self) -> RuntimeRoutes:
        return self.config.routes

    def router(self, *, question_interface: QuestionInterface | None = None) -> object:
        try:
            from fervis.interfaces.fastapi.router import fervis_fastapi_router
        except ImportError as exc:
            raise RuntimeError(
                "FastAPIIntegration.router() requires fastapi to be installed."
            ) from exc

        return fervis_fastapi_router(
            question_interface=self._question_interface(question_interface),
            read_context_capture=self.read_context_capture,
            delegated_credential_capture=self.delegated_credential_capture,
            principal_dependency=self._principal_dependency(),
            principal_id_attr=self.principal_id_attr,
            require_read_context=self.require_read_context,
        )

    def mount(
        self,
        app: object,
        *,
        question_interface: QuestionInterface | None = None,
    ) -> object:
        include_router = getattr(app, "include_router", None)
        if not callable(include_router):
            raise TypeError("FastAPIIntegration.mount() requires a FastAPI app.")
        interface = self._question_interface(question_interface)
        prefix = self.routes.prefix.rstrip("/")
        include_router(
            self.router(question_interface=interface),
            prefix=prefix,
            include_in_schema=False,
        )
        _close_question_interface_with_fastapi_app(app, interface)
        return app

    def _principal_dependency(self) -> Callable[..., object] | None:
        if self.principal_dependency is not None:
            return self.principal_dependency
        if self.principal_dependency_factory is None:
            return None
        return self.principal_dependency_factory()

    def _question_interface(
        self,
        explicit: QuestionInterface | None,
    ) -> QuestionInterface:
        if explicit is not None:
            return explicit
        if self.question_interface_factory is None:
            raise RuntimeError(
                "FastAPIIntegration requires a question_interface for mounted routes."
            )
        return self.question_interface_factory()


def _close_question_interface_with_fastapi_app(
    app: object,
    question_interface: QuestionInterface,
) -> None:
    router = getattr(app, "router", None)
    host_lifespan = getattr(router, "lifespan_context", None)
    if router is None or not callable(host_lifespan):
        raise TypeError("FastAPIIntegration.mount() requires an application lifespan.")

    @asynccontextmanager
    async def lifespan_with_fervis(host_app: object):
        async with host_lifespan(host_app) as lifespan_state:
            try:
                yield lifespan_state
            finally:
                await asyncio.to_thread(question_interface.close)

    router.lifespan_context = lifespan_with_fervis


register_framework_integration("fastapi", FastAPIIntegration)
