from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fervis.integrations.registry import register_framework_integration
from fervis.project.integration import FervisConfig, RuntimeRoutes

if TYPE_CHECKING:
    from fervis.interfaces.common.questions import QuestionInterface


@dataclass(frozen=True)
class FlaskIntegration:
    config: FervisConfig
    framework: str = "flask"
    question_interface_factory: Callable[[], QuestionInterface] | None = None
    read_context_capture: Callable[[object], object] | None = None
    delegated_credential_capture: Callable[[object], object] | None = None
    require_read_context: bool = False

    @property
    def routes(self) -> RuntimeRoutes:
        return self.config.routes

    def blueprint(
        self,
        *,
        question_interface: QuestionInterface | None = None,
    ) -> object:
        try:
            from fervis.interfaces.flask import fervis_flask_blueprint
        except ImportError as exc:
            raise RuntimeError(
                "FlaskIntegration.blueprint() requires flask to be installed."
            ) from exc

        return fervis_flask_blueprint(
            question_interface=question_interface,
            question_interface_factory=(
                None
                if question_interface is not None
                else lambda: self._question_interface(None)
            ),
            read_context_capture=self.read_context_capture,
            delegated_credential_capture=self.delegated_credential_capture,
            require_read_context=self.require_read_context,
        )

    def init_app(
        self,
        app: object,
        *,
        question_interface: QuestionInterface | None = None,
    ) -> object:
        register_blueprint = getattr(app, "register_blueprint", None)
        if not callable(register_blueprint):
            raise TypeError("FlaskIntegration.init_app() requires a Flask app.")
        register_blueprint(
            self.blueprint(question_interface=question_interface),
            url_prefix=self.routes.prefix.rstrip("/"),
        )
        return app

    def _question_interface(
        self,
        explicit: QuestionInterface | None,
    ) -> QuestionInterface:
        if explicit is not None:
            return explicit
        if self.question_interface_factory is None:
            raise RuntimeError(
                "FlaskIntegration requires a question_interface for mounted routes."
            )
        return self.question_interface_factory()


register_framework_integration("flask", FlaskIntegration)
