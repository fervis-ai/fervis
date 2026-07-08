"""Fervis project integration contract.

This is the small host-owned object that future package extraction can expose
as ``fervis``. It is intentionally declarative: no imports here should inspect
business apps, patch frameworks, or initialize runtime services.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class HostConfig:
    timezone: str
    organization_name: str = ""
    about_api: str = ""


@dataclass(frozen=True)
class RuntimeRoutes:
    prefix: str = "/fervis/"

    @property
    def django_path(self) -> str:
        return f"{self.prefix.strip('/')}/"


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    allowed_model_keys: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ModelConfig:
    default_provider: str
    default_model_key: str
    providers: list[ProviderConfig] = field(default_factory=list)

    @property
    def default_model_ref(self) -> str:
        return f"{self.default_provider}:{self.default_model_key}"


@dataclass(frozen=True)
class SQLitePersistence:
    path: str = ".fervis/fervis.sqlite3"


@dataclass(frozen=True)
class DjangoDatabasePersistence:
    database: str = "default"


@dataclass(frozen=True)
class DatabaseUrlPersistence:
    url_env: str = "FERVIS_DATABASE_URL"


PersistenceTarget = (
    SQLitePersistence | DjangoDatabasePersistence | DatabaseUrlPersistence
)


@dataclass(frozen=True)
class DjangoAppSource:
    name: str
    app_modules: list[str]
    path_prefixes: list[str]
    framework: str = "django"


@dataclass(frozen=True)
class FastAPIAppSource:
    name: str
    import_paths: list[str]
    path_prefixes: list[str]
    framework: str = "fastapi"


@dataclass(frozen=True)
class FlaskAppSource:
    name: str
    app: str
    app_args: list[object] = field(default_factory=list)
    app_kwargs: dict[str, object] = field(default_factory=dict)
    path_prefixes: list[str] = field(default_factory=list)
    blueprints: list[str] = field(default_factory=list)
    framework: str = "flask"


@dataclass(frozen=True)
class FervisConfig:
    host: HostConfig
    routes: RuntimeRoutes
    model: ModelConfig
    sources: list[DjangoAppSource | FastAPIAppSource | FlaskAppSource]
    schema_version: str = "v0.1"
    persistence: PersistenceTarget = field(default_factory=SQLitePersistence)

    @classmethod
    def from_schema(cls, payload: dict[str, object]) -> FervisConfig:
        from .config_schema import config_from_schema

        return config_from_schema(payload)

    def to_schema(self, *, framework: str) -> dict[str, object]:
        from .config_schema import config_to_schema

        return config_to_schema(self, framework=framework)


@dataclass(frozen=True)
class FervisAuthConfig:
    schema: dict[str, object]

    @classmethod
    def from_schema(cls, payload: dict[str, object]) -> FervisAuthConfig:
        from .config_versions.auth import validate_auth_schema

        validate_auth_schema(payload)
        return cls(schema=dict(payload))


@dataclass(frozen=True)
class DjangoIntegration:
    config: FervisConfig
    framework: str = "django"

    @property
    def routes(self) -> RuntimeRoutes:
        return self.config.routes

    @property
    def urls(self) -> str:
        return "fervis.django.urls"


@dataclass(frozen=True)
class FastAPIIntegration:
    config: FervisConfig
    framework: str = "fastapi"
    question_interface_factory: Callable[[], object] | None = None
    read_context_capture: Callable[[object], object] | None = None
    delegated_credential_capture: Callable[[object], object] | None = None
    principal_dependency: Callable[..., object] | None = None
    principal_id_attr: str = "id"
    require_read_context: bool = False

    @property
    def routes(self) -> RuntimeRoutes:
        return self.config.routes

    def router(self, *, question_interface: object | None = None) -> object:
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
            principal_dependency=self.principal_dependency,
            principal_id_attr=self.principal_id_attr,
            require_read_context=self.require_read_context,
        )

    def mount(self, app: object, *, question_interface: object | None = None) -> object:
        include_router = getattr(app, "include_router", None)
        if not callable(include_router):
            raise TypeError("FastAPIIntegration.mount() requires a FastAPI app.")
        prefix = self.routes.prefix.rstrip("/")
        include_router(
            self.router(question_interface=question_interface),
            prefix=prefix,
            include_in_schema=False,
        )
        return app

    def _question_interface(self, explicit: object | None) -> object:
        if explicit is not None:
            return explicit
        if self.question_interface_factory is None:
            raise RuntimeError(
                "FastAPIIntegration requires a question_interface for mounted routes."
            )
        return self.question_interface_factory()


@dataclass(frozen=True)
class FlaskIntegration:
    config: FervisConfig
    framework: str = "flask"
    question_interface_factory: Callable[[], object] | None = None
    read_context_capture: Callable[[object], object] | None = None
    delegated_credential_capture: Callable[[object], object] | None = None
    require_read_context: bool = False

    @property
    def routes(self) -> RuntimeRoutes:
        return self.config.routes

    def blueprint(self, *, question_interface: object | None = None) -> object:
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
        question_interface: object | None = None,
    ) -> object:
        register_blueprint = getattr(app, "register_blueprint", None)
        if not callable(register_blueprint):
            raise TypeError("FlaskIntegration.init_app() requires a Flask app.")
        register_blueprint(
            self.blueprint(question_interface=question_interface),
            url_prefix=self.routes.prefix.rstrip("/"),
        )
        return app

    def _question_interface(self, explicit: object | None) -> object:
        if explicit is not None:
            return explicit
        if self.question_interface_factory is None:
            raise RuntimeError(
                "FlaskIntegration requires a question_interface for mounted routes."
            )
        return self.question_interface_factory()
