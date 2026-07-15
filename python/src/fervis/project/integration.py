"""Fervis project integration contract.

This is the small host-owned object that future package extraction can expose
as ``fervis``. It is intentionally declarative: no imports here should inspect
business apps, patch frameworks, or initialize runtime services.
"""

from __future__ import annotations

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
