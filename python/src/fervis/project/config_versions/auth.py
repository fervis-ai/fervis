"""Current Fervis auth JSON schema contract."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

from fervis.project.config_versions.common import (
    reject_unknown_keys,
    require_mapping,
    require_string,
)

AUTH_CONFIG_SCHEMA_VERSION = "v0.1"


def normalize_auth_schema(
    payload: Mapping[str, object],
) -> tuple[dict[str, object], str | None]:
    """Return the current auth schema plus the source version, if upgraded."""

    version = payload.get("schema_version")
    if version != AUTH_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Fervis auth schema_version must be {AUTH_CONFIG_SCHEMA_VERSION}."
        )
    schema = deepcopy(dict(payload))
    validate_auth_schema(schema)
    return schema, None


def validate_auth_schema(schema: Mapping[str, object]) -> None:
    reject_unknown_keys(
        schema,
        allowed={
            "schema_version",
            "framework",
            "security",
            "principal",
            "environments",
        },
    )
    if schema.get("schema_version") != AUTH_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Fervis auth schema_version must be {AUTH_CONFIG_SCHEMA_VERSION}."
        )
    require_string(schema, "framework")
    _validate_security(require_mapping(schema, "security"))
    _validate_principal(
        require_mapping(schema, "principal"),
        framework=schema["framework"],
    )
    environments = require_mapping(schema, "environments")
    if not environments:
        raise ValueError("environments must contain at least one environment.")
    for name, environment in environments.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("environment names must be non-empty strings.")
        if not isinstance(environment, Mapping):
            raise ValueError(f"environment {name!r} must be an object.")
        _validate_environment(environment, name=name, schema=schema)


def active_auth_schema(
    schema: Mapping[str, object],
    *,
    environment_name: str,
) -> dict[str, object]:
    environments = require_mapping(schema, "environments")
    environment = environments.get(environment_name)
    if not isinstance(environment, Mapping):
        raise ValueError(
            f"Fervis auth environment {environment_name!r} is not declared."
        )
    return {
        "schema_version": schema["schema_version"],
        "framework": schema["framework"],
        "security": deepcopy(schema["security"]),
        "principal": deepcopy(schema["principal"]),
        "transport": deepcopy(environment["transport"]),
        "credentials": deepcopy(environment.get("credentials", {})),
    }


def _validate_security(security: Mapping[str, object]) -> None:
    reject_unknown_keys(security, allowed={"mode"}, prefix="security")
    mode = require_string(security, "mode")
    if mode != "principal_reauthorization":
        raise ValueError("security.mode must be principal_reauthorization.")


def _validate_principal(principal: Mapping[str, object], *, framework: object) -> None:
    source = require_string(principal, "source")
    if source == "django_request_user":
        reject_unknown_keys(
            principal,
            allowed={"source", "id_attr"},
            prefix="principal",
        )
        require_string(principal, "id_attr")
        if framework != "django":
            raise ValueError("django_request_user principal requires framework django.")
        return
    if source == "fastapi_dependency":
        reject_unknown_keys(
            principal,
            allowed={"source", "dependency", "id_attr", "resolver"},
            prefix="principal",
        )
        require_string(principal, "dependency")
        require_string(principal, "id_attr")
        if framework != "fastapi":
            raise ValueError("fastapi_dependency principal requires framework fastapi.")
        return
    if source in {"flask_login_current_user", "flask_g"}:
        reject_unknown_keys(
            principal,
            allowed={"source", "id_attr", "resolver"},
            prefix="principal",
        )
        require_string(principal, "id_attr")
        if principal.get("resolver") is not None:
            require_string(principal, "resolver")
        if framework != "flask":
            raise ValueError(f"{source} principal requires framework flask.")
        return
    if source == "callable":
        reject_unknown_keys(
            principal,
            allowed={"source", "capture", "resolver"},
            prefix="principal",
        )
        require_string(principal, "capture")
        require_string(principal, "resolver")
        if framework != "flask":
            raise ValueError("callable principal requires framework flask.")
        return
    raise ValueError(f"Unsupported principal source {source!r}.")


def _validate_environment(
    environment: Mapping[str, object],
    *,
    name: str,
    schema: Mapping[str, object],
) -> None:
    reject_unknown_keys(
        environment,
        allowed={"transport", "credentials"},
        prefix=f"environments.{name}",
    )
    credentials = environment.get("credentials")
    if credentials is not None:
        _validate_credentials(credentials, name=name)
    transport = require_mapping(environment, "transport")
    mode = require_string(transport, "mode")
    if mode == "in_process":
        reject_unknown_keys(
            transport,
            allowed={"mode"},
            prefix=f"environments.{name}.transport",
        )
        _validate_in_process(schema, name=name)
        return
    if mode == "http":
        reject_unknown_keys(
            transport,
            allowed={
                "mode",
                "base_url_env",
                "request_overlay_source",
                "auth_query_params",
            },
            prefix=f"environments.{name}.transport",
        )
        require_string(transport, "base_url_env")
        require_string(transport, "request_overlay_source")
        query_params = transport.get("auth_query_params", [])
        if not isinstance(query_params, list) or any(
            not isinstance(param, str) or not param.strip() for param in query_params
        ):
            raise ValueError(
                f"environments.{name}.transport.auth_query_params "
                "must be a list of strings."
            )
        return
    raise ValueError(f"Unsupported auth transport mode {mode!r}.")


def _validate_credentials(value: object, *, name: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"environments.{name}.credentials must be an object.")
    reject_unknown_keys(
        value,
        allowed={"source", "headers", "ttl_seconds", "encryption_key_env"},
        prefix=f"environments.{name}.credentials",
    )
    source = require_string(value, "source")
    if source != "captured_request_headers":
        raise ValueError(
            f"environments.{name}.credentials.source must be captured_request_headers."
        )
    headers = value.get("headers")
    if not isinstance(headers, list) or any(
        not isinstance(header, str) or not header.strip() for header in headers
    ):
        raise ValueError(
            f"environments.{name}.credentials.headers must be a list of strings."
        )
    if "ttl_seconds" in value:
        try:
            ttl_seconds = int(value["ttl_seconds"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"environments.{name}.credentials.ttl_seconds must be an integer."
            ) from exc
        if ttl_seconds < 1:
            raise ValueError(
                f"environments.{name}.credentials.ttl_seconds must be positive."
            )
    if "encryption_key_env" in value:
        require_string(value, "encryption_key_env")


def _validate_in_process(schema: Mapping[str, object], *, name: str) -> None:
    principal = require_mapping(schema, "principal")
    if principal.get("source") == "fastapi_dependency":
        require_string(principal, "resolver")
    elif principal.get("source") == "django_request_user":
        return
    elif principal.get("source") in {
        "flask_login_current_user",
        "flask_g",
        "callable",
    }:
        if principal.get("source") in {"flask_login_current_user", "flask_g"}:
            resolver = principal.get("resolver")
            if resolver is not None:
                require_string(principal, "resolver")
            return
        require_string(principal, "capture")
        require_string(principal, "resolver")
    else:
        raise ValueError(f"environments.{name}.transport principal is invalid.")
