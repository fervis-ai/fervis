"""Delegated host-read credential capture and replay."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Mapping

from cryptography.fernet import Fernet, InvalidToken

from fervis.host_api.contracts.credentials import DelegatedReadCredential
from fervis.host_api.contracts.execution import ReadTransportOverlay

DEFAULT_CREDENTIAL_TTL_SECONDS = 900
DEFAULT_CREDENTIAL_KEY_ENV = "FERVIS_READ_CREDENTIAL_KEY"
RUNTIME_CONTEXT_CREDENTIAL_KEY = "_fervis_delegated_read_credential"


@dataclass(frozen=True)
class CapturedHeaderCredentialPolicy:
    headers: tuple[str, ...]
    ttl_seconds: int = DEFAULT_CREDENTIAL_TTL_SECONDS
    encryption_key_env: str = DEFAULT_CREDENTIAL_KEY_ENV

    def __post_init__(self) -> None:
        headers = tuple(_header_name(header) for header in self.headers)
        if not headers:
            raise ValueError("captured header credential policy requires headers")
        if len({_header_lookup_key(header) for header in headers}) != len(headers):
            raise ValueError("captured header credential policy has duplicate headers")
        object.__setattr__(self, "headers", headers)
        ttl_seconds = int(self.ttl_seconds)
        if ttl_seconds < 1:
            raise ValueError("captured header credential ttl_seconds must be positive")
        object.__setattr__(self, "ttl_seconds", ttl_seconds)
        key_env = str(self.encryption_key_env or "").strip()
        if not key_env:
            raise ValueError(
                "captured header credential encryption_key_env is required"
            )
        object.__setattr__(self, "encryption_key_env", key_env)


def credential_policy_from_auth_schema(
    schema: Mapping[str, object] | None,
) -> CapturedHeaderCredentialPolicy | None:
    credentials = _mapping((schema or {}).get("credentials"))
    if credentials.get("source") != "captured_request_headers":
        return None
    headers = credentials.get("headers")
    if not isinstance(headers, list):
        return None
    return CapturedHeaderCredentialPolicy(
        headers=tuple(str(header) for header in headers),
        ttl_seconds=int(
            credentials.get("ttl_seconds") or DEFAULT_CREDENTIAL_TTL_SECONDS
        ),
        encryption_key_env=str(
            credentials.get("encryption_key_env") or DEFAULT_CREDENTIAL_KEY_ENV
        ),
    )


def credential_key_env_from_auth_schema(
    schema: Mapping[str, object] | None,
) -> str | None:
    policy = credential_policy_from_auth_schema(schema)
    if policy is None:
        return None
    return policy.encryption_key_env


def capture_header_credential(
    *,
    request_headers: Mapping[str, Any],
    policy: CapturedHeaderCredentialPolicy | None,
    now: datetime | None = None,
) -> DelegatedReadCredential | None:
    if policy is None:
        return None
    captured = _selected_headers(request_headers, policy.headers)
    if not captured:
        from fervis.host_api.contracts.ports import EndpointExecutionError

        raise EndpointExecutionError(
            "Delegated read credential is missing configured auth headers: "
            + ", ".join(policy.headers)
        )
    current = now or datetime.now(UTC)
    expires_at = current + timedelta(seconds=policy.ttl_seconds)
    payload = json.dumps({"headers": captured}, sort_keys=True, separators=(",", ":"))
    encrypted = _fernet(policy.encryption_key_env).encrypt(payload.encode()).decode()
    return DelegatedReadCredential(
        scheme="captured_headers",
        encrypted_payload=encrypted,
        expires_at=_format_timestamp(expires_at),
    )


def overlay_from_header_credential(
    credential: DelegatedReadCredential | None,
    *,
    policy: CapturedHeaderCredentialPolicy | None,
    now: datetime | None = None,
) -> ReadTransportOverlay:
    if credential is None or policy is None:
        return ReadTransportOverlay()
    current = now or datetime.now(UTC)
    if _parse_timestamp(credential.expires_at) <= current:
        from fervis.host_api.contracts.ports import EndpointExecutionError

        raise EndpointExecutionError("Delegated read credential expired.")
    try:
        decoded = _fernet(policy.encryption_key_env).decrypt(
            credential.encrypted_payload.encode()
        )
    except InvalidToken as exc:
        from fervis.host_api.contracts.ports import EndpointExecutionError

        raise EndpointExecutionError(
            "Delegated read credential could not be decrypted."
        ) from exc
    payload = json.loads(decoded.decode())
    headers = _mapping(payload.get("headers"))
    allowed_lookup_keys = {_header_lookup_key(item) for item in policy.headers}
    replay_headers = {
        _header_name(name): str(value)
        for name, value in headers.items()
        if _header_lookup_key(name) in allowed_lookup_keys and str(value or "").strip()
    }
    return ReadTransportOverlay(headers=replay_headers)


def delegated_credential_from_auth_schema(
    *,
    schema: Mapping[str, object] | None,
    request_headers: Mapping[str, Any],
) -> DelegatedReadCredential | None:
    return capture_header_credential(
        request_headers=request_headers,
        policy=credential_policy_from_auth_schema(schema),
    )


def delegated_credential_from_request(
    *,
    schema: Mapping[str, object] | None,
    request: Any,
) -> DelegatedReadCredential | None:
    return delegated_credential_from_auth_schema(
        schema=schema,
        request_headers=getattr(request, "headers", {}) or {},
    )


def credential_overlay_from_auth_schema(
    *,
    schema: Mapping[str, object] | None,
    credential: DelegatedReadCredential | None,
) -> ReadTransportOverlay:
    return overlay_from_header_credential(
        credential,
        policy=credential_policy_from_auth_schema(schema),
    )


def runtime_context_with_delegated_credential(
    runtime_context: Mapping[str, Any] | None,
    credential: DelegatedReadCredential | None,
) -> dict[str, Any]:
    context = dict(runtime_context or {})
    if credential is None:
        context.pop(RUNTIME_CONTEXT_CREDENTIAL_KEY, None)
    else:
        context[RUNTIME_CONTEXT_CREDENTIAL_KEY] = credential.to_storage_dict()
    return context


def delegated_credential_from_runtime_context(
    runtime_context: Mapping[str, Any] | None,
) -> DelegatedReadCredential | None:
    value = dict(runtime_context or {}).get(RUNTIME_CONTEXT_CREDENTIAL_KEY)
    if not isinstance(value, Mapping):
        return None
    return DelegatedReadCredential.from_storage_dict(value)


def _selected_headers(
    request_headers: Mapping[str, Any],
    names: tuple[str, ...],
) -> dict[str, str]:
    lookup = {
        _header_lookup_key(key): str(value)
        for key, value in request_headers.items()
        if str(value or "").strip()
    }
    return {
        name: lookup[_header_lookup_key(name)]
        for name in names
        if _header_lookup_key(name) in lookup
        and str(lookup[_header_lookup_key(name)] or "").strip()
    }


def _header_name(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("captured credential header names must not be empty")
    return text


def _header_lookup_key(value: object) -> str:
    return _header_name(value).lower()


def _fernet(env_name: str) -> Fernet:
    secret = os.getenv(env_name, "")
    if not secret:
        from fervis.host_api.contracts.ports import EndpointExecutionError

        raise EndpointExecutionError(
            f"Delegated read credential encryption key is missing from {env_name}."
        )
    key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())
    return Fernet(key)


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError("delegated credential expires_at is required")
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}
