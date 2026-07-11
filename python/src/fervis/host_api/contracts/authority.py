"""Host API read authority contracts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Literal, Mapping

from fervis.host_api.contracts.credentials import DelegatedReadCredential


ReadContextScheme = Literal[
    "anonymous",
    "django_principal",
    "fastapi_principal",
    "flask_principal",
    "delegated_capability",
]

_READ_CONTEXT_REF_KEYS = frozenset({"scheme", "key", "tenant_key"})


@dataclass(frozen=True)
class ReadContextRef:
    """Persisted non-secret host-owned context for reauthorizing reads."""

    scheme: ReadContextScheme
    key: str | None = None
    tenant_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "scheme", _read_context_scheme(self.scheme))
        object.__setattr__(self, "key", _optional_string(self.key))
        object.__setattr__(self, "tenant_key", _optional_string(self.tenant_key))

    def to_storage_dict(self) -> dict[str, str | None]:
        return {
            "scheme": self.scheme,
            "key": self.key,
            "tenant_key": self.tenant_key,
        }

    def matches_storage_dict(self, value: Mapping[str, Any]) -> bool:
        try:
            stored = type(self).from_storage_dict(value or {})
        except (TypeError, ValueError):
            return False
        return stored.to_storage_dict() == self.to_storage_dict()

    @classmethod
    def from_storage_dict(cls, value: Mapping[str, Any]) -> "ReadContextRef":
        keys = set(value)
        unexpected = keys - _READ_CONTEXT_REF_KEYS
        if unexpected:
            raise ValueError(
                "unexpected ReadContextRef keys: " + ", ".join(sorted(unexpected))
            )
        return cls(
            scheme=value.get("scheme"),
            key=value.get("key"),
            tenant_key=value.get("tenant_key"),
        )


@dataclass(frozen=True)
class ReadAuthority:
    """Durable subject scope for Fervis-owned state and host reads."""

    tenant_id: str
    read_context_ref: ReadContextRef
    delegated_credential: DelegatedReadCredential | None = None

    def __post_init__(self) -> None:
        tenant_id = str(self.tenant_id or "").strip()
        if not tenant_id:
            raise ValueError("read authority tenant_id is required")
        object.__setattr__(self, "tenant_id", tenant_id)
        if not isinstance(self.read_context_ref, ReadContextRef):
            object.__setattr__(
                self,
                "read_context_ref",
                ReadContextRef.from_storage_dict(self.read_context_ref),
            )
        if self.delegated_credential is not None and not isinstance(
            self.delegated_credential,
            DelegatedReadCredential,
        ):
            object.__setattr__(
                self,
                "delegated_credential",
                DelegatedReadCredential.from_storage_dict(self.delegated_credential),
            )

    @classmethod
    def from_principal(cls, principal) -> "ReadAuthority":
        return cls(
            tenant_id=principal.tenant_id,
            read_context_ref=principal.read_context_ref,
            delegated_credential=getattr(principal, "delegated_credential", None),
        )

    @property
    def evidence_ref(self) -> str:
        payload = json.dumps(
            {
                "tenant_id": self.tenant_id,
                "read_context_ref": self.read_context_ref.to_storage_dict(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return "authority:sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def _read_context_scheme(value: Any) -> ReadContextScheme:
    scheme = str(value or "").strip()
    if scheme not in {
        "anonymous",
        "django_principal",
        "fastapi_principal",
        "flask_principal",
        "delegated_capability",
    }:
        raise ValueError(f"unsupported ReadContextRef scheme: {scheme}")
    return scheme  # type: ignore[return-value]


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def read_context_ref_matches(
    value: Mapping[str, Any],
    expected: ReadContextRef | Mapping[str, Any],
) -> bool:
    expected_ref = (
        expected
        if isinstance(expected, ReadContextRef)
        else ReadContextRef.from_storage_dict(expected)
    )
    return expected_ref.matches_storage_dict(value)
