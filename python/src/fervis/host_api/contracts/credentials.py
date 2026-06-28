"""Credential contracts carried across Fervis lifecycle boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping


@dataclass(frozen=True)
class DelegatedReadCredential:
    scheme: str
    encrypted_payload: str
    expires_at: str

    def __post_init__(self) -> None:
        if self.scheme != "captured_headers":
            raise ValueError(f"unsupported delegated credential scheme: {self.scheme}")
        if not str(self.encrypted_payload or "").strip():
            raise ValueError("delegated credential encrypted_payload is required")
        _parse_expires_at(self.expires_at)

    def to_storage_dict(self) -> dict[str, str]:
        return {
            "scheme": self.scheme,
            "encrypted_payload": self.encrypted_payload,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_storage_dict(
        cls,
        value: Mapping[str, Any] | None,
    ) -> "DelegatedReadCredential | None":
        if not value:
            return None
        unexpected = set(value) - {"scheme", "encrypted_payload", "expires_at"}
        if unexpected:
            raise ValueError(
                "unexpected DelegatedReadCredential keys: "
                + ", ".join(sorted(unexpected))
            )
        return cls(
            scheme=str(value.get("scheme") or ""),
            encrypted_payload=str(value.get("encrypted_payload") or ""),
            expires_at=str(value.get("expires_at") or ""),
        )


def _parse_expires_at(value: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError("delegated credential expires_at is required")
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)
