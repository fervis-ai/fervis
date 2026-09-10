"""Non-secret origin context shared by native HTTP request transports."""

from urllib.parse import urlsplit, urlunsplit


def normalize_request_origin(value: str | None) -> str | None:
    if value is None:
        return None
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or any(character.isspace() for character in value)
    ):
        raise ValueError(
            "Request origin must be an HTTP(S) origin without credentials or a path."
        )
    # Validate the port before persisting context used to reconstruct a request.
    parsed.port
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def origin_from_request_url(value: str) -> str:
    parsed = urlsplit(value)
    origin = normalize_request_origin(
        urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
    )
    assert origin is not None
    return origin


def in_process_request_origin(value: str | None) -> str:
    """CLI reads have no inbound request and use the local origin."""
    return normalize_request_origin(value) or "http://localhost"


def request_origin_headers(headers: dict[str, str], origin: str) -> dict[str, str]:
    return {
        **{key: value for key, value in headers.items() if key.lower() != "host"},
        "Host": urlsplit(origin).netloc,
    }
