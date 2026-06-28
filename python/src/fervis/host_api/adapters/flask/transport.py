"""Flask in-process read transport.

Fervis executes reads through Flask's request dispatch boundary so routing,
request hooks, and response handling stay owned by Flask. WSGI middleware is
owned by HTTP mode; in-process principal injection needs Flask's request
context before Flask request hooks run.
"""

from __future__ import annotations

from io import BytesIO, StringIO
from typing import Any
from urllib.parse import urlencode

from ..response_body import response_body
from ..runtime_output import suppress_host_output


class FlaskInProcessReadTransport:
    def __init__(self, app: Any) -> None:
        self.app = app

    def get(
        self,
        url: str,
        query_params: dict[str, Any],
        *,
        principal: object | None = None,
        headers: dict[str, str] | None = None,
        cookies: dict[str, str] | None = None,
    ) -> tuple[int, Any]:
        environ = _wsgi_environ(
            method="GET",
            path=url,
            query_params=query_params,
            headers=dict(headers or {}),
            cookies=dict(cookies or {}),
        )
        with suppress_host_output():
            response = _dispatch_request(self.app, environ, principal=principal)
        return response.status_code, response_body(response)


def _dispatch_request(
    app: Any,
    environ: dict[str, Any],
    *,
    principal: object | None,
) -> Any:
    with app.request_context(environ):
        if principal is not None:
            from flask import g

            g.current_user = principal
            g.user = principal
        return app.full_dispatch_request()


def _wsgi_environ(
    *,
    method: str,
    path: str,
    query_params: dict[str, Any],
    headers: dict[str, str],
    cookies: dict[str, str],
) -> dict[str, Any]:
    environ = {
        "REQUEST_METHOD": method,
        "SCRIPT_NAME": "",
        "PATH_INFO": path,
        "QUERY_STRING": urlencode(query_params, doseq=True),
        "SERVER_NAME": "localhost",
        "SERVER_PORT": "80",
        "SERVER_PROTOCOL": "HTTP/1.1",
        "wsgi.version": (1, 0),
        "wsgi.url_scheme": "http",
        "wsgi.input": BytesIO(b""),
        "wsgi.errors": StringIO(),
        "wsgi.multithread": False,
        "wsgi.multiprocess": False,
        "wsgi.run_once": False,
        "CONTENT_LENGTH": "0",
    }
    for name, value in headers.items():
        key = "HTTP_" + name.upper().replace("-", "_")
        if key not in {"HTTP_CONTENT_TYPE", "HTTP_CONTENT_LENGTH"}:
            environ[key] = value
    if cookies:
        environ["HTTP_COOKIE"] = "; ".join(
            f"{name}={value}" for name, value in cookies.items()
        )
    return environ
