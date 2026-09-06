"""Origin context reaches ASGI and WSGI request dispatch, not only headers."""

from pathlib import Path
import pytest

from fervis.host_api.contracts import EndpointContract
from fervis.host_api.contracts.authority import ReadContextRef


@pytest.mark.parametrize(
    "origin", ["https://api.example.test:8443", "http://localhost"]
)
def test_asgi_native_origin_controls_generated_links(tmp_path: Path, origin):
    from fastapi import FastAPI, Request
    from fervis.host_api.adapters.fastapi.executor import FastAPIApplicationRuntime

    app = FastAPI()

    @app.get("/resources/", name="resources")
    def resources(request: Request):
        return {"url": str(request.url_for("resources"))}

    contract = EndpointContract(
        endpoint_name="resources",
        url_name="resources",
        method="GET",
        path_template="/resources/",
        docstring="Resources",
        view_class="resources",
    )
    runtime = FastAPIApplicationRuntime(app, project_root=tmp_path)
    try:
        result = runtime.execute_get(contract=contract, origin=origin)
    finally:
        runtime.close()
    assert result.response_status == 200
    assert result.response_body == {"url": origin + "/resources/"}


@pytest.mark.parametrize(
    "origin", ["https://api.example.test:8443", "http://localhost"]
)
def test_wsgi_native_origin_controls_generated_links(origin):
    from flask import Flask, url_for
    from fervis.host_api.adapters.flask.transport import FlaskInProcessReadTransport

    app = Flask(__name__)

    @app.get("/resources/")
    def resources():
        return {"url": url_for("resources", _external=True)}

    page = FlaskInProcessReadTransport(app).get("/resources/", {}, origin=origin)
    assert page.status == 200
    assert page.body == {"url": origin + "/resources/"}


@pytest.mark.parametrize(
    "origin",
    [
        "ftp://example.test",
        "https://user:secret@example.test",
        "https://example.test/path",
        "https://example.test?token=secret",
        "https://example.test/#part",
        "https://example.test:bad",
    ],
)
def test_persisted_origin_rejects_non_origin_urls(origin):
    with pytest.raises(ValueError):
        ReadContextRef(scheme="anonymous", origin=origin)


def test_origins_partition_persisted_read_authority():
    context = ReadContextRef(
        scheme="django_principal", key="reader", origin="https://first.example.test"
    )
    other = ReadContextRef(
        scheme="django_principal", key="reader", origin="https://second.example.test"
    )
    assert not context.matches_storage_dict(other.to_storage_dict())
