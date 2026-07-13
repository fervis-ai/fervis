from types import SimpleNamespace

from fervis.host_api.contracts.authority import ReadContextRef
from fervis.interfaces.fastapi.principal import principal_from_request as fastapi_principal
from fervis.interfaces.flask.principal import principal_from_request as flask_principal


def test_framework_principals_preserve_captured_tenant_authority() -> None:
    read_context = ReadContextRef(
        scheme="fastapi_principal",
        key="user_1",
        tenant_key="tenant_east",
    )
    fastapi = fastapi_principal(
        SimpleNamespace(),
        read_context_capture=lambda request: read_context,
    )
    flask = flask_principal(
        SimpleNamespace(),
        read_context_capture=lambda request: read_context,
    )

    assert fastapi.tenant_id == "tenant_east"
    assert flask.tenant_id == "tenant_east"
