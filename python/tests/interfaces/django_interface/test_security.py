from types import SimpleNamespace

import pytest

from fervis.interfaces.django.security import (
    MissingAuthenticationError,
    require_fervis_access,
    require_authenticated,
)
from tests.interfaces.django_interface.helpers import questions_url


@pytest.mark.django_db
def test_fervis_api_requires_authentication(
    anon_client,
    fervis_foundation_reset,
):
    response = anon_client.post(
        questions_url(),
        {"question": "no auth"},
        format="json",
    )

    assert response.status_code == 401
    body = response.json()
    assert "error" in body
    assert body["error"]["code"] == "read_context_required"


def test_authenticated_subject_is_sufficient() -> None:
    request = SimpleNamespace(user=SimpleNamespace(is_authenticated=True))

    require_authenticated(request)


def test_fervis_access_uses_host_policy(settings) -> None:
    request = SimpleNamespace(user=SimpleNamespace(is_authenticated=True))

    def deny_fervis_access(received_request):
        assert received_request is request
        return False

    settings.FERVIS_ACCESS_CHECK = deny_fervis_access

    with pytest.raises(MissingAuthenticationError):
        require_fervis_access(request)


def test_fervis_does_not_derive_authorization_from_scope_header_or_role() -> None:
    request = SimpleNamespace(
        user=SimpleNamespace(
            is_authenticated=True,
            role="BUYER",
        ),
        headers={"X-Requester-Scopes": "fervis:write fervis:read"},
    )

    require_authenticated(request)


def test_missing_authenticated_subject_fails_closed() -> None:
    request = SimpleNamespace(user=SimpleNamespace(is_authenticated=False))

    with pytest.raises(MissingAuthenticationError):
        require_authenticated(request)
