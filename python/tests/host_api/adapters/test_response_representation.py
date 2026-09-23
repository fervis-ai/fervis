import pytest
from httpx import Response
from fervis.host_api.adapters.response_body import response_page


@pytest.mark.parametrize("value", [None, False, 0, "", [], {}])
def test_json_values_preserve_their_exact_representation(value):
    page = (
        response_page(Response(200, json=value))
        if value is not None
        else response_page(
            Response(200, content=b"null", headers={"content-type": "application/json"})
        )
    )
    assert page.format.value == "json"
    assert page.body == value


def test_non_json_content_does_not_become_a_json_record():
    page = response_page(
        Response(
            200, text="<html>Dashboard</html>", headers={"content-type": "text/html"}
        )
    )
    assert page.format.value == "text"
    assert page.body == "<html>Dashboard</html>"


def test_explicit_plain_text_is_not_reinterpreted_as_a_json_number():
    page = response_page(
        Response(200, text="123", headers={"content-type": "text/plain"})
    )
    assert page.format.value == "text"
    assert page.body == "123"


def test_older_django_text_surface_preserves_body_and_charset():
    from types import SimpleNamespace

    page = response_page(
        SimpleNamespace(
            status_code=200,
            headers={"Content-Type": "text/plain"},
            content=b"caf\xe9",
            charset="iso-8859-1",
        )
    )
    assert page.format.value == "text"
    assert page.body == "café"
