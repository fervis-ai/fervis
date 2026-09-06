"""Framework-native reads preserve the request origin used by serializers."""

from types import SimpleNamespace

from django.test import override_settings, RequestFactory
from django.urls import path
from rest_framework.views import APIView
from rest_framework.response import Response

from fervis.host_api.adapters.django.executor import _get_page
from fervis.host_api.adapters.django.principal import capture_django_read_context
from fervis.host_api.contracts.authority import ReadContextRef


class OriginView(APIView):
    authentication_classes = []
    permission_classes = []

    def get(self, request):
        return Response({"url": request.build_absolute_uri("/resources/1/")})


urlpatterns = [path("resources/", OriginView.as_view())]


@override_settings(
    ROOT_URLCONF=__name__, ALLOWED_HOSTS=["api.example.test", "localhost"]
)
def test_native_read_preserves_https_authority_and_nondefault_port():
    status, body = _get_page(
        user=SimpleNamespace(is_authenticated=True),
        url="/resources/",
        query_params={},
        headers={},
        cookies={},
        origin="https://api.example.test:8443",
    )
    assert status == 200
    assert body == {"url": "https://api.example.test:8443/resources/1/"}


@override_settings(ROOT_URLCONF=__name__, ALLOWED_HOSTS=[], DEBUG=True)
def test_cli_without_inbound_request_uses_local_origin():
    status, body = _get_page(
        user=SimpleNamespace(is_authenticated=True),
        url="/resources/",
        query_params={},
        headers={},
        cookies={},
    )
    assert status == 200
    assert body == {"url": "http://localhost/resources/1/"}


@override_settings(ALLOWED_HOSTS=["api.example.test"])
def test_captured_origin_survives_persisted_read_authority():
    request = RequestFactory().get("/", secure=True, HTTP_HOST="api.example.test:8443")
    captured = capture_django_read_context(request)
    assert captured.origin == "https://api.example.test:8443"
    assert ReadContextRef.from_storage_dict(captured.to_storage_dict()) == captured
