from __future__ import annotations

from django.urls import include, path

urlpatterns = [
    path("v1/", include("fervis.interfaces.django.urls")),
    path("retail/", include("tests.fixtures.django_drf_retail_ops.urls")),
]
