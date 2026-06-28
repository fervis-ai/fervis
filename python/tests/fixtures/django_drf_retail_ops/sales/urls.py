from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import CancelOrderAPIView, OrderItemViewSet, OrderViewSet

app_name = "sales"

router = DefaultRouter()
router.register("orders", OrderViewSet, basename="orders")
router.register("order-items", OrderItemViewSet, basename="order-items")

urlpatterns = [
    path("", include(router.urls)),
    path("orders/<int:order_id>/cancel/", CancelOrderAPIView.as_view(), name="cancel-order"),
]

