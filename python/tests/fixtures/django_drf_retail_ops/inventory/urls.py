from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import InventoryLocationViewSet, StockRecordViewSet, StoreViewSet

app_name = "inventory"

router = DefaultRouter()
router.register("stores", StoreViewSet, basename="stores")
router.register("locations", InventoryLocationViewSet, basename="locations")
router.register("stock-records", StockRecordViewSet, basename="stock-records")

urlpatterns = [path("", include(router.urls))]
