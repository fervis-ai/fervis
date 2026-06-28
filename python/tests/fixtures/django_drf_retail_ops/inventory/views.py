from django.db.models import F
from rest_framework import viewsets

from .models import InventoryLocation, StockRecord, Store
from .serializers import (
    InventoryLocationSerializer,
    StockRecordQuerySerializer,
    StockRecordSerializer,
    StoreSerializer,
)


class StoreViewSet(viewsets.ModelViewSet):
    queryset = Store.objects.all()
    serializer_class = StoreSerializer


class InventoryLocationViewSet(viewsets.ModelViewSet):
    queryset = InventoryLocation.objects.select_related("store").all()
    serializer_class = InventoryLocationSerializer


class StockRecordViewSet(viewsets.ModelViewSet):
    queryset = StockRecord.objects.select_related("product", "location").all()
    serializer_class = StockRecordSerializer
    query_serializer_class = StockRecordQuerySerializer

    def get_queryset(self):
        queryset = super().get_queryset()
        product_id = self.request.query_params.get("product_id")
        location_id = self.request.query_params.get("location_id")
        below_reorder_point = self.request.query_params.get("below_reorder_point")
        if product_id:
            queryset = queryset.filter(product_id=product_id)
        if location_id:
            queryset = queryset.filter(location_id=location_id)
        if below_reorder_point == "true":
            queryset = queryset.filter(quantity_on_hand__lt=F("reorder_point"))
        return queryset
