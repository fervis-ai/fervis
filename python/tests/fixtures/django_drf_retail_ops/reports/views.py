from django.db.models import Count, Sum
from django.db.models import F
from rest_framework.response import Response
from rest_framework.views import APIView

from tests.fixtures.django_drf_retail_ops.inventory.models import StockRecord
from tests.fixtures.django_drf_retail_ops.sales.models import Order

from .serializers import (
    LowStockQuerySerializer,
    LowStockRowSerializer,
    SalesSummaryQuerySerializer,
    SalesSummaryRowSerializer,
)


class SalesSummaryAPIView(APIView):
    query_serializer_class = SalesSummaryQuerySerializer
    response_serializer_class = SalesSummaryRowSerializer

    def get(self, request):
        query = SalesSummaryQuerySerializer(data=request.query_params)
        query.is_valid(raise_exception=True)
        params = query.validated_data
        group_by = params.get("group_by") or "store"
        queryset = Order.objects.select_related("store").all()
        if params.get("store_id"):
            queryset = queryset.filter(store_id=params["store_id"])
        if params.get("start_date"):
            queryset = queryset.filter(ordered_at__gte=params["start_date"])
        if params.get("end_date"):
            queryset = queryset.filter(ordered_at__lte=params["end_date"])
        if group_by == "status":
            rows = queryset.values("status").annotate(
                total_orders=Count("id"),
                total_amount=Sum("total_amount"),
            )
            payload = [
                {
                    "label": row["status"],
                    "total_orders": row["total_orders"],
                    "total_amount": row["total_amount"],
                }
                for row in rows
            ]
        else:
            rows = queryset.values("store__name").annotate(
                total_orders=Count("id"),
                total_amount=Sum("total_amount"),
            )
            payload = [
                {
                    "label": row["store__name"],
                    "total_orders": row["total_orders"],
                    "total_amount": row["total_amount"],
                }
                for row in rows
            ]
        return Response(SalesSummaryRowSerializer(payload, many=True).data)


class LowStockAPIView(APIView):
    query_serializer_class = LowStockQuerySerializer
    response_serializer_class = LowStockRowSerializer

    def get(self, request):
        query = LowStockQuerySerializer(data=request.query_params)
        query.is_valid(raise_exception=True)
        params = query.validated_data
        queryset = StockRecord.objects.select_related("product", "location").filter(
            quantity_on_hand__lt=F("reorder_point")
        )
        if params.get("location_id"):
            queryset = queryset.filter(location_id=params["location_id"])
        if params.get("category_id"):
            queryset = queryset.filter(product__category_id=params["category_id"])
        payload = [
            {
                "product_id": row.product_id,
                "product_sku": row.product.sku,
                "location_id": row.location_id,
                "location_name": row.location.name,
                "quantity_on_hand": row.quantity_on_hand,
                "reorder_point": row.reorder_point,
            }
            for row in queryset
        ]
        return Response(LowStockRowSerializer(payload, many=True).data)
