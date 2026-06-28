from rest_framework import status, viewsets
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Order, OrderItem
from .serializers import OrderItemSerializer, OrderQuerySerializer, OrderSerializer


class OrderViewSet(viewsets.ModelViewSet):
    queryset = Order.objects.select_related("store").prefetch_related("items").all()
    serializer_class = OrderSerializer
    query_serializer_class = OrderQuerySerializer

    def get_queryset(self):
        queryset = super().get_queryset()
        store_id = self.request.query_params.get("store_id")
        status_value = self.request.query_params.get("status")
        start_date = self.request.query_params.get("start_date")
        end_date = self.request.query_params.get("end_date")
        if store_id:
            queryset = queryset.filter(store_id=store_id)
        if status_value:
            queryset = queryset.filter(status=status_value)
        if start_date:
            queryset = queryset.filter(ordered_at__gte=start_date)
        if end_date:
            queryset = queryset.filter(ordered_at__lte=end_date)
        return queryset


class OrderItemViewSet(viewsets.ModelViewSet):
    queryset = OrderItem.objects.select_related("order", "product").all()
    serializer_class = OrderItemSerializer


class CancelOrderAPIView(APIView):
    def post(self, request, order_id: int):
        order = Order.objects.get(pk=order_id)
        order.status = "cancelled"
        order.save(update_fields=["status"])
        return Response(OrderSerializer(order).data, status=status.HTTP_200_OK)

