from rest_framework import status, viewsets
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import StockTransfer
from .serializers import StockTransferQuerySerializer, StockTransferSerializer


class StockTransferViewSet(viewsets.ModelViewSet):
    queryset = StockTransfer.objects.select_related(
        "product",
        "from_location",
        "to_location",
    ).all()
    serializer_class = StockTransferSerializer
    query_serializer_class = StockTransferQuerySerializer

    def get_queryset(self):
        queryset = super().get_queryset()
        status_value = self.request.query_params.get("status")
        product_id = self.request.query_params.get("product_id")
        to_location_id = self.request.query_params.get("to_location_id")
        if status_value:
            queryset = queryset.filter(status=status_value)
        if product_id:
            queryset = queryset.filter(product_id=product_id)
        if to_location_id:
            queryset = queryset.filter(to_location_id=to_location_id)
        return queryset


class ReceiveTransferAPIView(APIView):
    def post(self, request, transfer_id: int):
        transfer = StockTransfer.objects.get(pk=transfer_id)
        transfer.status = "received"
        transfer.save(update_fields=["status"])
        return Response(
            StockTransferSerializer(transfer).data,
            status=status.HTTP_200_OK,
        )

