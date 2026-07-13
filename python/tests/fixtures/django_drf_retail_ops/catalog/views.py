from rest_framework import viewsets
from rest_framework.response import Response

from .models import Category, Product
from .serializers import CategorySerializer, ProductQuerySerializer, ProductSerializer


class CategoryViewSet(viewsets.ModelViewSet):
    queryset = Category.objects.all()
    serializer_class = CategorySerializer


class ProductViewSet(viewsets.ModelViewSet):
    queryset = Product.objects.select_related("category").all()
    serializer_class = ProductSerializer
    query_serializer_class = ProductQuerySerializer

    def list(self, request, *args, **kwargs):
        fields_param = request.query_params.get("fields")
        fields = (
            tuple(part.strip() for part in fields_param.split(",") if part.strip())
            if fields_param
            else None
        )
        queryset = self.filter_queryset(self.get_queryset())
        serializer = self.get_serializer(queryset, many=True, fields=fields)
        return Response(serializer.data)

    def get_queryset(self):
        queryset = super().get_queryset()
        category_id = self.request.query_params.get("category_id")
        active = self.request.query_params.get("active")
        ordering = self.request.query_params.get("ordering")
        if category_id:
            queryset = queryset.filter(category_id=category_id)
        if active in {"true", "false"}:
            queryset = queryset.filter(active=(active == "true"))
        if ordering:
            queryset = queryset.order_by(ordering)
        return queryset
