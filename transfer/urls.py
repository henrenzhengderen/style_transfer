from django.urls import path
from .views import index, train_view, style_transfer_view

urlpatterns = [
    path('', index, name='index'),
    path('train/', train_view, name='train'),
    path('style-transfer/', style_transfer_view, name='style_transfer'),
]