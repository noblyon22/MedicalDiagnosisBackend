from django.urls import path
from .views import tb_detection, leukemia_detection, health_check, index_view

urlpatterns = [
    path('', index_view, name='index'),  # Frontend interface
    path('tb/', tb_detection, name='tb_detection'),
    path('leukemia/', leukemia_detection, name='leukemia_detection'),
    path('health/', health_check, name='health_check'),
]