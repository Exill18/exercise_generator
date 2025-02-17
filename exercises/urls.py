from django.urls import path
from . import views

urlpatterns = [
    path('', views.exam_view, name='exam'),
    path('results/', views.exam_view, name='exam'),
]