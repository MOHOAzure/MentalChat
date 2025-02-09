from django.urls import path
from . import views

urlpatterns = [
    path('', views.chat, name='chat'),
    path('generate-response/', views.generate_response, name='generate_response'),
]