from django.urls import path
from . import views


urlpatterns = [
    path('keyboard/',views.keyboard),
]