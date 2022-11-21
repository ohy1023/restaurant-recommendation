from django.urls import path
from . import views


urlpatterns = [
    path('',views.home, name = 'home'),
    path('store/',views.findNearRestaurant),
    # path('test/',views.logic)
]