from django.contrib import admin

# Register your models here.
from content.models import restaurant_info


@admin.register(restaurant_info)
class restaurantAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'name',
        'x',
        'y',
        'address',
        'url',
    )

    list_display_links = (
        'id',
        'name',
        'x',
        'y',
        'address',
        'url',
    )

    search_fields = [
        'id',
        'name',
        'x',
        'y',
        'address',
        'url',
    ]