from django.http import JsonResponse  # 카카오톡과 연동하기 위해선 JsonResponse로 출력
from django.views.decorators.csrf import csrf_exempt
from .models import restaurant_review
import osmnx as ox, networkx as nx, geopandas as gpd, matplotlib.pyplot as plt


# JsonResponse 출력 테스트용
def keyboard(request):
    return JsonResponse({
        'type': 'text'
    })


@csrf_exempt
def viewList(request):
    querySet = restaurant_review.objects.filter(restaurant_id=438979, restaurant_id__id=438979).select_related(
        'restaurant_id').prefetch_related('restaurant_id__restaurant_review_set')

    result = [{
        "restaurant_id": data.restaurant_id.id,
        "name": data.restaurant_id.name,
        "x": data.restaurant_id.x,
        "y": data.restaurant_id.y,
        "address": data.restaurant_id.address,
        "url": data.restaurant_id.url,
        "id": data.id,
        "score": data.score,
        "review": data.review
    } for data in querySet]
    return JsonResponse({
        'message': result,
        'keyboard': {
            'type': 'text',
        }
    }, json_dumps_params={'ensure_ascii': False}, status=200)

