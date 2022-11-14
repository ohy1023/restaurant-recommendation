from django.http import JsonResponse  # 카카오톡과 연동하기 위해선 JsonResponse로 출력
from django.views.decorators.csrf import csrf_exempt

from .models import restaurant_info, restaurant_review


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
        "restaurant_id": review.restaurant_id.id,
        "name": review.restaurant_id.name,
        "x": review.restaurant_id.x,
        "y": review.restaurant_id.y,
        "address": review.restaurant_id.address,
        "url": review.restaurant_id.url,
        "id": review.id,
        "score": review.score,
        "review": review.review
    } for review in querySet]
    return JsonResponse({
        'message': result,
        'keyboard': {
            'type': 'text',
        }
    }, json_dumps_params={'ensure_ascii': False}, status=200)

# else:
# return JsonResponse({
#
#     'message': {
#         'text': '잘못된 입력입니다.'
#     },
#     'keyboard': {
#         'type': 'text',
#     }
#
# }, json_dumps_params={'ensure_ascii': False}, status=200)
