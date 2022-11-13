from django.http import JsonResponse # 카카오톡과 연동하기 위해선 JsonResponse로 출력
from django.views.decorators.csrf import csrf_exempt

from .models import restaurant_info


# JsonResponse 출력 테스트용
def keyboard(request):
    return JsonResponse({
        'type': 'text'
    })

@csrf_exempt
def viewList(request):
    querySet = restaurant_info.objects.filter(type__icontains='치킨').values()
    # model_instance = querySet.get(id=438979) #하나의 객체만 반환
    data = list(querySet)
    return JsonResponse({
        'message': data,
        'keyboard': {
            'type': 'text',

        }
    },json_dumps_params={'ensure_ascii': False}, status=200)


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
