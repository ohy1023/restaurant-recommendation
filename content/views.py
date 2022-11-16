import numpy as np
from django.http import JsonResponse  # 카카오톡과 연동하기 위해선 JsonResponse로 출력
from django.views.decorators.csrf import csrf_exempt

import my_settings
from .models import restaurant_review, restaurant_info
import requests
from tqdm import tqdm
import osmnx as ox, networkx as nx
import pandas as pd
import json


# JsonResponse 출력 테스트용
def keyboard(request):
    return JsonResponse({
        'type': 'text'
    })


# 현재 위치 좌표로 가져오기
def get_my_place_google():
    url = f'https://www.googleapis.com/geolocation/v1/geolocate?key={my_settings.GOOGLE_API_KEY}'
    data = {
        'considerIp': True,
    }

    result = requests.post(url, data)

    lat = result.json()['location']['lat']
    lng = result.json()['location']['lng']

    return lat, lng


# 핵심 로직
@csrf_exempt
def findNearRestaurant(request):
    # 요청 받아야하는 값 : ex)피자
    # 추후 카카오톡 obt도면 수정 예정
    querySet = restaurant_info.objects.filter(type__contains='피자').values()

    querySet = list(querySet)

    pk = [i['id'] for i in querySet]
    X = [j['x'] for j in querySet]
    Y = [k['y'] for k in querySet]

    df2 = pd.DataFrame({'pk': pk, 'X': X, 'Y': Y})
    df2.reset_index(drop=True, inplace=True)

    # 신촌역 좌표
    point = 37.5598, 126.9425
    G = ox.graph_from_point(point, network_type='bike', dist=1500)
    Gs = ox.utils_graph.get_largest_component(G, strongly=True)

    # 요청 받아야하는 값 : 사용자 위치
    user_x, user_y = get_my_place_google()

    road_li = []  # 도로 기준 최단 거리

    for i in tqdm(range(len(df2))):
        if (df2['X'][i] == user_y) & (df2['Y'][i] == user_x):
            pass
        else:
            orig_node = ox.nearest_nodes(Gs, X=user_y, Y=user_x)  # 출발지
            dest_node = ox.nearest_nodes(Gs, X=df2['X'][i], Y=df2['Y'][i])  # 목적지
            len_road = nx.shortest_path_length(Gs, orig_node, dest_node, weight='length')
            road_li.append(len_road)

    df2['minDist'] = road_li

    df2 = df2.sort_values(by='minDist', axis=0, ascending=True)

    # 추후 최적화 가능해 보임
    temp = list()

    for i in list(df2['pk'][:10]):

    # querySet2 = restaurant_review.objects.filter(restaurant_id=438979).select_related('restaurant_id')\
    #     .prefetch_related('restaurant_id__restaurant_review_set')

        querySet2 = restaurant_info.objects.filter(id=i)

        querySet3 = restaurant_review.objects.filter(restaurant_id=i).values_list('review', flat=True)

        data = querySet2.get()
        reviews = list(querySet3)

        test = json.dumps({
            'message': {
                "restaurant_id": data.id,
                "name": data.name,
                "x": data.x,
                "y": data.y,
                "address": data.address,
                "url": data.url,
                'reviews': reviews
            },
            'keyboard': {
                'type': 'text',
            }},ensure_ascii=False)

        temp.append(test)
    print(temp)


    return JsonResponse({
        'info' : temp,
        'keyboard': {
            'type': 'text',
        }
    }, json_dumps_params={'ensure_ascii': False}, status=200)
