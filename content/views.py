import numpy as np
from django.http import JsonResponse  # 카카오톡과 연동하기 위해선 JsonResponse로 출력
from django.views.decorators.csrf import csrf_exempt
from .models import restaurant_review, restaurant_info

from tqdm import tqdm
import osmnx as ox, networkx as nx
import pandas as pd


# JsonResponse 출력 테스트용
def keyboard(request):
    return JsonResponse({
        'type': 'text'
    })


@csrf_exempt
def viewList(request):
    # 피자 파는 식당의 정보 및 리뷰 조인 후 데이터 조회
    querySet = restaurant_review.objects.filter(restaurant_id__type__contains='치킨').select_related('restaurant_id').prefetch_related('restaurant_id__restaurant_review_set')

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

@csrf_exempt
def findNearRestaurant(request):
    # 요청 받아야하는 값 : 피자
    querySet = restaurant_info.objects.filter(type__contains='피자').all().values()

    querySet = list(querySet)

    pk = [i['id'] for i in querySet]
    X = [j['x'] for j in querySet]
    Y = [k['y'] for k in querySet]

    df2 = pd.DataFrame({'pk': pk, 'X': X, 'Y': Y})

    # 신촌역 좌표
    point = 37.5598, 126.9425
    G = ox.graph_from_point(point, network_type='bike', dist=500)
    Gs = ox.utils_graph.get_largest_component(G, strongly=True)

    # 요청 받아야하는 값 : 사용자 위치
    user_x = 37.5085162
    user_y = 126.8843116

    df2.reset_index(drop=True, inplace=True)
    road_li = []  # 도로 기준 최단 거리

    for i in tqdm(range(len(df2))):
        if (df2['X'][i] == user_y) & (df2['Y'][i] == user_x):
            pass
        else:
            orig_node = ox.nearest_nodes(Gs, X=user_y, Y=user_x)  # 출발지
            dest_node = ox.nearest_nodes(Gs, X=df2['X'][i], Y=df2['Y'][i])  # 목적지
            len_road = nx.shortest_path_length(Gs, orig_node, dest_node, weight='length')
            road_li.append(str(round(len_road, 1)) + 'm')

    # road_li3 = pd.DataFrame(road_li)
    # road_li3 = ['최단거리', '최단시간']

    t = []  # 최단거리 bike 기준 시간
    bike_sp = 15000  # 자전거 속도 15km/h
    road_li = pd.DataFrame(road_li)

    road_li.columns = ['최단거리']
    split = road_li['최단거리'].str.split('m')
    road_li['최단거리'] = split.str.get(0)

    for i in range(len(road_li)):
        etc = float(road_li['최단거리'][i]) / bike_sp
        hour = int(etc)
        mint = int((etc - hour) * 60)
        sec = int(((etc - hour) * 60 - mint) * 60)
        if hour == 0:
            t.append([mint, sec])
        else:
            t.append([hour, mint, sec])

    road_li['최단시간'] = t

    user_loc = pd.DataFrame({'위도': [user_x], '경도': [user_y]})
    user_loc

    df_diff = pd.DataFrame()
    stores = []
    diff = []
    # df_3 = pd.DataFrame()
    df3_rank1_stores = []
    df3_rank1_diff = []
    df3_rank2_stores = []
    df3_rank2_diff = []

    for i in range(len(user_loc)):
        for j in tqdm(range(len(df2))):
            stores.append(df2['pk'][j])
            diff.append(np.sqrt((user_loc['위도'][i] - df2['Y'][j]) ** 2 +
                                (user_loc['경도'][i] - df2['X'][j]) ** 2))

        df_diff['식당 이름'] = stores
        df_diff['차이'] = diff
        df_diff['순위'] = df_diff['차이'].rank(ascending=True)
        df_diff.sort_values('순위', inplace=True)
        df3_rank1_stores.append(df_diff.iloc[0]['식당 이름'])
        df3_rank1_diff.append(df_diff.iloc[0]['차이'])
        df3_rank2_stores.append(df_diff.iloc[1]['식당 이름'])
        df3_rank2_diff.append(df_diff.iloc[1]['차이'])
        stores = []
        diff = []

    user_loc['1순위'] = df3_rank1_stores
    user_loc['1순위 차이'] = df3_rank1_diff
    user_loc['2순위'] = df3_rank2_stores
    user_loc['2순위 차이'] = df3_rank2_diff
    print(user_loc)

    return JsonResponse({
        'message': user_loc['1순위'][0],
        'keyboard': {
            'type': 'text',
        }
    }, json_dumps_params={'ensure_ascii': False}, status=200)
