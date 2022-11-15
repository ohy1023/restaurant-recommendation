import numpy as np
from django.test import TestCase
from .models import restaurant_review, restaurant_info
import my_settings
import osmnx as ox, networkx as nx
import pandas as pd



# Create your tests here.
class DBAndModelTestClass(TestCase):

    @classmethod
    def setUpTestData(cls):
        restaurant_info.objects.create(id=1, name='춘천닭갈비막국수 본점', x=124.34, y=36.64,
                                       address='서울 영등포구 신길로 18', url='https://place.map.kakao.com/10834702',
                                       type='닭갈비')
        restaurant_info.objects.create(id=2, name='은행골 본점신관', x=124.34, y=36.64,
                                       address='서울 관악구 조원로 10-1', url='https://place.map.kakao.com/16622909',
                                       type='초밥')
        restaurant_info.objects.create(id=3, name='은행골 본점신관', x=124.34, y=36.64,
                                       address='서울 관악구 조원로 10-1', url='https://place.map.kakao.com/16622909',
                                       type='닭갈비')
        restaurant_review.objects.create(id=1, score=3, review='사장님은 예쁜데 싸가지가 존나 없어요.',
                                         restaurant_id=restaurant_info.objects.get(id=1))
        restaurant_review.objects.create(id=2, score=4, review='진짜 개맛있음요.',
                                         restaurant_id=restaurant_info.objects.get(id=1))
        restaurant_review.objects.create(id=3, score=1, review='개노맛.',
                                         restaurant_id=restaurant_info.objects.get(id=2))

    def test_select_type(self):
        query = restaurant_info.objects.get(id=1)

        self.assertEquals(query.type,'닭갈비')

    def test_select_review(self):
        query = restaurant_review.objects.get(id=3)

        self.assertEquals(query.review,'개노맛.')

    def test_select_join(self):
        query_set = restaurant_review.objects.filter(restaurant_id__type__contains='닭갈비').select_related('restaurant_id').prefetch_related('restaurant_id__restaurant_review_set')
        result = [{
            "restaurant_id": review.restaurant_id.id,
            "name":review.restaurant_id.name,
            "x":review.restaurant_id.x,
            "y": review.restaurant_id.y,
            "address": review.restaurant_id.address,
            "url": review.restaurant_id.url,
            "id":review.id,
            "score":review.score,
            "review":review.review
        }for review in query_set]
        print(result)
        self.assertEquals(len(result),2)

    def test_type_df(self):
        querySet = restaurant_info.objects.filter(type__contains='닭갈비').all()[:2].values()

        querySet = list(querySet)

        id = [i['id'] for i in querySet]
        X = [j['x'] for j in querySet]
        Y = [k['y'] for k in querySet]

        df = pd.DataFrame({'id': id, 'X': X, 'Y': Y})
        print(df)

class OsmnxTestClass(TestCase):


    def test_my_place_google(self):
        import requests

        url = f'https://www.googleapis.com/geolocation/v1/geolocate?key={my_settings.GOOGLE_API_KEY}'
        data = {
            'considerIp': True,
        }

        result = requests.post(url, data)

        print(result.text)

    def test_findNearRestaurant(self):
        place = '서대문구, 서울, 대한민국'
        G = ox.graph_from_place(place, network_type='bike', simplify=False)
        Gs = ox.utils_graph.get_largest_component(G, strongly=True)

        user_x = 37.5085162
        user_y = 126.8843116

        df2 = pd.read_csv('신촌 음식점.csv', encoding='utf8', index_col=0)

        from tqdm import tqdm

        # df2 = df[0:10000]
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

        road_li3 = pd.DataFrame(road_li)
        road_li3 = ['최단거리', '최단시간']

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
        df_3 = pd.DataFrame()
        df3_rank1_stores = []
        df3_rank1_diff = []
        df3_rank2_stores = []
        df3_rank2_diff = []

        for i in range(len(user_loc)):
            for j in tqdm(range(len(df2))):
                stores.append(df2['식당 이름'][j])
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