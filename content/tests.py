import numpy as np
from django.test import TestCase
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV

from .models import restaurant_review, restaurant_info
import my_settings
import osmnx as ox, networkx as nx
import pandas as pd

# 자연어 분석 처리 긍정/부정 비율
# 기본
import matplotlib.pyplot as plt
import seaborn as sns

# 경고 뜨지 않게 설정
import warnings

import re
# 한국어 형태소 분석
from konlpy.tag import Okt, Hannanum, Kkma, Mecab, Komoran


def text_clearing(text):
    hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')
    # 지정한 정규식에 해당하지 않은 것은 길이가 0인 문자열로 변환한다.
    result = hangul.sub('', text)
    return result

# konlpy 라이브러리로 텍스트 데이터에서 형태소를 추출한다.
def get_pos(x):
    tagger = Okt()
    pos = tagger.pos(x)

    # 단어와 품사를 합쳐서 하나의 단어로 만들어준다.
    result = []

    # 형태소의 수만큼 반복한다.
    # 조사인 것과 명사인 것이 같을 수 있기 때문에 구분해준다.
    # 형태소 벡터를 만들때 추후 사용
    for a1 in pos:
        result.append(f'{a1[0]}/{a1[1]}')

    return result


def bad_feature_sep(bottom50, text_data_dict):
    bad_feature = []
    bad_feature_temp = []

    for value, idx in bottom50:
        bad_feature.append(text_data_dict[idx])

    for i in bad_feature:
        st_li = i.split('/')
        bad_feature_temp.append(st_li[0])

    return bad_feature_temp


def get_bad_feature_keywords(bad_feature_temp, review):
    feature_temp = []
    for i in bad_feature_temp:
        for j in review:
            if i in j:
                feature_temp.append(i)

    return feature_temp


def good_feature_sep(top50, text_data_dict):
    good_feature = []
    good_feature_temp = []

    for value, idx in top50:
        good_feature.append(text_data_dict[idx])

    for i in good_feature:
        st_li = i.split('/')
        good_feature_temp.append(st_li[0])

    return good_feature_temp


def get_good_feature_keywords(good_feature_temp, review):
    feature_temp = []
    for i in good_feature_temp:
        for j in review:
            if i in j:
                feature_temp.append(i)

    return feature_temp


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

        self.assertEquals(query.type, '닭갈비')

    def test_select_review(self):
        query = restaurant_review.objects.get(id=3)

        self.assertEquals(query.review, '개노맛.')

    def test_select_join(self):
        query_set = restaurant_review.objects.filter(restaurant_id__type__contains='닭갈비').select_related(
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
        } for review in query_set]
        print(result)
        self.assertEquals(len(result), 2)

    def test_type_df(self):
        querySet = restaurant_info.objects.filter(type__contains='닭갈비').all()[:2].values()

        querySet = list(querySet)

        id = [i['id'] for i in querySet]
        X = [j['x'] for j in querySet]
        Y = [k['y'] for k in querySet]

        df = pd.DataFrame({'id': id, 'X': X, 'Y': Y})
        print(df)
        print(type(df['X'][0]))

        df2 = pd.read_csv('신촌 음식점.csv', encoding='utf8', index_col=0)
        print(df2)
        print(type(df2['X'][0]))
        # print(type(df2[0]['X']))


class OsmnxTestClass(TestCase):

    def test_my_place_google(self):
        import requests

        url = f'https://www.googleapis.com/geolocation/v1/geolocate?key={my_settings.GOOGLE_API_KEY}'
        data = {
            'considerIp': True,
        }

        result = requests.post(url, data)

        lat = result.json()['location']['lat']
        lng = result.json()['location']['lng']

        print(lat)
        print(lng)

    def test_findNearRestaurant(self):
        point = 37.5598, 126.9425
        G = ox.graph_from_point(point, network_type='bike', dist=500)
        Gs = ox.utils_graph.get_largest_component(G, strongly=True)

        user_x = 37.5085162
        user_y = 126.8843116

        df2 = pd.read_csv('신촌 음식점.csv', encoding='utf8', index_col=0)

        from tqdm import tqdm

        df2.reset_index(drop=True, inplace=True)
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

        print(df2.info())


class scoreTestClass(TestCase):

    def test_score(self):
        df = pd.read_csv('신촌 음식점 정보.csv', encoding='utf8', index_col=0)

        total_food = pd.DataFrame(columns=['store', 'kind', 'score', 'review', 'y'])
        s = ['1', '2', '3', '4', '5']

        for i in range(len(df)):
            if pd.isna(df['리뷰'][i]):
                pass
            else:
                n = df['식당 이름'][i]
                m = df['종류'][i]
                k = df['리뷰'][i].split('\\ ')
                del k[-1]

                review = []
                score = []
                store = []
                kind = []

                for i in k:
                    j = i.split(' - ')
                    store.append(n)
                    kind.append(m)
                    if j[0] in s:
                        score.append(j[0])
                    else:
                        score.append(-1)
                    review.append(j[-1])
                food_df = pd.DataFrame(columns=['store', 'kind', 'score', 'review', 'y'])
                food_df['store'] = store
                food_df['kind'] = kind
                food_df['score'] = score
                food_df['review'] = review

            total_food = pd.concat([total_food, food_df])

        total_food = total_food.astype({'score': 'int'})
        total_food.reset_index(drop=True, inplace=True)
        total_food['score'] >= 4
        from tqdm import tqdm

        for i in tqdm(range(len(total_food))):
            if total_food['score'][i] >= 4:
                total_food['y'][i] = 1
            else:
                total_food['y'][i] = 0

        sample = total_food['score'] >= 0
        sample = total_food[sample]

        sample["ko_review"] = sample["review"].apply(lambda x: text_clearing(x))
        del sample['review']
        sample = sample.astype({'y': 'int'})

        index_vectorizer = CountVectorizer(tokenizer=lambda x: get_pos(x))
        X = index_vectorizer.fit_transform(sample["ko_review"].tolist())
        X

        # TFidf 변환 모델 생성
        tfidf_vectorizer = TfidfTransformer()
        # 형태소 벡터 변환하기
        X = tfidf_vectorizer.fit_transform(X)
        y = sample["y"]

        # LogisticRegression
        # penalty : 규제의 종류(l1, l2, elasticnet, none)
        # C : 규제의 강도
        params = {
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
        }

        model = LogisticRegression()
        kfold = KFold(n_splits=10, shuffle=True, random_state=1)
        grid_clf = GridSearchCV(model, param_grid=params, scoring='f1', cv=kfold)
        grid_clf.fit(X, y)

        print(f'최적의 하이퍼 파라미터 : {grid_clf.best_params_}')
        print(f'최적의 모델 평균 성능 : {grid_clf.best_score_}')

        model = grid_clf.best_estimator_

        # 상관관계수 구하기
        a1 = (model.coef_[0])
        a2 = list(enumerate(a1))
        a3 = []

        for idx, value in a2:
            a3.append((value, idx))

        coef_pos_index = sorted(a3, reverse=True)

        # 새로운 딕셔너리 생성
        text_data_dict = {}

        # 단어 사전에 있는 단어의 수만큼 반복한다.
        for key in index_vectorizer.vocabulary_:
            # 현재 key에 해당하는 값을 가져온다.
            value = index_vectorizer.vocabulary_[key]

            # 위의 딕셔너리에 담는다.
            text_data_dict[value] = key

        text_data_dict

        # 긍정적인 어조 (상관계수가 1에 가장 큰)
        top50 = coef_pos_index[:50]
        # 부정적인 어조
        bottom50 = coef_pos_index[-50:]

        sample.reset_index(drop=True, inplace=True)
        restaurant_review = []
        for i in tqdm(range(len(sample))):
            temp_dict = {}
            temp_dict['_id'] = sample['store'][i]
            temp_dict['review'] = sample['ko_review'][i]
            if temp_dict['review'] == []:
                continue
            restaurant_review.append(temp_dict)

        store_unique = pd.DataFrame(columns=['store'])

        unique_li = sample['store'].drop_duplicates()
        unique_li.reset_index(drop=True, inplace=True)

        store_unique['store'] = unique_li

        total_list = pd.DataFrame(columns=['Store', 'Review', 'Score'])
        score_li = []
        review_li = []
        store_li = []

        for i in range(len(store_unique)):
            sample_df = sample[sample['store'] == store_unique['store'][i]]
            sample_df.reset_index(drop=True, inplace=True)
            bad_feature_temp = bad_feature_sep(bottom50, text_data_dict)
            bad_list = get_bad_feature_keywords(bad_feature_temp, sample_df['ko_review'])
            good_feature_temp = good_feature_sep(top50, text_data_dict)
            good_list = get_good_feature_keywords(good_feature_temp, sample_df['ko_review'])
            store_li.append(sample_df['store'][0])
            review_li.append(len(sample_df['ko_review']))
            if (len(good_list) + len(bad_list)) == 0:
                score_li.append(round(0, 2))
            else:
                score_li.append(round((len(good_list) / (len(good_list) + len(bad_list)) * 100), 2))

        total_list['Store'] = store_li
        total_list['Score'] = score_li
        total_list['Review'] = review_li

        print(total_list) #성공, 정리해야함, 갯수가 0일때 오류나는지 확인 필요!
