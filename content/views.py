from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import re
from konlpy.tag import Okt
import my_settings
from .models import restaurant_review, restaurant_info, good_word, bad_word
import requests
from tqdm import tqdm
import osmnx as ox, networkx as nx
import pandas as pd
from django.db.models import Q
import streamlit as st
from collections import OrderedDict


def whole_region(keyword, start_x, start_y, end_x, end_y):
    page_num = 1
    # 데이터가 담길 리스트
    all_data_list = []

    while (1):
        url = 'https://dapi.kakao.com/v2/local/search/keyword.json'
        params = {'query': keyword, 'page': page_num,
                  'rect': f'{start_x},{start_y},{end_x},{end_y}'}
        headers = {"Authorization": my_settings.KAKAO_API_KEY}
        ## 입력예시 -->> headers = {"Authorization": "KakaoAK f64acbasdfasdfasf70e4f52f737760657"}
        resp = requests.get(url, params=params, headers=headers)

        search_count = resp.json()['meta']['total_count']

        if search_count > 45:
            dividing_x = (start_x + end_x) / 2
            dividing_y = (start_y + end_y) / 2
            ## 4등분 중 왼쪽 아래
            all_data_list.extend(whole_region(keyword, start_x, start_y, dividing_x, dividing_y))
            ## 4등분 중 오른쪽 아래
            all_data_list.extend(whole_region(keyword, dividing_x, start_y, end_x, dividing_y))
            ## 4등분 중 왼쪽 위
            all_data_list.extend(whole_region(keyword, start_x, dividing_y, dividing_x, end_y))
            ## 4등분 중 오른쪽 위
            all_data_list.extend(whole_region(keyword, dividing_x, dividing_y, end_x, end_y))
            return all_data_list

        else:
            if resp.json()['meta']['is_end']:
                all_data_list.extend(resp.json()['documents'])
                return all_data_list
            # 아니면 다음 페이지로 넘어가서 데이터 저장
            else:
                page_num += 1
                all_data_list.extend(resp.json()['documents'])


def overlapped_data(keyword, start_x, start_y, next_x, next_y, num_x, num_y):
    # 최종 데이터가 담길 리스트
    overlapped_result = []

    # 지도를 사각형으로 나누면서 데이터 받아옴
    for i in range(1, num_x + 1):  ## 1,10
        end_x = start_x + next_x
        initial_start_y = start_y
        for j in range(1, num_y + 1):  ## 1,6
            end_y = initial_start_y + next_y
            each_result = whole_region(keyword, start_x, initial_start_y, end_x, end_y)
            overlapped_result.extend(each_result)
            initial_start_y = end_y
        start_x = end_x

    return overlapped_result


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


def home(request):
    # st.title('신촌에서 뭐 먹지?')
    # if st.button("크롤링"):
    #     st.write("Data Loading..")
    #     # 데이터 로딩 함수는 여기에!
    #
    #     keyword = '음식점'
    #     start_x = 126.93
    #     start_y = 37.55
    #     next_x = 0.01
    #     next_y = 0.01
    #     num_x = 2
    #     num_y = 2
    #
    #     overlapped_result = overlapped_data(keyword, start_x, start_y, next_x, next_y, num_x, num_y)
    #
    #     # 최종 데이터가 담긴 리스트 중복값 제거
    #     results = list(map(dict, OrderedDict.fromkeys(tuple(sorted(d.items())) for d in overlapped_result)))
    #     for i in results:
    #         restaurant_info(id=i['id'], name=i['place_name'], x=i['x'], y=i['y'], address=i['road_address_name'],
    #                         url=i['place_url'], type=(i['category_name'].split('>')[-1])).save()
    return render(request, 'home.html')


# 핵심 로직
@csrf_exempt
def findNearRestaurant(request):
    food_type = request.GET.get('food_field')
    # 요청 받아야하는 값 : ex)피자
    # 추후 카카오톡 obt도면 수정 예정
    querySet = restaurant_info.objects.filter(Q(name__contains=food_type) | Q(type__contains=food_type)).values()

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

    total_food = pd.DataFrame(columns=['store', 'kind', 'score', 'review', 'y'])

    review = []
    score = []
    store = []
    kind = []

    for i in list(df2['pk'][:10]):

        querySet2 = restaurant_info.objects.filter(id=i)

        querySet3 = restaurant_review.objects.filter(restaurant_id=i).values_list('review', flat=True)

        querySet4 = restaurant_review.objects.filter(restaurant_id=i).values_list('score', flat=True)

        data = querySet2.get()
        reviews = list(querySet3)
        scores = list(querySet4)

        for _ in range(len(reviews)):
            store.append(data.name)
            kind.append(data.type)
        for j in scores:
            score.append(j)
        for k in reviews:
            review.append(k)

    total_food['store'] = store
    total_food['kind'] = kind
    total_food['score'] = score
    total_food['review'] = review

    total_food["ko_review"] = total_food["review"].apply(lambda x: text_clearing(x))
    del total_food['review']

    total_food.reset_index(drop=True, inplace=True)
    restaurant_reviews = []
    for i in tqdm(range(len(total_food))):
        temp_dict = {}
        temp_dict['_id'] = total_food['store'][i]
        temp_dict['review'] = total_food['ko_review'][i]
        if temp_dict['review'] == []:
            continue
        restaurant_reviews.append(temp_dict)

    store_unique = pd.DataFrame(columns=['store'])

    unique_li = total_food['store'].drop_duplicates()
    unique_li.reset_index(drop=True, inplace=True)

    store_unique['store'] = unique_li

    good_feature_temp = list(good_word.objects.values_list('word', flat=True))

    bad_feature_temp = list(bad_word.objects.values_list('word', flat=True))

    total_list = pd.DataFrame(columns=['Store', 'Review', 'Score'])
    score_li = []
    review_li = []
    store_li = []

    for i in range(len(store_unique)):
        sample_df = total_food[total_food['store'] == store_unique['store'][i]]
        sample_df.reset_index(drop=True, inplace=True)
        good_list = get_good_feature_keywords(good_feature_temp, sample_df['ko_review'])
        bad_list = get_bad_feature_keywords(bad_feature_temp, sample_df['ko_review'])
        store_li.append(sample_df['store'][0])
        review_li.append(len(sample_df['ko_review']))
        if (len(good_list) + len(bad_list)) == 0:
            score_li.append(round(0, 2))
        else:
            score_li.append(round((len(good_list) / (len(good_list) + len(bad_list)) * 100), 2))

    total_list['Store'] = store_li
    total_list['Score'] = score_li
    total_list['Review'] = review_li

    result = total_list.sort_values('Score', ascending=False)[:3]

    return JsonResponse({
        "가게 이름": list(result['Store']),
        "점수": list(result['Score']),
    }, json_dumps_params={'ensure_ascii': False}, status=200)
