
# 1번 파일이 실행될 때 환경변수에 현재 자신의 프로젝트의 settings.py파일 경로를 등록.
import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "djangoProject4.settings")

# 2번 실행파일에 Django 환경을 불러오는 작업.
import django

django.setup()

# 3번 크롤링을 하고 DB model에 저장.
from content.models import restaurant_info
import my_settings
import requests
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

# 이태원 126.99 37.53
# 신촌 126.93 37.55
# 천안 야우리 127.1478 36.8124

if __name__=='__main__':
    keyword = '음식점'
    start_x = 127.1478
    start_y = 36.8124
    next_x = 0.01
    next_y = 0.01
    num_x = 2
    num_y = 2

    overlapped_result = overlapped_data(keyword, start_x, start_y, next_x, next_y, num_x, num_y)

    # 최종 데이터가 담긴 리스트 중복값 제거
    results = list(map(dict, OrderedDict.fromkeys(tuple(sorted(d.items())) for d in overlapped_result)))
    for i in results:
        restaurant_info(id=i['id'],name=i['place_name'], x=i['x'], y=i['y'], address=i['road_address_name'],
                        url=i['place_url'],type=(i['category_name'].split('>')[-1])).save()


