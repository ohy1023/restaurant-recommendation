import requests
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from config.settings import KAKAO_API_KEY, CHROME_OPTIONS


def whole_region(keyword, start_x, start_y, end_x, end_y):
    """재귀적으로 지역을 분할하여 카카오 API로 검색"""
    page_num = 1
    all_data_list = []

    while True:
        url = 'https://dapi.kakao.com/v2/local/search/keyword.json'
        params = {
            'query': keyword,
            'page': page_num,
            'rect': f'{start_x},{start_y},{end_x},{end_y}'
        }
        headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
        resp = requests.get(url, params=params, headers=headers)

        search_count = resp.json()['meta']['total_count']

        if search_count > 45:
            dividing_x = (start_x + end_x) / 2
            dividing_y = (start_y + end_y) / 2

            # 4등분하여 재귀 호출
            all_data_list.extend(whole_region(keyword, start_x, start_y, dividing_x, dividing_y))
            all_data_list.extend(whole_region(keyword, dividing_x, start_y, end_x, dividing_y))
            all_data_list.extend(whole_region(keyword, start_x, dividing_y, dividing_x, end_y))
            all_data_list.extend(whole_region(keyword, dividing_x, dividing_y, end_x, end_y))
            return all_data_list
        else:
            if resp.json()['meta']['is_end']:
                all_data_list.extend(resp.json()['documents'])
                return all_data_list
            else:
                page_num += 1
                all_data_list.extend(resp.json()['documents'])


def overlapped_data(keyword, center_x, center_y, dx, dy, steps):
    """
    중심 좌표에서 사방으로 지도 영역을 나누어 카카오 API로 검색
    - center_x, center_y : 중심 좌표
    - dx, dy : 한 격자 폭
    - steps : 중심 기준으로 몇 칸까지 확장할지
    """
    overlapped_result = []

    for i in range(-steps, steps + 1):       # X축 이동
        for j in range(-steps, steps + 1):   # Y축 이동
            start_x = center_x + i * dx
            start_y = center_y + j * dy
            end_x = start_x + dx
            end_y = start_y + dy

            each_result = whole_region(keyword, start_x, start_y, end_x, end_y)
            overlapped_result.extend(each_result)

    # 중복 제거
    return remove_duplicates(overlapped_result)


def remove_duplicates(data_list):
    seen = set()
    unique_list = []
    for d in data_list:
        key = d.get('id') or d.get('place_name')
        if key not in seen:
            seen.add(key)
            unique_list.append(d)
    return unique_list


def setup_chrome_driver():
    """Chrome 드라이버 설정"""
    options = Options()
    for option in CHROME_OPTIONS:
        options.add_argument(option)
    return webdriver.Chrome(options=options, service_log_path='selenium.log')

def scroll_reviews(driver, max_reviews):
    """무한 스크롤로 리뷰 더 로딩"""
    scroll_pause = 1.5
    scroll_count = 0
    max_scrolls = 50  # 안전 장치
    last_count = 0
    no_change_count = 0

    while scroll_count < max_scrolls:
        elements = driver.find_elements(By.CSS_SELECTOR, "div.group_review ul li")
        current_count = len(elements)
        if current_count >= max_reviews:
            break

        if elements:
            driver.execute_script(
                "arguments[0].scrollIntoView({block: 'end', behavior: 'smooth'});",
                elements[-1]
            )
        time.sleep(scroll_pause)

        # 페이지 끝 체크
        if current_count == last_count:
            no_change_count += 1
            if no_change_count >= 3:
                break
        else:
            no_change_count = 0

        last_count = current_count
        scroll_count += 1

def scrape_restaurant_review(driver, place_url, max_reviews=500):
    """음식점 리뷰 크롤링"""
    review_url = f"{place_url}#review"
    driver.get(review_url)
    time.sleep(3)

    # 총 후기 수 가져오기
    try:
        total_review_elem = driver.find_element(
            By.CSS_SELECTOR,
            "#mainContent > div.top_basic > div.info_main > div:nth-child(2) > div:nth-child(2) > a > span.info_num"
        )
        total_reviews = int(total_review_elem.text.replace(",", "").strip())
    except:
        total_reviews = 0

    if total_reviews == 0:
        return []

    # 실제 크롤링할 최대 리뷰 개수 설정
    actual_max = min(max_reviews, total_reviews)

    # 무한 스크롤
    scroll_reviews(driver, actual_max)

    # 리뷰 요소 가져오기
    all_elements = driver.find_elements(By.CSS_SELECTOR, "div.group_review ul li")
    reviews = []

    for elem in all_elements:
        if len(reviews) >= actual_max:
            break

        # 리뷰 내용
        try:
            content_elem = elem.find_element(By.CSS_SELECTOR, "div.wrap_review p")
            try:
                btn_more = content_elem.find_element(By.CSS_SELECTOR, "span.btn_more")
                if btn_more.is_displayed():
                    driver.execute_script("arguments[0].click();", btn_more)
                    time.sleep(0.2)
            except NoSuchElementException:
                pass

            content = content_elem.text.replace("접기", "").replace("\n", " ").strip()
            if not content:
                continue
        except:
            continue

        # 별점
        try:
            star_container = elem.find_element(
                By.CSS_SELECTOR,
                "div.review_detail > div.info_grade > span.starred_grade > span.wrap_grade"
            )
            filled_stars = star_container.find_elements(By.CSS_SELECTOR, "span.figure_star.on")
            rating = len(filled_stars)
        except:
            rating = None

        reviews.append({
            "content": content,
            "rating": rating
        })

    return reviews