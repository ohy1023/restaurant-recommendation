from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import json
import re


class KakaoMapCrawler:
    def __init__(self, headless=False):
        options = Options()
        if headless:
            options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        self.driver = webdriver.Chrome(options=options)
        self.wait = WebDriverWait(self.driver, 20)

    def crawl_reviews(self, place_id, max_reviews=50):
        url = f"https://place.map.kakao.com/{place_id}#review"
        self.driver.get(url)
        time.sleep(3)  # 초기 로딩 대기

        # 총 후기 수 가져오기
        try:
            total_review_elem = self.driver.find_element(
                By.CSS_SELECTOR,
                "#mainContent > div.top_basic > div.info_main > div:nth-child(2) > div:nth-child(2) > a > span.info_num"
            )
            total_reviews = int(total_review_elem.text.replace(",", "").strip())
        except:
            total_reviews = 0
        print(f"총 후기 수: {total_reviews}")

        # 무한 스크롤로 리뷰 더 로딩
        self.scroll_reviews(max_reviews)

        # 리뷰 요소 선택
        review_elements = self.driver.find_elements(
            By.CSS_SELECTOR,
            "div.group_review ul li"
        )

        reviews = []
        for elem in review_elements[:max_reviews]:
            try:
                # 리뷰 내용
                content_elem = elem.find_element(
                    By.CSS_SELECTOR,
                    "div.wrap_review > a > p"
                )
                content = content_elem.text.strip()

                # 별점
                rating_elem = elem.find_element(
                    By.CSS_SELECTOR,
                    "div.info_grade > span.starred_grade > span.screen_out"
                )

                rating_text = rating_elem.text.strip()
                rating = float(rating_text) if rating_text else None

                reviews.append({
                    "content": content,
                    "rating": rating
                })
            except:
                continue

        return total_reviews, reviews

    def scroll_reviews(self, max_reviews=50):
        """무한 스크롤로 리뷰 더 로딩"""
        scroll_pause = 2
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        scroll_count = 0
        while True:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(scroll_pause)
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            scroll_count += 1
            if new_height == last_height or scroll_count > 20:
                break
            last_height = new_height

    def close(self):
        self.driver.quit()


if __name__ == "__main__":
    crawler = KakaoMapCrawler(headless=False)
    place_id = "2036408963"  # 카카오맵 장소 ID
    total_reviews, reviews = crawler.crawl_reviews(place_id, max_reviews=50)

    print(f"\n총 후기 수: {total_reviews}")
    for i, r in enumerate(reviews, 1):
        print(f"리뷰 {i}: 별점 {r['rating']}, 내용: {r['content'][:100]}...")

    with open(f"reviews_{place_id}.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_reviews": total_reviews,
            "reviews": reviews
        }, f, ensure_ascii=False, indent=2)

    crawler.close()
