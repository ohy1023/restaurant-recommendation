from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
import time
import json


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
        time.sleep(3)

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
        self.scroll_reviews(max_reviews, total_reviews)

        # 모든 li 요소 가져오기
        all_elements = self.driver.find_elements(
            By.CSS_SELECTOR,
            "div.group_review ul li"
        )

        print(f"전체 li 요소 수: {len(all_elements)}")

        reviews = []
        seen_contents = set()
        skipped = 0

        # ⭐ 핵심 수정: 모든 요소를 순회하되, 최대 개수만큼 리뷰를 수집
        for idx, elem in enumerate(all_elements):
            # 이미 충분한 리뷰를 수집했으면 종료
            if len(reviews) >= max_reviews:
                break

            try:
                # 더보기 버튼 클릭 시도
                try:
                    more_button = elem.find_element(By.CSS_SELECTOR, "a.link_more")
                    self.driver.execute_script("arguments[0].click();", more_button)
                    time.sleep(0.2)
                except:
                    pass

                # 리뷰 내용 찾기 (여러 선택자 시도)
                # 리뷰 내용 가져오기
                content = None
                try:
                    content_elem = elem.find_element(By.CSS_SELECTOR, "div.wrap_review p")

                    # ⭐ '더보기' span이 있으면 클릭
                    try:
                        btn_more = content_elem.find_element(By.CSS_SELECTOR, "span.btn_more")
                        if btn_more.is_displayed():
                            self.driver.execute_script("arguments[0].click();", btn_more)
                            time.sleep(0.2)
                    except NoSuchElementException:
                        pass

                    # 최종 텍스트 가져오기
                    content = content_elem.text.replace("접기", "").replace("\n", " ").strip()
                except:
                    content = None

                if not content:
                    continue

                # 별점
                rating = None
                try:
                    star_container = elem.find_element(
                        By.CSS_SELECTOR,
                        "div.review_detail > div.info_grade > span.starred_grade > span.wrap_grade"
                    )
                    filled_stars = star_container.find_elements(By.CSS_SELECTOR, "span.figure_star.on")
                    rating = len(filled_stars)  # ★ 개수
                except:
                    rating = None

                reviews.append({
                    "content": content,
                    "rating": rating,
                })

                print(f"✓ 리뷰 {len(reviews)}/{total_reviews} 수집 완료: {content}")

            except Exception as e:
                skipped += 1
                continue

        print(f"\n전체 요소: {len(all_elements)}개")
        print(f"수집된 리뷰: {len(reviews)}개")
        print(f"스킵된 요소: {skipped}개")

        return total_reviews, reviews

    def scroll_reviews(self, max_reviews=50, total_reviews=0):
        """무한 스크롤로 리뷰 더 로딩"""
        scroll_pause = 1.5
        scroll_count = 0
        # 총 리뷰 수만큼 충분히 스크롤
        max_scrolls = min(40, (total_reviews // 3) + 10)

        try:
            review_container = self.driver.find_element(By.CSS_SELECTOR, "div.group_review")
        except:
            print("리뷰 컨테이너를 찾을 수 없습니다.")
            return

        last_review_count = 0
        no_change_count = 0

        print("\n스크롤 시작...")
        while scroll_count < max_scrolls:
            # 현재 로딩된 요소 수 확인
            current_elements = self.driver.find_elements(
                By.CSS_SELECTOR,
                "div.group_review ul li"
            )
            current_count = len(current_elements)

            # 스크롤 실행
            if current_elements:
                self.driver.execute_script(
                    "arguments[0].scrollIntoView({block: 'end', behavior: 'smooth'});",
                    current_elements[-1]
                )
            time.sleep(scroll_pause)

            # 페이지 끝까지 스크롤
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(0.5)

            # 변화 체크
            if current_count == last_review_count:
                no_change_count += 1
                if no_change_count >= 4:  # 4번 연속 변화 없으면 종료
                    print(f"스크롤 완료: 총 {current_count}개 요소 로딩됨")
                    break
            else:
                no_change_count = 0
                print(f"  스크롤 {scroll_count + 1}: {current_count}개 요소")

            last_review_count = current_count
            scroll_count += 1

    def close(self):
        self.driver.quit()


if __name__ == "__main__":
    crawler = KakaoMapCrawler(headless=False)
    place_id = "2036408963"
    total_reviews, reviews = crawler.crawl_reviews(place_id, max_reviews=100)  # 넉넉하게 설정

    print(f"\n{'=' * 50}")
    print(f"=== 최종 수집 결과 ===")
    print(f"{'=' * 50}")
    print(f"총 후기 수: {total_reviews}")
    print(f"실제 수집된 후기 수: {len(reviews)}")
    print(f"수집률: {len(reviews) / total_reviews * 100:.1f}%" if total_reviews > 0 else "N/A")

    with open(f"reviews_{place_id}.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_reviews": total_reviews,
            "collected_reviews": len(reviews),
            "reviews": reviews
        }, f, ensure_ascii=False, indent=2)

    print(f"\n파일 저장 완료: reviews_{place_id}.json")
    crawler.close()