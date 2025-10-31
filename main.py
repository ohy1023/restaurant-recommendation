import streamlit as st
import pandas as pd
from streamlit_folium import folium_static
from sklearn.model_selection import train_test_split

# 로컬 모듈 import
from config.settings import STOPWORDS
from data.database import (
    init_connection, init_db, insert_restaurant_info,
    insert_restaurant_review, insert_good_words, insert_bad_words,
    get_all_reviews, get_all_scores, get_all_restaurant_info
)
from data.crawler import (
    overlapped_data, setup_chrome_driver, scrape_restaurant_review
)
from data.preprocessor import (
    create_review_dataframe, label_reviews,
    filter_valid_scores, clean_reviews, tokenize_reviews,
    build_vocabulary, remove_empty_samples
)
from models.word_extractor import WordExtractor
from models.sentiment_analyzer import SentimentAnalyzer
from services.location_service import get_current_location_ip
from services.distance_calculator import DistanceCalculator
from services.restaurant_recommender import RestaurantRecommender
from utils.map_utils import create_restaurant_map

# 페이지 설정
st.set_page_config(
    page_title="신촌에서 뭐 먹지?",
    page_icon="🍽️",
    layout="wide"
)

# 세션 상태 초기화
if 'sentiment_analyzer' not in st.session_state:
    st.session_state.sentiment_analyzer = None
if 'distance_calculator' not in st.session_state:
    st.session_state.distance_calculator = None


def section_crawling():
    """데이터 크롤링 섹션"""
    st.header("1. 데이터 크롤링 및 초기 설정")

    # 식당 정보 크롤링
    if st.button("식당 정보 크롤링 후 DB에 넣기"):
        with st.spinner('크롤링 중입니다... 잠시만 기다려주세요'):
            try:
                # 크롤링 파라미터
                keyword = '음식점'
                center_x, center_y = 126.9362, 37.5555  # 신촌역
                dx, dy = 0.01, 0.01
                steps = 2

                # 데이터 수집
                data = overlapped_data(keyword, center_x, center_y, dx, dy, steps)

                # DB 저장
                conn = init_connection()
                insert_restaurant_info(conn, data)
                conn.close()

                st.success(f"{len(data)}개 식당 정보가 저장되었습니다!")
            except Exception as e:
                st.error(f"오류 발생: {str(e)}")

    st.markdown("---")

    # 리뷰 크롤링
    if st.button("리뷰 크롤링"):
        with st.spinner('리뷰 크롤링 중입니다... 많은 시간이 소요될 수 있습니다'):
            try:
                conn = init_connection()
                info_list = get_all_restaurant_info(conn)
                conn.close()

                driver = setup_chrome_driver()

                results = []

                for info in info_list:
                    restaurant_id = info['id']
                    url = info['url']

                    # 각 음식점 리뷰 크롤링
                    reviews = scrape_restaurant_review(driver, url)

                    # 각 리뷰에 restaurant_id 추가
                    for review in reviews:
                        review['restaurant_id'] = restaurant_id
                        results.append(review)

                driver.quit()

                result_df = pd.DataFrame(results)
                st.dataframe(result_df)
                st.success("리뷰 크롤링 완료!")

                conn = init_connection()
                insert_restaurant_review(conn, results)
                conn.close()
                st.success("크롤링된 리뷰 정보가 DB에 저장되었습니다!")
            except Exception as e:
                st.error(f"오류 발생: {str(e)}")

    st.markdown("---")

    # 긍정/부정 단어 추출
    if st.button("긍정/부정 단어 DB에 넣기"):
        with st.spinner('단어 추출 중...'):
            try:
                conn = init_connection()
                reviews = get_all_reviews(conn)
                conn.close()

                # DataFrame으로 변환
                df = pd.DataFrame(reviews)

                # 레이블링
                sample = filter_valid_scores(df)
                sample = label_reviews(sample)
                sample = clean_reviews(sample)

                st.dataframe(sample)

                # 단어 추출
                extractor = WordExtractor()
                extractor.fit(sample['ko_review'].tolist(), sample['y'].tolist())

                good_words, bad_words = extractor.extract_feature_words(percentile=10)

                # DB 저장
                conn = init_connection()
                insert_good_words(conn, good_words)
                insert_bad_words(conn, bad_words)
                conn.close()

                st.write("**긍정 단어**")
                st.write(good_words)
                st.write("**부정 단어**")
                st.write(bad_words)

                st.success("단어 추출 완료!")
            except Exception as e:
                st.error(f"오류 발생: {str(e)}")


def section_core_logic():
    """핵심 로직 섹션"""
    st.header("2. 핵심 로직")

    # 거리 계산
    st.subheader("OSMnx를 이용한 거리 계산")
    st.write("신촌역에서 창천근린공원까지의 거리 계산")

    if st.button("거리 계산하기"):
        with st.spinner('계산 중...'):
            try:
                calculator = DistanceCalculator(
                    center_point=(37.5555, 126.9362),
                    dist=1500
                )

                origin = (37.5555, 126.9362)
                destination = (37.5578737346181, 126.94028032885905)  # 창천근린공원

                distance = calculator.get_distance(origin, destination)
                fig, ax = calculator.plot_route(origin, destination)

                st.pyplot(fig)
                st.success(f"거리: {distance} km")
            except Exception as e:
                st.error(f"오류 발생: {str(e)}")

    st.markdown("---")

    # 감성 분석
    st.subheader("LSTM을 이용한 감성 분류")
    input_text = st.text_input(
        '리뷰를 입력하세요',
        max_chars=100,
        placeholder='예: 음식이 정말 맛있어요!'
    )

    if st.button("감성 분석 실행"):
        if not input_text:
            st.warning("리뷰를 입력해주세요")
        else:
            with st.spinner('분석 중...'):
                try:
                    # 데이터 로드
                    conn = init_connection()
                    reviews = get_all_reviews(conn)
                    conn.close()

                    # 데이터 준비
                    df = create_review_dataframe(reviews)
                    df = label_reviews(df)

                    # 학습/테스트 분리
                    train_data, test_data = train_test_split(
                        df,
                        stratify=df['y'],
                        random_state=25,
                        test_size=0.2
                    )

                    # 토큰화
                    train_data['review'] = train_data['review'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", regex=True)
                    X_train = tokenize_reviews(train_data['review'], STOPWORDS)
                    X_test = tokenize_reviews(test_data['review'], STOPWORDS)

                    # 모델 학습
                    analyzer = SentimentAnalyzer()
                    X_train_pad, X_test_pad = analyzer.preprocess_data(X_train, X_test)

                    y_train = train_data['y'].values
                    y_test = test_data['y'].values

                    # 빈 샘플 제거
                    X_train_pad, y_train = remove_empty_samples(X_train_pad, y_train)

                    # 모델 학습
                    analyzer.build_model()
                    analyzer.train(X_train_pad, y_train, epochs=15, batch_size=64)
                    analyzer.load_best_model()

                    # 예측
                    result = analyzer.predict_sentiment(input_text)
                    st.success(f"{result}")

                    # 세션에 저장
                    st.session_state.sentiment_analyzer = analyzer

                except Exception as e:
                    st.error(f"오류 발생: {str(e)}")


def section_recommendation():
    """맛집 추천 섹션"""
    st.header("3. 맛집 찾기 서비스")
    st.write("본 서비스는 사용자의 위치 정보를 사용합니다")

    agree = st.checkbox("위치 정보 사용에 동의하십니까?")

    if agree:
        # 음식 종류 입력
        food_type = st.text_input(
            '드시고 싶은 음식을 입력하세요',
            max_chars=20,
            placeholder='예: 피자, 치킨, 한식'
        )

        # 위치 설정 (실제로는 GPS나 IP 기반으로 가져와야 함)
        user_lat, user_lng = get_current_location_ip()
        if user_lat is None or user_lng is None:
            st.warning("위치를 가져오지 못했습니다. 수동으로 입력해주세요.")
            col1, col2 = st.columns(2)
            with col1:
                user_lat = st.number_input("위도", value=36.8509, format="%.4f")
            with col2:
                user_lng = st.number_input("경도", value=127.1531, format="%.4f")
        else:
            st.info(f"현재 위치: 위도 {user_lat}, 경도 {user_lng}")

        if st.button("맛집 찾기"):
            if not food_type:
                st.warning("음식 종류를 입력해주세요")
            else:
                with st.spinner('맛집을 찾는 중입니다... 잠시만 기다려주세요'):
                    try:
                        # 데이터베이스 연결
                        conn = init_connection()

                        # 감성 분석기 준비
                        if st.session_state.sentiment_analyzer is None:
                            st.info("감성 분석 모델을 로드하는 중...")

                            # 데이터 로드 및 학습
                            reviews = get_all_reviews(conn)
                            df = create_review_dataframe(reviews)
                            df = label_reviews(df)

                            train_data, test_data = train_test_split(
                                df, stratify=df['y'], random_state=25, test_size=0.2
                            )

                            train_data['review'] = train_data['review'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", regex=True)
                            X_train = tokenize_reviews(train_data['review'], STOPWORDS)
                            X_test = tokenize_reviews(test_data['review'], STOPWORDS)

                            analyzer = SentimentAnalyzer()
                            X_train_pad, X_test_pad = analyzer.preprocess_data(X_train, X_test)
                            y_train = train_data['y'].values
                            X_train_pad, y_train = remove_empty_samples(X_train_pad, y_train)

                            analyzer.build_model()
                            analyzer.train(X_train_pad, y_train, epochs=15, batch_size=64)
                            analyzer.load_best_model()

                            st.session_state.sentiment_analyzer = analyzer

                        # 거리 계산기 준비
                        if st.session_state.distance_calculator is None:
                            calculator = DistanceCalculator(
                                center_point=(user_lat, user_lng),
                                dist=5500
                            )
                            st.session_state.distance_calculator = calculator

                        # 추천 시스템 실행
                        recommender = RestaurantRecommender(
                            conn,
                            st.session_state.sentiment_analyzer,
                            st.session_state.distance_calculator
                        )

                        result = recommender.recommend(
                            food_type,
                            (user_lat, user_lng),
                            top_n=3
                        )

                        conn.close()

                        if result is None or len(result) == 0:
                            st.warning("해당 음식을 찾을 수 없습니다. 다른 음식을 검색해보세요.")
                        else:
                            st.subheader("추천 맛집 TOP 3")

                            # 데이터프레임 스타일링
                            styled_df = result[
                                ['Store', 'Total Score', 'Review', 'Score', 'Pre_Score']].style.highlight_max(
                                subset=['Total Score'],
                                color='lightgreen'
                            ).format({
                                'Total Score': '{:.2f}',
                                'Score': '{:.2f}',
                                'Pre_Score': '{:.2f}'
                            })

                            st.dataframe(styled_df, use_container_width=True)

                            # 지도 표시
                            st.subheader("맛집 위치")
                            m = create_restaurant_map(
                                center_location=[user_lat, user_lng],
                                user_location=[user_lat, user_lng],
                                restaurants_df=result
                            )
                            folium_static(m, width=700, height=500)

                            # 개별 음식점 정보
                            st.subheader("상세 정보")
                            for idx in range(len(result)):
                                restaurant = result.iloc[idx]
                                with st.expander(f"{idx + 1}. {restaurant['Store']}"):
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("총점", f"{restaurant['Total Score']:.2f}")
                                    with col2:
                                        st.metric("리뷰 수", restaurant['Review'])
                                    with col3:
                                        st.metric("단어 분석 점수", f"{restaurant['Score']:.2f}")

                                    st.write(f"**LSTM 분석 점수:** {restaurant['Pre_Score']:.2f}")
                                    st.write(f"[카카오맵에서 보기]({restaurant['url']})")

                            st.success("추천 완료!")

                    except Exception as e:
                        st.error(f"오류 발생: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())


def main():
    """메인 함수"""
    conn = init_connection()
    init_db(conn)
    conn.close()

    st.title("신촌에서 뭐 먹지?")
    st.markdown("---")

    # 사이드바
    with st.sidebar:
        st.header("메뉴")
        menu = st.radio(
            "기능 선택",
            ["데이터 크롤링", "핵심 로직 테스트", "맛집 추천"],
            index=2
        )

        st.markdown("---")
        st.info("""
        **사용 방법**
        1. 데이터 크롤링으로 식당 정보 수집
        2. 핵심 로직에서 기능 테스트
        3. 맛집 추천으로 최적의 식당 찾기
        """)

    # 선택된 메뉴 실행
    if menu == "데이터 크롤링":
        section_crawling()
    elif menu == "핵심 로직 테스트":
        section_core_logic()
    elif menu == "맛집 추천":
        section_recommendation()


if __name__ == '__main__':
    main()
