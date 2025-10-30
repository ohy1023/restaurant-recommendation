import pandas as pd
from tqdm import tqdm
from data.database import (
    get_restaurants_by_type, get_restaurant_details,
    get_restaurant_reviews, get_restaurant_scores,
    get_good_words, get_bad_words
)
from utils.text_utils import text_clearing, extract_feature_keywords


class RestaurantRecommender:
    """맛집 추천 시스템"""

    def __init__(self, conn, sentiment_analyzer, distance_calculator):
        """
        Args:
            conn: 데이터베이스 연결
            sentiment_analyzer: SentimentAnalyzer 인스턴스
            distance_calculator: DistanceCalculator 인스턴스
        """
        self.conn = conn
        self.sentiment_analyzer = sentiment_analyzer
        self.distance_calculator = distance_calculator
        self.good_words = get_good_words(conn)
        self.bad_words = get_bad_words(conn)

    def get_nearby_restaurants(self, food_type, user_location, top_n=10):
        """
        음식 종류와 사용자 위치 기반으로 가까운 음식점 찾기

        Args:
            food_type: 음식 종류
            user_location: (latitude, longitude)
            top_n: 상위 N개 선택

        Returns:
            DataFrame: 음식점 정보
        """
        # DB에서 해당 음식 종류 검색
        restaurants = get_restaurants_by_type(self.conn, food_type)

        if not restaurants:
            return None

        # DataFrame 생성
        df = pd.DataFrame(restaurants, columns=['pk', 'X', 'Y'])
        df.reset_index(drop=True, inplace=True)

        # 거리 계산
        distances = self.distance_calculator.calculate_distances_to_restaurants(
            user_location, df
        )
        df['minDist'] = distances

        # 거리순 정렬 후 상위 N개 선택
        df = df.sort_values(by='minDist', ascending=True).head(top_n)
        df.reset_index(drop=True, inplace=True)

        return df

    def collect_restaurant_reviews(self, restaurant_ids):
        """
        여러 음식점의 리뷰 수집

        Args:
            restaurant_ids: 음식점 ID 리스트

        Returns:
            DataFrame: 음식점별 리뷰 정보
        """
        data = {
            'store': [],
            'kind': [],
            'score': [],
            'review': [],
            'lat': [],
            'lng': [],
            'url': []
        }

        for rest_id in restaurant_ids:
            # 음식점 상세 정보
            details = get_restaurant_details(self.conn, rest_id)
            if not details:
                continue

            name, kind, lat, lng, url = details

            # 리뷰와 점수 가져오기
            reviews = get_restaurant_reviews(self.conn, rest_id)
            scores = get_restaurant_scores(self.conn, rest_id)

            # 데이터 추가
            for review, score in zip(reviews, scores):
                data['store'].append(name)
                data['kind'].append(kind)
                data['score'].append(score)
                data['review'].append(review)
                data['lat'].append(lat)
                data['lng'].append(lng)
                data['url'].append(url)

        df = pd.DataFrame(data)
        return df

    def analyze_sentiment(self, reviews_df):
        """
        리뷰 감성 분석

        Args:
            reviews_df: 리뷰 DataFrame

        Returns:
            DataFrame: 감성 점수 추가된 DataFrame
        """
        # 텍스트 정제
        reviews_df['ko_review'] = reviews_df['review'].apply(text_clearing)

        # 감성 예측
        predictions = []
        for review in tqdm(reviews_df['ko_review'], desc="감성 분석 중"):
            pred = self.sentiment_analyzer.predict_score(review)
            predictions.append(pred)

        reviews_df['pre_score'] = predictions
        return reviews_df

    def calculate_scores(self, reviews_df):
        """
        음식점별 점수 계산

        Args:
            reviews_df: 리뷰 DataFrame

        Returns:
            DataFrame: 음식점별 종합 점수
        """
        unique_stores = reviews_df['store'].unique()
        results = []

        for store in unique_stores:
            store_df = reviews_df[reviews_df['store'] == store]

            # 긍정/부정 단어 기반 점수
            good_count = len(extract_feature_keywords(
                self.good_words,
                store_df['ko_review'].tolist()
            ))
            bad_count = len(extract_feature_keywords(
                self.bad_words,
                store_df['ko_review'].tolist()
            ))

            total_words = good_count + bad_count
            word_score = (good_count / total_words * 100) if total_words > 0 else 0

            # LSTM 기반 점수
            lstm_score = (store_df['pre_score'].sum() / len(store_df) * 100)

            # 종합 점수
            total_score = (word_score + lstm_score) / 2

            results.append({
                'Store': store_df['store'].iloc[0],
                'Score': round(word_score, 2),
                'Pre_Score': round(lstm_score, 2),
                'Total Score': round(total_score, 2),
                'Review': len(store_df),
                'lat': store_df['lat'].iloc[0],
                'lng': store_df['lng'].iloc[0],
                'url': store_df['url'].iloc[0]
            })

        result_df = pd.DataFrame(results)
        return result_df

    def recommend(self, food_type, user_location, top_n=3):
        """
        맛집 추천 메인 함수

        Args:
            food_type: 음식 종류
            user_location: (latitude, longitude)
            top_n: 추천할 맛집 개수

        Returns:
            DataFrame: 추천 맛집 정보
        """
        # 1. 가까운 음식점 찾기
        nearby_df = self.get_nearby_restaurants(food_type, user_location, top_n=10)

        if nearby_df is None or len(nearby_df) == 0:
            return None

        # 2. 리뷰 수집
        reviews_df = self.collect_restaurant_reviews(nearby_df['pk'].tolist())

        if len(reviews_df) == 0:
            return None

        # 3. 감성 분석
        reviews_df = self.analyze_sentiment(reviews_df)

        # 4. 점수 계산
        scores_df = self.calculate_scores(reviews_df)

        # 5. 정렬 및 상위 N개 선택
        result = scores_df.sort_values(
            by=['Review', 'Total Score'],
            ascending=[False, False]
        ).head(top_n)

        result.reset_index(drop=True, inplace=True)
        return result