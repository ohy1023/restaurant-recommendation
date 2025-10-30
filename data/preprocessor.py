import pandas as pd
import numpy as np
from tqdm import tqdm
from utils.text_utils import text_clearing, tokenize_with_okt


def create_review_dataframe(reviews, scores):
    """리뷰와 점수로 데이터프레임 생성"""
    df = pd.DataFrame({
        'score': scores,
        'review': reviews
    })
    df = df.astype({'score': 'int'})
    df.reset_index(drop=True, inplace=True)
    return df


def label_reviews(df, positive_threshold=4):
    """리뷰에 긍정/부정 레이블 추가"""
    df['y'] = df['score'].apply(lambda x: 1 if x >= positive_threshold else 0)
    return df


def filter_valid_scores(df, min_score=0):
    """유효한 점수만 필터링"""
    return df[df['score'] >= min_score].copy()


def clean_reviews(df, column='review'):
    """리뷰 텍스트 정제"""
    df[f'ko_{column}'] = df[column].apply(lambda x: text_clearing(x))
    return df


def tokenize_reviews(reviews, stopwords):
    """리뷰 목록 토큰화"""
    tokenized = []
    for review in tqdm(reviews, desc="Tokenizing"):
        tokens = tokenize_with_okt(review, stopwords)
        tokenized.append(tokens)
    return tokenized


def build_vocabulary(tokenizer, X_train, threshold=3):
    """어휘 사전 구축"""
    tokenizer.fit_on_texts(X_train)

    total_cnt = len(tokenizer.word_index)
    rare_cnt = 0
    total_freq = 0
    rare_freq = 0

    for key, value in tokenizer.word_counts.items():
        total_freq += value
        if value < threshold:
            rare_cnt += 1
            rare_freq += value

    vocab_size = total_cnt - rare_cnt + 1
    return vocab_size


def remove_empty_samples(X, y):
    """빈 샘플 제거"""
    drop_indices = [idx for idx, sentence in enumerate(X) if len(sentence) < 1]
    X_filtered = np.delete(X, drop_indices, axis=0)
    y_filtered = np.delete(y, drop_indices, axis=0)
    return X_filtered, y_filtered


def prepare_restaurant_data(df):
    """음식점 정보로부터 리뷰 데이터 준비"""
    total_food = pd.DataFrame(columns=['store', 'kind', 'score', 'review', 'y'])

    for idx in range(len(df)):
        if pd.isna(df['리뷰'][idx]):
            continue

        store_name = df['식당 이름'][idx]
        food_kind = df['종류'][idx]
        reviews = df['리뷰'][idx].split('\\ ')[:-1]

        for review_item in reviews:
            parts = review_item.split(' - ')
            if len(parts) >= 2:
                try:
                    score = int(parts[0])
                    review_text = parts[1]

                    temp_df = pd.DataFrame([{
                        'store': store_name,
                        'kind': food_kind,
                        'score': score,
                        'review': review_text,
                        'y': None
                    }])
                    total_food = pd.concat([total_food, temp_df], ignore_index=True)
                except ValueError:
                    continue

    total_food = total_food.astype({'score': 'int'})
    return total_food