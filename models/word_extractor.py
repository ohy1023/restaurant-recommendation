from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV
from utils.text_utils import get_pos


class WordExtractor:
    """긍정/부정 단어 추출 클래스"""

    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.tfidf_transformer = None
        self.text_data_dict = {}

    def fit(self, reviews, labels):
        """모델 학습"""
        # CountVectorizer로 형태소 벡터화
        self.vectorizer = CountVectorizer(tokenizer=lambda x: get_pos(x))
        X = self.vectorizer.fit_transform(reviews)

        # TF-IDF 변환
        self.tfidf_transformer = TfidfTransformer()
        X = self.tfidf_transformer.fit_transform(X)

        # LogisticRegression 하이퍼파라미터 튜닝
        params = {
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
        }

        kfold = KFold(n_splits=10, shuffle=True, random_state=1)
        grid_clf = GridSearchCV(
            LogisticRegression(max_iter=1000),
            param_grid=params,
            scoring='f1',
            cv=kfold
        )
        grid_clf.fit(X, labels)

        self.model = grid_clf.best_estimator_

        # 단어 사전 생성
        for key, value in self.vectorizer.vocabulary_.items():
            self.text_data_dict[value] = key

        return self

    def get_coefficients(self):
        """상관계수 정렬"""
        coefficients = self.model.coef_[0]
        indexed_coefs = [(coef, idx) for idx, coef in enumerate(coefficients)]
        return sorted(indexed_coefs, reverse=True)

    def extract_positive_words(self, percentile=10):
        """긍정 단어 추출"""
        coef_pos_index = self.get_coefficients()
        n = int(len(coef_pos_index) / percentile)
        top_features = coef_pos_index[:n]

        positive_words = []
        for _, idx in top_features:
            word_tag = self.text_data_dict[idx]
            word = word_tag.split('/')[0]
            positive_words.append(word)

        return positive_words

    def extract_negative_words(self, percentile=10):
        """부정 단어 추출"""
        coef_pos_index = self.get_coefficients()
        n = int(len(coef_pos_index) / percentile)
        bottom_features = coef_pos_index[-n:]

        negative_words = []
        for _, idx in bottom_features:
            word_tag = self.text_data_dict[idx]
            word = word_tag.split('/')[0]
            negative_words.append(word)

        return negative_words

    def extract_feature_words(self, percentile=10):
        """긍정/부정 단어 모두 추출"""
        positive = self.extract_positive_words(percentile)
        negative = self.extract_negative_words(percentile)
        return positive, negative


def count_feature_occurrences(feature_words, reviews):
    """리뷰에서 특정 단어 출현 횟수 계산"""
    count = 0
    for word in feature_words:
        for review in reviews:
            if word in review:
                count += 1
    return count


def calculate_sentiment_score(good_words, bad_words, reviews):
    """긍정/부정 단어 기반 감성 점수 계산"""
    good_count = count_feature_occurrences(good_words, reviews)
    bad_count = count_feature_occurrences(bad_words, reviews)

    total = good_count + bad_count
    if total == 0:
        return 0.0

    return round((good_count / total) * 100, 2)