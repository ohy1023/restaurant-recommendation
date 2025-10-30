import re
from konlpy.tag import Okt


def text_clearing(text):
    """한글만 남기고 나머지 제거"""
    hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')
    result = hangul.sub('', text)
    return result


def get_pos(text):
    """형태소 분석 및 품사 태깅"""
    tagger = Okt()
    pos = tagger.pos(text)
    result = [f'{word}/{tag}' for word, tag in pos]
    return result


def tokenize_with_okt(sentence, stopwords, stem=True):
    """Okt를 사용한 토큰화 및 불용어 제거"""
    okt = Okt()
    tokenized = okt.morphs(sentence, stem=stem)
    filtered = [word for word in tokenized if word not in stopwords]
    return filtered


def clean_review_text(text):
    """리뷰 텍스트 정제 (한글, 공백만 남김)"""
    return re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', '', text)


def extract_feature_keywords(feature_words, reviews):
    """리뷰에서 특정 키워드 추출"""
    keywords = []
    for word in feature_words:
        for review in reviews:
            if word in review:
                keywords.append(word)
    return keywords


def separate_score_and_text(review_string):
    """점수와 텍스트를 분리 (예: '5 - 맛있어요\\ ' 형식)"""
    reviews = review_string.split('\\ ')
    scores = []
    texts = []

    for review in reviews:
        if not review:
            continue
        parts = review.split(' - ')
        if len(parts) >= 2:
            try:
                score = int(parts[0])
                text = parts[1]
                scores.append(score)
                texts.append(text)
            except ValueError:
                continue

    return scores, texts


def calculate_text_length_distribution(nested_list, max_len):
    """텍스트 길이 분포 계산"""
    count = sum(1 for sentence in nested_list if len(sentence) <= max_len)
    percentage = (count / len(nested_list)) * 100
    return count, percentage