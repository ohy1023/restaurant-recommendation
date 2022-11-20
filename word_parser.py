# 1번 파일이 실행될 때 환경변수에 현재 자신의 프로젝트의 settings.py파일 경로를 등록.
import os
import pandas as pd
import re
from tqdm import tqdm
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "djangoProject4.settings")

# 2번 실행파일에 Django 환경을 불러오는 작업.
import django

django.setup()

# 3번 크롤링을 하고 DB model에 저장.
from content.models import good_word, bad_word


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

def good_feature_sep(top50, text_data_dict):
    good_feature = []
    good_feature_temp = []

    for value, idx in top50:
        good_feature.append(text_data_dict[idx])

    for i in good_feature:
        st_li = i.split('/')
        good_feature_temp.append(st_li[0])
    return good_feature_temp

if __name__ == '__main__':
    df = pd.read_csv('신촌 음식점 정보.csv', encoding='utf8', index_col=0)

    total_food = pd.DataFrame(columns=['store', 'kind', 'score', 'review', 'y'])
    s = ['1', '2', '3', '4', '5']

    for i in range(len(df)):
        if pd.isna(df['리뷰'][i]):
            pass
        else:
            n = df['식당 이름'][i]
            m = df['종류'][i]
            k = df['리뷰'][i].split('\\ ')
            del k[-1]

            review = []
            score = []
            store = []
            kind = []

            for i in k:
                j = i.split(' - ')
                store.append(n)
                kind.append(m)
                if j[0] in s:
                    score.append(j[0])
                else:
                    score.append(-1)
                review.append(j[-1])
            food_df = pd.DataFrame(columns=['store', 'kind', 'score', 'review', 'y'])
            food_df['store'] = store
            food_df['kind'] = kind
            food_df['score'] = score
            food_df['review'] = review

        total_food = pd.concat([total_food, food_df])

    total_food = total_food.astype({'score': 'int'})
    total_food.reset_index(drop=True, inplace=True)

    for i in tqdm(range(len(total_food))):
        if total_food['score'][i] >= 4:
            total_food['y'][i] = 1
        else:
            total_food['y'][i] = 0

    sample = total_food['score'] >= 0
    sample = total_food[sample]

    sample["ko_review"] = sample["review"].apply(lambda x: text_clearing(x))
    del sample['review']
    sample = sample.astype({'y': 'int'})

    index_vectorizer = CountVectorizer(tokenizer=lambda x: get_pos(x))
    X = index_vectorizer.fit_transform(sample["ko_review"].tolist())

    # TFidf 변환 모델 생성
    tfidf_vectorizer = TfidfTransformer()
    # 형태소 벡터 변환하기
    X = tfidf_vectorizer.fit_transform(X)
    y = sample["y"]

    # LogisticRegression
    # penalty : 규제의 종류(l1, l2, elasticnet, none)
    # C : 규제의 강도
    params = {
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    }

    model = LogisticRegression()
    kfold = KFold(n_splits=10, shuffle=True, random_state=1)
    grid_clf = GridSearchCV(model, param_grid=params, scoring='f1', cv=kfold)
    grid_clf.fit(X, y)

    model = grid_clf.best_estimator_

    # 상관관계수 구하기
    a1 = (model.coef_[0])
    a2 = list(enumerate(a1))
    a3 = []

    for idx, value in a2:
        a3.append((value, idx))

    coef_pos_index = sorted(a3, reverse=True)

    # 새로운 딕셔너리 생성
    text_data_dict = {}

    # 단어 사전에 있는 단어의 수만큼 반복한다.
    for key in index_vectorizer.vocabulary_:
        # 현재 key에 해당하는 값을 가져온다.
        value = index_vectorizer.vocabulary_[key]

        # 위의 딕셔너리에 담는다.
        text_data_dict[value] = key

    # 긍정적인 어조 (상관계수가 1에 가장 큰)
    top50 = coef_pos_index[:50]
    # 부정적인 어조
    bottom50 = coef_pos_index[-50:]

    good = good_feature_sep(top50, text_data_dict)
    bad = bad_feature_sep(bottom50, text_data_dict)

    for i in good:
        good_word(word=i).save()

    for j in bad:
        bad_word(word=j).save()
