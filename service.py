import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
# 설정 및 라이브러리
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import re
from konlpy.tag import Okt
import osmnx as ox, networkx as nx
import pandas as pd
from django.db.models import Q
import streamlit as st
from tqdm import tqdm
import my_settings
import requests
from collections import OrderedDict
import django

# 설정
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "djangoProject4.settings")
# 2번 실행파일에 Django 환경을 불러오는 작업.
django.setup()

from content.models import restaurant_info, restaurant_review, good_word, bad_word


# 함수 구간
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


# 자연어 처리
def text_clearing(text):
    hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')
    result = hangul.sub('', text)
    return result


# 형태소 추출
def get_pos(x):
    tagger = Okt()
    pos = tagger.pos(x)
    result = []
    for a1 in pos:
        result.append(f'{a1[0]}/{a1[1]}')

    return result


# 긍정 / 부정 리뷰 사전
def good_feature_sep(top50, text_data_dict):
    good_feature = []
    good_feature_temp = []

    for value, idx in top50:
        good_feature.append(text_data_dict[idx])

    for i in good_feature:
        st_li = i.split('/')
        good_feature_temp.append(st_li[0])

    return good_feature_temp


def get_good_feature_keywords(good_feature_temp, review):
    feature_temp = []
    for i in good_feature_temp:
        for j in review:
            if i in j:
                feature_temp.append(i)

    return feature_temp


def bad_feature_sep(bottom50, text_data_dict):
    bad_feature = []
    bad_feature_temp = []

    for value, idx in bottom50:
        bad_feature.append(text_data_dict[idx])

    for i in bad_feature:
        st_li = i.split('/')
        bad_feature_temp.append(st_li[0])

    return bad_feature_temp


def get_bad_feature_keywords(bad_feature_temp, review):
    feature_temp = []
    for i in bad_feature_temp:
        for j in review:
            if i in j:
                feature_temp.append(i)

    return feature_temp


def below_threshold_len(max_len, nested_list):
    count = 0
    for sentence in nested_list:
        if (len(sentence) <= max_len):
            count = count + 1
    print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s' % (max_len, (count / len(nested_list)) * 100))


def review_predict(new_sentence):
    new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', '', new_sentence)
    new_sentence = okt.morphs(new_sentence, stem=True)  # 토큰화
    new_sentence = [word for word in new_sentence if not word in stopwords]  # 불용어 제거
    encoded = tokenizer.texts_to_sequences([new_sentence])  # 정수 인코딩
    pad_new = pad_sequences(encoded, maxlen=max_len)  # 패딩
    score = float(loaded_model.predict(pad_new))  # 예측
    if (score > 0.5):
        print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(score * 100))
    else:
        print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - score) * 100))


# 현재 위치 좌표로 가져오기
def get_my_place_google():
    url = f'https://www.googleapis.com/geolocation/v1/geolocate?key={my_settings.GOOGLE_API_KEY}'
    data = {
        'considerIp': True,
    }

    result = requests.post(url, data)

    lat = result.json()['location']['lat']
    lng = result.json()['location']['lng']

    return lat, lng


def extract_review():
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    # 첫 페이지 리뷰 목록 찾기
    review_lists = soup.select('.list_evaluation > li')
    # 리뷰가 있는 경우
    return review_lists


if __name__ == '__main__':

    space = """
            <br/>
        """

    st.title('신촌에서 뭐 먹지?')

    if st.button("식당 정보 크롤링 후 db에 넣기"):
        st.write("Data Loading..")
        keyword = '음식점'
        start_x = 126.93
        start_y = 37.55
        next_x = 0.01
        next_y = 0.01
        num_x = 2
        num_y = 2

        overlapped_result = overlapped_data(keyword, start_x, start_y, next_x, next_y, num_x, num_y)

        # 최종 데이터가 담긴 리스트 중복값 제거
        results = list(map(dict, OrderedDict.fromkeys(tuple(sorted(d.items())) for d in overlapped_result)))
        for i in results:
            restaurant_info(id=i['id'], name=i['place_name'], x=i['x'], y=i['y'], address=i['road_address_name'],
                            url=i['place_url'], type=(i['category_name'].split('>')[-1])).save()

        st.success("Success")

    st.markdown(space, unsafe_allow_html=True)

    if st.button("리뷰 크롤링"):
        df = pd.read_csv('신촌 음식점.csv', encoding='utf-8-sig')

        options = webdriver.ChromeOptions()  # 크롬 옵션 객체 생성
        options.add_argument('headless')  # headless 모드 설정
        options.add_argument("window-size=1920x1080")  # 화면크기(전체화면)
        options.add_argument("disable-gpu")
        options.add_argument("disable-infobars")
        options.add_argument("--disable-extensions")

        # 속도 향상을 위한 옵션 해제
        prefs = {'profile.default_content_setting_values': {'cookies': 2, 'images': 2, 'plugins': 2, 'popups': 2,
                                                            'geolocation': 2, 'notifications': 2,
                                                            'auto_select_certificate': 2, 'fullscreen': 2,
                                                            'mouselock': 2, 'mixed_script': 2, 'media_stream': 2,
                                                            'media_stream_mic': 2, 'media_stream_camera': 2,
                                                            'protocol_handlers': 2, 'ppapi_broker': 2,
                                                            'automatic_downloads': 2, 'midi_sysex': 2,
                                                            'push_messaging': 2, 'ssl_cert_decisions': 2,
                                                            'metro_switch_to_desktop': 2,
                                                            'protected_media_identifier': 2, 'app_banner': 2,
                                                            'site_engagement': 2, 'durable_storage': 2}}
        options.add_experimental_option('prefs', prefs)

        driver = webdriver.Chrome('D:/공주대학교/4-2/종합 설계/python_project/chromedriver.exe', options=options)
        driver.maximize_window()

        info = pd.DataFrame(columns=['종류', '별점', '리뷰 개수', '오픈 시간', '마감 시간', '해시 태그', '리뷰'])

        cnt = 0
        for i in tqdm(df['웹 주소']):
            kakao_map_search_url = i
            driver.implicitly_wait(3)
            driver.get(kakao_map_search_url)

            try:
                restaurantType = driver.find_element(By.XPATH,
                                                     '//*[@id="mArticle"]/div[1]/div[1]/div[2]/div/div/span[1]').text
            except:
                restaurantType = ''
            try:
                rateNum = driver.find_element(By.XPATH,
                                              '//*[@id="mArticle"]/div[1]/div[1]/div[2]/div/div/a[1]/span[1]').text
                rateNum = int(rateNum)
            except:
                rateNum = 0
            try:
                reviewCnt = driver.find_element(By.XPATH, '//*[@id="mArticle"]/div[6]/strong[1]/span').text
                reviewCnt = int(reviewCnt)
            except:
                reviewCnt = 0
            try:
                restaurantInfo = driver.find_element(By.XPATH,
                                                     '//*[@id="mArticle"]/div[1]/div[2]/div[2]/div/div[1]/ul/li/span/span').text
                restaurantOpenInfo = restaurantInfo.split("~")[0].strip()
                restaurantCloseInfo = restaurantInfo.split("~")[1].strip()
            except:
                restaurantOpenInfo = ''
                restaurantCloseInfo = ''
            try:
                hashTag = driver.find_element(By.XPATH, '//*[@id="mArticle"]/div[1]/div[2]/div[5]/div/div/span').text
            except:
                hashTag = ''

            try:
                if reviewCnt > 3:
                    clickCnt = (reviewCnt - 3) // 5 + 1
                    for _ in range(clickCnt):
                        driver.find_element(By.CLASS_NAME, 'txt_more').click()
                        time.sleep(1)
            except:
                pass

            review_content = ""
            if reviewCnt > 0:
                for review in extract_review():
                    comment = review.find('p', 'txt_comment')
                    comment = comment.select_one('span').string
                    wid = review.find('span', 'ico_star inner_star')
                    wid = wid.get('style')
                    if wid != None and wid.startswith('width'):
                        wid = wid[6:]
                        wid = wid.strip("%;")
                        wid = int(wid)
                        if wid > 80:
                            rating = 5
                        elif wid > 60:
                            rating = 4
                        elif wid > 40:
                            rating = 3
                        elif wid > 20:
                            rating = 2
                        else:
                            rating = 1
                    if comment == None:
                        text = "내용이 없습니다"
                        review_content += str(rating) + " - " + text + "\ "
                    else:
                        review_content += str(rating) + " - " + comment + "\ "

            print(review_content)

            info.loc[cnt] = [restaurantType, rateNum, reviewCnt, restaurantOpenInfo, restaurantCloseInfo, hashTag,
                             review_content]

            cnt += 1
            if cnt == 5:
                break

        driver.quit()

        info

        st.success("Success")

    st.markdown(space, unsafe_allow_html=True)

    if st.button("긍정 부정 단어 db에 넣기"):
        st.write("Data Loading..")
        # 데이터 로딩 함수는 여기에!
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

        st.success("Success")

    st.markdown(space, unsafe_allow_html=True)

    if st.button("감정 분석"):

        st.write("Data Loading..")
        # 데이터 로딩 함수는 여기에!
        df = pd.read_csv('신촌 음식점 정보.csv', encoding='utf8', index_col=0)

        total_food = pd.DataFrame(columns=['store', 'kind', 'score', 'review', 'y'])
        s = ['1', '2', '3', '4', '5']

        for i in range(len(df)):
            if pd.isna(df['리뷰'][i]):
                pass
            else:
                k = df['리뷰'][i].split('\\')
                n = df['식당 이름'][i]
                m = df['종류'][i]
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

        # y 값 라벨링
        for i in tqdm(range(len(total_food))):
            if total_food['score'][i] >= 4:
                total_food['y'][i] = 1
            elif total_food['score'][i] == -1:
                total_food['y'][i] = -1
            else:
                total_food['y'][i] = 0

        # 전처리 함수 중 리뷰 없는 부분 제거
        sample = total_food['score'] >= 0
        sample = total_food[sample]
        sample["ko_review"] = sample["review"].apply(lambda x: text_clearing(x))
        del sample['review']
        sample = sample.astype({'y': 'int'})

        sample.reset_index(drop=True, inplace=True)

        # y 예측 모델링
        train_data, test_data = train_test_split(sample)

        train_data['ko_review'] = train_data['ko_review'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
        train_data[:5]
        stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

        okt = Okt()

        # 학습데이터
        X_train = []
        for sentence in tqdm(train_data['ko_review']):
            tokenized_sentence = okt.morphs(sentence, stem=True)
            stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords]
            X_train.append(stopwords_removed_sentence)

        # 검증데이터
        X_test = []
        for sentence in tqdm(test_data['ko_review']):
            tokenized_sentence = okt.morphs(sentence, stem=True)  # 토큰화
            stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords]  # 불용어 제거
            X_test.append(stopwords_removed_sentence)

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X_train)
        threshold = 3
        total_cnt = len(tokenizer.word_index)
        rare_cnt = 0
        total_freq = 0
        rare_freq = 0

        for key, value in tokenizer.word_counts.items():
            total_freq = total_freq + value
            if (value < threshold):
                rare_cnt = rare_cnt + 1
                rare_freq = rare_freq + value

        vocab = total_cnt - rare_cnt + 1
        tokenizer = Tokenizer(vocab)
        tokenizer.fit_on_texts(X_train)
        X_train = tokenizer.texts_to_sequences(X_train)
        X_test = tokenizer.texts_to_sequences(X_test)
        y_train = np.array(train_data['y'])
        y_test = np.array(test_data['y'])
        del_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]
        X_train = np.delete(X_train, del_train, axis=0)
        y_train = np.delete(y_train, del_train, axis=0)

        max_len = 30
        below_threshold_len(max_len, X_train)

        X_train = pad_sequences(X_train, maxlen=max_len)
        X_test = pad_sequences(X_test, maxlen=max_len)
        X_train = np.asarray(X_train).astype(np.int)
        X_test = np.asarray(X_train).astype(np.int)
        y_train = np.asarray(y_train).astype(np.int)
        y_test = np.asarray(y_train).astype(np.int)

        embedding_dim = 100
        hidden_units = 128

        # 예측 모델
        model = Sequential()
        model.add(Embedding(vocab, embedding_dim))
        model.add(LSTM(hidden_units))
        model.add(Dense(1, activation='sigmoid'))

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
        mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
        history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=64, validation_split=0.2)
        loaded_model = load_model('best_model.h5')

        review_predict('의자도 편하고 다시 방문하고 싶습니다')
        review_predict('지나가다 눈에 보였서 어쩔수 없이 들어갔는데 괜찮습니다')
        review_predict('다시는 안가')
        review_predict('사장님은 예쁜데 싸가지가 존나 없어요')
        review_predict('김치는 맛있는데 딴거는 별로')
        review_predict('이게 음식이냐')

        st.success("Success")

    st.markdown(space, unsafe_allow_html=True)

    if st.button("감정 분석 (초기 버전)"):
        st.write("Data Loading..")
        # 데이터 로딩 함수는 여기에!
        df = pd.read_csv('신촌 음식점 정보.csv', encoding='utf8', index_col=0)

        total_food = pd.DataFrame(columns=['store', 'kind', 'score', 'review', 'y'])
        s = ['1', '2', '3', '4', '5']

        for i in range(len(df)):
            if pd.isna(df['리뷰'][i]):
                pass
            else:
                k = df['리뷰'][i].split('\ ')
                n = df['식당 이름'][i]
                m = df['종류'][i]
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

        total_food.reset_index(drop=True, inplace=True)
        total_food = total_food.astype({'score': 'int'})

        for i in tqdm(range(len(total_food))):
            if total_food['score'][i] >= 4:
                total_food['y'][i] = 1
            else:
                total_food['y'][i] = 0

        # 학습용과 검증용으로 나누는 함수
        from sklearn.model_selection import train_test_split

        train_data, test_data = train_test_split(total_food)

        train_data['review'] = train_data['review'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
        stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

        okt = Okt()

        # 훈련 데이터 토큰화
        X_train = []
        for sentence in tqdm(train_data['review']):
            tokenized_sentence = okt.morphs(sentence, stem=True)  # 토큰화
            stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords]  # 불용어 제거
            X_train.append(stopwords_removed_sentence)

        # 테스트 데이터 토큰화
        X_test = []
        for sentence in tqdm(test_data['review']):
            tokenized_sentence = okt.morphs(sentence, stem=True)  # 토큰화
            stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords]  # 불용어 제거
            X_test.append(stopwords_removed_sentence)

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X_train)

        threshold = 3
        total_cnt = len(tokenizer.word_index)  # 단어의 수
        rare_cnt = 0  # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
        total_freq = 0  # 훈련 데이터의 전체 단어 빈도수 총 합
        rare_freq = 0  # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

        # 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
        for key, value in tokenizer.word_counts.items():
            total_freq = total_freq + value

            # 단어의 등장 빈도수가 threshold보다 작으면
            if (value < threshold):
                rare_cnt = rare_cnt + 1
                rare_freq = rare_freq + value

        # 전체 단어 개수 중 빈도수 2이하인 단어는 제거.
        # 0번 패딩 토큰을 고려하여 + 1
        vocab_size = total_cnt - rare_cnt + 1

        tokenizer = Tokenizer(vocab_size)
        tokenizer.fit_on_texts(X_train)
        X_train = tokenizer.texts_to_sequences(X_train)
        X_test = tokenizer.texts_to_sequences(X_test)

        y_train = np.array(train_data['y'])
        y_test = np.array(test_data['y'])

        drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]

        # 빈 샘플들을 제거
        X_train = np.delete(X_train, drop_train, axis=0)
        y_train = np.delete(y_train, drop_train, axis=0)

        plt.hist([len(review) for review in X_train], bins=50)
        plt.xlabel('length of samples')
        plt.ylabel('number of samples')
        plt.show()

        max_len = 30
        below_threshold_len(max_len, X_train)

        X_train = pad_sequences(X_train, maxlen=max_len)
        X_test = pad_sequences(X_test, maxlen=max_len)

        X_train = np.asarray(X_train).astype(np.int)
        X_test = np.asarray(X_train).astype(np.int)
        y_train = np.asarray(y_train).astype(np.int)
        y_test = np.asarray(y_train).astype(np.int)

        embedding_dim = 100
        hidden_units = 128

        model = Sequential()
        model.add(Embedding(vocab_size, embedding_dim))
        model.add(LSTM(hidden_units))
        model.add(Dense(1, activation='sigmoid'))

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
        mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
        history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=64, validation_split=0.2)

        loaded_model = load_model('best_model.h5')

        review_predict('의자도 편하고 다시 방문하고 싶습니다')
        review_predict('지나가다 눈에 보였서 어쩔수 없이 들어갔는데 괜찮습니다')
        review_predict('다시는 안가')
        review_predict('사장님은 예쁜데 싸가지가 존나 없어요')
        review_predict('김치는 맛있는데 딴거는 별로')
        review_predict('이게 음식이냐')

        st.success("Success")

    st.markdown("---")

    with st.container():
        st.subheader("맛집 찾기 서비스")

        input_food = st.text_input('input food : ', key='food_name')
        if st.button("찾기"):
            food_type = st.session_state.food_name

            # 요청 받아야하는 값 : ex)피자
            # 추후 카카오톡 obt도면 수정 예정
            querySet = restaurant_info.objects.filter(
                Q(name__contains=food_type) | Q(type__contains=food_type)).values()

            querySet = list(querySet)
            print(querySet)

            pk = [i['id'] for i in querySet]
            X = [j['x'] for j in querySet]
            Y = [k['y'] for k in querySet]

            df2 = pd.DataFrame({'pk': pk, 'X': X, 'Y': Y})
            df2.reset_index(drop=True, inplace=True)

            # 신촌역 좌표
            point = 37.5598, 126.9425
            G = ox.graph_from_point(point, network_type='bike', dist=1500)
            Gs = ox.utils_graph.get_largest_component(G, strongly=True)

            # 요청 받아야하는 값 : 사용자 위치
            user_x, user_y = get_my_place_google()

            road_li = []  # 도로 기준 최단 거리

            for i in tqdm(range(len(df2))):
                if (df2['X'][i] == user_y) & (df2['Y'][i] == user_x):
                    pass
                else:
                    orig_node = ox.nearest_nodes(Gs, X=user_y, Y=user_x)  # 출발지
                    dest_node = ox.nearest_nodes(Gs, X=df2['X'][i], Y=df2['Y'][i])  # 목적지
                    len_road = nx.shortest_path_length(Gs, orig_node, dest_node, weight='length')
                    road_li.append(len_road)

            df2['minDist'] = road_li

            df2 = df2.sort_values(by='minDist', axis=0, ascending=True)
            total_food = pd.DataFrame(columns=['store', 'kind', 'score', 'review', 'y'])

            review = []
            score = []
            store = []
            kind = []

            for i in list(df2['pk'][:10]):

                querySet2 = restaurant_info.objects.filter(id=i)

                querySet3 = restaurant_review.objects.filter(restaurant_id=i).values_list('review', flat=True)

                querySet4 = restaurant_review.objects.filter(restaurant_id=i).values_list('score', flat=True)

                data = querySet2.get()
                reviews = list(querySet3)
                scores = list(querySet4)

                for _ in range(len(reviews)):
                    store.append(data.name)
                    kind.append(data.type)
                for j in scores:
                    score.append(j)
                for k in reviews:
                    review.append(k)

            total_food['store'] = store
            total_food['kind'] = kind
            total_food['score'] = score
            total_food['review'] = review

            total_food["ko_review"] = total_food["review"].apply(lambda x: text_clearing(x))
            del total_food['review']

            total_food.reset_index(drop=True, inplace=True)
            restaurant_reviews = []
            for i in tqdm(range(len(total_food))):
                temp_dict = {}
                temp_dict['_id'] = total_food['store'][i]
                temp_dict['review'] = total_food['ko_review'][i]
                if temp_dict['review'] == []:
                    continue
                restaurant_reviews.append(temp_dict)

            store_unique = pd.DataFrame(columns=['store'])

            unique_li = total_food['store'].drop_duplicates()
            unique_li.reset_index(drop=True, inplace=True)

            store_unique['store'] = unique_li

            store_unique

            good_feature_temp = list(good_word.objects.values_list('word', flat=True))

            bad_feature_temp = list(bad_word.objects.values_list('word', flat=True))

            total_list = pd.DataFrame(columns=['Store', 'Review', 'Score'])
            score_li = []
            review_li = []
            store_li = []

            for i in range(len(store_unique)):
                sample_df = total_food[total_food['store'] == store_unique['store'][i]]
                sample_df.reset_index(drop=True, inplace=True)
                good_list = get_good_feature_keywords(good_feature_temp, sample_df['ko_review'])
                bad_list = get_bad_feature_keywords(bad_feature_temp, sample_df['ko_review'])
                store_li.append(sample_df['store'][0])
                review_li.append(len(sample_df['ko_review']))
                if (len(good_list) + len(bad_list)) == 0:
                    score_li.append(round(0, 2))
                else:
                    score_li.append(round((len(good_list) / (len(good_list) + len(bad_list)) * 100), 2))

            total_list['Store'] = store_li
            total_list['Score'] = score_li
            total_list['Review'] = review_li

            result = total_list.sort_values('Score', ascending=False)[:3]
            st.write(result)

            st.success("Success")

# @st.cache
# def load_data(nrows):
#     data = pd.read_csv(DATA_URL, nrows=nrows)
#     lowercase = lambda x: str(x).lower()
#     data.rename(lowercase, axis='columns', inplace=True)
#     data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
#     return data
#
#
# data_load_state = st.text('Loading data...')
# data = load_data(10000)
# data_load_state.text("Done! (using st.cache)")
#
# if st.checkbox('Show raw data'):
#     st.subheader('Raw data')
#     st.write(data)
#
# st.subheader('Number of pickups by hour')
# hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0, 24))[0]
# st.bar_chart(hist_values)
#
# hour_to_filter = st.slider('hour', 0, 23, 17)
# filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
#
# st.subheader('Map of all pickups at %s:00' % hour_to_filter)
# st.map(filtered_data)
