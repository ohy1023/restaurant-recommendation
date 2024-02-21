# 졸업 작품

참여 학생 : 최승하, 오형상

## 사용 환경
Python 3.7

Mysql

Streamlit 1.15.1

## End Point
https://ohy1023-chatbot-restaurant-recommendation-service-m6tmie.streamlit.app/

## 실행 화면
![image](https://user-images.githubusercontent.com/110380812/204121588-fae2d5ee-8cdf-4b69-902c-82a74a96e0a9.png)
![image](https://user-images.githubusercontent.com/110380812/204121602-166b6d6f-a62c-43eb-9bdd-6c208ed3b335.png)
![image](https://user-images.githubusercontent.com/110380812/204121614-6612e3ad-7bd2-481b-a2b0-c6c0491e6ac2.png)
![image](https://user-images.githubusercontent.com/110380812/204121620-0043424a-82bd-48e1-8b60-486711eb77b8.png)


## 1️⃣ 프로젝트 취지 및 목적

 기존에는 최적화된 맞춤 맛집 추천 시스템이 존재하지 않아 스케줄링에 시간 소모가 심각하다고 여겨, 사용자의 효율적인 시간 사용과 인터넷의 과대광고로 인한 실질적인 맛집인가에 대한 효용성 문제로 정확하게 구분하고자 하였습니다.

---

## 2️⃣ 상세 과정

### 음식점 정보 / 리뷰 크롤링

 KAKAO API를 이용하여 시작 좌표에서 경도/위도를 움직이며 해당 지역에 음식점 정보를 크롤링 하였고, 음식점 정보에 있는 가게 url에 Selenium 라이브러리를 통해 접근하여 가게 리뷰를 크롤링 하여음식점 정보 테이블과 음식점 리뷰 테이블을 만들어 DB화하였습니다.

![화면 캡처 2022-11-27 145401](https://user-images.githubusercontent.com/110380812/204121464-18437727-b1ce-4497-9c20-70cf75d928b0.png)


### Sentiment Dictionary

 음식과 관련하여, 사용되는 사람의 보편적인 기본 감정 표현을 나타내는 긍정/부정어로 구성되는 사전을 구축하여 학습 모델이 보다 쉽게 학습하고, 예측하기 위한 용도로 감성 사전을 제작하였습니다.빠른 예측을 위해 DB화하였습니다.

### Osmnx

 OpenStreetMap의 도로망 데이터를 기반으로 NetworkX를 이용하며, 실제 거리 네트워크 및 기타 지리 공간을 모델링, 시각화, 분석 등을 할 수 있는 Python Package입니다. 해당 Package를 활용하여, 도보, 자전거, 자동차 등의 거리 시간 계산을 쉽게 이용할 수 있으며, 다른 인프라 유형을 보기 쉽게 다룰 수 있습니다. 따라서 사용자의 위치와 조건에 맞는 음식점 간의 거리를 계산하는 용도로 사용하였습니다.

### 감성분류

 문서 또는 문장을 긍정 / 부정으로 나누는 이진 분류에 해당하는 감성 분석을 사용하였습니다. LSTM을 사용한 이유는 RNN Sequence으로 사용할 경우 긍정 / 부정 판단에 누적 학습에 왜곡 현상이 발생하여 판단 오류가 생기는 Vanishing 현상이 발생할 수 밖에 없었고, 이를 예방하고자 LSTM(Long Short Term Memory)를 사용하였습니다.

### 맛집 추천

 사용자가 위치 정보 사용에 동의하면 음식 종류를 입력받는 칸이 생성되며 그 값을 통해 DB에 음식점 종류가 같거나 음식점 이름에 사용자 입력값이 포함된 결과를 가져와 해당 음식점들과 현재 사용자 위치를 Osmnx 라이버르리를 이용하여 사용자와 가까운 음식점 10개를 선정.

DB에서 선정된 음식점들의 리뷰를 가져와  DB에 저장된 감성 사전의 긍정/부정 데이터를 통하여 빈도 계산 + RNN을 이용한 긍정/부정 예측 후 빈도 계산하여 최종 점수를 구하였고 그 결과를 점수와 리뷰 개수를 내림차순 정렬하면 최종 3개의 음식점을 추천하도록 설계하였습니다.

---

### 3️⃣ 기대효과

 학습을 통한 예측을 기반 추천 시스템을 통해서 시간적 효율성과 한눈에 보기 쉬운 비교 분석을 할 수 있습니다. 또한, 서버를 통하여 지역을 확대해나갈 수 있는 확장성을 염두에 두고 개발을 진행하여 원하는 지역을 간단하게 추가 작업할 수 있는 방안으로 제작되었습니다.

✔️ 내용 : [신촌에서 뭐 먹지? (notion.so)](https://www.notion.so/e9abe8ef8c8042658c73c7faf4c2555a)

✔️ 코드 보기 : [https://github.com/ohy1023/chatbot-restaurant-recommendation](https://github.com/ohy1023/chatbot-restaurant-recommendation)
