import streamlit as st

# API 키 설정
KAKAO_API_KEY = st.secrets["KEY"]["KAKAO_API_KEY"]
GOOGLE_API_KEY = st.secrets["KEY"]["GOOGLE_API_KEY"]

# 데이터베이스 설정
DB_CONFIG = st.secrets["DATABASES"]

# 모델 설정
MAX_LEN = 30
EMBEDDING_DIM = 100
HIDDEN_UNITS = 128
VOCAB_THRESHOLD = 3

# 불용어 설정
STOPWORDS = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

# 크롤링 설정
CHROME_OPTIONS = [
    "--headless",
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-gpu",
    "--disable-features=NetworkService",
    "--window-size=1920x1080",
    "--disable-features=VizDisplayCompositor"
    "--blink-settings=imagesEnabled=false"
]