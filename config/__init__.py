# config/__init__.py
"""설정 모듈"""
from .settings import (
    KAKAO_API_KEY,
    GOOGLE_API_KEY,
    DB_CONFIG,
    MAX_LEN,
    EMBEDDING_DIM,
    HIDDEN_UNITS,
    VOCAB_THRESHOLD,
    STOPWORDS,
    CHROME_OPTIONS
)

__all__ = [
    'KAKAO_API_KEY',
    'GOOGLE_API_KEY',
    'DB_CONFIG',
    'MAX_LEN',
    'EMBEDDING_DIM',
    'HIDDEN_UNITS',
    'VOCAB_THRESHOLD',
    'STOPWORDS',
    'CHROME_OPTIONS'
]