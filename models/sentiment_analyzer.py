import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from config.settings import MAX_LEN, EMBEDDING_DIM, HIDDEN_UNITS, VOCAB_THRESHOLD, STOPWORDS
from utils.text_utils import clean_review_text, tokenize_with_okt


class SentimentAnalyzer:
    """LSTM 기반 감성 분석 모델"""

    def __init__(self, max_len=MAX_LEN, embedding_dim=EMBEDDING_DIM,
                 hidden_units=HIDDEN_UNITS, vocab_threshold=VOCAB_THRESHOLD):
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.vocab_threshold = vocab_threshold
        self.tokenizer = None
        self.model = None
        self.vocab_size = None
        self.stopwords = STOPWORDS

    def preprocess_data(self, X_train, X_test):
        """데이터 전처리"""
        # Tokenizer 학습
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(X_train)

        # 어휘 크기 계산
        total_cnt = len(self.tokenizer.word_index)
        rare_cnt = sum(1 for count in self.tokenizer.word_counts.values()
                       if count < self.vocab_threshold)
        self.vocab_size = total_cnt - rare_cnt + 1

        # 새 Tokenizer로 재학습
        self.tokenizer = Tokenizer(self.vocab_size)
        self.tokenizer.fit_on_texts(X_train)

        # 텍스트를 시퀀스로 변환
        X_train_seq = self.tokenizer.texts_to_sequences(X_train)
        X_test_seq = self.tokenizer.texts_to_sequences(X_test)

        # 패딩
        X_train_pad = pad_sequences(X_train_seq, maxlen=self.max_len)
        X_test_pad = pad_sequences(X_test_seq, maxlen=self.max_len)

        return X_train_pad, X_test_pad

    def build_model(self):
        """LSTM 모델 구축"""
        model = Sequential([
            Embedding(self.vocab_size, self.embedding_dim),
            LSTM(self.hidden_units),
            Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['acc']
        )
        self.model = model
        return model

    def train(self, X_train, y_train, epochs=15, batch_size=64, validation_split=0.2):
        """모델 학습"""
        if self.model is None:
            self.build_model()

        # 콜백 설정
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
        mc = ModelCheckpoint(
            'best_model.h5',
            monitor='val_acc',
            mode='max',
            verbose=1,
            save_best_only=True
        )

        # 학습
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[es, mc]
        )

        return history

    def load_best_model(self, model_path='best_model.h5'):
        """저장된 최적 모델 로드"""
        self.model = load_model(model_path)
        return self.model

    def predict_sentiment(self, text):
        """텍스트의 감성 예측"""
        # 텍스트 정제
        cleaned = clean_review_text(text)
        # 토큰화
        tokens = tokenize_with_okt(cleaned, self.stopwords)
        # 시퀀스 변환
        sequence = self.tokenizer.texts_to_sequences([tokens])
        # 패딩
        padded = pad_sequences(sequence, maxlen=self.max_len)
        # 예측
        score = float(self.model.predict(padded, verbose=0)[0][0])

        if score > 0.5:
            return f"{score * 100:.2f}% 확률로 긍정 리뷰입니다."
        else:
            return f"{(1 - score) * 100:.2f}% 확률로 부정 리뷰입니다."

    def predict_score(self, text):
        """텍스트의 긍정/부정 점수 반환 (0 or 1)"""
        cleaned = clean_review_text(text)
        tokens = tokenize_with_okt(cleaned, self.stopwords)
        sequence = self.tokenizer.texts_to_sequences([tokens])
        padded = pad_sequences(sequence, maxlen=self.max_len)
        score = float(self.model.predict(padded, verbose=0)[0][0])

        return 1 if score > 0.5 else 0

    def predict_batch(self, texts):
        """여러 텍스트 일괄 예측"""
        predictions = []
        for text in texts:
            pred = self.predict_score(text)
            predictions.append(pred)
        return predictions