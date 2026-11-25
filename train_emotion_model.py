import pandas as pd
import joblib

from sklearn.model_selection  import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# 학습 데이터 불러오기 
df = pd.read_csv("data/emotion_travel.csv")

#전체 학습 데이터 개수 확인 
print("전체 데이터 개수 확인", len(df))

#감정 코드 종류 확인 
print("감정 종류:", df["emotion_code"].nunique())

#학습 데이터 미리 보기 
print(df.head())

#입력 데이터 와 정답 데이터 분리 
x = df["text"]
y= df["emotion_code"]

print("입력 데이터 개수: ", len(x))
print("정답 데이터 개수: ", len(y))


#학습용 데이터와 검증용 데이터 분리 
x_train , x_val, y_train , y_val = train_test_split(
    x, 
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("학습 테이터 개수 ", len(x_train))
print("검증 데이터 개수", len(x_val))

# 감정 분류 모델 구성 
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer= "char",
        ngram_range=(2,4),
        max_features=30000
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])
#감정 ai 모델 학습 시작
print("감정 AI 모델 학습 시작")

model.fit(x_train, y_train)

print("감정 AI 모델 학습 완료")

# 검증 데이터 감정 예측
y_pred  = model.predict(x_val)

#모델 정확도 확인 
accuracy = accuracy_score(y_val, y_pred)
print("모델 정확도",  accuracy)

# 감정별 상세 성능 평가 
print(classification_report(
    y_val,
    y_pred,
    zero_division=0
))

#학습된 감정  AI 모델 저장 
joblib.dump(
    model, 
    "data/emotion_model.pkl"
)
print("감정 AI 모델 저장 완료")


# 새로운 문장 감정 예측 테스트 
test_sentences = [
    "요즘 너무 지치고 우울해",
    "내일이 너무 걱정돼",
    "오늘 정말 행복하고 기분이 좋아",
    "아무것도 내 마음을 이해해 주지 않아",
    "너무 화가 나고 짜증이 나"
]

predictions = model.predict(test_sentences)

#감정 예측 결과 출력 
print("\n 감정 예측 테스트 ")

for text, prediction in zip (test_sentences, predictions):
    print(f"문장 : {text}")
    print(f"문장 예측 :{prediction}")
    print("-" * 40)
