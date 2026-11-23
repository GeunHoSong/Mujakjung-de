import pandas as pd
import json

file_path = "data/감성대화말뭉치(최종데이터)_Training (1).xlsx"

df = pd.read_excel(file_path)

emotions = (
    df[["감정_대분류", "감정_소분류"]]
    .drop_duplicates()
    .sort_values(["감정_대분류", "감정_소분류"])
)

print("전체 학습 데이터", len(df))
print("감정 분류 등록")
print(emotions.to_string(index=False))


with open("data/emotion_mapping.json", "r", encoding="utf-8") as f:
    emotion_mapping = json.load(f)
df["mapped_emotion"] = df.apply(
    lambda row: emotion_mapping
    .get(row["감정_대분류"], {})
    .get(row["감정_소분류"]),
    axis=1
)
print("매핑 성공", df["mapped_emotion"].notna().sum())
print("매핑 실패", df["mapped_emotion"].isna().sum())
print("감정별 매핑 개수")
print(df["mapped_emotion"].value_counts())


train_df = df.dropna(subset=["mapped_emotion"])

train_df = train_df[
    ["사람문장1", "mapped_emotion"]
].dropna()

train_df.to_csv(
    "data/emotion_train_ready.csv",
    index=False,
    encoding="utf-8-sig"
)

print("학습용 데이터 저장 완료", len(train_df))

print("\n학습 데이터 미리보기")
print(train_df.head(5).to_string(index=False))
