import pandas as pd #엑셀과 CSV 데이터를 처리하는 pandas 라이브러리를 불러와.
import json
#df : 읽어 온 데이터를 저장하는 변수야. DataFrame의 줄임말로 많이 사용해.
df = pd.read_excel("data/감성대화말뭉치(최종데이터)_Training (1).xlsx")#pd.read_excel() : 엑셀 파일을 읽어서 표 형태의 데이터로 변환해.
print(df.head())  #엑셀 데이터의 첫 5행을 출력

with open("data/emotion_mapping.json", "r", encoding= "utf-8") as f:
    emotion_mapping = json.load(f)

# 감정 분석 AI 학습에 필요한 두 개의 열 선택
emotion_df = df[[
    "사람문장1", 
    "감정_대분류",
    "감정_소분류"
    ]].copy() #감정 분석 학습에 사용할 데이터를 저장하는 새로운 변수야.

#선택한 데이터의첫 5행 출력 
print(emotion_df.head())


#감정 문장과 감정 라벨의 빈 값 개수 확인 
print(emotion_df.isnull().sum())

#빈값이 포함한 행 제거 
emotion_df= emotion_df.dropna()

#감정 이름 앞뒤로 공백 지우기 
emotion_df["감정_대분류"] = emotion_df["감정_대분류"].str.strip()
emotion_df["감정_소분류"] = emotion_df["감정_소분류"].str.strip()
# 감정 코드 매핑 
emotion_df["emotion_code"] = emotion_df.apply(
    lambda row : emotion_mapping.get(
        row["감정_대분류"], {}
    ).get(row["감정_소분류"]),
    axis=1
)

# 감정 코드 매핑 누락 검사 
missing_count =emotion_df["emotion_code"].isnull().sum()

print("매핑 누락 개수:", missing_count)

#누락된 감정 확인 
if missing_count > 0:
    print (
        emotion_df.loc[
            emotion_df["emotion_code"].isnull(),
            ["감정_대분류", "감정_소분류"]
        ].drop_duplicates()
    )
    raise ValueError("감정 매핑 누락 발생")

#빈값을 제거한후 데이터 개수 확인 
print("정리된 데이터 수 : ", len(emotion_df))

#감정 종류별 데이터 개수 확인
print(emotion_df["감정_대분류"].value_counts())

# 학습에 사용할 열 이름 변경 
emotion_df = emotion_df.rename(columns={
    "사람문장1" : "text",
    "감정_대분류" : "label",
    "감정_소분류" : "sub_label"
})

#학습용 csv 파일에 저장 
emotion_df.to_csv(
    "data/emotion_travel.csv",
    index=False,
    encoding="utf-8-sig"
)

print("감정 학습용 csv 저장 완료")
print(df.columns.tolist())
print(df[["감정_대분류", "감정_소분류"]].drop_duplicates().sort_values(["감정_대분류", "감정_소분류"]).to_string(index=False))
