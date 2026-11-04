import pandas as pd
import json

# 전처리 감정 데이터 불러오기
df = pd.read_csv("data/emotion_travle.csv")

# 데이터 개수 확인 
print("전체 데이터 개수: ", len(df))

#처음 5개 문장과 감정확인 
print(df[["text", "label"]].head())
