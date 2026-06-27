import os
import time
import pandas as pd # pandas를 여기서 불러와야 해!
from google import genai

# 1. 클라이언트 초기화 (API 키는 .env에서 가져오는 게 좋아)
client = genai.Client(api_key="AIzaSyDe0hyTvo8tFkmPtw9GtjfW5C4x3C1LKFE")

# 데이터 파일 불러오기
file_path = os.path.join(os.path.dirname(__file__), '전국관광지정보표준데이터.csv')
try: 
    df = pd.read_csv(file_path, encoding='cp949') 
    print(" --- 컬럼 목록 --- ")
    print(df.columns.tolist())
    ("\n-- 데이터 상위 3개 ---")
    print(df.head(3)) # 수정됨
except FileExistsError:
    print("파일을 못찾았어~ 경로를 다시 확인 해봐 : {file_path}")

def ask_gemini(user_query):
    # API 호출 전엔 쉬어주기 (무료 티어 제한 방지)
    time.sleep(5) 
    
    # 2. 최신 모델 호출 방식
    prompt = f"질문: {user_query}"
    
    # 여기서 response 변수를 선언해야 해!
    response = client.models.generate_content(
        model='gemini-2.0-flash', 
        contents=prompt,
    )
    
    return response.text
def preprocess_data(df):
    # 정확한 컬럼명으로 수정했어 (터미널 출력값 기준)
    columns_to_keep = ['관광지명', '소재지도로명주소', '위도', '경도', '관광지소개']
    df_clean = df[columns_to_keep].copy()

    # subset 이름도 정확하게!
    df_clean = df_clean.dropna(subset=['관광지명', '소재지도로명주소'])

    print("-- 전처리 완료된 데이터 샘플 --")
    print(df_clean.head())
    return df_clean


    

if __name__ == "__main__":
    print(ask_gemini("당신은 누구인가요?"))