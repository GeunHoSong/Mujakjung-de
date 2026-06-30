import os
import time
import pandas as pd 
import json 
from sklearn.metrics.pairwise import cosine_similarity
import random
import numpy as np

# --- 1. 설정 및 초기화 ---
# API 호출은 테스트 모드에서는 사용하지 않습니다.
CACHE_FILE = 'gemini_cache.json'

# [테스트 모드] API 호출 제거: 랜덤 벡터 반환
def get_query_embedding(query):
    return [random.random() for _ in range(768)]

# 2. 가장 비슷한 관광지를 찾는 함수
def search_similar_tour(user_query, embedded_df):
    query_vec = get_query_embedding(user_query)
    data_vecs = list(embedded_df['embedding'])
    similarities = cosine_similarity([query_vec], data_vecs)
    best_idx = np.argmax(similarities)
    return embedded_df.iloc[best_idx]

# [테스트 모드] API 호출 제거: 랜덤 벡터 반환
def get_embedding(text):
    return [random.random() for _ in range(768)]

def train_and_save_data(df):
    print("데이터 임베딩 중... (테스트 모드: 랜덤 벡터 생성)")
    
    # 1. 원본을 안전하게 복사하고 임베딩 컬럼 생성
    df_embedded = df.copy()
    
    # [수정] apply를 한 번만 깔끔하게 호출
    df_embedded['embedding'] = df_embedded['관광지소개'].apply(get_embedding)
    
    # 2. JSON 저장
    df_embedded.to_json('tour_data_embedding.json', orient='records', force_ascii=False)
    
    print("학습 완료! 'tour_data_embedding.json' 파일이 생성되었습니다.")
    return df_embedded # 결과를 반환해서 메인에서 바로 쓰게 함
# --- 3. Gemini 연동 (가짜 답변 모드) ---
def ask_gemini(user_query):
    return "가짜 답변: API 할당량이 풀리면 정말 멋진 추천을 해줄게!"

# --- 4. 데이터 전처리 ---
def preprocess_data(df):
    columns_to_keep = ['관광지명', '소재지도로명주소', '위도', '경도', '관광지소개']
    df_clean = df[columns_to_keep].dropna(subset=['관광지명', '소재지도로명주소']).copy()
    return df_clean

# --- 5. 추천 시스템 로직 ---
def recommend_tour_site(user_preference, df):
    data_sample = df[['관광지명', '관광지소개']].head(10).to_string()
    return ask_gemini(f"참고 정보: {data_sample} / 사용자 취향: {user_preference}")

# --- 6. 메인 실행부 ---
if __name__ == "__main__" :
    file_path = os.path.join(os.path.dirname(__file__), '전국관광지정보표준데이터.csv')
    
    try:
        df = pd.read_csv(file_path, encoding='cp949')
        df_clean = preprocess_data(df)
        
        # 1. 학습 테스트
        df_clean = train_and_save_data(df_clean.head(3))
        # [핵심 추가!] 파일에서 다시 불러오거나, 
        # 방금 만든 임베딩 값을 df_clean에 넣어줘야 에러가 안 나.
        # 가장 간단한 방법:
        df_clean.loc[df_clean.index[:3], 'embedding'] = df_clean.head(3)['embedding']
        
        # 2. 검색 테스트
        print("\n--- 테스트 3: 검색 시스템(벡터 유사도) ---")
        best_tour = search_similar_tour("아이들과 가기 좋은 곳", df_clean)
        print(f"추천된 관광지: {best_tour['관광지명']}")
        
    except Exception as e:
        print(f"에러 발생: {e}")