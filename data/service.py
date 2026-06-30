import os
import time
import pandas as pd 
import json 
from google import genai
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. 설정 및 초기화 ---
# Gemini API 초기화 (실제 사용 시 API 키 확인 필수)
client = genai.Client(api_key="AIzaSyDe0hyTvo8tFkmPtw9GtjfW5C4x3C1LKFE")
CACHE_FILE = 'gemini_cache.json'

# 1. 질문을 숫자로 바꾸는 함수 (API 호출)
def get_query_embedding(query):
    # API를 통해 질문을 벡터로 변환
    result = client.models.embed_content(
        model="text-embedding-004", 
        contents=query
    )
    return result.embeddings[0].values

# 2. 가장 비슷한 관광지를 찾는 함수
def search_similar_tour(user_query, embedded_df):
    # 질문을 벡터로 변환
    query_vec = get_query_embedding(user_query)
    
    # 데이터프레임의 모든 벡터를 리스트로 가져오기 (반드시 리스트로 변환해야 함)
    data_vecs = list(embedded_df['embedding'])
    
    # 유사도 계산 (질문 벡터와 모든 데이터 벡터 비교)
    # [query_vec] 처럼 리스트로 감싸서 보내야 함
    similarities = cosine_similarity([query_vec], data_vecs)
    
    # 가장 높은 점수를 가진 데이터의 인덱스 찾기
    best_idx = np.argmax(similarities)
    
    return embedded_df.iloc[best_idx]
# 테스트 
def get_embodding(text):
    result = client.models.embed_content(
        model = "text-embedding-004",
        contents= text, 
    )
    return result.embeddings[0].value

def train_and_save_data(df):
    print("데이터 임베딩 중... (데이터 양에 따라 시간이 걸릴 수 있어요)")
    
    # 1. '관광지소개' 컬럼을 숫자의 배열(벡터)로 변환
    # 만약 데이터가 너무 많으면 중간에 API 할당량 에러가 날 수 있으니 
    # 처음엔 .head(5)로 5개만 먼저 테스트해보는 걸 추천해!
    df['embedding'] = df['관광지소개'].apply(get_embedding)
    
    # 2. 결과를 JSON으로 저장 (이 파일이 나중에 AI의 '지식 저장소'가 됨)
    df.to_json('tour_data_embedding.json', orient='records', force_ascii=False)
    
    print("학습 완료! 'tour_data_embedding.json' 파일이 생성되었습니다.")
# --- 2. 캐시 관리 함수 (할당량 절약용) ---
def load_cache():
    """저장된 캐시 파일을 불러옵니다."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    """결과를 캐시 파일에 저장합니다."""
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=4)

# --- 3. Gemini 연동 함수 ---
def ask_gemini(user_query):
    """
    Gemini 모델에 질문을 보냅니다.
    현재는 API 할당량 제한으로 인해 '가짜 답변'을 반환하는 테스트 모드입니다.
    """
    # [테스트 모드] API 할당량이 풀리면 아래 return문을 주석 처리하세요.
    return "가짜 답변: API 할당량이 풀리면 정말 멋진 추천을 해줄게!"
    
    # --- [실제 호출 모드] 할당량 복구 후 아래 주석을 해제하세요 ---
    # cache = load_cache()
    # if user_query in cache: 
    #     print(f"[캐시] 답변을 불러옵니다.")
    #     return cache[user_query]
    #
    # print(f"[API] 질문 중: {user_query[:20]}...")
    # time.sleep(5) # API 호출 간격 확보
    # response = client.models.generate_content(model='gemini-2.0-flash', contents=user_query)
    # cache[user_query] = response.text
    # save_cache(cache)
    # return response.text

# --- 4. 데이터 전처리 함수 ---
def preprocess_data(df):
    """필요한 컬럼만 추출하고 기본 전처리를 수행합니다."""
    columns_to_keep = ['관광지명', '소재지도로명주소', '위도', '경도', '관광지소개']
    df_clean = df[columns_to_keep].dropna(subset=['관광지명', '소재지도로명주소']).copy()
    
    # 주소에서 시군구 정보 분리
    df['시군구'] = df['소재지도로명주소'].str.split(' ').str[1]
    return df_clean

# --- 5. 추천 시스템 로직 ---
def recommend_tour_site(user_preference, df):
    """사용자 취향을 분석하여 적합한 관광지 정보를 Gemini에게 전달합니다."""
    # 데이터가 너무 많으면 API 비용/용량 문제가 생기므로 상위 10개만 샘플링
    data_sample = df[['관광지명', '관광지소개']].head(10).to_string()
    
    prompt = f"""
    아래 관광지 정보들을 참고해서 사용자의 취향에 맞는 관광지를 추천해줘.
    
    [관광지 정보]
    {data_sample}
    
    [사용자 취향]
    {user_preference}
    
    답변은 추천 이유를 포함해서 친절하게 작성해줘.
    """
    return ask_gemini(prompt)


# --- 6. 메인 실행부 ---
if __name__ == "__main__" :
    file_path = os.path.join(os.path.dirname(__file__), '전국관광지정보표준데이터.csv')
    
    try:
        # 데이터 로드
        df = pd.read_csv(file_path, encoding='cp949')
        df_clean = preprocess_data(df)

        train_and_save_data(df_clean.head(3))
        
        # 테스트 1: 기본 질문
        print("--- 테스트 1: 질문 ---")
        print(ask_gemini("당신은 누구인가요?"))
        
        # 테스트 2: 추천 시스템
        print("\n--- 테스트 2: 관광지 추천 ---")
        my_taste = "아이들과 함께 가기 좋은, 체험 활동이 많은 관광지를 추천해줘."
        print(recommend_tour_site(my_taste, df_clean))
        
    except FileNotFoundError:
        print("에러: 데이터 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    except Exception as e:
        print(f"예기치 못한 에러 발생: {e}")