import os
import time
import pandas as pd 
import json  # 추가
from google import genai

# 1. Gemini API 클라이언트 초기화
client = genai.Client(api_key="AIzaSyDe0hyTvo8tFkmPtw9GtjfW5C4x3C1LKFE")

# 캐시 파일 설정
CACHE_FILE = 'gemini_cache.json'

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=4)

# 3. 수정된 Gemini AI 질문 함수 (캐싱 적용)
def ask_gemini(user_query):
    cache = load_cache()
    
    # 이미 물어본 적이 있으면 API 호출 안 함
    if user_query in cache:
        print(f"[캐시] '{user_query}'에 대한 답변을 가져옵니다.")
        return cache[user_query]
    
    # 캐시에 없으면 API 호출
    print(f"[API] '{user_query}' 질문 중...")
    time.sleep(5) 
    
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash', 
            contents=f"질문: {user_query}",
        )
        answer = response.text
        
        # 결과 저장
        cache[user_query] = answer
        save_cache(cache)
        return answer
    except Exception as e:
        return f"에러 발생: {e}"

# 4. 데이터 전처리 함수 (내용 그대로 유지)
def preprocess_data(df):
    columns_to_keep = ['관광지명', '소재지도로명주소', '위도', '경도', '관광지소개']
    df_clean = df[columns_to_keep].copy()
    df_clean = df_clean.dropna(subset=['관광지명', '소재지도로명주소'])
    
    # 시군구 추출
    df['시군구'] = df['소재지도로명주소'].str.split(' ').str[1]
    
    return df_clean

# 5. 프로그램 시작
if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(__file__), '전국관광지정보표준데이터.csv')
    try:
        df = pd.read_csv(file_path, encoding='cp949')
        df_clean = preprocess_data(df)
        
        # 테스트: 이제 10번을 실행해도 첫 번째만 API를 부르고 나머지는 캐시에서 가져옴!
        print(ask_gemini("당신은 누구인가요?"))
        
    except FileNotFoundError:
        print("파일을 찾을 수 없습니다.")

# 수정 포인트
# 1. '관광지명 ' -> '관광지명' (뒤에 공백 제거)
# 2. '관광지 소개' -> '관광지소개' (우리가 위에서 정의한 컬럼명과 일치시켜야 해)
# 3. to.string() -> to_string() (언더바가 맞아)
# 4. data_samle -> data_sample (오타 수정)
# 5. if __name__ == "__name__" -> if __name__ == "__main__" (이게 핵심!)

def recommend_tour_site(user_preference, df):
    # 컬럼명과 메서드 이름 정확히 확인
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

if __name__ == "__main__":
    # 데이터가 로드된 상태인 df를 사용해야 함
    df_clean = preprocess_data(df)

    # 추천 요청 
    my_taste = "아이들과 함께 가기 좋은, 체험 활동이 많은 관광지를 추천해줘."
    print(recommend_tour_site(my_taste, df_clean))
