import os
import pickle
import numpy as np 
from dotenv import load_dotenv
from openai import OpenAI

# .env 파일 로드 
load_dotenv()

EMBEDDINGS_INPUT_PATH = 'tour_embeddings_openai.pkl'
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)
# 1. 검색어 임베딩 생성 
def get_embedding(text):
    # 1. 검색어 정리
    text = text.replace("\n", " ").strip()

    # 2. OpenAI 임베딩 생성
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    # 3. 임베딩 벡터 반환
    return response.data[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 2. 관련 관광지 검색 
def search_tour(query, top_k=3, destination= None):
    with open(EMBEDDINGS_INPUT_PATH, 'rb') as f:
        embedding_data = pickle.load(f)

    query_vector = get_embedding(query)
    results = []
    for item in embedding_data:
        tour = item["data"]
        address=  tour.get("소재지도로명주소") or ""
        # 사용자 지정한 지역만 검색 
        if destination :
            if destination not in address:
                continue
        # 유사도 계산 
        sim = cosine_similarity(
            query_vector,
            item["embedding"]
        )
        results.append({
            "tour": item["data"],
            "similarity": sim
        })
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]

# 3. 검색된 데이터를 참고하여 Gemini로 일정표 생성 
def generate_travel_schedule(user_request , destination=None):
    print(f"사용자 요청: {user_request}\n" + "_" * 50)
    
    # Step 1: 사용자의 요청과 관련된 관광지 정보 검색
    retrieved_results = search_tour(
        user_request, 
        top_k=3,
        destination= destination
        ) 
    #지역 필터링 결과 확인 
    print("선택한 목적지", destination)
    print("검색된 관광지 수", len(retrieved_results))
    for result in retrieved_results:
        tour = result["tour"]
        print("관광지:", tour.get("관광지명"))
        print("주소:", tour.get("소재지도로명주소"))

    # Step 2: 검색된 장소 정보를 프롬프트의 참고 자료(Context)로 가공
    context_text = ""
    for i, res in enumerate(retrieved_results, 1):
        tour = res["tour"]
        context_text += f"[{i}] 관광지명: {tour.get('관광지명')}\n"
        context_text += f"    주소: {tour.get('소재지도로명주소')}\n"
        context_text += f"    소개: {tour.get('관광지소개')}\n\n"

    # Step 3: Gemini에게 전달할 프롬프트 작성
    prompt = f"""
당신은 전문 여행 플래너입니다. 아래 제공된 [참고 관광지 데이터]를 바탕으로 사용자의 요청에 맞는 맞춤형 여행 일정표를 작성해 주세요.

[참고 관광지 데이터]
{context_text}

[사용자 요청]
{user_request}

[출력 조건]
- 제공된 참고 관광지 데이터를 반드시 일정에 포함해 주세요.
- 오전, 오후, 저녁 시간대별로 알맞은 동선으로 구성해 주세요.
- 친절하고 깔끔한 마크다운 형식으로 작성해 주세요. 
"""

    print("AI가 맞춤 일정을 짜는 중입니다...")
    
    # Step 4: Gemini 생성 모델 호출 (gemini-2.0-flash로 변경)
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output_text

if __name__ == "__main__":
    # 테스트 요청 
    user_query = "부산으로 바다를 보며 힐링할 수 있는 1박 2일 여행 코스 짜줘"
    schedule = generate_travel_schedule( 
        user_query,
        destination= "부산"
    )
    print("=== AI 맞춤 일정표 ===")
    print(schedule)