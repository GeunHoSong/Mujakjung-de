import os 
from dotenv import load_dotenv
from google import genai
from google.genai import types
from schemas.recommend_schema import RecommendRequest, RecommendResponse

# .env 파일에서 환경 변수(API 키 등) 로드 
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# 제미나이(Google GenAI) 클라이언트 초기화 
client = genai.Client(api_key=api_key)

async def generate_travel_recommendation(request: RecommendRequest, candidate_places: list) -> RecommendResponse:
    """
    사용자의 여행 조건과 실제 검색된 장소 목록(candidate_places)을 받아
    Gemini 모델을 통해 맞춤형 여행 코스 추천 결과를 생성합니다.
    
    - candidate_places: 카카오 검색 및 공공데이터를 통해 실제로 확보한 장소 리스트 (이름, 주소, 카테고리 등 포함)
    """
    
    # 1. 실제 검색된 장소 목록을 Gemini가 읽기 편한 텍스트 형태로 변환합니다.
    places_text = "\n".join([
        f"- 장소명 : {p.get('name')}, 주소: {p.get('address')}, 카테고리: {p.get('category', '기타')}"
        for p in candidate_places
    ])
    
    # 2. AI에게 전달할 프롬프트 작성 (할루시네이션 방지 가드레일 포함)
    prompt = f"""
    당신은 전문 여행 플래너 입니다. 아래의 [실제 검색된 장소 목록] 데이터만을 활용하여 사용자의 조건에 맞는 맞춤형 여행 코스를 추천해 주세요.

    [주의사항 - 중요!]
    1. 아래 제공된 [실제 검색된 장소 목록]에 포함된 장소만 사용해야 합니다. 
    2. 목록에 없는 장소를 임의로 지어내거나(Hallucination) 다른 곳을 추천하면 절대 안 됩니다.
    3. 사용자의 기분, 예산, 음식 취향, 이동 거리를 고려하여 가장 적합한 곳을 골라주세요.

    [여행 조건]
    - 목적지: {request.destination}
    - 여행 기간: {request.duration}일
    - 여행 테마: {request.theme}
    - 동행인: {request.companion}
    - 기분/분위기: {getattr(request, 'mood', '상관 없음')}
    - 예산: {getattr(request, 'budget', '상관 없음')}
    - 음식 취향: {getattr(request, 'food_preference', '상관 없음')}
    - 이동 거리 선호: {getattr(request, 'distance_pref', '상관 없음')}

    [실제 검색된 장소 목록]
    {places_text}
    
    반드시 지정된 응답 형식(JSON)에 맞춰서 작성해주세요. 다른 부가적인 설명 텍스트 없이 순수하게 JSON 구조로만 응답해야 합니다.
    """
    
    # 3. Gemini 모델 호출 (구조화된 출력 Pydantic 모델 지정)
    response = client.models.generate_content(
        model='gemini-3.8-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RecommendResponse,
            temperature=0.3,  # 창의성을 낮추고, 제공된 데이터 안에서 정확하게 고르도록 유도
        )
    )
    
    # 4. 응답으로 온 텍스트를 RecommendResponse Pydantic 객체로 파싱하여 반환
    return response.parsed