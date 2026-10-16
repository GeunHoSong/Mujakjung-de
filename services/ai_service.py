import os 
from dotenv import load_dotenv
from google import genai
from google.genai import types
from schemas.recommend_schema import RecommendRequest, RecommendResponse

# .env 파일에서 API 키 로드 
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# 제미나이 클라이언트 초기화 
client = genai.Client(api_key=api_key)

async def generate_travel_recommendation(request: RecommendRequest) -> RecommendResponse:
    # 💡 문자열 앞에 'f'를 붙여야 변수가 정상적으로 치환됩니다!
    prompt = f"""
    당신은 전문 여행 플래너 입니다. 아래 조건에 맞는 맞춤형 여행 코스를 추천해 주세요.

    [여행 조건]
    - 목적지: {request.destination}
    - 여행 기간: {request.duration}일
    - 여행 테마: {request.theme}
    - 동행인: {request.companion}
    
    반드시 지정된 응답 형식(JSON)에 맞춰서 작성해주세요. 다른 부가적인 설명 텍스트 없이 순수하게 JSON 구조로만 응답해야 합니다.
    """
    
    # Gemini 모델 호출 (구조화된 출력 Pydantic 모델 지정)
    response = client.models.generate_content(
        model='gemini-2.5-flash', # 모델명을 사용 가능한 버전으로 지정
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RecommendResponse, # 스키마 이름 오타 수정
            temperature=0.7,
        )
    )
    
    # 응답으로 온 텍스트를 RecommendResponse 객체로 파싱하여 반환
    return response.parsed