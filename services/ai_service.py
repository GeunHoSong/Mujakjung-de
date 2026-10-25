
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


async def generate_travel_recommendation(
    request: RecommendRequest,
    candidate_places: list
) -> RecommendResponse:
    """
    사용자의 여행 조건과 실제 검색된 장소 목록(candidate_places)을 받아
    Gemini 모델을 통해 맞춤형 여행 코스 추천 결과를 생성합니다.

    - candidate_places: 카카오 검색 및 공공데이터를 통해 확보한 장소 리스트
    """

    # 1. 실제 검색된 장소 목록을 Gemini가 읽기 편한 텍스트 형태로 변환
    places_text = "\n".join([
        f"- 장소명: {p.get('name')}, "
        f"주소: {p.get('address')}, "
        f"카테고리: {p.get('category', '기타')},"
        f"위도: {p.get('y')},"
        f"경도: {p.get('x')}"
        for p in candidate_places
    ])

    # 2. AI에게 전달할 프롬프트 작성 (할루시네이션 방지)
    
    prompt = f"""
    당신은 사용자의 감정과 실제 이동 조건을 고려하는 감성 여행 플래너입니다.

    [핵심 목표]
    여행지와 맛집을 모두 포함한 하나의 여행 코스를 추천하세요.

    [추천 규칙]
    1. 실제 검색된 장소 목록에 있는 장소만 사용하세요.
    2. 여행지와 맛집을 모두 포함하세요.
    3. 각 장소의 category를 확인해 여행지와 맛집을 구분하세요.
    4. 사용자의 감정과 원하는 분위기에 맞는 장소를 우선하세요.
    5. 예산이 제공되면 예산을 고려하세요.
    6. 이동 가능 거리가 제공되면 그 범위를 고려하세요.
    7. 장소의 주소나 좌표가 없다면 임의로 만들지 마세요.
        위도나 경도는 실제 검색된 목록의 좌표를 그대로 사용을 하세요.
        좌표가 있는 장소는 latiude 와 longitude 을 null로 반환 하세요
    8. 검색된 장소 중 여행지 또는 맛집이 부족하면
       존재하지 않는 장소를 만들어 채우지 마세요.
    9. 각 장소를 추천하는 이유를 description에 설명하세요.
    10. 응답은 지정된 JSON 형식으로만 작성하세요.

    [여행 조건]
    - 현재 위치: 위도 {request.latitude}, 경도 {request.longitude}
    - 목적지: {request.destination or "미지정"}
    - 여행 기간: {request.duration or 1}일
    - 여행 테마: {request.theme or "미지정"}
    - 동행인: {request.companion or "미지정"}
    - 감정 및 요청: {request.query}
    - 분위기 태그: {", ".join(request.tags) if request.tags else "미지정"}
    - 예산: {getattr(request, "budget", None) or "미지정"}
    - 최대 이동 거리(km): {
        getattr(request, "max_distance_km", None) or "미지정"
    }

    [실제 검색된 장소 목록]
    {places_text}

    일정에는 장소 이름, 장소 종류(category), 시간,
    설명(description), 주소(address), 좌표(latitude, longitude),
    추천 이유(recommendation_reason)를 가능한 범위에서 포함하세요.
    검색 데이터에 없는 주소나 좌표는 추측하지 마세요.
    """


    # 3. Gemini 모델 호출 (구조화된 출력 Pydantic 모델 지정)
    print("Gemini 모델 호출 시작")

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RecommendResponse,
            temperature=0.3,
        )
    )

    # 4. Gemini 응답 확인
    print("Gemini 응답 수신 완료")
    print("파싱 결과:", response.parsed)

    # 5. 파싱된 추천 결과를 호출한 곳으로 반환
    if response.parsed is None:
        raise ValueError("Gemini 응답을 RecommendResponse 형식으로 파싱하지 못했습니다.")

    return response.parsed
