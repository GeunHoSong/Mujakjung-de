
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
    # 여행 기간 설정 

    """
    사용자의 여행 조건과 실제 검색된 장소 목록(candidate_places)을 받아
    Gemini 모델을 통해 맞춤형 여행 코스 추천 결과를 생성합니다.

    - candidate_places: 카카오 검색 및 공공데이터를 통해 확보한 장소 리스트
    """
    duration = request.duration or 1

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
    당신은 실제 검색 데이터만 사용하여 여행 일정을 만드는 감성 여행 플래너입니다.

    [핵심 규칙]
    1. 실제 검색된 장소 목록에 있는 장소만 사용하세요.
    2. 장소 이름, 주소, 위도, 경도는 검색 데이터에 있는 값을 사용하세요.
    3. 검색 결과에 없는 장소나 좌표를 추측해서 만들지 마세요.
    4. category를 기준으로 관광지와 맛집을 구분하고 두 종류를 모두 포함하도록 노력하세요.
    5. 사용자의 감정, 테마, 분위기 태그, 예산을 고려하세요.
    6. 장소를 추천하는 이유는 recommendation_reason에 작성하세요.
    7. description에는 장소에서 할 수 있는 활동을 설명하세요.
    8. 검색된 장소가 부족하면 억지로 채우지 마세요.
    9. 응답은 RecommendResponse 스키마에 맞는 JSON으로 반환하세요.

    [일정 생성 규칙]
    - 여행 기간은 총 {duration}일입니다.
    - itinerary에는 1일부터 {duration}일까지의 일정을 모두 포함하세요.
    - 각 날짜의 day 값은 1부터 {duration}까지 순서대로 작성하세요.
    - 검색된 장소 중에서 각 날짜에 방문할 장소를 배정하세요.
    - 같은 장소를 특별한 이유 없이 여러 날짜에 반복하지 마세요.
    - 실제 검색된 장소가 부족하면 날짜별 코스 수가 적어도 괜찮습니다.
    - 여행 기간을 임의로 하루로 줄이지 마세요.

    [여행 조건]
    - 현재 위치 위도: {request.latitude}
    - 현재 위치 경도: {request.longitude}
    - 목적지: {request.destination or "미지정"}
    - 여행 기간: {duration}일
    - 여행 테마: {request.theme or "미지정"}
    - 동행인: {request.companion or "미지정"}
    - 감정 및 요청: {request.query}
    - 분위기 태그: {", ".join(request.tags) if request.tags else "미지정"}
    - 예산: {getattr(request, "budget", None) or "미지정"}
    - 최대 이동 거리(km): {getattr(request, "max_distance_km", None) or "미지정"}

    [실제 검색된 장소 목록]
    {places_text}

    각 일정 항목에는 가능한 범위에서 다음 정보를 포함하세요.
    장소명, category, time, description, address,
    latitude, longitude, recommendation_reason.

    위도와 경도는 검색된 장소 목록의 값을 사용하세요.
    좌표를 찾을 수 없는 경우에만 null을 사용하세요.
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
