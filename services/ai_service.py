
import os
import time

from dotenv import load_dotenv
from openai import AsyncOpenAI

from schemas.recommend_schema import RecommendRequest, RecommendResponse


# 환경 변수 불러오기
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = AsyncOpenAI(api_key=api_key)


async def generate_travel_recommendation(
    request: RecommendRequest,
    candidate_places: list
) -> RecommendResponse:

    """
    사용자의 여행 조건과 실제 검색된 장소 목록(candidate_places)을 받아
    OpenAI 모델을 통해 맞춤형 여행 코스 추천 결과를 생성합니다.

    - candidate_places: 카카오 검색 및 공공데이터를 통해 확보한 장소 리스트
    """

    # 1. 여행 기간 설정
    duration = request.duration or 1

    # 2. 실제 검색된 장소 목록을 AI가 읽기 편한 텍스트 형태로 변환
    places_text = "\n".join([
        f"- 장소명: {p.get('name')}, "
        f"주소: {p.get('address')}, "
        f"카테고리: {p.get('category', '기타')}, "
        f"위도: {p.get('y')}, "
        f"경도: {p.get('x')}"
        for p in candidate_places
    ])

    # 3. AI에게 전달할 프롬프트 작성
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

    [감정 정보 처리 규칙]
    1. 사용자가 표현한 감정은 여행 추천을 위한 참고 정보로만 활용하세요.
    2. 감정 표현만으로 우울증, 불안장애 등 정신질환을 진단하지 마세요.
    3. 사용자의 감정을 의학적 사실처럼 단정하지 마세요.
    4. 감정이 불분명하면 임의로 감정을 추측하지 마세요.
    5. 사용자의 감정을 존중하고 판단하거나 비난하지 마세요.
    6. 감정 상태에 따라 여행을 강요하지 말고 다양한 선택지를 제안하세요.
    7. 감정 분석 결과가 없거나 불확실하면 일반적인 여행 조건을 중심으로 추천하세요.
    8. 사용자가 표현한 감정을 바탕으로 편안함, 휴식, 활력 등 여행 분위기를 고려하세요.

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

    [응답 데이터 규칙]
    - 각 일정 항목에는 가능한 범위에서 다음 정보를 포함하세요.
      장소명, category, time, description, address,
      latitude, longitude, recommendation_reason.
    - 위도와 경도는 검색된 장소 목록의 값을 사용하세요.
    - 좌표를 찾을 수 없는 경우에만 null을 사용하세요.
    """

    # 4. OpenAI 모델 호출
    print("OpenAI 모델 호출 시작")

    start_time = time.perf_counter()

    response = await client.responses.parse(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": (
                    "당신은 실제 검색 데이터만 사용하는 감성 여행 플래너입니다. "
                    "사용자가 제공한 장소 목록을 벗어나지 마세요. "
                    "감정 표현은 여행 추천을 위한 참고 정보로만 활용하고 "
                    "의학적 진단을 하지 마세요."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        text_format=RecommendResponse,
        temperature=0.3,
    )

    elapsed_time = time.perf_counter() - start_time

    # 5. OpenAI 응답 확인
    print(f"OpenAI 응답 수신 완료: {elapsed_time:.2f}초")
    print(f"검색 장소 수: {len(candidate_places)}개")
    print(f"프롬프트 길이: {len(prompt)}자")
    print("파싱 결과:", response.output_parsed)

    # 6. 파싱된 결과 반환
    if response.output_parsed is None:
        raise ValueError(
            "OpenAI 응답을 RecommendResponse 형식으로 파싱하지 못했습니다."
        )

    return response.output_parsed
