
# ============================================================
# 1. 필요한 라이브러리 및 함수 가져오기
# ============================================================

from fastapi import APIRouter, HTTPException

# 프론트엔드 요청 및 AI 응답 데이터 형식
from schemas.recommend_schema import RecommendRequest, RecommendResponse

# OpenAI를 이용한 여행 일정 생성
from services.ai_service import generate_travel_recommendation

# 카카오 장소 검색 및 감정 기반 관광지 검색
from search import search_nearby_restaurants, search_tour

# 거리 계산 및 방문 경로 정렬
from services.geo_service import optimize_route, calculate_distance

# 공공데이터 기반 장소 검색
from services.public_data_service import search_public_recommendations

from services.emotion_service import analyze_emotion

# ============================================================
# 2. API 라우터 설정
# ============================================================

router = APIRouter(
    prefix="/api/ai",
    tags=["AI Travel Recommendation"]
)


# ============================================================
# 3. 사용자 감정 태그를 검색어로 변환하는 함수
# ============================================================

def make_search_keyword(query: str, selected_tags: list[str]) -> str:

    # 사용자가 선택한 태그를 검색 키워드로 변환
    tags_keyword = {
        "조용한 곳": "조용한 카페 휴식",
        "출발 하기 좋은 곳": "여행 출발하기 좋은 장소",
        "매운 음식": "매운 음식",
        "사색/휴식": "산책 휴식 명소",
        "분위기 좋은": "분위기 좋은 카페 맛집",
        "빠르게 먹는": "빠르게 식사할 수 있는 식당",
    }

    # 사용자가 작성한 문장을 기본 검색어로 사용
    search_keyword = query.strip() if query and query.strip() else "맛집"

    # 선택한 태그를 검색어로 변환
    mapped_tags = [
        tags_keyword[tag]
        for tag in selected_tags
        if tag in tags_keyword
    ]

    # 변환된 태그가 있으면 검색어에 추가
    if mapped_tags:
        search_keyword = f"{search_keyword} {' '.join(mapped_tags)}"

    return search_keyword


# ============================================================
# 4. 카카오 장소 데이터를 AI용 공통 형식으로 변환
# ============================================================

def convert_kakao_place(place: dict) -> dict:

    # 카카오 API의 필드명을 무작정에서 사용하는 이름으로 변환
    return {
        "name": place.get("place_name"),
        "address": (
            place.get("road_address_name")
            or place.get("address_name")
        ),
        "category": place.get("category_name", "기타"),

        # 카카오 x = 경도, y = 위도
        "x": place.get("x"),
        "y": place.get("y"),

        "source": "Kakao API"
    }


# ============================================================
# 5. AI 여행 추천 API
# ============================================================

@router.post("/recommend", response_model=RecommendResponse)
async def recommend_travel_course(request: RecommendRequest):

    try:
        # 사용자 문장 감정 분석
        emotion_result = analyze_emotion(request.query)

        # 감정 코드 추출
        request.emotions = [
            emotion["code"]
            for emotion in emotion_result.get("emotions", [])
        ]

        # 감정 판단 이유 저장
        request.emotion_reason = emotion_result.get("reason")

        # 감정 이름 추출
        emotion_names = [
            emotion["name"]
            for emotion in emotion_result.get("emotions", [])
        ]

        print("감정 분석 결과 확인:", emotion_result)
        print("분석된 감정 이름:", emotion_names)

        # ----------------------------------------------------
        # 5-1. 사용자 입력 정보 가져오기
        # ---------------------------------------------------- 
        print("===== 수정한 recommend.py 실행 확인 =====")

        # 사용자가 선택한 감정 및 분위기 태그
        selected_tags = request.tags or []

        # 감정 문장과 태그를 검색어로 변환
        search_keyword = make_search_keyword(
            request.query,
            selected_tags
        )

        # 개발 중 확인용 로그
        print("사용자 요청:", request.query)
        print("선택한 태그:", selected_tags)
        print("최종 검색어:", search_keyword)


        # ----------------------------------------------------
        # 5-2. 감정 기반 관광지 RAG 검색
        # ----------------------------------------------------

        # 관광지 임베딩을 이용하여 관련 관광지 3개 검색
        # 기존 search_tour의 유사도 계산 방식 활용

        combined_tags = selected_tags + emotion_names
        destination_query =(
            f"{request.destination} {request.query}"
            if request.destination
            else request.query
        )

        tour_results = search_tour(
            query=destination_query,
            top_k=20,
            tags=combined_tags
        )

        # 최종적으로 AI에 전달할 장소 목록
        candidate_places = []

        # 좌표 확인에 성공한 관광지 목록
        matched_tours = []


        # ----------------------------------------------------
        # 5-3. 추천 관광지의 카카오 좌표 확인
        # ----------------------------------------------------

        for result in tour_results:

            # 관광지 원본 데이터
            tour = result["tour"]

            # 관광지 이름과 주소
            tour_name = tour.get("관광지명")
            tour_address = tour.get("소재지도로명주소")

            # 관광지 이름이 없으면 다음 결과로 이동
            if not tour_name:
                continue
            # 사용자가 목적지를 지정했다면 해당 지역 관광지만 선택
            if request.destination:
                destination = request.destination.strip()
                address = tour_address or ""

                if destination not in address:
                    print(f"목적지 불일치 관광지 제외: {tour_name}")
                    continue



            # 카카오 API에서 관광지 이름 검색
            # 이 단계는 좌표 확인을 위한 이름 검색이므로
            # nearby_only=False를 사용
            kakao_tours = search_nearby_restaurants(
                query=tour_name,
                size=5,
                nearby_only=False
            )

            # 이름이 비슷하고 좌표가 존재하는 검색 결과 찾기
            matched_tour = next(
                (
                    place for place in kakao_tours
                    if (
                        place.get("place_name")
                        and (
                            tour_name in place["place_name"]
                            or place["place_name"] in tour_name
                        )
                        and place.get("x") is not None
                        and place.get("y") is not None
                    )
                ),
                None
            )

            # 관광지 좌표 연결 성공
            if matched_tour:
                # 사용자 위치에서 관광까지 거리 거리 계산(km)
                distance_from_user = calculate_distance(
                    request.latitude,
                    request.longitude, 
                    float(matched_tour["y"]),
                    float(matched_tour["x"])
                )

                #사용자가 설정한 최대 이동 거리 확인
                max_distance = request.max_distance_km
                # 최대 이동걸이 초괴하면 추천 후보에서 제외 
                if max_distance is not None and max_distance> 0:
                    if distance_from_user > max_distance:
                        print(
                            f"거리 초과 관광지 제외: {tour_name}"
                            f"({distance_from_user:.2f}km)"
                        )
                        continue


                tour_place = {
                    "name": tour_name,
                    "address": (
                        tour_address
                        or matched_tour.get("road_address_name")
                        or matched_tour.get("address_name")
                    ),
                    "category": "관광지",
                    "x": matched_tour.get("x"),
                    "y": matched_tour.get("y"),
                    "source": "관광지 RAG + Kakao API",

                    # 관광지와 사용자 감정의 유사도
                    "similarity": result["similarity"]
                }

                # 좌표가 확인된 관광지를 목록에 저장
                matched_tours.append(tour_place)

                print(f"관광지 좌표 연결 성공: {tour_name}")

            else:
                print(f"관광지 좌표를 찾지 못함: {tour_name}")


        # ----------------------------------------------------
        # 5-4. 기준 관광지 선택
        # ----------------------------------------------------

        # 좌표를 확인한 관광지가 없다면 추천 진행 불가
        if not matched_tours:
            raise HTTPException(
                status_code=404,
                detail="좌표를 확인할 수 있는 추천 관광지가 없습니다."
            )
        print("===RAG 관광지 후보=== ")
        for tour in matched_tours:
            print(tour["name"], tour.get("similarity"))

        # 현재는 좌표가 확인된 관광지 중
        # 추천 순서가 가장 높은 관광지를 기준으로 선택
        main_tour = matched_tours[0]

        # 기준 관광지 위도와 경도
        tour_latitude = float(main_tour["y"])
        tour_longitude = float(main_tour["x"])

        # 기준 관광지를 최종 후보에 추가
        candidate_places.append(main_tour)

        print(f"기준 관광지: {main_tour['name']}")


        # ----------------------------------------------------
        # 5-5. 같은 지역의 다른 관광지 추가
        # ----------------------------------------------------

        # 나머지 관광지가 기준 관광지에서
        # 직선거리 5km 이내인 경우에만 추가
        for tour_place in matched_tours[1:]:

            distance = calculate_distance(
                tour_latitude,
                tour_longitude,
                float(tour_place["y"]),
                float(tour_place["x"])
            )

            if distance <= 5:
                candidate_places.append(tour_place)

                print(
                    f"주변 관광지 추가: {tour_place['name']}, "
                    f"거리: {distance:.2f}km"
                )


        # ----------------------------------------------------
        # 5-6. 기준 관광지 주변 5km 맛집 검색
        # ----------------------------------------------------

        # 현재 위치가 아닌 관광지 좌표를 사용
        nearby_restaurants = search_nearby_restaurants(
            query="맛집",
            size=10,
            latitude=tour_latitude,
            longitude=tour_longitude,
            nearby_only=True
        )

        print(
            f"{main_tour['name']} 주변 맛집: "
            f"{len(nearby_restaurants)}개"
        )

        # 검색된 맛집을 공통 장소 형식으로 변환
        for restaurant in nearby_restaurants:

            restaurant_place = convert_kakao_place(
                restaurant
            )

            candidate_places.append(restaurant_place)


        # ----------------------------------------------------
        # 5-7. 공공데이터 기반 추천 장소 추가
        # ----------------------------------------------------

        # 공공데이터에서도 관련 장소 검색
        public_places = search_public_recommendations(
            query_text=search_keyword,
            top_k=5
        )

        for p in public_places:

            # 공공데이터 좌표 가져오기
            latitude = p.get("latitude")
            longitude = p.get("longitude")

            # 좌표가 없다면 거리 계산 불가능
            if latitude is None or longitude is None:
                continue

            # 잘못된 좌표 형식 방지
            try:
                distance = calculate_distance(
                    tour_latitude,
                    tour_longitude,
                    float(latitude),
                    float(longitude)
                )
            except (TypeError, ValueError):
                continue

            # 기준 관광지에서 5km 이내일 때만 추가
            if distance <= 5:

                candidate_places.append({
                    "name": p.get("name"),
                    "address": p.get("address"),
                    "category": p.get(
                        "category",
                        "공공데이터 추천 장소"
                    ),
                    "x": longitude,
                    "y": latitude,
                    "source": "공공데이터"
                })


        # ----------------------------------------------------
        # 5-8. 중복 장소 제거
        # ----------------------------------------------------

        unique_places = []
        seen_places = set()

        for place in candidate_places:

            place_name = place.get("name")
            place_address = place.get("address")

            # 이름이 없는 장소는 제외
            if not place_name:
                continue

            # 이름과 주소를 묶어 중복 확인
            place_key = (place_name, place_address)

            if place_key not in seen_places:

                seen_places.add(place_key)
                unique_places.append(place)

        print(f"최종 후보 장소 수: {len(unique_places)}개")

        # AI에 전달하기 전 후보 장소 목록 확인
        for place in unique_places:
            print(
                f"후보 장소: {place['name']} | "
                f"주소: {place['address']} | "
                f"출처: {place['source']}"
            )

        # ----------------------------------------------------
        # 5-9. 방문 경로 최적화
        # ----------------------------------------------------

        # 기준 관광지에서 가까운 순서로 장소 정렬
        optimized = optimize_route(
            unique_places,
            start_lat=tour_latitude,
            start_lng=tour_longitude
        )

        #AI에 전달할 장소 목록 확인 
        print("=== AI 전달 장소 목록 ===")
        for place in optimized:
            print(
                f"장소: {place.get('name')} | "
                f"주소: {place.get('address')} | "
                f"출처: {place.get('source')}"
            )
        print("=======")


        # ----------------------------------------------------
        # 5-10. OpenAI 여행 일정 생성
        # ----------------------------------------------------

        # 실제 검색된 장소만 AI에 전달
        result = await generate_travel_recommendation(
            request,
            optimized
        )

        # Pydantic 객체를 딕셔너리로 변환
        result_data = result.model_dump()


        # ----------------------------------------------------
        # 5-11. AI 일정의 장소 검증 및 좌표 연결
        # ----------------------------------------------------

        # 실제 후보에 있는 장소 이름만 허용
        valid_names = {
            p["name"] for p in unique_places
        }

        for day in result_data.get("itinerary", []):

            valid_courses = []

            for course in day.get("courses", []):

                course_name = course.get("place")

                # 검색 후보에 없는 장소는 제외
                if course_name not in valid_names:
                    print(
                        f"후보에 없는 AI 장소 제외: {course_name}"
                    )
                    continue

                # 이름이 일치하는 실제 장소 찾기
                matched_place = next(
                    (
                        p for p in unique_places
                        if p["name"] == course_name
                    ),
                    None
                )

                if matched_place:

                    # AI가 생성한 주소 대신 실제 검색 주소 사용
                    course["address"] = matched_place["address"]

                    # 실제 검색된 좌표 사용
                    course["latitude"] = float(matched_place["y"])
                    course["longitude"] = float(matched_place["x"])

                    valid_courses.append(course)

            # 검증된 코스만 일정에 저장
            day["courses"] = valid_courses


        # ----------------------------------------------------
        # 5-12. 최종 결과 반환
        # ----------------------------------------------------

        return RecommendResponse.model_validate(result_data)


    # 의도적으로 발생시킨 HTTP 오류는 그대로 전달
    # 의도적으로 발생시킨 HTTP 오류는 그대로 전달
    except HTTPException:
        raise

    # 그 외 예외 처리
    except Exception as e:
        print(f"AI 코스 추천 중 오류 발생 상세: {e}")

        raise HTTPException(
            status_code=500,
            detail=f"AI 코스 추천 중 오류가 발생했습니다: {str(e)}"
        )
