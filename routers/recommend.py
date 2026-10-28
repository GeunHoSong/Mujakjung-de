
# routers/recommend.py

from fastapi import APIRouter, HTTPException
from schemas.recommend_schema import RecommendRequest, RecommendResponse
from services.ai_service import generate_travel_recommendation
from search import search_nearby_restaurants, search_tour
from services.geo_service import optimize_route
from services.public_data_service import search_public_recommendations

router = APIRouter(
    prefix="/api/ai",
    tags=["AI Travel Recommendation"]
)


@router.post("/recommend", response_model=RecommendResponse)
async def recommend_travel_course(request: RecommendRequest):
    try:
        # 1. 프론트엔드에서 전달받은 검색어 설정
        search_keyword = request.query if request.query else "맛집"

        # 2. 카카오 API에서 주변 장소 검색
        raw_restaurants = search_nearby_restaurants(
            query=search_keyword,
            size=4,
            latitude=request.latitude,
            longitude=request.longitude,
            nearby_only= False
        )

        # 검색 결과가 없으면 기본 검색어로 다시 검색
        if not raw_restaurants:
            raw_restaurants = search_nearby_restaurants(
                query="맛집",
                size=4,
                latitude=request.latitude,
                longitude=request.longitude,
                nearby_only= False
            )

        # 3. 카카오 검색 결과를 공통 형식으로 변환
        candidate_places = []

        for r in raw_restaurants:
            candidate_places.append({
                "name": r.get("place_name"),
                "address": (
                    r.get("road_address_name")
                    or r.get("address_name")
                ),
                "category": r.get("category_name", "기타"),
                "x": r.get("x"),  # 경도
                "y": r.get("y"),  # 위도
                "source": "Kakao API"
            })

        # 4. 공공데이터 검색 결과 추가
        public_places = search_public_recommendations(
            query_text=search_keyword,
            top_k=2
        )

        for p in public_places:
            candidate_places.append({
                "name": p.get("name"),
                "address": p.get("address"),
                "category": p.get("category", "공공 인증 맛집"),
                "x": p.get("longitude"),
                "y": p.get("latitude"),
                "source": "공공데이터"
            })
        # 관광지 RAG 검색 결과 추가 
         # 관광지 RAG 검색 결과 추가
        tour_results = search_tour(
            query=search_keyword,
            top_k=3
        )
        print("사용자가 요청한 검색어:", search_keyword)

        for result in tour_results:
            # 관광지 원본 데이터 가져오기
            tour = result["tour"]

            tour_name = tour.get("관광지명")
            tour_address = tour.get("소재지도로명주소")

            if not tour_name:
                continue

            # 카카오 API로 관광지 좌표 검색
            kakao_tours = search_nearby_restaurants(
                query=tour_name,
                size=5,
                nearby_only=False
            )

            # 이름이 일치하고 좌표가 있는 결과 찾기
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

            if matched_tour:
                candidate_places.append({
                    "name": tour_name,
                    "address": (
                        tour_address
                        or matched_tour.get("road_address_name")
                        or matched_tour.get("address_name")
                    ),
                    "category": "관광지",
                    "x": matched_tour.get("x"),
                    "y": matched_tour.get("y"),
                    "source": "관광지 RAG + Kakao API"
                })

                print(f"관광지 좌표 연결 성공: {tour_name}")
            else:
                print(f"관광지 좌표를 찾지 못함: {tour_name}")
        # 5. 현재 위치를 기준으로 경로 최적화
        optimized = optimize_route(
            candidate_places,
            start_lat=request.latitude,
            start_lng=request.longitude
        )

        # 6. Gemini AI로 여행 일정 생성
        result = await generate_travel_recommendation(
            request,
            optimized
        )

        # 7. AI 결과를 딕셔너리로 변환
        result_data = result.model_dump()

        # 8. AI가 추천한 장소에 원본 검색 좌표 연결
        for day in result_data.get("itinerary", []):
            for course in day.get("courses", []):
                course_name = course.get("place")
                course_address = course.get("address")

                matched_place = next(
                    (
                        p for p in candidate_places
                        if (
                            p.get("name") == course_name
                            or (
                                course_address
                                and p.get("address") == course_address
                            )
                        )
                        and p.get("x") is not None
                        and p.get("y") is not None
                    ),
                    None
                )

                if matched_place:
                    # 카카오 API: x는 경도, y는 위도
                    course["latitude"] = float(matched_place["y"])
                    course["longitude"] = float(matched_place["x"])
                else:
                    # 실제 좌표를 찾지 못하면 임의로 만들지 않기
                    course["latitude"] = None
                    course["longitude"] = None

        # 9. 수정된 결과를 응답 모델로 변환해 반환
        return RecommendResponse.model_validate(result_data)

    except Exception as e:
        print(f"AI 코스 추천 중 오류 발생 상세: {e}")

        raise HTTPException(
            status_code=500,
            detail=f"AI 코스 추천 중 오류가 발생했습니다: {str(e)}"
        )


