# routers/recommend.py

from fastapi import APIRouter, HTTPException
from schemas.recommend_schema import RecommendRequest, RecommendResponse
from services.ai_service import generate_travel_recommendation
from search import search_nearby_restaurants 
from services.geo_service import optimize_route 
from services.public_data_service import search_public_recommendations  # 🚀 공공데이터 서비스 추가

router = APIRouter(
    prefix="/api/ai",  
    tags=["AI Travel Recommendation"]
)

@router.post("/recommend", response_model=RecommendResponse)
async def recommend_travel_course(request: RecommendRequest):
    try: 
        # 1. request.destination 대신 프론트엔드가 보낸 request.query를 검색어로 활용
        search_keyword = request.query if request.query else "인천 맛집"
        raw_restaurants = search_nearby_restaurants(query=search_keyword, size=4)
        
        # 검색 결과가 없을 경우 기본 키워드로 한 번 더 방어 코드 적용
        if not raw_restaurants:
            raw_restaurants = search_nearby_restaurants(query="인천 맛집", size=4)

        # 2. ai_service에서 요구하는 규격에 맞춰 데이터 가공 (Kakao API 실시간 결과)
        candidate_places = [] 
        for r in raw_restaurants:
            candidate_places.append({
                "name": r.get("place_name"),
                "address": r.get("road_address_name") or r.get("address_name"),
                "category": r.get("category_name", "기타"),
                "x": r.get("x"),  # 경도
                "y": r.get("y"),  # 위도
                "source": "Kakao API"
            })

        # 3. 🚀 공공데이터 셋(모범음식점 및 관광지 임베딩/CSV) 검색 결과 결합
        public_places = search_public_recommendations(query_text=search_keyword, top_k=2)
        for p in public_places:
            candidate_places.append({
                "name": p.get("name"),
                "address": p.get("address", "인천광역시"),
                "category": p.get("category", "공공 인증 맛집"),
                "x": str(request.longitude) if request.longitude else "126.7052",  # 좌표 미보유 시 기본 위치(인천) 활용
                "y": str(request.latitude) if request.latitude else "37.4563",
                "source": "공공데이터"
            })

        # 4. 경로 최적화 및 AI 추천 생성
        optimized = optimize_route(candidate_places) 
        result = await generate_travel_recommendation(request, optimized)
        return result
        
    except Exception as e:
        print(f"AI 코스 추천 중 오류 발생 상세: {e}")
        raise HTTPException(status_code=500, detail=f"AI 코스 추천 중 오류가 발생했습니다: {str(e)}")