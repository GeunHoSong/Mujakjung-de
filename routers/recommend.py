# routers/recommend.py

from fastapi import APIRouter, HTTPException
from schemas.recommend_schema import RecommendRequest, RecommendResponse
from services.ai_service import generate_travel_recommendation
from search import search_nearby_restaurants 
from services.geo_service import optimize_route 

# 👇 'perfix'가 아니라 'prefix'로 수정!
router = APIRouter(
    prefix="/api/ai",  
    tags=["AI Travel Recommendation"]
)

@router.post("/recommend", response_model=RecommendResponse)
async def recommend_travel_course(request: RecommendRequest):
    try: 
        # 1. 사용자가 입력한 목적지나 테마를 키워드로 삼아 카카오 API로 실제 주변 장소/맛집 검색
        # (request.destination이나 테마를 검색어로 활용)
        query  = f"{request.destination} 맛집"
        raw_restaurants = search_nearby_restaurants(query= query, size=5 )
        # 2. ai_service에서 요구하는 규격(name, address, category)에 맞춰 데이터 가공

        candidate_places = [] 
        for r in raw_restaurants:
            candidate_places.append({
                "name": r.get("place_name"),
                "address": r.get("road_address_name") or r.get("address_name"),
                "category": r.get("category_name", "기타"),
                "x": r.get("x"),  #  2. 경도 추가
                "y": r.get("y")   #  2. 위도 추가
            })

        # 3. 가공된 실제 장소 리스트(candidate_places)를 AI 추천 함수로 전달
        optimized = optimize_route(candidate_places)  # 👈 optimized_route -> optimize_route 로 수정!
        result = await generate_travel_recommendation(request, optimized)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI 코스 추천 중 오류가 발생했습니다: {str(e)}")