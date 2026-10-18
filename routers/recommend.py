# routers/recommend.py

from fastapi import APIRouter, HTTPException
from schemas.recommend_schema import RecommendRequest, RecommendResponse
from services.ai_service import generate_travel_recommendation

# 👇 'perfix'가 아니라 'prefix'로 수정!
router = APIRouter(
    prefix="/api/ai",  
    tags=["AI Travel Recommendation"]
)

@router.post("/recommend", response_model=RecommendResponse)
async def recommend_travel_course(request: RecommendRequest):
    try:
        result = await generate_travel_recommendation(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI 코스 추천 중 오류가 발생했습니다: {str(e)}")