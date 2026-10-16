from fastapi import APIRouter , HTTPException
from schemas.recommend_schema import RecommendRequest , RecommendResponse
from services.ai_service import generate_travel_recommendation

# APIRouter 생성
router = APIRouter(
    perfix= "/api/ai",
    tage=["AI Travel Recommendation"]
)

@router.post("/recommend", response_model= RecommendResponse)
async def recommend_travel_course(request: RecommendRequest):
    try:
        # AI 서비스 호출하여 여행 코스 생성
        result = await generate_travel_recommendation(request)
        return result
    except Exception as  e:
        # 에러 발생 시 500 에러 반환
        raise HTTPException(status_code=500, detail=f"AI 코스 추천 중 오류가 발생했습니다: {str(e)}")