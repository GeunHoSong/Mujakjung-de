
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

# 사용자가 보내는 데이터 형식
class EmotionRequest(BaseModel):
    text: str

# 감정 분석 요청을 받는 API
@router.post("/analyze-emotion")
def analyze_emotion(request: EmotionRequest):
    return {
        "message": "감정 분석 요청 성공",
        "text": request.text
    }
