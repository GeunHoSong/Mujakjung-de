
from pydantic import BaseModel, Field
from typing import List, Optional


# 1. 여행 추천 요청 DTO
class RecommendRequest(BaseModel):
    latitude: float
    longitude: float

    # 사용자가 원하는 감정이나 분위기
    tags: List[str] = Field(default_factory=list)
    query: str

    # 기존 필드 유지
    destination: Optional[str] = None
    duration: Optional[int] = None
    theme: Optional[str] = None
    companion: Optional[str] = None

    # 추가 여행 조건
    budget: Optional[int] = None
    max_distance_km: Optional[float] = None


# 2. 여행 일정 상세 항목
class ScheduleItem(BaseModel):
    time: str
    place: str
    description: str

    # 장소 종류: 여행지, 맛집, 카페 등
    category: str = "기타"

    # 장소 주소와 좌표
    address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    # 사용자에게 이 장소를 추천하는 이유
    recommendation_reason: Optional[str] = None


# 3. 일자별 일정
class DaySchedule(BaseModel):
    day: int
    courses: List[ScheduleItem]


# 4. 여행 추천 응답 DTO
class RecommendResponse(BaseModel):
    title: str
    summary: str
    itinerary: List[DaySchedule]
