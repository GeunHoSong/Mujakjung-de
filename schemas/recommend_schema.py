from pydantic import BaseModel
from typing import List, Optional

# 여행 추천 요청 DTO (프론트엔드 페이로드 및 기존 필드 통합 지원)
class RecommendRequest(BaseModel):
    latitude: float
    longitude: float
    tags: List[str] = []
    query: str
    # 기존에 사용하던 필드들도 혹시 모를 충돌을 방지하기 위해 Optional로 추가
    destination: Optional[str] = None
    duration: Optional[int] = None
    theme: Optional[str] = None
    companion: Optional[str] = None

# 여행 일정 상세 항목 (코스 한 개)
class ScheduleItem(BaseModel):
    time: str              # 시간대 (예: "10:00 - 12:00")
    place: str             # 장소 이름
    description: str       # 활동 내용 및 설명

# 일자별 일정 
class DaySchedule(BaseModel):
    day: int               # 몇 일차 (예: 1, 2)
    courses: List[ScheduleItem]  # 해당 일자의 코스 목록

# 여행 추천 응답 DTO
class RecommendResponse(BaseModel):
    title: str             # 추천 여행 제목
    summary: str           # 전체 여행 요약
    itinerary: List[DaySchedule] # 일자별 코스 목록