from pydantic import BaseModel
from typing import List, Optional

# 여행 추천 요청 DTO
class RecommendRequest(BaseModel):
    destination: str     # 여행 지역 (예: 서울, 부산, 후쿠오카 등)
    duration: int        # 여행 기간 (예: 2박 3일이면 3, 1박 2일이면 2)
    theme: str           # 테마 (예: 힐링, 맛집 등)
    companion: str       # 동행인 (예: 친구, 가족 등)

# 여행 일정 상세 항목 (코스 한 개)
class ScheduleItem(BaseModel):
    time: str            # 시간대 (예: "10:00 - 12:00")
    place: str           # 장소 이름
    description: str     # 활동 내용 및 설명

# 일자별 일정 
class DaySchedule(BaseModel):
    day: int             # 몇 일차 (예: 1, 2)
    courses: List[ScheduleItem]  # 해당 일자의 코스 목록

# 여행 추천 응답 DTO (Gemini가 이 구조대로 JSON을 반환하게 됨)
class RecommendResponse(BaseModel):
    title: str           # 추천 여행 제목
    summary: str         # 전체 여행 요약
    itinerary: List[DaySchedule] # 일자별 코스 목록