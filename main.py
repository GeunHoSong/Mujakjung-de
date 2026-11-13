import os
import time
import pickle
import numpy as np
from datetime import date
from typing import List, Optional
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI , APIConnectionError, APITimeoutError, RateLimitError
from routers.emotion import router as emotion_router

# 외부 라우터 불러오기 (실시간 카카오 맛집 검색 및 AI 큐레이션 담당)
from routers import recommend
import traceback
from routers import chat

# ==========================================
# 1. OpenAI API 키 설정
# ==========================================
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY가 설정되어 있지 않습니다."
    )

# ==========================================
# 2. OpenAI 클라이언트 초기화
# ==========================================
client = OpenAI(api_key=api_key)

print("OpenAI 클라이언트 초기화 완료")

# ==========================================
# 3. FastAPI 애플리케이션 인스턴스 생성 및 설정
# ==========================================
app = FastAPI(
    title="Mujakjung AI Backend", 
    version="1.0.0",
    description="무작정 여행 서비스 AI 및 일정 관리 백엔드 서버"
)
app.include_router(emotion_router)
# AI 여행 추천 라우터
app.include_router(recommend.router)
# AI 채팅 상담 라우터
app.include_router(chat.router)

# CORS 설정 (프론트엔드 연동을 위한 모든 오리진 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 4. Pydantic 요청/응답 데이터 모델 정의
# ==========================================
class TextRequest(BaseModel):
    content: str  # RAG 검색 및 여행 추천을 위한 사용자 입력 텍스트

class PlaceItem(BaseModel):
    name: str
    latitude: float
    longitude: float
    order_index: int
    memo: Optional[str] = None

class TodayScheduleCreate(BaseModel):
    trip_id: str
    visit_date: date
    places: Optional[List[PlaceItem]] = None


# ==========================================
# 5. RAG용 관광지 임베딩 데이터 로드 및 검색 함수
# ==========================================
TOUR_DATA = []
TOUR_EMBEDDINGS = []

def load_tour_data():
    """서버 실행 시 pickle 파일에서 관광지 데이터와 임베딩 벡터를 메모리에 로드"""
    global TOUR_DATA, TOUR_EMBEDDINGS
    pkl_path = "tour_embeddings.pkl"
    
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    TOUR_DATA = data.get("texts", [])
                    TOUR_EMBEDDINGS = data.get("embeddings", [])
                elif isinstance(data, list):
                    TOUR_DATA = data
            print(f"관광지 임베딩 로드 완료! (총 {len(TOUR_DATA)}건)")
        except Exception as e:
            print(f"임베딩 데이터 로드 실패: {e}")
    else:
        print("경고! tour_embeddings.pkl 파일이 없습니다. 전체 데이터를 담은 피클 파일을 생성해 주세요.")

# 서버 구동 시 데이터 로드 실행
load_tour_data()

def search_tour_spots(query: str, top_k=3):
    """사용자의 질문 속 키워드를 분석하여 로드된 관광지 데이터 중 연관된 정보를 추출하는 함수"""
    if not TOUR_DATA:
        return "관광지 데이터가 없습니다."

    matched = []
    keywords = query.split()
    
    for item in TOUR_DATA:
        textcontent = str(item)
        score = sum(1 for kw in keywords if kw in textcontent)
        if score > 0 or len(matched) < top_k:
            matched.append((score, textcontent))

    matched.sort(key=lambda x: x[0], reverse=True)
    result = [content for _, content in matched[:top_k]]
    return "\n".join(result) if result else "관련 관광지를 찾지 못했습니다."


# ==========================================
# 6. API 엔드포인트 정의
# ==========================================

@app.get("/")
def read_root():
    """서버 헬스 체크용 루트 엔드포인트"""
    return {"message": "Mujakjung AI Server is running"}



@app.post("/generate")
def generate(request: TextRequest):
    """
    관광지 RAG 데이터를 이용한 OpenAI 여행 추천
    """
    max_retries = 3

    for attempt in range(max_retries):
        try:
            # 1. 관련 관광지 검색
            relevant_data = search_tour_spots(request.content)

            # 2. 프롬프트 생성
            enriched_content = (
                f"[참고 관광지 데이터]\n{relevant_data}\n\n"
                f"[사용자 요청]\n{request.content}\n\n"
                "참고 관광지 데이터를 바탕으로 "
                "사용자에게 맞는 국내 여행 코스를 추천해 주세요. "
                "참고 데이터에 없는 장소는 만들어내지 마세요."
            )

            print(f"OpenAI API 호출 시작 (시도 {attempt + 1})")

            # 3. OpenAI 모델 호출
            response = client.responses.create(
                model="gpt-4.1-mini",
                input=[
                    {
                        "role": "system",
                        "content": (
                            "당신은 무작정의 감성 여행 플래너입니다. "
                            "제공된 관광지 데이터를 바탕으로 "
                            "여행 일정을 추천합니다."
                        )
                    },
                    {
                        "role": "user",
                        "content": enriched_content
                    }
                ],
                temperature=0.7
            )

            # 4. 응답 확인
            result = response.output_text

            if not result:
                raise ValueError("OpenAI 응답이 비어 있습니다.")

            print("OpenAI API 호출 완료")

            # 5. 기존 응답 형식 유지
            return {
                "analysis_result": result
            }

        except (APIConnectionError, APITimeoutError) as e:
            print(f"OpenAI 연결 오류: {e}")

            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue

            raise HTTPException(
                status_code=503,
                detail="OpenAI 서버 연결에 실패했습니다."
            )

        except RateLimitError as e:
            print(f"OpenAI 사용량 또는 결제 한도 오류: {e}")
            raise HTTPException(
                status_code=429,
                detail="OpenAI API 사용량 또는 결제 한도를 확인해 주세요."
            )

        except Exception as e:
            print(f"AI 추천 오류: {e}")
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail="AI 추천 생성 중 오류가 발생했습니다."
            )



@app.post("/api/v1/schedules/today")
def save_today_schedule(schedule: TodayScheduleCreate):
    """[오늘 일정 저장 API] 사용자가 구성한 당일 여행 장소 리스트와 메모를 받아 처리"""
    try:
        print(f"[오늘 일정 저장 요청] 여행 ID: {schedule.trip_id}, 날짜: {schedule.visit_date}")
        place_count = len(schedule.places) if schedule.places else 0
        print(f"장소 개수: {place_count}개")
        
        return {
            "status": "success",
            "message": "오늘 일정이 성공적으로 처리 되었습니다",
            "data": schedule
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 7. 서버 실행부
# ==========================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=False)