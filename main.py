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

from google import genai
from google.genai import types

# 외부 라우터 불러오기 (실시간 카카오 맛집 검색 및 AI 큐레이션 담당)
from routers import recommend

# ==========================================
# 1. 환경 변수 로드 및 API 키 검증
# ==========================================
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

print(f"디버깅용 API 키 확인: {api_key[:10] if api_key else '키 없음'}...")

if not api_key:
    raise ValueError("GEMINI_API_KEY가 설정되어 있지 않습니다. .env 파일을 확인해 주세요.")

# ==========================================
# 2. Google GenAI 클라이언트 초기화
# ==========================================
# 최신 google-genai SDK 방식 적용
client = genai.Client(api_key=api_key)

# ==========================================
# 3. FastAPI 애플리케이션 인스턴스 생성 및 설정
# ==========================================
app = FastAPI(
    title="Mujakjung AI Backend", 
    version="1.0.0",
    description="무작정 여행 서비스 AI 및 일정 관리 백엔드 서버"
)

# 라우터 등록 (routers/recommend.py 연결)
app.include_router(recommend.router)

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
    [RAG 기반 여행 코스 추천 엔드포인트]
    - 사용자 입력과 연관된 관광지 정보를 피클 데이터에서 검색 후 Gemini에 전달
    - 503(과부하/일시적 오류) 발생 시 최대 3회까지 자동 재시도(Retry) 로직 적용
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 1. 관련 관광지 데이터 검색 (RAG)
            relevant_data = search_tour_spots(request.content)

            # 2. 프롬프트 조합
            enriched_content = (
                f"[참고 관광지 데이터]\n{relevant_data}\n\n"
                f"[사용자 요청]\n{request.content}\n\n"
                "위의 참고 데이터를 바탕으로 사용자에게 알맞은 국내 여행지 및 맞춤 일정을 친근하고 자세하게 추천해주세요."
            )

            print(f"Gemini API 호출 시작... (시도 {attempt + 1})")
            
            # 3. Gemini AI 모델 호출 (안정적인 최신 플래시 모델 사용)
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=enriched_content,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                )
            )
            
            print("Gemini API 호출 완료!")

            # 4. 결과 반환
            return {
                "analysis_result": response.text
            }
            
        except Exception as e:
            print(f"시도 {attempt + 1} 실패 - 에러: {e}")

            # 503 또는 일시적 서비스 불가 에러 발생 시 0.5초 대기 후 재시도
            if ("503" in str(e) or "UNAVAILABLE" in str(e)) and attempt < max_retries - 1:
                time.sleep(0.5)
                continue       
            
            # 마지막 시도까지 실패하면 500 에러 반환
            if attempt == max_retries - 1:
                raise HTTPException(status_code=500, detail=str(e))


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