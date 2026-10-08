from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import time
import pickle
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. 환경 변수 로드
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

print(f"디버깅용 API 키 확인: {api_key[:10] if api_key else '키 없음'}...")

if not api_key:
    raise ValueError("GEMINI_API_KEY가 설정되어 있지 않습니다. .env 파일을 확인해 주세요.")

# 2. Google GenAI 클라이언트 초기화
client = genai.Client(api_key=api_key)

# 3. FastAPI 애플리케이션 인스턴스 생성
app = FastAPI()

# 4. CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    content: str

# 5. RAG용 임베딩 데이터 로드
TOUR_DATA = []
TOUR_EMBEDDINGS = []

def load_tour_data():
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
        print("경고! tour_embeddings.pkl 파일이 없습니다.")

load_tour_data()

def search_tour_spots(query: str, top_k=3):
    """사용자의 질문 속 키워드를 분석하여 연관된 관광지 정보를 찾아주는 함수"""
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


@app.post("/generate")
def generate(request: TextRequest):
    """
    503 과부하 발생 시 자동으로 재시도하여 안정성을 높인 고속 엔드포인트
    """
    max_retries = 3  # 최대 3번까지 재시도
    for attempt in range(max_retries):
        try:
            # 1. 관련 관광지 데이터 검색
            relevant_data = search_tour_spots(request.content)

            # 2. 프롬프트 조합
            enriched_content = (
                f"[참고 관광지 데이터]\n{relevant_data}\n\n"
                f"[사용자 요청]\n{request.content}\n\n"
                "위의 참고 데이터를 바탕으로 사용자에게 알맞은 국내 여행지 및 맞춤 일정을 친근하고 자세하게 추천해주세요."
            )

            print(f"Gemini API 호출 시작... (시도 {attempt + 1})")
            
            # 3. 단발성 고속 생성 요청
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

            # 503 또는 UNAVAILABLE 에러 발생 시 0.5초 대기 후 재시도
            if ("503" in str(e) or "UNAVAILABLE" in str(e)) and attempt < max_retries - 1:
                time.sleep(0.5)
                continue       
            
            # 마지막 시도까지 실패하면 500 에러 반환
            if attempt == max_retries - 1:
                raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)