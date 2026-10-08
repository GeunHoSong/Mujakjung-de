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

# 1. 환경 변수 로드 (.env 파일에서 GEMINI_API_KEY 또는 GOOGLE_API_KEY를 읽어옵니다)
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

print(f"디버깅용 API 키 확인: {api_key[:10] if api_key else '키 없음'}...")

if not api_key:
    raise ValueError("GEMINI_API_KEY가 설정되어 있지 않습니다. .env 파일을 확인해 주세요.")

# 2. Google GenAI 클라이언트 초기화
client = genai.Client(api_key=api_key)

# 3. FastAPI 애플리케이션 인스턴스 생성
app = FastAPI()

# 4. CORS 설정 (프론트엔드와 백엔드가 원활하게 통신할 수 있도록 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 모든 도메인 허용
    allow_credentials=True,
    allow_methods=["*"],          # 모든 HTTP 메서드 허용 (GET, POST 등)
    allow_headers=["*"],          # 모든 헤더 허용
)

# 클라이언트로부터 받을 요청 데이터 포맷 정의
class TextRequest(BaseModel):
    content: str

# 5. RAG용 임베딩 데이터 및 텍스트를 담을 전역 변수 설정
TOUR_DATA = []
TOUR_EMBEDDINGS = []

def load_tour_data():
    """서버가 시작될 때 로컬에 있는 'tour_embeddings.pkl' 파일을 읽어오는 함수"""
    global TOUR_DATA, TOUR_EMBEDDINGS
    pkl_path = "tour_embeddings.pkl"  # 파일명 통일
    
    if os.path.exists(pkl_path):
        try:
            # 바이너리 읽기("rb") 모드로 피클 파일 열기
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
                # 데이터 구조에 맞게 리스트 혹은 딕셔너리 형태로 파싱
                if isinstance(data, dict):
                    TOUR_DATA = data.get("texts", [])
                    TOUR_EMBEDDINGS = data.get("embeddings", [])
                elif isinstance(data, list):
                    TOUR_DATA = data
            print(f"관광지 임베딩 로드 완료! (총 {len(TOUR_DATA)}건)")
        except Exception as e:
            print(f"임베딩 데이터 로드 실패: {e}")
    else:
        print("경고! tour_embeddings.pkl 파일이 없습니다. 일반 대화 모드로 동작합니다.")

# 서버 시작 시 관광지 데이터 로드 함수 실행
load_tour_data()


def search_tour_spots(query: str, top_k=3):
    """
    사용자의 질문(query) 속 키워드를 분석하여, 
    데이터셋에서 가장 연관성이 높은 관광지 정보 상위 top_k(기본 3개)개를 찾아 텍스트로 묶어주는 함수
    """
    if not TOUR_DATA:
        return "관광지 데이터가 없습니다."

    matched = []
    keywords = query.split()  # 사용자의 문장을 띄어쓰기 기준으로 단어 단위로 쪼갭니다.
    
    for item in TOUR_DATA:
        textcontent = str(item)
        # 문장 안에 사용자의 검색 키워드가 포함된 개수만큼 점수(score)를 매깁니다.
        score = sum(1 for kw in keywords if kw in textcontent)
        
        # 점수가 0보다 크거나, 아직 상위 추천 개수(top_k)를 채우지 못했다면 후보군에 넣습니다.
        if score > 0 or len(matched) < top_k:
            matched.append((score, textcontent))

    # 점수가 높은 순(내림차순, reverse=True)으로 정렬합니다.
    matched.sort(key=lambda x: x[0], reverse=True)

    # 상위 top_k개의 텍스트 내용만 추출합니다.
    result = [content for _, content in matched[:top_k]]
    
    # 찾은 결과를 줄바꿈 문자("\n")로 합쳐서 하나의 문자열로 반환합니다.
    return "\n".join(result) if result else "관련 관광지를 찾지 못했습니다."


# 대화형 세션을 서버 메모리에 유지하기 위한 전역 변수
chat_session = None

def get_chat_session():
    """Gemini AI와 채팅 세션을 생성하거나, 이미 만들어진 세션을 재사용하는 함수"""
    global chat_session
    if chat_session is None:
        system_instruction = (
            "사용자의 감정과 고민을 따뜻하게 위로하고, "
            "제공되는 국내 관광지 참고 데이터를 활용하여 그에 딱 맞는 국내 여행지와 맞춤 일정을 추천해주는 다정한 AI 가이드입니다. "
            "이전 대화 내용을 기억하며 자연스럽고 친근한 대화형 어조로 답변해주세요."
        )
        chat_session = client.chats.create(
            model="gemini-2.0-flash",  # 안정적이고 빠른 최신 플래시 모델 지정
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,       # 창의성과 답변의 일관성을 조율하는 온도 설정
            )
        )
    return chat_session


@app.post("/generate")
def generate(request: TextRequest):
    """
    프론트엔드로부터 사용자의 메시지를 받아 
    1) RAG 데이터 검색 -> 2) 프롬프트 결합 -> 3) Gemini 요청 및 재시도 처리를 수행하는 메인 엔드포인트
    """
    
    # 1. 사용자의 질문과 관련된 국내 관광지 데이터를 파일(데이터셋)에서 검색합니다.
    relevant_data = search_tour_spots(request.content)

    # 2. 검색된 참고 데이터와 사용자 요청을 조합하여 AI에게 보낼 프롬프트를 완성합니다.
    enriched_content = (
        f"[참고 관광지 데이터]\n{relevant_data}\n\n"
        f"[사용자 요청]\n{request.content}\n\n"
        "위의 참고 데이터를 바탕으로 사용자에게 알맞은 국내 여행지 및 맞춤 일정을 친근하고 자세하게 추천해주세요."
    )

    max_retries = 5  # 구글 API 과부하(503) 발생 시 최대 5번까지 재시도합니다.
    for attempt in range(max_retries):
        try: 
            # 3. 채팅 세션을 가져와 완성된 프롬프트 메시지를 전송합니다.
            chat = get_chat_session()
            response = chat.send_message(enriched_content)
            
            # 4. 성공 시 AI의 답변 텍스트를 프론트엔드가 받을 수 있는 JSON 형태로 반환합니다.
            return {
                "analysis_result": response.text
            }
            
        except Exception as e:
            print(f"시도 {attempt + 1} 실패 - 에러: {e}")

            # 에러 발생 시 대화 세션을 초기화하여 다음 시도 때 새로운 연결을 맺도록 합니다.
            global chat_session
            chat_session = None

            # 만약 에러 내용에 '503' 또는 'UNAVAILABLE'(서버 과부하)이 포함되어 있고 재시도 횟수가 남았다면
            if ("503" in str(e) or "UNAVAILABLE" in str(e)) and attempt < max_retries - 1:
                print("일시적인 과부하 발생, 3초 뒤에 재시도합니다....")
                time.sleep(3)  # 3초 동안 숨을 고른 뒤
                continue       # 다음 시도로 넘어갑니다.
            
            # 최대 재시도 횟수를 모두 소모했는데도 실패했다면 500 에러를 발생시킵니다.
            if attempt == max_retries - 1:
                raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # uvicorn 라이브러리를 통해 로컬 서버를 구동합니다. (모듈명:앱객체 형식인 "main:app" 사용)
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)