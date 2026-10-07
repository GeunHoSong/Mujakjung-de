from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 환경 변수 로드
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

print(f"디버깅용 API 키 확인: {api_key[:10] if api_key else '키 없음'}...")

if not api_key:
    raise ValueError("GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")

# Google GenAI 클라이언트 초기화
client = genai.Client(api_key=api_key)

app = FastAPI()

# CORS 설정 (프론트엔드 연동용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    content: str

# 대화 세션을 서버 메모리에 유지
chat_session = None

def get_chat_session():
    global chat_session
    if chat_session is None:
        system_instruction = (
            "사용자의 감정과 고민을 따뜻하게 위로하고, "
            "그에 딱 맞는 국내 여행지와 맞춤 일정을 추천해주는 다정한 AI 가이드입니다. "
            "이전 대화 내용을 기억하며 자연스럽고 친근한 대화형 어조로 답변해주세요."
        )
        chat_session = client.chats.create(
            model="gemini-3.8-flash",
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
            )
        )
    return chat_session

@app.post("/generate")
def generate(request: TextRequest):
    max_retries = 100
    for attempt in range(max_retries):
        try:
            chat = get_chat_session()
            response = chat.send_message(request.content)
            return {
                "analysis_result": response.text
            }
        except Exception as e:
            print(f"시도 {attempt + 1} 실패 - 에러: {e}")
            
            # 에러 발생 시 세션을 초기화하여 다음 시도나 요청 때 새로 연결되도록 함
            global chat_session
            chat_session = None
            
            if "503" in str(e) and attempt < max_retries - 1:
                time.sleep(3)
                continue
            if attempt == max_retries - 1:
                raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)