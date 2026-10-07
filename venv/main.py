import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
# 1. 여기서 변경!
from google import genai 

load_dotenv(".env")

# 2. 클라이언트 초기화 방식 변경
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

class TextRequest(BaseModel):
    content: str

@app.post("/generate")
async def generate_text(request: TextRequest):
    try:
        # 1. 감성 AI 가이드 역할 정의
        system_instruction = (
            "사용자의 감정과 고민을 따뜻하게 위로하고, 그에 딱 맞는 국내 여행지와 맞춤 일정을 추천해주는 AI 가이드입니다."
        )

        # 2. Gemini 모델 호출 (config에 system_instruction 적용)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=request.content,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
            )
        )
        
        # 3. 프론트엔드와 맞출 수 있도록 키 이름을 analysis_result로 반환
        return {"analysis_result": response.text}

    except Exception as e:
        print(f"에러 발생: {e}")
        return {"error": str(e)}