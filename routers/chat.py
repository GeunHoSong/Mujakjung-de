from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from services.chat_service import generate_chat_response

router = APIRouter(
    prefix="/api/chat",
    tags=["AI chat"]
)

# AI 상담 요청 데이터
class ChatRequest(BaseModel):
    content: str = Field(min_length=1)


# AI 상담 응답 데이터
class ChatResponse(BaseModel):
    analysisResult: str


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        message = request.content.strip()

        if not message:
            raise HTTPException(
                status_code=400,
                detail="메시지를 입력하세요."
            )

        answer = await generate_chat_response(message)

        return ChatResponse(
            analysisResult=answer
        )

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="AI 상담 응답을 생성하지 못했습니다."
        )
# Spring Boot에서 전송한 요청 데이터 확인용
@router.post("/debug")
async def debug_chat(request: Request):

    # HTTP 요청 본문 읽기
    body = await request.body()

    # FastAPI 콘솔에 출력
    print("요청 본문:", repr(body))
    print("Content-Type:", request.headers.get("content-type"))

    return {
        "body": body.decode("utf-8", errors="replace"),
        "content_type": request.headers.get("content-type")
    }