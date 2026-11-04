
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.chat_service import generate_chat_response

router = APIRouter(
    prefix="/api/chat",
    tags=["AI chat"]
)


class ChatRequest(BaseModel):
    content: str = Field(min_length=1)


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
