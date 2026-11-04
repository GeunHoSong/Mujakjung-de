from openai import AsyncOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def generate_chat_response(message: str) -> str:
    """
    무작정 AI 여행 상담 답변 생성
    """
    response = await client.responses.create(
        model="gpt-4.1-mini",
        instructions=(
            "당신은 무작정 AI 여행 서비스의 친절한 상담사입니다. "
            "국내 여행, 여행 일정, 맛집 추천, "
            "서비스 이용 방법에 대해 안내하세요. "
            "확인되지 않은 예약 정보나 가격은 "
            "사실처럼 말하지 마세요. "
            "실제 상담사 연결을 요청하면 "
            "상담사 연결 버튼을 안내하세요."
        ),
        input=message,
        max_output_tokens=500,
    )

    return response.output_text