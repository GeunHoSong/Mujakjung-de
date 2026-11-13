import os
from dotenv import load_dotenv
from openai import OpenAI
import json
from pathlib import Path

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# 현재 파일 위치를 기준으로 JSON 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
LABEL_PATH = BASE_DIR / "data" / "emotion_labels.json"

# 감정 라벨 데이터 읽기
def load_emotion_labels():
    with open(LABEL_PATH, "r", encoding="utf-8") as file:
        labels = json.load(file)

    return labels

# 전체 세부 감정 개수 계산
def count_emotions():
    labels = load_emotion_labels()

    total = 0 

    for category in labels.values():
        total += len(category["emotions"])
    return total


def analyze_emotion(text: str):
    labels = load_emotion_labels()

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": (
                    "당신은 무작정 AI 감정 분석가입니다.\n"
                    "사용자의 문장에서 나타나는 감정을 분석하세요.\n"
                    "제공된 감정 라벨 데이터만 사용하세요.\n"
                    "복합 감정은 여러 개 반환할 수 있습니다.\n"
                    "각 감정은 가장 적절한 세부 감정 하나만 선택하세요.\n"
                    "반드시 JSON 형식으로 응답하세요.\n"
                    "다른 설명이나 Markdown은 출력하지 마세요.\n"
                    "응답 형식:\n"
                    "{"
                    '"emotions": ['
                    '{"category": "SAD", "code": "SAD_05", "name": "무기력"}'
                    '],'
                    '"reason": "감정 판단 이유"'
                    "}"
                    "\n감정 라벨 데이터:\n"
                    f"{json.dumps(labels, ensure_ascii=False)}"
                )
            },
            {
                "role": "user",
                "content": text
            }
        ]
    )

    result = json.loads(response.output_text)

    return {
        "text": text,
        "emotions": result["emotions"],
        "reason": result["reason"]
    }
