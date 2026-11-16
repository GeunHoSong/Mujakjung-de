
# ==========================================
# 1. 필요한 라이브러리 불러오기
# ==========================================

import os
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# ==========================================
# 2. 환경 변수 및 OpenAI 클라이언트 설정
# ==========================================

# .env 파일에 저장된 환경 변수를 불러옴
load_dotenv()

# 환경 변수에서 OpenAI API 키를 가져와 클라이언트 생성
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ==========================================
# 3. 감정 라벨 JSON 파일 경로 설정
# ==========================================

# 현재 파일의 위치를 기준으로 상위 폴더 경로 확인
BASE_DIR = Path(__file__).resolve().parent.parent

# data 폴더 안에 있는 emotion_labels.json 파일 경로 설정
LABEL_PATH = BASE_DIR / "data" / "emotion_labels.json"


# ==========================================
# 4. 감정 라벨 데이터 불러오기
# ==========================================

def load_emotion_labels():
    """
    emotion_labels.json 파일을 읽어서
    감정 분류 데이터를 파이썬 딕셔너리로 반환하는 함수
    """

    # JSON 파일을 UTF-8 인코딩으로 열기
    with open(LABEL_PATH, "r", encoding="utf-8") as file:

        # JSON 데이터를 파이썬 딕셔너리로 변환
        labels = json.load(file)

    # 불러온 감정 라벨 데이터 반환
    return labels


# ==========================================
# 5. 전체 세부 감정 개수 계산
# ==========================================

def count_emotions():
    """
    감정 라벨 JSON에 저장된
    전체 세부 감정 개수를 계산하는 함수
    """

    # JSON에서 감정 라벨 데이터 불러오기
    labels = load_emotion_labels()

    # 전체 감정 개수를 저장할 변수
    total = 0

    # 각 감정 대분류를 하나씩 반복
    for category in labels.values():

        # 각 대분류 안에 있는 세부 감정 개수를 더함
        total += len(category["emotions"])

    # 계산한 전체 감정 개수 반환
    return total


# ==========================================
# 6. 사용자 문장 감정 분석
# ==========================================

def analyze_emotion(text: str):
    """
    사용자가 입력한 문장을 OpenAI 모델에 전달하여
    감정을 분석하고 JSON 형태로 반환하는 함수

    Args:
        text (str): 사용자가 입력한 감정 표현 문장

    Returns:
        dict: 입력 문장, 감정 분석 결과, 판단 이유
    """

    # 감정 분석에 사용할 라벨 데이터 불러오기
    labels = load_emotion_labels()

    # OpenAI Responses API를 사용하여 감정 분석 요청
    response = client.responses.create(

        # 사용할 AI 모델 지정
        model="gpt-4.1-mini",

        # AI에게 전달할 대화 내용
        input=[
            {
                # AI의 역할과 감정 분석 규칙 설정
                "role": "system",

                "content": (
                    "당신은 무작정 AI 감정 분석가입니다.\n"

                    # 사용자 문장에서 감정 파악
                    "사용자의 문장에서 나타나는 감정을 분석하세요.\n"

                    # JSON에 정의된 감정 라벨만 사용
                    "제공된 감정 라벨 데이터만 사용하세요.\n"

                    # 여러 감정이 함께 나타나는 경우 허용
                    "복합 감정은 여러 개 반환할 수 있습니다.\n"

                    # 각 감정에 가장 적절한 세부 감정 선택
                    "각 감정은 가장 적절한 세부 감정 하나만 선택하세요.\n"

                    # JSON 형식으로만 응답하도록 지시
                    "반드시 JSON 형식으로 응답하세요.\n"
                    "다른 설명이나 Markdown은 출력하지 마세요.\n"

                    # AI가 따라야 할 JSON 응답 구조
                    "응답 형식:\n"
                    "{"
                    '"emotions": ['
                    '{"category": "SAD", "code": "SAD_05", "name": "무기력"}'
                    '],'
                    '"reason": "감정 판단 이유"'
                    "}"

                    # 실제 감정 라벨 데이터를 AI에게 전달
                    "\n감정 라벨 데이터:\n"
                    f"{json.dumps(labels, ensure_ascii=False)}"
                )
            },
            {
                # 사용자가 입력한 실제 문장 전달
                "role": "user",

                "content": text
            }
        ]
    )

    # AI가 반환한 JSON 문자열을 파이썬 딕셔너리로 변환
    result = json.loads(response.output_text)

    # 분석 결과를 딕셔너리 형태로 반환
    return {
        "text": text,                       # 사용자가 입력한 문장
        "emotions": result["emotions"],     # 분석된 감정 목록
        "reason": result["reason"]         # 감정 판단 이유
    }
