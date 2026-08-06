import os
from dotenv import load_dotenv
from google import genai

# 환경 변수 로드 (.env에 있는 AQ... 키 자동 연동)
load_dotenv()
client = genai.Client()

print("--- 사용 가능한 임베딩 모델 조회 중 ---")
try:
    for model in client.models.list():
        # 이름에 embed가 들어가는 모델 찾기
        if "embed" in model.name.lower():
            print(f"지원 모델 이름: {model.name}")
            print(f"지원 기능: {model.supported_generation_methods}")
            print("-" * 30)
except Exception as e:
    print(f"에러 발생: {e}")