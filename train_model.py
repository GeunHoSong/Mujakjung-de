import json
import pickle
import os

from dotenv import load_dotenv
from google import genai

# ==========================
# 환경 변수 로드
# ==========================
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY가 없습니다.")

client = genai.Client(api_key=API_KEY)

# ==========================
# 파일 경로
# ==========================
DATA_FILE_PATH = "tour_data_embedding.json"
EMBEDDINGS_OUTPUT_PATH = "tour_embeddings.pkl"

# ==========================
# JSON 읽기
# ==========================
print("데이터 읽는 중...")

with open(DATA_FILE_PATH, "r", encoding="utf-8") as f:
    tour_data = json.load(f)

# ==========================
# 임베딩 함수
# ==========================
def get_embedding(text):
    text = text.replace("\n", " ")

    response = client.models.embed_content(
        model="gemini-embedding-2",  # 최신 공식 모델명으로 변경
        contents=text
    )

    return response.embeddings[0].values

# ==========================
# 임베딩 생성
# ==========================
embeddings = []

print("임베딩 생성 시작...")

for idx, item in enumerate(tour_data):

    target_text = (
        item.get("관광지소개", "")
        or item.get("관광지명", "")
    )

    if not target_text:
        continue

    try:

        vector = get_embedding(target_text)

        embeddings.append({
            "id": idx,
            "data": item,
            "embedding": vector
        })

        print(f"{idx + 1}/{len(tour_data)} 완료")

    except Exception as e:

        print(f"{idx + 1}번째 실패 : {e}")

# ==========================
# 저장
# ==========================
with open(EMBEDDINGS_OUTPUT_PATH, "wb") as f:
    pickle.dump(embeddings, f)

print()
print("=" * 40)
print("임베딩 완료")
print(f"저장 개수 : {len(embeddings)}")
print(f"저장 위치 : {EMBEDDINGS_OUTPUT_PATH}")
print("=" * 40)