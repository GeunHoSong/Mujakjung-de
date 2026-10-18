import csv
import os
import pickle
from dotenv import load_dotenv
from google import genai

# ==========================
# 1. 환경 변수 로드
# ==========================
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY가 없습니다.")

client = genai.Client(api_key=API_KEY)

# ==========================
# 2. 파일 경로 설정
# (VS Code 탐색기에 있는 정확한 파일명 확인 필요)
# ==========================
DATA_FILE_PATH = "data/전국관광지정보표준데이터.csv"
EMBEDDINGS_OUTPUT_PATH = "tour_embeddings.pkl"

# ==========================
# 3. CSV 데이터 읽기
# ==========================
print("CSV 데이터 읽는 중...")
tour_data = []

# 한글 인코딩 에러 방지를 위해 utf-8-sig 또는 cp949 시도
try:
    with open(DATA_FILE_PATH, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 실제 CSV 컬럼명에 맞춰 키값 조회 (공백 포함 가능성 대비)
            name = row.get("관광지명", "").strip()
            desc = row.get("관광지소개", "").strip() or row.get("관광지 소개", "").strip()
            address = row.get("소재지도로명주소", "").strip() or row.get("소재지 도로명주소", "").strip()
            
            if name or desc:
                tour_data.append({
                    "관광지명": name,
                    "관광지소개": desc,
                    "소재지도로명주소": address
                })
except UnicodeDecodeError:
    with open(DATA_FILE_PATH, "r", encoding="cp949") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("관광지명", "").strip()
            desc = row.get("관광지소개", "").strip() or row.get("관광지 소개", "").strip()
            address = row.get("소재지도로명주소", "").strip() or row.get("소재지 도로명주소", "").strip()
            
            if name or desc:
                tour_data.append({
                    "관광지명": name,
                    "관광지소개": desc,
                    "소재지도로명주소": address
                })

print(f"총 {len(tour_data)}개의 관광지 데이터를 로드했습니다.")

# ==========================
# 4. 임베딩 함수 정의
# ==========================
def get_embedding(text):
    text = text.replace("\n", " ")
    response = client.models.embed_content(
        model="gemini-embedding-2",  
        contents=text
    )
    return response.embeddings[0].values

# ==========================
# 5. 임베딩 생성 및 저장
# ==========================
embeddings = []
print("임베딩 생성 시작...")

for idx, item in enumerate(tour_data):
    target_text = item.get("관광지소개", "") or item.get("관광지명", "")
    
    if not target_text:
        continue

    try:
        vector = get_embedding(target_text)

        embeddings.append({
            "id": idx,
            "data": item,
            "embedding": vector
        })

        # 너무 많을 수 있으니 10개 단위나 전체 진행상황 출력
        if (idx + 1) % 10 == 0 or (idx + 1) == len(tour_data):
            print(f"{idx + 1}/{len(tour_data)} 완료")

    except Exception as e:
        print(f"{idx + 1}번째 실패 : {e}")

# ==========================
# 6. 파일 저장
# ==========================
with open(EMBEDDINGS_OUTPUT_PATH, "wb") as f:
    pickle.dump(embeddings, f)

print()
print("=" * 40)
print("임베딩 완료!")
print(f"저장 개수 : {len(embeddings)}")
print(f"저장 위치 : {EMBEDDINGS_OUTPUT_PATH}")
print("=" * 40)