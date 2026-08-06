import json
import os
import pickle
from google import genai

# 수정 후 (폴더명 빼고 파일명만 입력)
DATA_FILE_PATH = 'tour_data_embedding.json'
EMBEDDINGS_OUTPUT_PATH = 'tour_embeddings.pkl'

# 구글 Gemini 클라이언트 초기화 (환경 변수 GEMINI_API_KEY 자동 인식)
client = genai.Client(api_key="AIzaSyDe0hyTvo8tFkmPtw9GtjfW5C4x3C1LKFE")

# 1. json 파일 읽어 오기 
print("데이터 파일 읽어 오는 중...")
with open(DATA_FILE_PATH, 'r', encoding='utf-8') as f: 
    tour_data = json.load(f)

# 2. 임베딩 생성 함수 정의 (Gemini text-embedding-004 모델 사용)
def get_embedding(text):
    text = text.replace("\n", " ")
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=text
    )
    # 응답에서 임베딩 벡터 추출
    return response.embeddings[0].values

# 3. 데이터 순회하며 임베딩 추출 

embeddings = []
print("임베딩 생성 시작 .....") 

for idx, item in enumerate(tour_data):
    # JSON 파일의 키 값('관광지소개', '관광지명')에 맞춰서 변경
    target_text = item.get('관광지소개', '') or item.get('관광지명', '')

    if target_text:
        vector = get_embedding(target_text)
        embeddings.append({
            "id": idx,
            "data": item, 
            "embedding": vector
        })

# 4. 피클 파일로 저장
#os.makedirs(os.path.dirname(EMBEDDINGS_OUTPUT_PATH), exist_ok=True)
with open(EMBEDDINGS_OUTPUT_PATH, 'wb') as f:
    pickle.dump(embeddings, f)

print(f"임베딩 완료! {EMBEDDINGS_OUTPUT_PATH}에 저장되었습니다.")

# 5. 실행 완료 후 바로 결과 확인
print("\n--- 생성된 결과 바로 확인하기 ---")
print(f"총 저장된 데이터 개수: {len(embeddings)}")

if len(embeddings) > 0:
    sample = embeddings[0]
    print(f"첫 번째 데이터 ID: {sample.get('id')}")
    print(f"원본 데이터 (일부): {sample.get('data')}")
    
    vector = sample.get('embedding')
    print(f"임베딩 벡터 차원 수: {len(vector)}")
    print(f"임베딩 벡터 앞부분 5개: {vector[:5]}")