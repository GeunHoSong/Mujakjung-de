import json 
import os
from openai import OpenAI
import numpy as np
import pickle

# 설정 부분 
DATA_FILE_PATH = 'Mujakjung-de/tour_data_embedding.json'
EMBEDDINGS_OUTPUT_PATH = 'Mujakjung-de/tour_embeddings.pkl'

# OpenAI API 키 설정 (환경 변수 OPENAI_API_KEY 사용)
client = OpenAI()

# 1. json 파일 읽어 오기 
print("데이터 파일 읽어 오는 중...")
with open(DATA_FILE_PATH, 'r', encoding='utf-8') as f: 
    tour_data = json.load(f)

# 2. 임베딩 생성 함수 정의 (오타 수정 반영)
def get_embedding(text, model="text-embedding-3-small"):
    # 공백이나 빈 문자열 처리 
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

# 3. 데이터 순회하며 임베딩 추출 
embeddings = []
print("임베딩 생성 시작 .....") 

for idx, item in enumerate(tour_data):
    # JSON 구조에 맞춰서 임베딩할 텍스트 필드 지정
    target_text = item.get('description', '') or item.get('title', '')

    if target_text:
        vector = get_embedding(target_text)
        embeddings.append({
            "id": idx,
            "data": item, 
            "embedding": vector
        })

# 4. 피클 파일로 저장
os.makedirs(os.path.dirname(EMBEDDINGS_OUTPUT_PATH), exist_ok=True)
with open(EMBEDDINGS_OUTPUT_PATH, 'wb') as f:
    pickle.dump(embeddings, f)

print(f"임베딩 완료! {EMBEDDINGS_OUTPUT_PATH}에 저장되었습니다.")