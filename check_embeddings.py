import pickle

# 저장했던 피클 파일 경로 (오타 수정: EMNBEDDINGS -> EMBEDDINGS)
EMBEDDINGS_OUTPUT_PATH = 'tour_embeddings.pkl'

# 1. 피클 파일 불러 오기
print("피클 파일 불러 오는 중...")
with open(EMBEDDINGS_OUTPUT_PATH, 'rb') as f:
    loaded_data = pickle.load(f)

# 2. 총 데이터 개수 확인 
print(f"총 저장된 데이터 갯수: {len(loaded_data)}")

# 3. 첫 번째 데이터 샘플 확인 
if len(loaded_data) > 0:
    sample = loaded_data[0]
    print("\n--- 첫 번째 데이터 샘플 ---")
    print(f"ID: {sample.get('id')}")
    print(f"원본 데이터 (일부): {sample.get('data')}")

    # 임베딩 벡터는 숫자가 1500개 넘게 들어있으니 길이(차원 수)만 확인
    vector = sample.get('embedding')
    print(f"임베딩 벡터 차원 수: {len(vector)}")
    print(f"임베딩 벡터 앞부분 5개: {vector[:5]}")