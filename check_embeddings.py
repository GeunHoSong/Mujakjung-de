
import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(
    BASE_DIR, "restaurant_embeddings_test.pkl"
)

with open(file_path, "rb") as file:
    embeddings = pickle.load(file)

print("저장된 음식점 수:", len(embeddings))
print("첫 번째 음식점:", embeddings[0]["data"]["name"])
print("벡터 숫자 개수:", len(embeddings[0]["embedding"]))
print("벡터 앞부분:", embeddings[0]["embedding"][:5])
