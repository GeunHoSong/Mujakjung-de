import os
import pickle
import numpy as np

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

EMBEDDINGS_INPUT_PATH = "tour_embeddings.pkl"


# 검색어 임베딩 생성
def get_embedding(text):

    text = text.replace("\n", " ")

    response = client.models.embed_content(
        model="text-embedding-004",
        contents=text
    )

    return response.embeddings[0].values


# 코사인 유사도
def cosine_similarity(a, b):

    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# 관광지 검색
def search_tour(query, top_k=3):

    with open(EMBEDDINGS_INPUT_PATH, "rb") as f:
        embedding_data = pickle.load(f)

    query_vector = get_embedding(query)

    results = []

    for item in embedding_data:

        sim = cosine_similarity(
            query_vector,
            item["embedding"]
        )

        results.append({
            "tour": item["data"],
            "similarity": float(sim)
        })

    results.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    return results[:top_k]


# 실행
if __name__ == "__main__":

    query = "바다와 백사장이 이쁜 섬 여행지"

    print("=" * 60)
    print("검색어 :", query)
    print("=" * 60)

    top_results = search_tour(query)

    for idx, res in enumerate(top_results, start=1):

        tour = res["tour"]

        print(f"{idx}. 관광지명 : {tour.get('관광지명')}")
        print(f"주소 : {tour.get('소재지도로명주소')}")
        print(f"소개 : {tour.get('관광지소개')}")
        print(f"유사도 : {res['similarity']:.4f}")
        print("-" * 60)