import os
import pickle
import numpy as np
from google import genai

# 파일 경로 설정
EMBEDDINGS_INPUT_PATH = 'tour_embeddings.pkl'

# 구글 Gemini 클라이언트 초기화 (환경 변수 방식으로 인증 오류 우회)
os.environ["GEMINI_API_KEY"] = "AQ.Ab8RN6Lnq9VgssirHh9TcwiD6tGg2StDW-2CxpbrtvAcKYsOBQ"
client = genai.Client()

# 1. 검색어 임베딩 생성 함수 
def get_embedding(text):
    text = text.replace("\n", "")
    
    response = client.models.embed_content(
        model="text-embedding-004", 
        contents=text
    )
    return response.embeddings[0].values

# 2. 코사인 유사도 계산 함수
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 3. 관광지 검색 함수
def search_tour(query, top_k=3):
    with open(EMBEDDINGS_INPUT_PATH, 'rb') as f:
        embedding_data = pickle.load(f)

        query_vector = get_embedding(query)

        results = []
        for item in embedding_data:
            sim = cosine_similarity(query_vector, item["embedding"])
            
            results.append({
                "tour": item["data"],
                "similarity": sim
            })
            
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

if __name__ == "__main__":
    query = "바다와 백사장이 이쁜 섬 여행지"
    print(f"검색어: {query} \n" + "=" * 50)
    
    top_results = search_tour(query, top_k=3)
    
    for i, res in enumerate(top_results, 1):
        tour = res["tour"]
        print(f"{i}. 관광지명 : {tour.get('관광지명')}")
        print(f"주소: {tour.get('소재지도로명주소')}")
        print(f"소개: {tour.get('관광지소개')}")
        print(f"유사도 점수: {res['similarity']:.4f}\n")