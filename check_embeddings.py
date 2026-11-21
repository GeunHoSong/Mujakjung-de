import json
import pickle

# 기존 관광지 임베딩 파일 확인 
file_path = "data/tour_data_embedding.json"

with open(file_path , "r" , encoding="utf-8") as f :
    data = json.load(f)
print("데이터 타입" , type(data).__name__)
print("전체 데이터 개수" , len(data))

#첫번째 관광지 확인 
first = data[0]

print("첫번째 데이터 필드", list(first.keys()))

embedding = first.get("embedding", [])

print("임베딩 차원", len(embedding))
print("임베딩 앞 5개 값" ,  embedding[:5])

with open("tour_embeddings_openai.pkl", "rb") as f:
    tour_data = pickle.load(f)
print("실제 RAG 관광지 개수: " , len(tour_data))
print("실제 임베딩 차원:", len(tour_data[0]["embedding"]))

# 기존 RAG 데이터 구조 확인 
first_tour = tour_data[0]

print("RAG 최상위 키 ", list(first_tour.keys()))
print("관광지 데이터 키", list(first_tour["data"].keys()))

#기존 관광지 ID 확인 
print("첫 번째 여행 관광지 ID: " , tour_data[0]["id"])
print("마지막 관광지 ID:", tour_data[-1]["id"])
print("ID 자료형 :", type(tour_data[0]["id"]).__name__)

# 기존 관광지 ID 중복 확인 
existing_ids = [tour["id"] for tour in tour_data ]

print("전체 ID 개수:" , len(existing_ids))
print("고유 ID 개수:", len(set(existing_ids)))
print("가장 큰 ID:", max(existing_ids))
