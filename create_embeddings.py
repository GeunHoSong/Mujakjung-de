import json

# 기존 관광지 임베딩 파일 확인 
file_path = "data/tour_data_embedding.json"

with open(file_path , "r" , encoding="utf=8") as f :
    data = json.load(f)
print("데이터 타입" , type(data).__name__)
print("전체 데이터 개수" , len(data))

#첫번째 관광지 확인 
first = data[0]

print("첫번째 데이터 필드", list(first.keys()))

embedding = first.get("embedding", [])

print("임베딩 차원", len(embedding))
print("임베딩 앞 5개 값" ,  embedding[:5])