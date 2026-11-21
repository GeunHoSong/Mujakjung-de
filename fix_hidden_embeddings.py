import pickle

# 기존 숨은 여행지 임베딩 불러오기 
with open("hidden_tour_embeddings.pkl", "rb") as f:
    hidden_data = pickle.load(f)

# 잘못 지정된 ID 키 수정 
for tour in hidden_data:
    if id in tour:
        tour["id"] =tour.pop(id)
print("수정된 관광지 수", len(hidden_data))
print("첫 번째 ID", hidden_data[0]["id"])
print("마지막 ID", hidden_data[-1]["id"])

# 수정된 임베딩을 새파일로 지정 
with open("hidden_embeddings_fixed.pkl", "wb") as f:
    pickle.dump(hidden_data , f)
print("복구 파일 저장 완료")