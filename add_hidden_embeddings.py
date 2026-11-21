import csv
import os
import pickle
from dotenv import load_dotenv
from openai import OpenAI
# 현재 python 파일이 있는 폴더 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#숨은 여행지 임베딩 저장 파일 완료 
HIDDEN_EMBEDDINGS_PATH = os.path.join(
    BASE_DIR,
    "hidden_tour_embeddings.pkl"
)
# 환경 변수 불러오기 
load_dotenv(os.path.join(BASE_DIR, ".env"))
# openai API 클라이어트 생서 
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)
# OpenAi 키 설정 확인
api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    print("OpenAI 키 설정 완료")
else:
    print("OPenAI 키 가 없습니다 ")


# 신규 숨은 여행지 csv 경로 
CSV_PATH = os.path.join(
    BASE_DIR,
    "data",
    "new_tours.csv"
)

#CSV 파일 읽기 
with open(CSV_PATH, "r" , encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    hidden_tours = list(reader)
print("숨은 여행지 데이터수", len(hidden_tours))

# 숨은 여행지 데이터 RAG 형식으로 변환 
rag_tours =[]

for tour in hidden_tours:
    rag_tour ={
        "관광지명": tour["관광지명"],
        "소재지도로명주소": tour["주소"],
        "관광지소개": f"{tour['관광지명']}은 비대면 관광지 안심관광지입니다"
    }
    rag_tours.append(rag_tour)
print("변환된 관광지 수", len(rag_tours))
print("첫 번째 여행지 수", rag_tours[0])


#신규 관광지 id 생성
new_tour_with_id =[]

for index , tour  in enumerate(rag_tours , start=854):
    new_tour_with_id.append({
        "id": index,
        "data": tour
    })
print("첫 번째 신규 관광지 ID:", new_tour_with_id[0]["id"])
print("마지막 신규 관광지 ID:", new_tour_with_id[-1]["id"])

# 첫 번쩨 숨은 여행지 임베딩 테스트 
first_tour = new_tour_with_id[0]["data"]

embedding_text = (
    f"관광지명: {first_tour['관광지명']}\n"
    f"주소: {first_tour['소재지도로명주소']}\n"
    f"소개:{first_tour['관광지소개']}\n"
)

print("임베딩 관광지: ", first_tour["관광지명"])
print("임벰딩 입력 문자:", embedding_text)

# # 아차산 1개 임베딩 생성 테스트 
# response = client.embeddings.create(
#     model="text-embedding-3-small",
#     input=embedding_text
# )

# # 생성된 임베딩 백터 추출 
# embedding_vector = response.data[0].embedding

# print("임베딩 생성 완료: ", first_tour["관광지명"])
# print("임베딩 자원:", len(embedding_vector))


# 저장은 숨은 여행지 임베딩 불러오기 
if os.path.exists(HIDDEN_EMBEDDINGS_PATH):
    with open(HIDDEN_EMBEDDINGS_PATH , "rb") as f:
        hidden_embeddings=  pickle.load(f)
else :
    hidden_embeddings =[]
print("저장된 임베딩 개수:", len(hidden_embeddings))