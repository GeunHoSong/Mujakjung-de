import csv
import os
import pickle
# 기존 관광지 임베딩 파일 경로 
embedding_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "tour_embeddings_openai.pkl"
)
with open(embedding_path, "rb") as f:
    embedding_data = pickle.load(f)

# 기존 관광지 개수 확인
print("기존 관광지 수:", len(embedding_data))

# CSV 파일 경로
file_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "한국관광공사_비대면 안심관광지 목록_20220829.csv"
)

# CSV 파일 읽기
with open(file_path, "r", encoding="cp949", newline="") as f:
    reader = csv.DictReader(f)

    # CSV 열 이름 확인
    print("데이터 컬럼:", reader.fieldnames)
    hidden_tours = list(reader)
    print("전체 관광지 수:", len(hidden_tours))

    #기존 관광지 이름을 집합으로 지정 
    existing_names = {
        item["data"].get("관광지명", "").strip()
        for item in embedding_data
    }

    # 관광지 데이터 5개 확인 
    for i, tour in enumerate(hidden_tours):
        if i >= 5:
         break
    print(tour)


    # 신규 관광지 이름을 집합으로 지정 
    hidden_names= {
        tour.get("관광지명", "").strip()
        for tour in hidden_tours
    }

    # 기존 관광지와 신규 관광지의 중복 확인 
    duplicate_names = existing_names & hidden_names

    #중복된 관광지 개수 출력 
    print("중복 관광지 수: ",len(duplicate_names))

    # 중복된 관광지 이름 출력 
    for name in sorted(duplicate_names):
        print(name)

    # 기존 관광지에 없는 신규 관광지 찾기 
    new_names = hidden_names - existing_names
    print("신규 관광지 후보 수 ", len(new_names))
    # 중복을 제외한 신규 관광지 데이터 
    new_tours =[
        tour for tour in hidden_tours
        if tour["관광지명"].strip() in new_names
    ]
    # 주소가 누락돤 황강 마실길 주소 보안 
    for tour in new_tours:
        if "황강 마실길" in tour["관광지명"]:
            tour["주소"] ="경상남도 합천군 합천읍 황강체육공원로 93"
            print("주소 보완 완료", tour["관광지명"])

    # 신규 관광지 csv 파일 저장 
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        "new_tours.csv"
    )

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=hidden_tours[0].keys())
        writer.writeheader()
        writer.writerows(new_tours)
    print("신규 관광지 csv  저장 완료", len(new_tours))


    # 주소가 없느 관광지 확인 
    missing_address = [
        tour for tour in new_tours
        if not tour["주소"].strip()
    ]
    print("주소 누락 관광지 수", len(missing_address))

    # 주소가 없는 관광지 이름 출력 
    for tour in missing_address:
        print("주소 누락 관광지:", tour["관광지명"])