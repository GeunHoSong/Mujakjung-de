import csv
import os
import pickle

from dotenv import load_dotenv
from google import genai

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(BASE_DIR, ".env"))

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY가 없습니다.")

client = genai.Client(api_key=API_KEY)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATHS = [
    os.path.join(BASE_DIR, "data", "\ubaa8\ubc94\uc74c\uc2dd\uc810\uc815\ubcf4.csv"),
    os.path.join(
        BASE_DIR,
        "data",
        "\ud55c\uad6d\ubb38\ud654\uc815\ubcf4\uc6d0_\uc804\uad6d \uc2dc\ud2f0\ud22c\uc5b4 \ucf54\uc2a4\uc640 \ud568\uaed8\ud558\ub294 \ub9db\uc9d1 \uc815\ubcf4(20261009).csv"
    ),
]
def load_restaurants():
    """두 CSV에서 음식점 정보를 수집한다."""
    restaurants = []

    for csv_path in CSV_PATHS:
        if not os.path.exists(csv_path):
            print(f"파일을 찾을 수 없습니다: {csv_path}")
            continue

        try:
            rows = read_csv_rows(csv_path)
        except Exception as error:
            print(f"CSV 읽기 실패: {error}")
            continue

        print(f"읽은 파일: {os.path.basename(csv_path)}")

        is_model_restaurant = "모범음식점정보" in os.path.basename(csv_path)

        for row in rows:
            if is_model_restaurant:
                name = find_value(row, ["업소명"])
                address = find_value(
                    row, ["도로명주소", "소재지주소"]
                )
                description = " ".join(filter(None, [
                    find_value(row, ["주된음식종류"]),
                    find_value(row, ["음식의유형"]),
                ]))
                status = find_value(row, ["영업상태명"])

                # 영업 중인 음식점만 수집
                if status and status != "영업":
                    continue

            else:
                name = find_value(row, ["TRATNM"])
                address = find_value(
                    row, ["ROADNMADDR", "LOTNOADDR"]
                )
                description = " ".join(filter(None, [
                    find_value(row, ["TRATCN"]),
                    find_value(row, ["TRATGDNCEXPLN"]),
                ]))

            if not name:
                continue

            restaurants.append({
                "name": name,
                "address": address,
                "description": description,
                "source": os.path.basename(csv_path),
            })

    # 이름과 주소가 같은 중복 데이터 제거
    unique = []
    seen = set()

    for item in restaurants:
        key = (
            item["name"].replace(" ", "").lower(),
            item["address"].replace(" ", "").lower()
        )

        if key not in seen:
            seen.add(key)
            unique.append(item)

    return unique

def read_csv_rows(file_path):
    """CSV 파일을 읽고 각 행을 반환한다."""
    for encoding in ("utf-8-sig", "cp949"):
        try:
            with open(
                file_path,
                "r",
                encoding=encoding,
                newline=""
            ) as file:
                return list(csv.DictReader(file))
        except UnicodeDecodeError:
            continue

    raise ValueError(f"CSV 인코딩을 확인해 주세요: {file_path}")


def find_value(row, possible_names):
    """후보 컬럼명 중 실제 존재하는 값을 찾는다."""
    normalized_row = {
        str(key).strip(): (value or "").strip()
        for key, value in row.items()
        if key is not None
    }

    for name in possible_names:
        value = normalized_row.get(name, "")
        if value:
            return value

    return ""
def get_embedding(text):
    """음식점 정보를 Gemini 임베딩 벡터로 변환한다."""
    text = text.replace("\n", " ")

    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=text
    )

    if not response.embeddings:
        raise ValueError("임베딩 결과가 없습니다.")

    return response.embeddings[0].values


def main():
    restaurants = load_restaurants()
    print(f"수집한 음식점 수: {len(restaurants)}")

    # 전체 데이터가 아니라 우선 5개만 테스트
    test_restaurants = restaurants[:5]
    embeddings = []
    total = len(test_restaurants)

    for index, item in enumerate(test_restaurants, start=1):
        print(f"진행 상황: {index}/{total}")
        target_text = (
            f"음식점 이름: {item['name']}\n"
            f"주소: {item['address']}\n"
            f"음식점 정보: {item['description']}"
        )

        try:
            vector = get_embedding(target_text)

            embeddings.append({
                "id": index,
                "data": item,
                "embedding": vector,
            })

            print(
                f"{index}/5 성공: {item['name']} "
                f"(벡터 차원: {len(vector)})"
            )

        except Exception as error:
            print(f"임베딩 실패: {item['name']} - {error}")

    if not embeddings:
        print("임베딩 결과가 없어 파일을 저장하지 않았습니다.")
        return

    output_path = os.path.join(
        BASE_DIR, "restaurant_embeddings_test.pkl"
    )

    with open(output_path, "wb") as file:
        pickle.dump(embeddings, file)

    print(f"테스트 완료: {len(embeddings)}개")
    print(f"저장 위치: {output_path}")


if __name__ == "__main__":
    main()

