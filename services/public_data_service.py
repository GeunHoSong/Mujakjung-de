
import os
import pandas as pd

# 1. 프로젝트 기준 경로
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EMBEDDING_PATH = os.path.join(BASE_DIR, "tour_embeddings.pkl")

# 2. 실제 data 폴더에 있는 CSV 파일 경로
CSV_PATHS = [
    os.path.join(BASE_DIR, "data", "모범음식점정보.csv"),
    os.path.join(
        BASE_DIR,
        "data",
        "한국문화정보원_전국 시티투어 코스와 함께하는 맛집 정보(20261009).csv"
    )
]

# 3. CSV 경로 확인용 디버깅 코드
print("현재 서비스 파일 위치:", os.path.abspath(__file__))
print("프로젝트 기준 경로:", BASE_DIR)

data_dir = os.path.join(BASE_DIR, "data")
print("data 폴더 존재 여부:", os.path.isdir(data_dir))

for path in CSV_PATHS:
    print("CSV 경로:", path)
    print("파일 존재 여부:", os.path.exists(path))


def read_csv_file(csv_path: str) -> pd.DataFrame:
    """CSV 파일을 여러 인코딩 방식으로 읽는다."""
    encodings = ["utf-8-sig", "cp949", "euc-kr"]
    last_error = None

    for encoding in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            print(f"CSV 읽기 성공: {os.path.basename(csv_path)} ({encoding})")
            return df
        except UnicodeDecodeError as e:
            last_error = e

    raise last_error


def search_public_recommendations(
    query_text: str,
    top_k: int = 3
):
    """공공데이터에서 음식점 후보를 검색한다."""
    result = []

    # 4. CSV 파일을 하나씩 불러온다
    for csv_path in CSV_PATHS:
        if not os.path.exists(csv_path):
            print(f"CSV 파일을 찾을 수 없습니다: {csv_path}")
            continue

        try:
            df = read_csv_file(csv_path)
            df.columns = df.columns.astype(str).str.strip()

            # 5. 음식점 이름 컬럼 찾기
            name_column = next(
                (
                    col for col in df.columns
                    if col in ["업소명", "음식점명", "TRATNM"]
                ),
                None
            )

            # 6. 주소 컬럼 찾기
            # 도로명 주소를 우선 사용하고, 없으면 지번 주소를 사용한다.
            if "ROADNMADDR" in df.columns:
                road_address = df["ROADNMADDR"].fillna("").astype(str).str.strip()
            else:
                road_address = pd.Series("", index=df.index)

            if "LOTNOADDR" in df.columns:
                lot_address = df["LOTNOADDR"].fillna("").astype(str).str.strip()
            else:
                lot_address = pd.Series("", index=df.index)

            if "ROADNMADDR" in df.columns or "LOTNOADDR" in df.columns:
                df["_search_address"] = road_address.where(
                    road_address != "",
                    lot_address
                )
                addr_column = "_search_address"
            else:
                addr_column = next(
                    (
                        col for col in df.columns
                        if "소재지" in col or "주소" in col
                    ),
                    None
                )

            # 필요한 컬럼이 없으면 다음 파일로 이동
            if name_column is None or addr_column is None:
                print(
                    f"음식점명 또는 주소 컬럼을 찾을 수 없습니다: {csv_path}"
                )
                print("실제 컬럼명:", df.columns.tolist())
                continue

            # 7. 이름과 주소를 문자열로 정리
            df = df.copy()

            df[name_column] = (
                df[name_column]
                .fillna("")
                .astype(str)
                .str.strip()
            )

            df[addr_column] = (
                df[addr_column]
                .fillna("")
                .astype(str)
                .str.strip()
            )

            # 8. 이름과 주소가 모두 있는 데이터만 사용
            df = df[
                (df[name_column] != "")
                & (df[addr_column] != "")
            ].copy()

            # 9. 모범음식점 CSV는 인천 주소를 우선 검색
            is_model_restaurant = (
                os.path.basename(csv_path) == "모범음식점정보.csv"
            )

            if is_model_restaurant:
                incheon_df = df[
                    df[addr_column].str.contains("인천", na=False)
                ]

                if not incheon_df.empty:
                    df = incheon_df

            # 10. 검색어와 일치하는 음식점 이름을 우선 배치
            keywords = [
                word.strip()
                for word in query_text.split()
                if len(word.strip()) >= 2
            ]

            if keywords:
                matched = df[
                    df[name_column].apply(
                        lambda name: any(
                            keyword in name for keyword in keywords
                        )
                    )
                ]

                candidates = pd.concat(
                    [matched, df],
                    ignore_index=True
                ).drop_duplicates()
            else:
                candidates = df

            # 11. 검색 결과 추가
            for _, row in candidates.iterrows():
                result.append({
                    "name": row[name_column],
                    "address": row[addr_column],
                    "category": (
                        "지자체 인증 모범음식점"
                        if is_model_restaurant
                        else "시티투어 연계 맛집"
                    ),
                    "source": os.path.basename(csv_path)
                })

        except Exception as e:
            print(
                f"CSV 공공데이터 검색 오류 ({csv_path}): {e}"
            )

    # 12. 음식점 이름과 주소가 같은 중복 결과 제거
    unique_result = []
    seen = set()

    for item in result:
        key = (
            item["name"].replace(" ", "").lower(),
            item["address"].replace(" ", "").lower()
        )

        if key not in seen:
            seen.add(key)
            unique_result.append(item)

    # 13. 요청한 개수만 반환
    return unique_result[:top_k]
