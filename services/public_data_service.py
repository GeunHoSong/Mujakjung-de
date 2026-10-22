
import os
import pickle
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EMBEDDING_PATH = os.path.join(BASE_DIR, "tour_embeddings.pkl")

CSV_PATHS = [
    os.path.join(BASE_DIR, "data", "모범 맛집음식점.csv"),
    os.path.join(
        BASE_DIR,
        "data",
        "한국 문화 정보원_전국 시티투어 코스와 함께하는 맛집 정보(20261009).csv"
    )
]


def search_public_recommendations(query_text: str, top_k: int = 3):
    """공공데이터에서 검색어와 관련된 음식점 후보를 반환한다."""
    result = []

    # 1. 등록된 CSV 파일을 하나씩 불러오기
    for csv_path in CSV_PATHS:
        if not os.path.exists(csv_path):
            print(f"CSV 파일을 찾을 수 없습니다: {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            df.columns = df.columns.str.strip()

            # 2. 음식점 이름 컬럼 찾기
            name_column = next(
                (
                    col for col in df.columns
                    if col in ["업소명", "음식점명"]
                ),
                None
            )

            # 3. 주소 컬럼 찾기
            addr_column = next(
                (
                    col for col in df.columns
                    if "소재지" in col or "주소" in col
                ),
                None
            )

            # 필요한 컬럼이 없으면 다음 파일로 이동
            if not name_column or not addr_column:
                print(f"음식점명 또는 주소 컬럼을 찾을 수 없습니다: {csv_path}")
                print("실제 컬럼명:", df.columns.tolist())
                continue

            # 4. 이름과 주소를 문자열로 정리
            df = df.copy()
            df[name_column] = (
                df[name_column].fillna("").astype(str).str.strip()
            )
            df[addr_column] = (
                df[addr_column].fillna("").astype(str).str.strip()
            )

            # 5. 이름과 주소가 모두 있는 데이터만 사용
            df = df[
                (df[name_column] != "")
                & (df[addr_column] != "")
            ].copy()

            # 6. 모범음식점 CSV는 인천 지역을 우선 검색
            if "모범 맛집음식점.csv" in os.path.basename(csv_path):
                incheon_df = df[
                    df[addr_column].str.contains("인천", na=False)
                ]

                if not incheon_df.empty:
                    df = incheon_df

            # 7. 검색어의 핵심 단어와 일치하는 상호명을 우선 배치
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

            # 8. 검색 결과 추가
            for _, row in candidates.iterrows():
                result.append({
                    "name": row[name_column],
                    "address": row[addr_column],
                    "category": (
                        "지자체 인증 모범음식점"
                        if "모범 맛집음식점.csv" in os.path.basename(csv_path)
                        else "시티투어 연계 맛집"
                    ),
                    "source": os.path.basename(csv_path)
                })

        except Exception as e:
            print(f"CSV 공공데이터 검색 오류 ({csv_path}): {e}")

    # 9. 두 파일에서 중복된 음식점 제거
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

    # 10. 요청한 개수만 반환
    return unique_result[:top_k]
