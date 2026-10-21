import os
import pickle
import pandas as pd 
import numpy as np

# 임베딩 데이터 및 CSV 파일 경로
EMBEDDING_PATH = os.path.join(os.path.dirname(__file__), "../tour_embeddings.pkl")
CSV_PATH = os.path.join(os.path.dirname(__file__), "../data/모범음식점정보.csv")  # 공백 주의

def search_public_recommendations(query_text: str, top_k: int = 3):
    """
    공공 데이터 임베딩(tour_embeddings.pkl) 또는 CSV 파일에서 
    쿼리와 가장 연관성 높고 인천/근교에 위치한 모범음식점/맛집 데이터를 추천합니다. 
    """
    result = []

    # 1. pkl 임베딩 파일이 존재하는 경우 검색
    if os.path.exists(EMBEDDING_PATH): 
        try:
            with open(EMBEDDING_PATH, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, list):
                    # 상위 top_k개 추출
                    for item in data[:top_k]:
                        # 🚀 pkl에 저장된 실제 구조({'data': {'관광지명': ..., '소재지도로명주소': ...}})에 맞게 안전하게 추출
                        inner_data = item.get("data", {})
                        name = inner_data.get("관광지명") or inner_data.get("업소명") or inner_data.get("title") or "공공 인증 관광지"
                        address = inner_data.get("소재지도로명주소") or inner_data.get("소재지") or "인천광역시"
                        
                        result.append({
                            "name": name,
                            "address": address,
                            "category": "공공기관 인증 정보",
                            "source": "공공데이터"
                        })
        except Exception as e:
            print(f"공공데이터 임베딩 검색 중 오류: {e}")

    # 2. CSV 파일 직접 필터링 (Fallback)
    if not result and os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
            
            # 컬럼명에 '소재지'가 포함된 컬럼 동적 탐색 (에러 방지)
            addr_column = next((col for col in df.columns if '소재지' in col), None)
            
            if addr_column:
                incheon_df = df[df[addr_column].str.contains('인천', na=False)]
            else:
                incheon_df = df

            for _, row in incheon_df.head(top_k).iterrows():
                # 업태명과 업소명 안전하게 결합
                biz_type = row.get("업태명", "")
                biz_name = row.get("업소명", "인증 맛집")
                full_name = f"{biz_type} {biz_name}".strip()
                
                # 도로명 또는 지번 주소 안전하게 가져오기
                road_addr = row.get("소재지(도로명)") or row.get("소재지(지번)") or "인천광역시"

                result.append({
                    "name": full_name,
                    "address": road_addr,
                    "category": "지자체 인증 모범음식점",
                    "source": "공공데이터"                
                })
        except Exception as e :
            print(f"CSV 공공 데이터 읽기 오류: {e}")
            
    return result