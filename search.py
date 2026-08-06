import pickle
import numpy as np
from google import genai

# 파일 경로 설정 (저장되어 있는 임베딩 피클 파일 경로)
EMBEDDINGS_INPUT_PATH = 'tour_embeddings.pkl'

# 구글 Gemini 클라이언트 초기화 (API 키 연동)
client = genai.Client(api_key="AIzaSyDe0hyTvo8tFkmPtw9GtjfW5C4x3C1LKFE")

# 1. 검색어 임베딩 생성 함수 
def get_embedding(text):
    # 텍스트 내의 줄바꿈('\n') 제거 (필요시 공백 등으로 처리)
    text = text.replace("\n", "")
    
    # Gemini의 text-embedding-004 모델을 이용해 텍스트를 벡터로 변환
    response = client.models.embed_content(
        model="text-embedding-004",  # [수정] 모델명
        contents=text                # [수정] 전달할 텍스트 (콤마 추가 완료)
    )
    # 변환된 숫자형 벡터 리스트 반환
    return response.embeddings[0].values

# 2. 코사인 유사도 계산 함수 (두 벡터 간의 방향성 일치 정도를 측정)
def cosine_similarity(a, b):
    # numpy를 이용해 내적 값을 두 벡터 크기의 곱으로 나누어 유사도 계산
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))  # [수정] lianlg -> linalg 오타 교정

# 3. 관광지 검색 함수 (사용자 쿼리와 가장 유사한 관광지 top_k개를 추천)
def search_tour(query, top_k=3):
    # 지정된 피클 파일 읽기 모드('rb')로 열기
    with open(EMBEDDINGS_INPUT_PATH, 'rb') as f:
        embedding_data = pickle.load(f)

        # 사용자 검색어를 임베딩 벡터로 변환
        query_vector = get_embedding(query)

        # 모든 관광지 데이터와 코사인 유사도 계산
        results = []
        for item in embedding_data:
            # 검색어 벡터와 각 관광지의 임베딩 벡터 간 유사도 계산
            sim = cosine_similarity(query_vector, item["embedding"])  # [수정] 함수명 오타 교정
            
            # 결과 리스트에 관광지 원본 데이터와 유사도 점수 저장
            results.append({
                "tour": item["data"],
                "similarity": sim
            })
            
        # [중요 수정] 정렬은 모든 데이터의 유사도를 다 계산한 뒤 루프 '밖'에서 수행해야 함!
        # 유사도가 높은 순(내림차순, reverse=True)으로 정렬
        results.sort(key=lambda x: x["similarity"], reverse=True)  # [수정] 키값 지정 올바르게 수정

        # 상위 top_k개 만큼만 잘라서 반환
        return results[:top_k]

if __name__ == "__main__":
    # 테스트 검색어 입력
    query = "바다와 백사장이 이쁜 섬 여행지"
    print(f"검색어: {query} \n" + "=" * 50)
    
    # 검색 함수 호출 (상위 3개 추출)
    top_results = search_tour(query, top_k=3)
    
    # 결과 출력
    for i, res in enumerate(top_results, 1):
        tour = res["tour"]
        print(f"{i}. 관광지명 : {tour.get('관광지명')}")
        print(f"주소: {tour.get('소재지도로명주소')}")       # JSON 키 값에 맞춰 띄어쓰기 조정 가능
        print(f"소개: {tour.get('관광지소개')}")           # JSON 키 값에 맞춰 띄어쓰기 조정 가능
        print(f"유사도 점수: {res['similarity']:.4f}\n")