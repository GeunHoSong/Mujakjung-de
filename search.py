
import os
import pickle
import numpy as np
import requests

from dotenv import load_dotenv
from google import genai


# ============================================================
# 0. 환경 설정 및 초기화
# ============================================================

# .env 파일에 저장한 환경 변수를 파이썬으로 불러온다.
# 예: GOOGLE_API_KEY=...
#     KAKAO_API_KEY=...
load_dotenv()

# 관광지 임베딩 데이터가 저장된 파일 경로
# 이 파일은 관광지 정보를 벡터로 변환해 미리 저장한 파일이다.
EMBEDDINGS_INPUT_PATH = "tour_embeddings.pkl"

# Gemini 클라이언트 초기화
# google-genai SDK는 기본적으로 GOOGLE_API_KEY 환경 변수를 읽는다.
client = genai.Client()


# ============================================================
# 1. 검색어를 벡터로 변환하는 함수
# ============================================================

def get_embedding(text):
    """
    입력받은 검색어를 Gemini 임베딩 모델로 벡터화한다.

    예:
        입력: "바다와 백사장이 예쁜 섬 여행지"
        출력: 검색어의 의미를 숫자로 표현한 벡터

    관광지 데이터도 같은 임베딩 모델과 호환되는 방식으로
    벡터화되어 있어야 유사도를 비교할 수 있다.
    """

    # 줄바꿈 문자를 제거해 검색어를 정리한다.
    text = text.replace("\n", " ").strip()

    # 검색어의 의미를 벡터로 변환한다.
    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=text
    )

    # 첫 번째 입력의 임베딩 벡터를 반환한다.
    return response.embeddings[0].values


# ============================================================
# 2. 코사인 유사도를 계산하는 함수
# ============================================================

def cosine_similarity(a, b):
    """
    두 벡터가 의미적으로 얼마나 비슷한지 계산한다.

    값이 1에 가까울수록 벡터의 방향이 비슷하다.
    값이 0에 가까우면 방향의 유사성이 낮다.
    값이 -1에 가까우면 방향이 반대다.

    주의:
    유사도 점수가 높다고 해서 실제 여행 만족도나
    관광지의 품질이 높다는 뜻은 아니다.
    """

    # 두 벡터의 길이가 0이면 나눗셈 문제가 발생할 수 있다.
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    # 벡터 내적을 두 벡터의 크기 곱으로 나눈다.
    return float(np.dot(a, b) / (norm_a * norm_b))


# ============================================================
# 3. 관광지 RAG 검색 함수
# ============================================================

def search_tour(query, top_k=3):
    """
    사용자의 검색어와 의미적으로 비슷한 관광지를 찾는다.

    RAG란?
    관련 정보를 검색한 뒤, 그 정보를 AI 답변 생성에
    활용하는 방식이다.

    이 함수는 검색 단계만 담당하며,
    Gemini가 최종 추천 문장을 생성하지는 않는다.

    매개변수:
        query: 사용자가 입력한 검색어
        top_k: 반환할 관광지 개수

    반환값:
        관광지 데이터와 유사도 점수를 담은 리스트
    """

    # 임베딩 데이터 파일이 있는지 확인한다.
    if not os.path.exists(EMBEDDINGS_INPUT_PATH):
        raise FileNotFoundError(
            f"관광지 임베딩 파일이 없습니다: "
            f"{EMBEDDINGS_INPUT_PATH}"
        )

    # pickle 파일에서 미리 저장한 관광지 데이터를 읽는다.
    with open(EMBEDDINGS_INPUT_PATH, "rb") as f:
        embedding_data = pickle.load(f)

    # 검색어를 벡터로 변환한다.
    query_vector = get_embedding(query)

    results = []

    # 저장된 모든 관광지를 하나씩 비교한다.
    for item in embedding_data:

        # 해당 관광지의 임베딩 벡터
        tour_vector = item["embedding"]

        # 검색어와 관광지의 의미적 유사도를 계산한다.
        similarity = cosine_similarity(
            query_vector,
            tour_vector
        )

        # 원본 관광지 정보와 유사도 점수를 함께 저장한다.
        results.append({
            "tour": item["data"],
            "similarity": similarity
        })

    # 유사도가 높은 관광지부터 정렬한다.
    results.sort(
        key=lambda item: item["similarity"],
        reverse=True
    )

    # 상위 top_k개만 반환한다.
    return results[:top_k]


# ============================================================
# 4. 카카오 로컬 API로 맛집을 검색하는 함수
# ============================================================

def search_kakao_restaurants(query, size=5):
    """
    카카오 로컬 키워드 검색 API로 실제 장소 정보를 가져온다.

    예:
        query = "제주도 맛집"

    반환 데이터에는 장소명, 주소, 좌표 등이 포함될 수 있다.

    중요:
    이 함수는 카카오 API 인증이 정상적으로 완료되어야
    실제 검색 결과를 반환할 수 있다.
    """

    # .env 파일에서 카카오 REST API 키를 가져온다.
    # API 키를 소스 코드에 직접 작성하지 않는다.
    kakao_api_key = os.getenv("KAKAO_API_KEY", "").strip()

    # 키가 없으면 API 요청을 보내지 않는다.
    if not kakao_api_key:
        print(
            "오류: .env 파일에 "
            "KAKAO_API_KEY가 설정되어 있지 않습니다."
        )
        return []

    # 카카오 로컬 키워드 검색 API 주소
    url = (
        "https://dapi.kakao.com/"
        "v2/local/search/keyword.json"
    )

    # 카카오 REST API 인증 헤더
    headers = {
        "Authorization": f"KakaoAK {kakao_api_key}"
    }

    # 검색어와 반환할 결과 개수
    params = {
        "query": query,
        "size": min(max(size, 1), 15)
    }

    try:
        # API 요청을 전송한다.
        # timeout은 응답을 무한정 기다리지 않도록 제한한다.
        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=10
        )

        # HTTP 200은 요청이 정상 처리되었음을 의미한다.
        if response.status_code == 200:

            # JSON 응답에서 장소 목록을 추출한다.
            data = response.json()
            places = data.get("documents", [])

            print(f"카카오 검색 결과: {len(places)}개")

            return places

        # 인증 오류 등의 원인을 확인하기 위해
        # HTTP 상태 코드와 응답 메시지를 출력한다.
        print(
            f"카카오 API 오류: HTTP "
            f"{response.status_code}"
        )

        print("오류 상세:", response.text[:500])

        return []

    except requests.RequestException as e:
        # 네트워크 연결 실패, 시간 초과 등을 처리한다.
        print(f"카카오 API 요청 실패: {e}")
        return []


# ============================================================
# 5. 프로그램 실행 및 테스트
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # A. 관광지 RAG 검색 테스트
    # --------------------------------------------------------

    # 사용자가 여행지에 관해 입력했다고 가정한 검색어
    query = "바다와 백사장이 예쁜 섬 여행지"

    print(f"관광지 검색어: {query}")
    print("=" * 50)

    try:
        # 의미적으로 유사한 관광지 3개를 검색한다.
        top_results = search_tour(
            query=query,
            top_k=3
        )

        print("[관광지 RAG 검색 결과]")

        for i, result in enumerate(top_results, 1):

            # 관광지의 원본 데이터
            tour = result["tour"]

            # 유사도 점수
            similarity = result["similarity"]

            # 관광지명과 주소를 출력한다.
            print(
                f"{i}. 관광지명: "
                f"{tour.get('관광지명', '정보 없음')}"
            )

            print(
                f"   주소: "
                f"{tour.get('소재지도로명주소', '정보 없음')}"
            )

            print(
                f"   유사도 점수: {similarity:.4f}\n"
            )

    except FileNotFoundError as e:
        print(f"관광지 데이터 파일 오류: {e}")

    except Exception as e:
        # Gemini 인증, 임베딩 모델, 데이터 형식 등의
        # 오류를 확인하기 위한 기본 예외 처리다.
        print(f"관광지 RAG 검색 오류: {e}")

    print("=" * 50)

    # --------------------------------------------------------
    # B. 카카오 맛집 검색 테스트
    # --------------------------------------------------------

    # 제주도 맛집을 검색한다.
    restaurant_query = "제주도 맛집"

    print(f"맛집 검색어: {restaurant_query}")
    print("=" * 50)

    restaurants = search_kakao_restaurants(
        query=restaurant_query,
        size=3
    )

    # 실제 검색된 식당이 있는 경우에만 출력한다.
    if restaurants:

        for i, place in enumerate(restaurants, 1):

            print(
                f"{i}. 식당 이름: "
                f"{place.get('place_name', '정보 없음')}"
            )

            print(
                f"   주소: "
                f"{place.get('address_name', '정보 없음')}"
            )

            print(
                f"   경도: {place.get('x', '정보 없음')}"
            )

            print(
                f"   위도: {place.get('y', '정보 없음')}\n"
            )

    else:
        print("검색된 맛집이 없습니다.")
        print("카카오 API 키와 검색어를 확인해 주세요.")
