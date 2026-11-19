
import os
import pickle
import numpy as np
import requests

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# 0. 환경 설정 및 초기화
# ============================================================

# .env 파일에 저장한 환경 변수를 파이썬으로 불러온다.
# 예: GOOGLE_API_KEY=...
#     KAKAO_API_KEY=...
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 관광지 임베딩 데이터가 저장된 파일 경로
# 이 파일은 관광지 정보를 벡터로 변환해 미리 저장한 파일이다.
EMBEDDINGS_INPUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "tour_embeddings_openai.pkl"
)

# Gemini 클라이언트 초기화
# google-genai SDK는 기본적으로 GOOGLE_API_KEY 환경 변수를 읽는다.


# ============================================================
# 1. 검색어를 벡터로 변환하는 함수
# ============================================================

def get_embedding(text):
    """
    입력받은 검색어를 OpenAI 임베딩 모델로 벡터화한다.

    예:
        입력: "바다와 백사장이 예쁜 섬 여행지"
        출력: 검색어의 의미를 숫자로 표현한 벡터

    관광지 데이터도 같은 임베딩 모델과 호환되는 방식으로
    벡터화되어 있어야 유사도를 비교할 수 있다.
    """

      # 1. 검색어 정리
    text = text.replace("\n", " ").strip()

    # 2. OpenAI 임베딩 생성
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    # 3. 생성된 벡터 반환
    return response.data[0].embedding

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
def classify_tour(tour):
    tour_name = tour.get("관광지명") or ""
    famous_places = [
        "해운대해수욕장",
        "광안리해수욕장",
        "송도해수욕장",
        "다대포해수욕장"
    ]
    hidden_places = []
    if tour_name in famous_places:
        return "famous"
    if tour_name in hidden_places:
        return "hidden"
    return "unclassified"
def search_tour(query, top_k=3 , tags=None, destination=None):
    """
    사용자의 검색어와 의미적으로 비슷한 관광지를 찾는다.

    RAG란?
    관련 정보를 검색한 뒤, 그 정보를 AI 답변 생성에
    활용하는 방식이다.

   이 함수는 관광지 검색 단계만 담당한다.
    OpenAI를 이용한 최종 여행 일정 생성은
    별도의 AI 서비스에서 처리한다.

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

     # 1. 여행 조건을 벡터로 변환
    query_vector = get_embedding(query)

    # 2. 감정 태그 초기화
    tag_vector = None

    # 3. 감정 태그가 있다면 벡터로 변환
    if tags:
        tag_text = " ".join(tags)
        tag_vector = get_embedding(tag_text)

   
   

    results = []
    # 전체 관광지 데이터 개수 확인
    print("전체 관광지 수 :" , len(embedding_data))
    
           
    #인천 관광지 개수 확인 
    incheon_count = sum (
        1
        for item in embedding_data
        if "인천" in str(item["data"])
    )
    print("인천 관광지 개수", incheon_count)
    incheon_count_address = sum(
        1 
        for item in embedding_data
        if "인천" in str (
            item["data"].get("소재지도로명주소") or ""
        )
    )
    print("인천 주소 기준 관광지 수", incheon_count_address)
    # 저장된 모든 관광지를 하나씩 비교한다.
    for item in embedding_data:
        
        # 관광지 원본 데이터 가져오기 
        tour = item["data"]

        # 관광지 주소 가져 오기
        tour_address = tour.get("소재지도로명주소") or ""

        # 사용자가 선택한 목적지 확인 
        if destination:
            destination_name = destination.strip()
            # 목적지와 관광지 주소가 일치 하지 않으면 제외 
            if destination_name not in tour_address:
                continue
        # 해당 관광지의 임베딩 벡터
        tour_vector = item["embedding"]
        # 사용자 요청에서 바다 여행 추향 확인 
        sea_keywords=["바다", "해변", "해수욕장", "해안" ]
        # 관광지 이름 가져오기 
        tour_name = tour.get("관광지명") or ""
        tour_description = tour.get("관광지소개") or ""
        travel_keywords = {
            "sea":["바다", "해변", "해수욕장", "해안"],
            "mountain": ["등산", "산행", "트레킹", "산 정상"],
            "forest": ["숲", "자연휴양림", "수목원"],
            "history": ["역사","유적", "궁궐", "박물관"], 
            "city": ["도시", "쇼핑", "변화가"]
        }
        preference_score= 0.0
        # 여행 유형별 키워드 비교
        for category, keywords in travel_keywords.items():

        # 사용자가 원하는 여행 유형인지 확인
            if any(keyword in query for keyword in keywords):

                # 관광지 이름과 소개를 합쳐서 검색
                tour_text = tour_name + " " + tour_description

                    # 관광지 이름과 소개에서 키워드 확인
                if any(keyword in tour_name for keyword in keywords):
                    preference_score = 1.0
                elif any(keyword in  tour_description for keyword in keywords):
                    preference_score = 0.3
                break
        
        travel_score = cosine_similarity(
            query_vector,
            tour_vector
        )
        # 감정 태그 가 있는지 확인 
        if tag_vector is not None:
            # 감정태그 와 관광지 유사도 계산 
            embedding_score =cosine_similarity(
                tag_vector,
                tour_vector
            )
            # 여행 조건 70% 감정 30%
            similarity =(
                travel_score *0.6
                + embedding_score *0.2
                + preference_score *0.2
            )
            
        else :
            #감정 태그 없으면 여행 조건만 반영 
            similarity = (
                travel_score *0.8
                + preference_score *0.2
            )
            
        # 원본 관광지 정보와 유사도 점수를 함께 저장한다.
        results.append({
           "tour": item["data"],
           "tour_type": classify_tour(tour),
           "similarity": similarity,
           "travel_score": travel_score,
           "emotion_score": embedding_score if tag_vector is not  None else None
        })

    # 유사도가 높은 관광지부터 정렬한다.
    results.sort(
        key=lambda item: item["similarity"],
        reverse=True
    )
    unclassified_resul = [
        result 
        for result in results
        if result["tour_type"] == "unclassified"
    ]
    print("\n [미분류 관광지 상위5곳]")
    for result  in unclassified_resul[:5]:
        print(result["tour"]["관광지명"])
    # 개발 테스트용: 상위 관광지 3곳 확인 
    print("\n [상위 관광지 추천 결과]")
    for result in results[:top_k]:
        tour_name = result["tour"].get("관광지명", "정보명 없음")
        travel = result["travel_score"]
        emotion = result["emotion_score"]
        final = result["similarity"]
        

        print(f"\n관광지명:{tour_name}" )
        tour_type= result["tour_type"]
        print(f"관광지 유형: {tour_type}")
        print(f"여행 점수: {travel:.4f}")
        if emotion is not None:
            print(f"감정 점수: {emotion:.4f}")
        print(f"최종 점수{final:.4f}")
        print(f"관광지 소개: {result["tour"].get('관광지소개', '')}")

    # 상위 top_k개만 반환한다.
    return results[:top_k]


# ============================================================
# 4. 카카오 로컬 API로 맛집을 검색하는 함수
# ============================================================


def search_nearby_restaurants(
    query,
    size=5,
    latitude=None,
    longitude=None,
    nearby_only=True
):
    """
    카카오 로컬 키워드 검색 API로 장소를 검색한다.
    현재 위치가 있으면 반경 5km 내에서 가까운 순으로 검색한다.
    """

    kakao_api_key = os.getenv("KAKAO_API_KEY")

    if not kakao_api_key:
        print("오류: .env 파일에 KAKAO_API_KEY가 없습니다.")
        return []

    url = "https://dapi.kakao.com/v2/local/search/keyword.json"

    headers = {
        "Authorization": f"KakaoAK {kakao_api_key}"
    }

    params = {
        "query": query,
        "size": min(max(size, 1), 15)
    }

    # 현재 위치가 전달된 경우에만 주변 검색 조건 추가
    if nearby_only and latitude is not None and longitude is not None:
        params["x"] = longitude
        params["y"] = latitude
        params["radius"] = 5000
        params["sort"] = "distance"

    try:
        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            places = data.get("documents", [])
            print(f"카카오 검색 결과: {len(places)}개")
            return places

        print(f"카카오 API 오류: HTTP {response.status_code}")
        print("오류 상세:", response.text[:500])
        return []

    except requests.RequestException as e:
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
    query = "부산에서 바다를 보며 힐링하고 싶어"

    print(f"관광지 검색어: {query}")
    print("=" * 50)

    try:
        # 의미적으로 유사한 관광지 3개를 검색한다.
        top_results = search_tour(
            query=query,
            top_k=10,
            tags=["힐링", "휴식"],
            destination="부산"
        )

        print("[관광지 RAG 검색 결과]")

        for i, result in enumerate(top_results, 1):

            # 관광지의 원본 데이터
            tour = result["tour"]
            print("관광지 데이터 필드 ", list(tour.keys()))

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
        import traceback
        traceback.print_exc()

    print("=" * 50)

    # --------------------------------------------------------
    # B. 카카오 맛집 검색 테스트
    # --------------------------------------------------------
    if top_results:
        best_tour = top_results[0]["tour"]
        address = best_tour.get("소재지도로명주소", "")
        address_parts = address.split()

        if len(address_parts) >= 2:
            region = address_parts[1]
            restaurant_query = f"{region} 맛집"
        else:
            restaurant_query = "맛집"
    else:
        restaurant_query = "맛집"

    print(f"맛집 검색어: {restaurant_query}")
    print("=" * 50)

    restaurants = search_nearby_restaurants(
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
print("\n[관광지 분류 테스트]")

test_tour = {"관광지명": "달음산자연휴양림"}

print(classify_tour(test_tour))