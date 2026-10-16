import os
import json 
import requests
from dotenv import load_dotenv
import geocoder
from google import genai  # Gemini 클라이언트 사용을 위해 추가

# 1. .env 파일에서 환경 변수(API 키 등)를 불러옴
load_dotenv()

# 2. 구글 Gemini 클라이언트 초기화 (최신 SDK 방식)
client = genai.Client()

# 3. 크롤링/검색된 데이터 및 학습용 데이터셋을 저장할 파일 경로 설정
OUTPUT_JSON_PATH = 'data/kakao-raw-restaurants.json'
OUTPUT_JSONL_PATH = 'data/kakao-train-dataset.jsonl'

# --- 📍 실시간 위치 (좌표) 자동 설정 함수 ---
def get_current_location():
    """
    현재 실행 중인 컴퓨터/서버의 IP 주소를 기반으로 
    위도(lat)와 경도(lng)를 자동으로 추정하여 반환하는 함수
    """
    g = geocoder.ip('me')
    
    if g.latlng:
        lat, lng = g.latlng
        print(f"현재 위치 자동 감지 성공! (위도: {lat}, 경도: {lng})")
        return lat, lng
    else: 
        # 위치 감지 실패 시 기본 위치(인천 시청 부근)로 대체
        lat, lng = 37.4563, 126.7052
        print("위치 감지 실패, 기본 위치(인천)로 설정을 합니다")
        return lat, lng

# --- 🔍 카카오 API로 실시간 주변 맛집 검색 함수 ---
def search_nearby_restaurants(query="맛집", size=5, sort="accuracy"):
    """
    카카오 로컬 키워드 검색 API를 호출하여 내 주변 맛집 정보를 가져옴
    - query: 검색 키워드 (예: "파스타", "카페")
    - size: 가져올 데이터 개수 (최대 15개)
    - sort: 정렬 방식 ('accuracy' = 추천순/정확도순, 'distance' = 거리순)
    """
    # .env 파일에서 카카오 REST API 키 로드 (변수명 호환성 고려)
    kakao_api_key = os.getenv("KAKAO_REST_API_KEY") or os.getenv("KAKAO_API_KEY")
    if not kakao_api_key:
        print("오류: .env에 KAKAO_REST_API_KEY가 없습니다.")
        return []

    # 현재 위치 좌표 가져오기
    lat, lng = get_current_location()

    # 카카오 로컬 키워드 검색 API 엔드포인트
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"

    # 인증 헤더 설정 (KakaoAK 스키마 사용)
    headers = {
        "Authorization": f"KakaoAK {kakao_api_key}"
    }

    # API 요청에 전달할 파라미터 설정
    params = {
        "query": query,
        "y": lat,             # 위도
        "x": lng,             # 경도
        "radius": 3000,       # 검색 반경 (미터 단위, 3km)
        "sort": sort,         # 정렬 기준 ('accuracy' 또는 'distance')
        "size": min(max(size, 1), 15)  # 가져올 개수 (1~15 사이)
    }

    try:
        # GET 요청 전송
        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=10
        )

        # 요청 성공 시 (HTTP 200)
        if response.status_code == 200:
            data = response.json()
            restaurants = data.get("documents", [])
            print(f"검색 결과 ({sort}): {len(restaurants)}개")
            return restaurants

        # 요청 실패 시 에러 코드 및 상세 내용 출력
        print(f"카카오 API 오류: HTTP {response.status_code}")
        try:
            print("오류 상세:", response.json())
        except ValueError:
            print("오류 응답:", response.text[:500])

        return []

    except requests.RequestException as e:
        print(f"카카오 API 요청 실패: {e}")
        return []


# --- 🤖 Gemini AI를 활용한 맞춤형 추천 브리핑 생성 함수 ---
def generate_ai_recommendation(query, restaurants):
    """
    카카오 API로 수집한 맛집 리스트를 보기 좋게 정리한 뒤,
    Gemini 모델에게 전달해 친절하고 매력적인 추천 브리핑 멘트를 생성함
    """
    if not restaurants: 
        return "주변에 추천할 맛집 데이터를 찾지 못했어요"
        
    # 카카오에서 가져온 맛집 리스트를 문자열 형태로 가독성 있게 조합
    restaurant_list_str = ""
    for i, f in enumerate(restaurants, 1):
        name = f.get('place_name')
        category = f.get('category_name')
        address = f.get('address_name')
        distance = f.get('distance')
        restaurant_list_str += f"{i}. 이름: {name} | 카테고리: {category} | 주소: {address} | 거리: {distance}m\n"
        
    # Gemini에게 지시할 프롬프트 작성
    prompt = f"""
    사용자가 '{query}'를 검색했어. 아래는 카카오 로컬 API로 가져온 내 주변 실제 맛집 리스트야.
    이 리스트를 바탕으로, 사용자에게 가장 가까운 곳과 추천할 만한 곳을 친절하고 매력적인 말투로 브리핑해줘.
    
    [맛집 리스트]
    {restaurant_list_str}
    """
    
    # 구글 Gemini AI 모델 호출 (안정적인 최신 2.5-flash 모델 사용)
    response = client.models.generate_content(
        model='gemini-3.8-flash', 
        contents=prompt
    )
    return response.text


# --- 🚀 메인 실행부 ---
if __name__ == "__main__":
    search_keyword = "파스타"  
    print(f"'{search_keyword}' 키워드로 내 주변 맛집 검색 및 AI 추천 시작...\n" + "=" * 50)
    
    # 1. 카카오 API로 주변 맛집 데이터 가져오기 
    # (sort="accuracy" -> 추천순/정확도순, sort="distance" -> 거리순)
    restaurants = search_nearby_restaurants(query=search_keyword, size=3, sort="accuracy")

    if restaurants:
        # 2. 수집한 원본 데이터를 JSON 파일로 안전하게 저장
        os.makedirs('data', exist_ok=True)
        with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(restaurants, f, ensure_ascii=False, indent=4)
        print(f"원본 데이터 저장 완료: {OUTPUT_JSON_PATH}\n" + "=" * 50)
        
        # 3. Gemini AI를 통한 맞춤형 추천 멘트 생성 및 콘솔 출력
        print("[Gemini AI 맛집 추천 브리핑]\n")
        ai_recommendation = generate_ai_recommendation(search_keyword, restaurants)
        print(ai_recommendation)
    else:
        print("검색된 맛집이 없습니다.")