import os
import json 
import requests
from dotenv import load_dotenv
import geocoder
from google import genai  # Gemini 클라이언트 사용을 위해 추가

# .env 파일에서 환경 변수 불러 오기 
load_dotenv()

# 구글 Gemini 클라이언트 초기화
client = genai.Client()

# 저장할 데이터 파일 경로 설정 (중복되지 않게 수정)
OUTPUT_JSON_PATH = 'data/kakao-raw-restaurants.json'
OUTPUT_JSONL_PATH = 'data/kakao-train-dataset.jsonl'

# --- 📍 실시간 위치 (좌표) 자동 설정 함수 ---
def get_current_location():
    # 파이썬이 실행 되는 순간 내 컴퓨터의 ip 를 기반으로 현재 위치를 찾아 낸다 
    g = geocoder.ip('me')
    
    if g.latlng:
        lat, lng = g.latlng
        print(f"현재 위치 자동 감지 성공! (위도: {lat}, 경도: {lng})")
        return lat, lng
    else: 
        # 위치 감지 실패시 기본 위치 (예 : 인천 시청 부근)
        lat, lng = 37.4563, 126.7052
        print("위치 감지 실패, 기본 위치(인천)로 설정을 합니다")
        return lat, lng

# 2. 카카오 API로 실시간 주변 맛집 검색

def search_nearby_restaurants(query="맛집", size=5):
    kakao_api_key = os.getenv("KAKAO_API_KEY")

    if not kakao_api_key:
        print("오류: .env에 KAKAO_REST_API_KEY가 없습니다.")
        return []

    lat, lng = get_current_location()

    url = "https://dapi.kakao.com/v2/local/search/keyword.json"

    headers = {
        "Authorization": f"KakaoAK {kakao_api_key}"
    }

    params = {
        "query": query,
        "y": lat,
        "x": lng,
        "radius": 3000,
        "sort": "distance",
        "size": min(max(size, 1), 15)
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            restaurants = data.get("documents", [])
            print(f"검색 결과: {len(restaurants)}개")
            return restaurants

        print(f"카카오 API 오류: HTTP {response.status_code}")

        try:
            print("오류 상세:", response.json())
        except ValueError:
            print("오류 응답:", response.text[:500])

        return []

    except requests.RequestException as e:
        print(f"카카오 API 요청 실패: {e}")
        return []


# 3. Gemini AI를 활용해 맛집 데이터를 자연스러운 추천 멘트로 변환
def generate_ai_recommendation(query, restaurants):
    if not restaurants: 
        return "주변에 추천할 맛집 데이터를 찾지 못했어요"
        
    # 카카오에서 가져온 맛집 리스트를 보기 쉽게 텍스트로 정리
    restaurant_list_str = ""
    for i, f in enumerate(restaurants, 1):
        name = f.get('place_name')
        category = f.get('category_name')
        address = f.get('address_name')
        distance = f.get('distance')
        restaurant_list_str += f"{i}. 이름: {name} | 카테고리: {category} | 주소: {address} | 거리: {distance}m\n"
        
    # Gemini에게 전달할 프롬프트 작성
    prompt = f"""
    사용자가 '{query}'를 검색했어. 아래는 카카오 로컬 API로 가져온 내 주변 실제 맛집 리스트야.
    이 리스트를 바탕으로, 사용자에게 가장 가까운 곳과 추천할 만한 곳을 친절하고 매력적인 말투로 브리핑해줘.
    
    [맛집 리스트]
    {restaurant_list_str}
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt
    )
    return response.text

if __name__ == "__main__":
    search_keyword = "파스타"  
    print(f"'{search_keyword}' 키워드로 내 주변 맛집 검색 및 AI 추천 시작...\n" + "=" * 50)
    
    # 1. 카카오 API로 주변 맛집 데이터 가져오기
    restaurants = search_nearby_restaurants(query=search_keyword, size=3)

    if restaurants:
        # 원본 데이터 저장 
        os.makedirs('data', exist_ok=True)
        with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(restaurants, f, ensure_ascii=False, indent=4)
        print(f"원본 데이터 저장 완료: {OUTPUT_JSON_PATH}\n" + "=" * 50)
        
        # 2. Gemini AI를 통한 맞춤형 추천 멘트 생성 및 출력
        print("[Gemini AI 맛집 추천 브리핑]\n")
        ai_recommendation = generate_ai_recommendation(search_keyword, restaurants)
        print(ai_recommendation)
    else:
        print("검색된 맛집이 없습니다.")