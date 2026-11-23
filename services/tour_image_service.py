import os
import requests
from dotenv import load_dotenv

load_dotenv()

tour_api_key = os.getenv("TOUR_API_KEY")

def get_tour_image(keyword: str):
    url = "https://apis.data.go.kr/B551011/KorService2/searchKeyword2"

    params = {
        "serviceKey": tour_api_key,
        "numOfRows": 10,
        "pageNo": 1,
        "MobileOS": "ETC",
        "MobileApp": "Mujakjung",
        "_type": "json",
        "keyword": keyword
    }

    try:
        response = requests.get(url, params=params, timeout=15)

        # 429 요청 제한 체크
        if response.status_code == 429:
            print("관광공사 API 요청 제한(429)")
            return None

        # 그 외 HTTP 오류 체크
        response.raise_for_status()

        data = response.json()

        items = (
            data.get("response", {})
            .get("body", {})
            .get("items", {})
            .get("item", [])
        )

        # item이 dict 하나로 올 수도 있어서 보정
        if isinstance(items, dict):
            items = [items]

        if not items:
            return None

        first_item = items[0]

        return first_item.get("firstimage") or first_item.get("firstimage2")

    except Exception as e:
        print(f"관광지 이미지 조회 오류: {e}")
        return None