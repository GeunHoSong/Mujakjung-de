import os 
import requests
from dotenv import load_dotenv

load_dotenv()


url="https://apis.data.go.kr/B551011/KorService2/searchKeyword2"

params= {
    "serviceKey" : os.getenv("TOUR_API_KEY"),
    "numOfRows" : 10,
    "MobileOs": "ETC",
    "MobileAPP": "Mujakjung",
    "_type": "json",
    "keyword" : "달음산자연휴양림"
}

response = requests.get(
    url,
    params=params,
    timeout=15
)

print("HTTP 상태", response.status_code)
print("응답",  response.text[:20])