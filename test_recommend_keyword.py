
print("테스트 파일 실행 시작!")

from routers.recommend import make_search_keyword

# 테스트 1: 검색어와 태그 두 개 전달
result = make_search_keyword(
    "인천",
    ["조용한 곳", "매운 음식"]
)
print("테스트 1 결과:", result)

# 테스트 2: 검색어만 전달
result2 = make_search_keyword("인천", [])
print("테스트 2 결과:", result2)

# 테스트 3: 검색어가 비어 있는 경우
result3 = make_search_keyword("", ["조용한 곳"])
print("테스트 3 결과:", result3)
