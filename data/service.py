import os
from google import genai

# 1. 최신 클라이언트 방식으로 초기화
client = genai.Client(api_key="AIzaSyDe0hyTvo8tFkmPtw9GtjfW5C4x3C1LKFE")

def ask_gemini(user_query):
    # 파일 경로 설정 (현재 파일 옆에 있는 data.txt)
    file_path = os.path.join(os.path.dirname(__file__), "data.txt")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            context = f.read()
    except Exception as e:
        return f"파일 읽기 오류: {e}"

    # 2. 최신 모델 호출 방식 (models.generate_content)
    prompt = f"다음 정보를 참고해서 답변해줘:\n\n{context}\n\n질문: {user_query}"
    
    response = client.models.generate_content(
        model='gemini-2.0-flash', 
        contents=prompt,
    )
    return response.text

if __name__ == "__main__":
    print(ask_gemini("당신은 누구인가요?"))