import os
import google.generativeai as genai
import msvcrt # 윈도우에서 파일 잠금 문제를 피하기 위한 모듈

genai.configure(api_key="여기에_발급받은_API_키를_넣어줘") 

def get_context_from_file():
    file_path = os.path.join(os.path.dirname(__file__), "data.txt")
    
    # 윈도우 환경에서 파일 공유 모드로 열기
    # os.O_RDONLY: 읽기 전용, os.O_BINARY: 바이너리 모드
    fd = os.open(file_path, os.O_RDONLY | os.O_BINARY)
    try:
        # 파일 전체를 읽기
        size = os.fstat(fd).st_size
        content = os.read(fd, size)
        return content.decode('utf-8')
    finally:
        os.close(fd) # 확실하게 닫기

def ask_gemini(user_query):
    try:
        context = get_context_from_file()
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"다음 정보를 참고해서 답변해줘:\n\n{context}\n\n질문: {user_query}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"오류 발생: {e}"

if __name__ == "__main__":
    print(ask_gemini("당신은 누구인가요?"))