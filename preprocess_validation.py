import json

# 검증용 json 파일 읽기 
with open(
    "data/감성대화말뭉치(최종데이터)_Validation.json", 
    "r",
    encoding= "utf-8"
)as file:
    validation_data = json.load(file)
# 전체 데이터 계수 확인 
print("검증데이터 계수:" , len(validation_data))

#첫번째 데이터의 문장과 감정 코드 확인 
print("문장", validation_data[0]["talk"]["content"]["HS01"])
print("감정코드", validation_data[0]["profile"]["emotion"]["type"])

# 검증 데이터에 어떤 감정 코드가 있는지 확인
emotion_codes = set()

for item in validation_data:
    emotion_codes.add(item["profile"]["emotion"]["type"])
print("감정 코드 종류 : ", sorted(emotion_codes))
print("감정 코드 개수:", len(emotion_codes))