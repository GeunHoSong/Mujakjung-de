
import csv
import glob
import os

files = [
    f for f in glob.glob("data/*.csv")
    if "모범음식점정보" in f or "시티투어" in f
]

for file_path in files:
    print("\n파일:", os.path.basename(file_path))

    for encoding in ("utf-8-sig", "cp949"):
        try:
            with open(
                file_path,
                encoding=encoding,
                newline=""
            ) as file:
                columns = next(csv.reader(file))

            print("인코딩:", encoding)
            print("컬럼명:", columns)
            break

        except UnicodeDecodeError:
            continue
    else:
        print("인코딩을 확인할 수 없어.")
