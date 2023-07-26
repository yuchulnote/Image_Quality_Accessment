import os
import shutil
import pandas as pd

# 시작 디렉토리 경로
start_dir = "C:/Users/PETNOW/Desktop/shadow_glare_motionblur"  # 이 경로를 실제 시작 디렉토리 경로로 변경

# 결과 파일들을 저장할 디렉토리 경로
dest_dir = "C:/Users/PETNOW/Desktop/glar_positive_motionblur_negative"  # 이 경로를 실제 결과 파일을 저장할 디렉토리 경로로 변경

# CSV 파일 경로
csv_file = "C:/Users/PETNOW/Desktop/glar_positive_motionblur_negative/image_name.csv"  # 이 경로를 실제 CSV 파일 경로로 변경

# CSV 파일 읽어오기
df = pd.read_csv(csv_file)

# 찾고자 하는 파일명들의 리스트
# 이 경우, CSV 파일의 첫번째 열에 파일명들이 있다고 가정
target_filenames = df.iloc[:, 0].tolist()

# 모든 하위 디렉토리를 탐색
for foldername, subfolders, filenames in os.walk(start_dir):
    for filename in filenames:
        if filename in target_filenames:
            # 원하는 파일명을 가진 파일을 찾았으면, 해당 파일을 목표 디렉토리에 복사
            shutil.copy(os.path.join(foldername, filename), dest_dir)
