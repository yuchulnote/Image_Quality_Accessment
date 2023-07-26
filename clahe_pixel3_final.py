import cv2
import numpy as np
import os

# 사진 순서를 기록하기 위한 전역 변수
image_order = 1

# 이미지를 불러올 폴더 경로
folder_path = "C:/Users/PETNOW/Desktop/ivan/Robert_ivan/result_his/dataset/output-negative"  # 경로를 실제 이미지 폴더 경로로 변경

# 결과 이미지를 저장할 폴더 경로
output_folder_path = "C:/Users/PETNOW/Desktop/ivan/Robert_ivan/result_his/dataset/output_test/result_negative"  # 경로를 실제 결과 이미지를 저장할 폴더 경로로 변경

# 특정 명암값
threshold_value = 200  # 이 값은 특정 명암값으로 변경할 수 있음

# 폴더 내의 모든 파일에 대해 반복
for filename in os.listdir(folder_path):
    if filename.endswith(".png") or filename.endswith(".jpg"):  # PNG 또는 JPG 파일인 경우
        # 원본 이미지 파일 불러오기
        original = cv2.imread(os.path.join(folder_path, filename))

        # 이미지가 제대로 로드되었는지 확인
        if original is None:
            print(f"Failed to load image file {filename}. Please check the image path.")
            continue

        # 이미지를 grayscale로 변환
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

        # CLAHE (Contrast Limited Adaptive Histogram Equalization) 적용
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)

        # 특정 명암값 이상인 픽셀은 원래 명암으로, 이하인 픽셀은 검정색으로 표시
        _, threshold_img = cv2.threshold(
            clahe_img, threshold_value, 255, cv2.THRESH_BINARY
        )
        threshold_img = cv2.cvtColor(
            threshold_img, cv2.COLOR_GRAY2BGR
        )  # thresholding 결과를 다시 컬러 이미지로 변환

        # 세 이미지를 하나의 이미지로 합치기
        concatenated_image = np.concatenate(
            (original, cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR), threshold_img),
            axis=1,
        )

        # 이미지 저장
        # 확장자를 제거한 원래의 파일 이름 가져오기
        original_name = os.path.splitext(filename)[0]
        cv2.imwrite(
            f"{output_folder_path}/Threshold-{threshold_value}-{image_order}-{original_name}.png",
            concatenated_image,
        )

        # 사진 순서 업데이트
        image_order += 1
