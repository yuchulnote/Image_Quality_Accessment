import os
import cv2


def apply_histogram_equalization(input_image_path, output_folder):
    # 이미지 로드
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    # 히스토그램 평탄화 수행
    equalized_image = cv2.equalizeHist(image)

    # 평탄화된 이미지 저장
    image_name = os.path.basename(input_image_path)
    output_image_path = os.path.join(output_folder, f"equalized_{image_name}")
    cv2.imwrite(output_image_path, equalized_image)


if __name__ == "__main__":
    input_folder = ""  # 입력 이미지 폴더 경로로 변경해주세요
    output_folder = ""  # 평탄화된 이미지를 저장할 폴더 경로로 변경해주세요

    # 폴더가 없으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 입력 폴더 내의 모든 이미지 파일에 대해 히스토그램 평탄화 수행 및 저장
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            apply_histogram_equalization(image_path, output_folder)
