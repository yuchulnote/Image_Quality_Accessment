import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def apply_threshold(image_path, threshold, save_folder, image_index):
    # 원본 이미지 로드
    original_image = cv2.imread(image_path)

    # 특정 명암값 이상인 값은 원래 명암으로, 특정 명암값 미만인 값은 검정색(0)으로 처리
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    _, thresholded_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

    # 파일명 설정
    image_name = os.path.basename(image_path)
    image_name_without_extension = os.path.splitext(image_name)[0]
    output_filename = f"{threshold}-{image_index+1}_{image_name_without_extension}.jpg"
    combined_image_path = os.path.join(save_folder, output_filename)

    # 원본 이미지와 처리된 이미지를 가로로 결합
    combined_image = np.concatenate(
        (original_image, cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2BGR)), axis=1
    )

    # 결합된 이미지를 저장
    cv2.imwrite(combined_image_path, combined_image)

    return combined_image_path


if __name__ == "__main__":
    input_folder = "C:/Users/PETNOW/Desktop/ivan/Robert_ivan/result_his/dataset/output_test/negative"  # 입력 이미지 폴더 경로로 변경해주세요
    threshold = 200  # 특정 명암값 설정
    save_folder = "C:/Users/PETNOW/Desktop/ivan/Robert_ivan/result_his/dataset/output_test/result_negative"  # 결과 이미지를 저장할 폴더 경로로 변경해주세요

    # 결과 이미지를 저장할 폴더가 없다면 생성
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # 입력 폴더 내의 모든 이미지 파일에 대해 처리 및 결과 저장 수행
    for image_index, filename in enumerate(os.listdir(input_folder)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            combined_image_path = apply_threshold(
                image_path, threshold, save_folder, image_index
            )
