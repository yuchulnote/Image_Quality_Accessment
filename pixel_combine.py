import os
import numpy as np
import cv2


def apply_threshold(image_path, threshold, save_folder):
    # 이미지를 그레이스케일로 변환
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 특정 명암값 이상인 값은 원래 명암으로, 특정 명암값 미만인 값은 검정색(0)으로 처리
    _, thresholded_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    # 원본 이미지와 처리된 이미지를 가로로 결합
    combined_image = np.concatenate((image, thresholded_image), axis=1)

    # 결합된 이미지를 저장
    image_name = os.path.basename(image_path)
    combined_image_path = os.path.join(save_folder, f"결합_{image_name}")
    cv2.imwrite(combined_image_path, combined_image)

    return combined_image_path


if __name__ == "__main__":
    input_folder = (
        "C:/Users/PETNOW/Desktop/ivan/test/threshold/input"  # 입력 이미지가 있는 폴더 이름으로 바꿔주세요
    )
    threshold = 150  # 특정 명암값 설정
    save_folder = "C:/Users/PETNOW/Desktop/ivan/test/threshold/output"  # 결과 이미지를 저장할 폴더 이름으로 바꿔주세요

    # 결과 이미지를 저장할 폴더가 없다면 생성
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # 입력 폴더 내의 모든 이미지 파일에 대해 처리 및 결과 저장 수행
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            combined_image_path = apply_threshold(image_path, threshold, save_folder)
