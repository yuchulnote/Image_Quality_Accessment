import cv2
import numpy as np
import os

# 이미지를 불러올 폴더 경로
folder_path = ""

# 이미지를 저장할 폴더 경로
save_folder_path = ""  # 실제 저장할 폴더 경로로 변경하세요

# 특정 명암값
threshold_value = 200

# 파일 리스트 불러오기
file_list = [
    f for f in os.listdir(folder_path) if f.endswith(".png") or f.endswith(".jpg")
]

index = 0  # 현재 이미지의 인덱스

def process_image(index, threshold_value):
    filename = file_list[index]

    # 원본 이미지 파일 불러오기
    original = cv2.imread(os.path.join(folder_path, filename))

    # 이미지가 제대로 로드되었는지 확인
    if original is None:
        print(f"Failed to load image file {filename}. Please check the image path.")
        return

    # 이미지를 grayscale로 변환
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # CLAHE (Contrast Limited Adaptive Histogram Equalization) 적용
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    # 특정 명암값 이상인 픽셀은 원래 명암으로, 이하인 픽셀은 검정색으로 표시
    _, threshold_img = cv2.threshold(clahe_img, threshold_value, 255, cv2.THRESH_BINARY)
    threshold_img = cv2.cvtColor(threshold_img, cv2.COLOR_GRAY2BGR)

    # 세 이미지를 하나의 이미지로 합치기
    concatenated_image = np.concatenate((original, cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR), threshold_img), axis=1)

    # 빈 공간 생성 및 텍스트 추가
    empty_space = np.zeros((50, concatenated_image.shape[1], 3), dtype=np.uint8)
    filename_text = f'Filename: {filename}'
    threshold_text = f'Threshold: {threshold_value}'
    cv2.putText(empty_space, filename_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(empty_space, threshold_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 빈 공간과 이미지 합치기
    final_image = np.concatenate((empty_space, concatenated_image), axis=0)

    return final_image

# 처음 이미지 처리 및 출력
final_image = process_image(index, threshold_value)
cv2.imshow("Image", final_image)

while True:
    key = cv2.waitKey(0)

    if key == ord("."):  # 'n' 키가 눌렸을 때 다음 이미지로 이동
        index += 1
        if index >= len(file_list):  # 마지막 이미지에서 'n' 키를 누르면 프로그램 종료
            break
        final_image = process_image(index, threshold_value)
        cv2.imshow("Image", final_image)
    elif key == ord(","):  # 'p' 키가 눌렸을 때 이전 이미지로 이동
        index -= 1
        if index < 0:  # 처음 이미지에서 'p' 키를 누르면 프로그램 종료
            break
        final_image = process_image(index, threshold_value)
        cv2.imshow("Image", final_image)
    elif key == ord("]"):  # '+' 키가 눌렸을 때 threshold_value 증가
        if threshold_value < 255:  # threshold_value 가 255보다 클 경우 증가시키지 않음
            threshold_value += 1
        final_image = process_image(index, threshold_value)
        cv2.imshow("Image", final_image)
    elif key == ord("["):  # '-' 키가 눌렸을 때 threshold_value 감소
        if threshold_value > 0:  # threshold_value 가 0보다 작을 경우 감소시키지 않음
            threshold_value -= 1
        final_image = process_image(index, threshold_value)
        cv2.imshow("Image", final_image)
    elif key == ord("s"):  # 's' 키가 눌렸을 때 이미지 저장
        save_path = os.path.join(save_folder_path, f"Threshold_{threshold_value}-Index_{index}-{file_list[index]}")
        cv2.imwrite(save_path, final_image)
        print(f"Image saved at {save_path}")
    elif key == ord("q"):  # 'q' 키가 눌렸을 때 프로그램 종료
        break

cv2.destroyAllWindows()
