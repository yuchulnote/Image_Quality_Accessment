import os
from PIL import Image


def apply_threshold(image_path, threshold, save_folder):
    # 이미지를 gray scale로 변환
    image = Image.open(image_path).convert("L")

    # 이미지의 너비와 높이를 가져옴
    width, height = image.size

    # 새로운 이미지를 생성 (결과 이미지)
    new_image = Image.new("L", (width, height))

    # 특정 명암값 이상인 값은 원래 명암으로, 특정 명암값 미만인 값은 검정색(0)으로 처리
    for x in range(width):
        for y in range(height):
            pixel = image.getpixel((x, y))
            if pixel >= threshold:
                new_image.putpixel((x, y), pixel)
            else:
                new_image.putpixel((x, y), 0)

    # 결과 이미지를 저장
    image_name = os.path.basename(image_path)
    new_image_path = os.path.join(save_folder, f"thresholded_{image_name}")
    new_image.save(new_image_path)


if __name__ == "__main__":
    input_folder = (
        ""  # 입력 이미지가 있는 폴더 이름으로 바꿔주세요
    )
    threshold = 150  # 특정 명암값 설정 (여기서는 128을 사용하겠습니다)
    save_folder = ""  # 결과 이미지를 저장할 폴더 이름으로 바꿔주세요

    # 결과 이미지를 저장할 폴더가 없다면 생성
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # 입력 폴더 내의 모든 이미지 파일에 대해 처리 수행
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            apply_threshold(image_path, threshold, save_folder)
