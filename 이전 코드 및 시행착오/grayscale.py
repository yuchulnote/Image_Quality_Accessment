import cv2
import os


def convert_images_to_grayscale(input_folder, output_folder):
    # 입력 폴더 내의 이미지 파일들 가져오기
    image_files = os.listdir(input_folder)

    for image_file in image_files:
        input_image_path = os.path.join(input_folder, image_file)
        output_image_path = os.path.join(output_folder, image_file)

        # 이미지를 그레이스케일로 로드
        image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

        if image is not None:
            # 그레이스케일로 변환된 이미지를 저장
            cv2.imwrite(output_image_path, image)
            print(f"{input_image_path} 이미지를 그레이스케일로 변환하여 {output_image_path}에 저장했습니다.")
        else:
            print(f"오류: {input_image_path} 이미지를 읽는 데 실패했습니다.")


# 입력 폴더와 출력 폴더 경로를 지정
input_folder = ""
output_folder = ""

# 이미지를 그레이스케일로 변환 수행
convert_images_to_grayscale(input_folder, output_folder)
