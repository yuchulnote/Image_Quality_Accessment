import cv2
import numpy as np
import os


# Image directory
folder_path = "C:/Users/PETNOW/Desktop/idea_glare/glare_positive_motionblur_negative"
save_folder_path = "C:/Users/PETNOW/Desktop/idea_glare/saved_image"
threshold_value = 190
max_size = 150

# Load all images
file_list = [
    f for f in os.listdir(folder_path) if f.endswith(".png") or f.endswith(".jpg")
]


def load_image(filename):
    image = cv2.imread(os.path.join(folder_path, filename))
    if image is None:
        return None

    # get the aspect ratio
    h, w = (image.shape[:2],)
    aspect_ratio = w / h

    if w > h:  # width > height
        new_w = 200
        new_h = int(new_w / aspect_ratio)
    else:
        new_h = 200
        new_w = int(new_h * aspect_ratio)

    # Resize the image, maintaining aspect ratio
    image = cv2.resize(image, (new_w, new_h))

    return image


def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def create_threshold_img(image, threshold_value):
    _, binary_img = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return binary_img


def remove_large_components(image, max_size):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        image, connectivity=8
    )
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > max_size:
            image[labels == i] = 0
    return image


def highlight_regions_before(image, binary_mask, color=(0, 255, 0)):
    if len(image.shape) == 2:
        highlighted_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        highlighted_image = image.copy()
    highlighted_image[np.where(binary_mask == 255)] = color
    return highlighted_image


def highlight_regions(image, binary_mask, color=(0, 255, 0)):
    highlighted_image = image.copy()
    highlighted_image[np.where(binary_mask == 255)] = color
    return highlighted_image


def concatenate_images(
    original,
    clahe_img,
    threshold_img,
    postprocessed_img,
    preprocessed_highlighted_img,
    postprocessed_highlighted_img,
):
    if len(preprocessed_highlighted_img.shape) == 2:
        preprocessed_highlighted_img_bgr = cv2.cvtColor(
            preprocessed_highlighted_img, cv2.COLOR_GRAY2BGR
        )
    else:
        preprocessed_highlighted_img_bgr = preprocessed_highlighted_img

    if len(postprocessed_highlighted_img.shape) == 2:
        postprocessed_highlighted_img_bgr = cv2.cvtColor(
            postprocessed_highlighted_img, cv2.COLOR_GRAY2BGR
        )
    else:
        postprocessed_highlighted_img_bgr = postprocessed_highlighted_img

    clahe_img_bgr = (
        cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)
        if len(clahe_img.shape) == 2
        else clahe_img
    )
    threshold_img_bgr = (
        cv2.cvtColor(threshold_img, cv2.COLOR_GRAY2BGR)
        if len(threshold_img.shape) == 2
        else threshold_img
    )
    postprocessed_img_bgr = (
        cv2.cvtColor(postprocessed_img, cv2.COLOR_GRAY2BGR)
        if len(postprocessed_img.shape) == 2
        else postprocessed_img
    )
    return np.concatenate(
        (
            original,
            clahe_img_bgr,
            threshold_img_bgr,
            postprocessed_img_bgr,
            preprocessed_highlighted_img_bgr,
            postprocessed_highlighted_img_bgr,
        ),
        axis=1,
    )


def add_image_info(image, filename, threshold_value, max_size, postprocessed_img):
    empty_space = np.zeros(
        (150, image.shape[1], 3), dtype=np.uint8
    )  # Empty space for text
    cv2.putText(
        empty_space,
        f"Filename: {filename}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )
    cv2.putText(
        empty_space,
        f"Threshold: {threshold_value}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )
    cv2.putText(
        empty_space,
        f"Max size: {max_size}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )

    # 이미지를 3 부분으로 나누기
    height, width = postprocessed_img.shape
    sections = np.array_split(postprocessed_img, 3, axis=0)
    section_names = ["Top", "Middle", "Bottom"]
    white_ratios = []

    for sec in sections:
        white_pixels = np.sum(sec == 255)
        total_pixels = sec.size
        white_ratios.append(white_pixels / total_pixels)

    # 가장 높은 비율 계산
    highest_ratio = max(white_ratios)
    highest_ratio_index = white_ratios.index(highest_ratio)
    highest_ratio_section = section_names[highest_ratio_index]

    cv2.putText(
        empty_space,
        f"Highest white pixel ratio: {highest_ratio * 100:.4f}% ({highest_ratio_section} section)",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )

    cv2.putText(
        empty_space,
        f"White pixel count in highest ratio section: {int(white_ratios[highest_ratio_index]*total_pixels)}",
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )

    return np.concatenate((empty_space, image), axis=0)


def process_image(index, threshold_value, max_size):
    filename = file_list[index]
    original = load_image(filename)
    if original is None:
        print(f"Failed to load image file {filename}. Please check the image path.")
        return
    gray = convert_to_gray(original)  # 원본 이미지 gray scale 변경
    clahe_img = apply_clahe(gray)  # 변경된 gray scale 이미지에 CLAHE 적용
    threshold_img = create_threshold_img(
        clahe_img, threshold_value
    )  # Threshold 초과 값 흰픽셀, 이하 값 검정 픽셀 처리
    preprocessed_img = threshold_img.copy()  # 하이라이트를 입히기 위한 Threshold 이미지 복사
    preprocessed_highlighted_img = highlight_regions_before(
        original, preprocessed_img
    )  # 원본이미지에, Threshold 흰색 픽셀이였던 부분 초록색으로 덧 씌우기
    postprocessed_img = remove_large_components(
        threshold_img.copy(), max_size
    )  # Threshold 적용된 이미지에 max_size 이하로 묶인 components 제거된 이미지
    postprocessed_highlighted_img = highlight_regions(
        original, postprocessed_img
    )  # 원본이미지에, Threshold + remove components 된 흰색 픽셀이었던 부분 초록색으로 덧 씌우기
    final_image = concatenate_images(  # 최종 이미지는 [원본, CLAHE, Threshold, Components제거, 제거처리 전 초록픽셀, 제거처리 후 초록픽셀] -> 6장
        original,
        clahe_img,
        threshold_img,
        postprocessed_img,
        preprocessed_highlighted_img,
        postprocessed_highlighted_img,
    )
    final_image = add_image_info(
        final_image, filename, threshold_value, max_size, postprocessed_img
    )
    return final_image


index = 0  # current image index
final_image = process_image(index, threshold_value, max_size)
cv2.imshow("Image", final_image)

while True:
    key = cv2.waitKey(0)
    if key == ord("."):  # If '.' is pressed, move to the next image
        index += 1
        if index >= len(
            file_list
        ):  # If '.' is pressed at the last image, exit the program
            break
        final_image = process_image(index, threshold_value, max_size)
        cv2.imshow("Image", final_image)
    elif key == ord(","):  # If ',' is pressed, move to the previous image
        index -= 1
        if index < 0:  # If ',' is pressed at the first image, exit the program
            break
        final_image = process_image(index, threshold_value, max_size)
        cv2.imshow("Image", final_image)
    elif key == ord("]"):  # If ']' is pressed, increase threshold_value
        if threshold_value < 255:
            threshold_value += 1
        final_image = process_image(index, threshold_value, max_size)
        cv2.imshow("Image", final_image)
    elif key == ord("["):  # If '[' is pressed, decrease threshold_value
        if threshold_value > 0:
            threshold_value -= 1
        final_image = process_image(index, threshold_value, max_size)
        cv2.imshow("Image", final_image)
    elif key == ord("m"):  # If 'm' is pressed, increase max_size
        max_size += 1
        final_image = process_image(index, threshold_value, max_size)
        cv2.imshow("Image", final_image)
    elif key == ord("n"):  # If 'n' is pressed, decrease max_size
        if max_size > 1:
            max_size -= 1
        final_image = process_image(index, threshold_value, max_size)
        cv2.imshow("Image", final_image)
    elif key == ord("s"):  # If 's' is pressed, save the image
        save_path = os.path.join(
            save_folder_path,
            f"Threshold_{threshold_value}-Index_{index}-{file_list[index]}",
        )
        cv2.imwrite(save_path, final_image)
        print(f"Image saved at {save_path}")
    elif key == ord("q"):  # If 'q' is pressed, exit the program
        break

cv2.destroyAllWindows()
