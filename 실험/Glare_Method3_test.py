import cv2
import numpy as np
import os
import csv

# Image directory
folder_path = r"C:\Users\PETNOW\Desktop\shadow_glare_motionblur\total"
threshold_value = 190

# Load all images
file_list = [
    f for f in os.listdir(folder_path) if f.endswith(".png") or f.endswith(".jpg")
]


def load_image(filename):
    image = cv2.imread(os.path.join(folder_path, filename))
    if image is None:
        return None

    # Resize the image
    h, w = image.shape[:2]
    aspect_ratio = w / h

    if w > h:
        new_w = 200
        new_h = int(new_w / aspect_ratio)
    else:
        new_h = 200
        new_w = int(new_h * aspect_ratio)

    return cv2.resize(image, (new_w, new_h))


def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def create_threshold_img(image, threshold_value):
    _, binary_img = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return binary_img


def calculate_white_ratio(index, threshold_value):
    filename = file_list[index]
    original = load_image(filename)
    if original is None:
        print(f"Failed to load image file {filename}. Please check the image path.")
        return

    gray = convert_to_gray(original)
    clahe_img = apply_clahe(gray)
    threshold_img = create_threshold_img(clahe_img, threshold_value)

    # 이미지를 3 부분으로 나누기
    height, width = threshold_img.shape
    sections = np.array_split(threshold_img, 3, axis=0)
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

    return {
        "filename": filename,
        "highest_ratio": highest_ratio,
        "highest_ratio_section": highest_ratio_section,
        "white_pixel_count": int(white_ratios[highest_ratio_index] * total_pixels),
    }


# Calculate white ratio for all images
all_images_white_ratio = []
for index in range(len(file_list)):
    result = calculate_white_ratio(index, threshold_value)
    all_images_white_ratio.append(result)

# Write to CSV
with open("white_ratio3.csv", "w", newline="") as csvfile:
    fieldnames = [
        "filename",
        "highest_ratio",
        "highest_ratio_section",
        "white_pixel_count",
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for data in all_images_white_ratio:
        writer.writerow(data)
