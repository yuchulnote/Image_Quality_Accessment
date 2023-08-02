import cv2
import numpy as np
import os
import csv

# Image directory
folder_path = r"C:\Users\PETNOW\Desktop\shadow_glare_motionblur\total"
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


def remove_large_components(image, max_size):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        image, connectivity=8
    )
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > max_size:
            image[labels == i] = 0
    return image


def calculate_white_ratio(index, threshold_value, max_size):
    filename = file_list[index]
    original = load_image(filename)
    if original is None:
        print(f"Failed to load image file {filename}. Please check the image path.")
        return
    gray = convert_to_gray(original)
    clahe_img = apply_clahe(gray)
    threshold_img = create_threshold_img(clahe_img, threshold_value)
    postprocessed_img = remove_large_components(threshold_img.copy(), max_size)

    white_pixels = np.sum(postprocessed_img == 255)
    total_pixels = postprocessed_img.size
    white_ratio = white_pixels / total_pixels

    return {
        "filename": filename,
        "white_ratio": white_ratio,
        "white_pixel_count": white_pixels,
    }


# Calculate white ratio for all images
all_images_white_ratio = []
for index in range(len(file_list)):
    result = calculate_white_ratio(index, threshold_value, max_size)
    all_images_white_ratio.append(result)

# Write to CSV
with open("white_ratio2.csv", "w", newline="") as csvfile:
    fieldnames = [
        "filename",
        "white_ratio",
        "white_pixel_count",
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for data in all_images_white_ratio:
        writer.writerow(data)
