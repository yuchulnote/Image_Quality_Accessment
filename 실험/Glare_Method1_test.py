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

    # get the aspect ratio
    h, w = image.shape[:2]
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


def calculate_white_ratio(binary_img):
    total_pixels = binary_img.shape[0] * binary_img.shape[1]
    white_pixels = np.sum(binary_img == 255)
    white_ratio = white_pixels / total_pixels
    return white_ratio


def save_to_csv(data, filename):
    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "white_ratio"])  # Add column headers
        writer.writerows(data)


# Store white ratio and filename
white_ratios = []

for index in range(len(file_list)):
    filename = file_list[index]
    original = load_image(filename)
    if original is None:
        print(f"Failed to load image file {filename}. Please check the image path.")
        continue
    gray = convert_to_gray(original)
    clahe_img = apply_clahe(gray)
    threshold_img = create_threshold_img(clahe_img, threshold_value)
    white_ratio = calculate_white_ratio(threshold_img)

    white_ratios.append([filename, white_ratio])

# Save the result to a csv file
save_to_csv(white_ratios, "white_ratio1.csv")
