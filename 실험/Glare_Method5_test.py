import cv2
import numpy as np
import os
import csv
from pathlib import Path

# Image directory
folder_path = r"C:\Users\PETNOW\Desktop\shadow_glare_motionblur\total"
# Change this to your image directory

# Threshold for binary image
THRESHOLD = 190  # Change this to your desired threshold

# Load all images
file_list = [
    f for f in os.listdir(folder_path) if f.endswith(".png") or f.endswith(".jpg")
]


def load_image(filename):
    path = os.path.join(folder_path, filename)
    return cv2.imread(path)


def apply_circular_mask(image, ratio=0.8):
    # Create a black image with same dimensions as our loaded image
    black_img = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Get the center coordinates of the image to create the circle
    circle_center_coordinates = (image.shape[1] // 2, image.shape[0] // 2)

    # Get the radius size in relation to the image size
    circle_radius = int(min(image.shape[0], image.shape[1]) * ratio // 2)

    # Create the circle on the black image
    mask = cv2.circle(
        black_img, circle_center_coordinates, circle_radius, (255), thickness=-1
    )

    return mask


def apply_clahe(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    return clahe_img


# Open CSV file for writing
with open("white_ratio.csv", "w", newline="") as csvfile:
    fieldnames = ["filename", "white_ratio"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    idx = 0
    while idx < len(file_list):
        # Load image
        filename = file_list[idx]
        image = load_image(filename)

        # Apply clahe to the image
        clahe_image = apply_clahe(image)

        # Generate mask
        mask = apply_circular_mask(image)

        # Apply mask on the image, only keep the part inside the mask
        masked_image = cv2.bitwise_and(clahe_image, clahe_image, mask=mask)

        # Threshold the masked image
        _, thresholded_image = cv2.threshold(
            masked_image, THRESHOLD, 255, cv2.THRESH_BINARY
        )

        # Create a white image
        white_img = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * 255

        # Subtract the mask from the white image
        white_img_outside_mask = cv2.bitwise_and(
            white_img, white_img, mask=cv2.bitwise_not(mask)
        )

        # Merge the thresholded image with the white image outside the mask
        result = cv2.add(
            cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2BGR), white_img_outside_mask
        )

        # Count the white pixels within the mask
        white_pixels = cv2.countNonZero(cv2.bitwise_and(thresholded_image, mask))

        # Count the total pixels within the mask
        total_pixels = cv2.countNonZero(mask)

        # Calculate the ratio of white pixels
        white_ratio = (white_pixels / total_pixels) * 100
        print(f"White pixels ratio for {filename}: {white_ratio:.2f}%")

        # Write the ratio to CSV
        writer.writerow({"filename": filename, "white_ratio": white_ratio})
        idx += 1
