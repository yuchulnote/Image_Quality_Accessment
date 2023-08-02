import cv2
import numpy as np
import os
import csv

# Image directory
folder_path = r"C:\Users\PETNOW\Desktop\shadow_glare_motionblur\total"  # Change this to your image directory

# Threshold for binary image
THRESHOLD = 200  # Change this to your desired threshold

# Max component size to remove
max_size = 190

# Load all images
file_list = [
    f for f in os.listdir(folder_path) if f.endswith(".png") or f.endswith(".jpg")
]


def load_image(filename):
    path = os.path.join(folder_path, filename)
    return cv2.imread(path)


def apply_circular_mask(image, ratio=0.8):
    black_img = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    circle_center_coordinates = (image.shape[1] // 2, image.shape[0] // 2)

    circle_radius = int(min(image.shape[0], image.shape[1]) * ratio // 2)

    mask = cv2.circle(
        black_img, circle_center_coordinates, circle_radius, (255), thickness=-1
    )

    return mask


def apply_clahe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    return clahe_img


def remove_components(image, max_size):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        image, connectivity=8
    )

    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    img2 = np.zeros((output.shape), dtype=np.uint8)

    for i in range(0, nb_components):
        if sizes[i] <= max_size:
            img2[output == i + 1] = 255

    return img2


# Create and open a new csv file to write the output
with open("white_ratio6-1.csv", "w", newline="") as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(["Image", "White Pixel Ratio"])

    for idx in range(len(file_list)):
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

        # Remove small components
        clean_image = remove_components(thresholded_image, max_size)

        # Count the white pixels within the mask
        white_pixels = cv2.countNonZero(cv2.bitwise_and(clean_image, mask))

        # Count the total pixels within the mask
        total_pixels = cv2.countNonZero(mask)

        # Calculate the ratio of white pixels
        white_ratio = (white_pixels / total_pixels) * 100

        print(f"White pixels ratio for {filename}: {white_ratio:.4f}%")

        # Write the image name and the white pixel ratio to the csv file
        writer.writerow([filename, white_ratio])
