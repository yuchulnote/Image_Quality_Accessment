import cv2
import numpy as np
import os

# Image directory
folder_path = "C:/Users/PETNOW/Desktop/idea_glare/glare_positive_motionblur_negative"
save_folder_path = "C:/Users/PETNOW/Desktop/idea_glare/saved_image"
threshold_value = 180
max_size = 500

# Load all images
file_list = [
    f for f in os.listdir(folder_path) if f.endswith(".png") or f.endswith(".jpg")
]


def load_image(filename):
    return cv2.imread(os.path.join(folder_path, filename))


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
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def concatenate_images(original, clahe_img, threshold_img, final_img):
    clahe_img_bgr = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)
    threshold_img_bgr = cv2.cvtColor(threshold_img, cv2.COLOR_GRAY2BGR)
    return np.concatenate(
        (original, clahe_img_bgr, threshold_img_bgr, final_img), axis=1
    )


def add_image_info(image, filename, threshold_value, max_size):
    empty_space = np.zeros((100, image.shape[1], 3), dtype=np.uint8)
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
    return np.concatenate((empty_space, image), axis=0)


def process_image(index, threshold_value, max_size):
    filename = file_list[index]
    original = load_image(filename)
    if original is None:
        print(f"Failed to load image file {filename}. Please check the image path.")
        return
    gray = convert_to_gray(original)
    clahe_img = apply_clahe(gray)
    threshold_img = create_threshold_img(clahe_img, threshold_value)
    final_image = remove_large_components(threshold_img.copy(), max_size)
    final_image = concatenate_images(original, clahe_img, threshold_img, final_image)
    final_image = add_image_info(final_image, filename, threshold_value, max_size)
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
