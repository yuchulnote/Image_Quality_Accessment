import cv2
import numpy as np
import os
import csv
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix,
)
import matplotlib.pyplot as plt

# Image directory
folder_path = r"C:\Users\PETNOW\Desktop\shadow_glare_motionblur\total"  # Change this to your image directory

# Threshold for binary image
THRESHOLD = 200  # Change this to your desired threshold

# Max component size to remove
max_size = 210

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
with open("white_ratio6-2.csv", "w", newline="") as file:
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


# Open the existing CSV file in read mode
with open("test.csv", "r") as existing_file:
    reader = csv.reader(existing_file)
    existing_data = list(reader)

# Open the new CSV file in read mode
with open("white_ratio6-2.csv", "r") as new_file:
    reader = csv.reader(new_file)
    new_data = list(reader)[1:]  # Skip the header

# Add new column title to the existing header
existing_data[0].append("White Pixel Ratio")

# Add the new data
for row in new_data:
    # Find the image name in the existing data
    for existing_row in existing_data[1:]:  # Skip the header
        if existing_row[0] == row[0]:
            # Append the white ratio to the end of the row
            existing_row.append(row[1])
            break
    else:
        # If the image name was not found, print an error and stop execution
        print(f"Error: The image {row[0]} was not found in the existing data.")
        exit(1)

# Write the merged data to a new CSV file
with open("glare_ture_pred6-2.csv", "w", newline="") as merged_file:
    writer = csv.writer(merged_file)
    writer.writerows(existing_data)


# read the csv file
with open(r"C:\Users\PETNOW\Desktop\ivan\Robert_ivan\glare_ture_pred6-2.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)  # Skip the header
    data = list(reader)

# Assign data
labels = [row[1] for row in data]
scores = [float(row[2]) for row in data]

# Convert labels to binary
binary_labels = [1 if label == "positive" else 0 for label in labels]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(binary_labels, scores)

# Compute the threshold that maximizes TPR - FPR
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold_tpr_fpr = thresholds[optimal_idx]

# Compute precision, recall and thresholds for PR curve
precision, recall, thresholds = precision_recall_curve(binary_labels, scores)

# Append the last precision and recall values
thresholds = np.append(thresholds, 1)

##### Hoo - accuracy max #####
accuracys = []
cms = []

for th in thresholds:
    predicted = [1 if score >= th else 0 for score in scores]
    TN, FP, FN, TP = confusion_matrix(binary_labels, predicted).ravel()
    _accuracy = (TP + TN) / (TP + FP + FN + TN)
    accuracys.append(_accuracy)
    cms.append(confusion_matrix(binary_labels, predicted))

acc_max_idx = np.argmax(accuracys)
[[TN, FP], [FN, TP]] = cms[acc_max_idx]

# print(f"Accuracy max idx: {acc_max_idx}")
print(f"accuracy: {accuracys[acc_max_idx]}")
print(f"precision: {TP / (TP + FP)}")
print(f"cm:\n{cms[acc_max_idx]}")
print(f"TN\tFP\nFN\tTP")

roc_auc = auc(fpr, tpr)
print(f"auc: {roc_auc}")
print("-" * 20)

# Compute confusion matrix and metrics at TPR - FPR max threshold
predicted = [1 if score >= optimal_threshold_tpr_fpr else 0 for score in scores]
TN, FP, FN, TP = confusion_matrix(binary_labels, predicted).ravel()
print(confusion_matrix(binary_labels, predicted).ravel())
accuracy_tpr_fpr = (TP + TN) / (TP + FP + FN + TN)
precision_tpr_fpr = TP / (TP + FP)
recall_tpr_fpr = TP / (TP + FN)
print(f"Threshold that maximizes TPR - FPR: {optimal_threshold_tpr_fpr}")
print(f"Accuracy at TPR - FPR: {accuracy_tpr_fpr}")
print(f"Precision at TPR - FPR: {precision_tpr_fpr}")
print(f"Recall at TPR - FPR: {recall_tpr_fpr}")
print(f"Confusion matrix(TP, FN, FP, TN) at TPR - FPR: {(TP, FN, FP, TN)}")

# Compute confusion matrix and metrics at max accuracy threshold
# predicted = [1 if score >= optimal_threshold_acc else 0 for score in scores]
predicted = [1 if score >= thresholds[acc_max_idx] else 0 for score in scores]
TN_acc, FP_acc, FN_acc, TP_acc = confusion_matrix(binary_labels, predicted).ravel()
print(confusion_matrix(binary_labels, predicted).ravel())
accuracy_acc = (TP_acc + TN_acc) / (TP_acc + FP_acc + FN_acc + TN_acc)
precision_acc = TP_acc / (TP_acc + FP_acc)
recall_acc = TP_acc / (TP_acc + FN_acc)


print(f"Threshold that maximizes Accuracy: {thresholds[acc_max_idx]}")
print(f"Accuracy at max Accuracy: {accuracy_acc}")
print(f"Precision at max Accuracy: {precision_acc}")
print(f"Recall at max Accuracy: {recall_acc}")
print(
    f"Confusion matrix(TP, FN, FP, TN) at max Accuracy: {(TP_acc, FN_acc, FP_acc, TN_acc)}"
)

lw = 2

plt.figure()
plt.plot(recall, precision, color="b", lw=lw, label="PR curve")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="lower right")
plt.show()

plt.figure()
plt.plot(fpr, tpr, "o-", ms=2, label=f"AUC = {roc_auc}")
plt.legend()
plt.plot([0, 1], [0, 1], "k--", label="random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC curve")
plt.show()

plt.figure()
scores = np.array(scores)
scores = np.log(scores + 0.7)
density = True
plt.hist(
    np.array(list(filter(lambda x: x[0] == 1, zip(binary_labels, scores))))[:, 1],  # p
    label="positive(reject)",
    density=density,
    bins=50,
    histtype="step",
)
plt.hist(
    np.array(list(filter(lambda x: x[0] == 0, zip(binary_labels, scores))))[:, 1],  # n
    label="negative(accept)",
    density=density,
    bins=50,
    histtype="step",
)
plt.xlabel("Glare score")
plt.ylabel("count")
plt.legend()
plt.title(f"Glare score plot")
plt.show()

plt.figure()
scores = np.array(scores)
scores = np.log(scores + 0.7)
density = False
plt.hist(
    np.array(list(filter(lambda x: x[0] == 1, zip(binary_labels, scores))))[:, 1],  # p
    label="positive(reject)",
    density=density,
    bins=50,
    histtype="step",
)
plt.hist(
    np.array(list(filter(lambda x: x[0] == 0, zip(binary_labels, scores))))[:, 1],  # n
    label="negative(accept)",
    density=density,
    bins=50,
    histtype="step",
)
plt.xlabel("Glare score")
plt.ylabel("count")
plt.legend()
plt.title(f"Glare score plot")
plt.show()
