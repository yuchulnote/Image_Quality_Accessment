import csv
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import numpy as np

# read the csv file
with open(r"C:\Users\PETNOW\Desktop\ivan\Robert_ivan\glare_ture_pred6-1.csv", "r") as f:
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
# _precision = TP / (TP + FP)
# _recall = TP / (TP + FN)
# print(f"precision: {_precision}")
# print(f"recall: {_recall}")
# print(f"ACC from PR: {(_precision * _recall) / (_precision + _recall)}")

##### Hoo - accuracy max #####

# Compute the threshold that maximizes accuracy
# accuracy = (precision * recall) / (precision + recall)
# optimal_idx_acc = np.argmax(accuracy)
# optimal_threshold_acc = thresholds[optimal_idx_acc]
# Compute ROC area
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

# Plot ROC curve
# plt.figure()
lw = 2
# plt.plot(
#     fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.4f)" % roc_auc
# )
# plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("Receiver Operating Characteristic")
# plt.legend(loc="lower right")
# plt.show()

# Plot PR curve
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
