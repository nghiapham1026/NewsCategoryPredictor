import numpy as np

# CNN Model Calculations
# Confusion matrix for CNN Model
cm_cnn = np.array([
    [180, 12, 15, 15],
    [9, 270, 17, 11],
    [16, 7, 61, 6],
    [28, 10, 11, 177]
])

# BERT Model Calculations
# Confusion matrix for BERT Model
cm_bert = np.array([
    [173, 14, 4, 30],
    [11, 286, 1, 11],
    [8, 17, 61, 5],
    [29, 18, 6, 171]
])

# Function to calculate precision, recall, F1 score, and accuracy
def calculate_metrics(cm):
    # True Positives are the diagonal elements
    TP = np.diag(cm)
    # False Positives are the sum of the column, minus the diagonal
    FP = np.sum(cm, axis=0) - TP
    # False Negatives are the sum of the row, minus the diagonal
    FN = np.sum(cm, axis=1) - TP
    # True Negatives are the sum of all elements minus (TP + FP + FN for each class)
    TN = np.sum(cm) - (FP + FN + TP)

    # Precision, Recall, and F1 Score calculations
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)
    # Accuracy calculation
    accuracy = np.sum(TP) / np.sum(cm)

    return precision, recall, f1_score, accuracy

# Calculate metrics
precision_cnn, recall_cnn, f1_cnn, accuracy_cnn = calculate_metrics(cm_cnn)
precision_bert, recall_bert, f1_bert, accuracy_bert = calculate_metrics(cm_bert)

# Print metrics
print("CNN Model Metrics:")
print("Precision:", precision_cnn)
print("Recall:", recall_cnn)
print("F1 Score:", f1_cnn)
print("Accuracy:", accuracy_cnn)

print("\nBERT Model Metrics:")
print("Precision:", precision_bert)
print("Recall:", recall_bert)
print("F1 Score:", f1_bert)
print("Accuracy:", accuracy_bert)