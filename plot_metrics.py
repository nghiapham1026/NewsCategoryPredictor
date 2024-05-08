import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def plot_roc_curves(classes, y_test_binarized, y_pred_proba):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i, label in enumerate(classes):
        fpr[label], tpr[label], _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
        roc_auc[label] = auc(fpr[label], tpr[label])

    plt.figure(figsize=(12, 6))
    for label in classes:
        plt.plot(fpr[label], tpr[label], linestyle='--', label=f'ROC curve (area = {roc_auc[label]:.2f}) for class {label}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves by Class')
    plt.legend(loc="lower right")
    plt.show()

def plot_precision_recall_curves(classes, y_test_binarized, y_pred_proba):
    precision = dict()
    recall = dict()
    pr_auc = dict()

    for i, label in enumerate(classes):
        precision[label], recall[label], _ = precision_recall_curve(y_test_binarized[:, i], y_pred_proba[:, i])
        pr_auc[label] = average_precision_score(y_test_binarized[:, i], y_pred_proba[:, i])

    plt.figure(figsize=(12, 6))
    for label in classes:
        plt.plot(recall[label], precision[label], linestyle='--', label=f'Precision-Recall curve (area = {pr_auc[label]:.2f}) for class {label}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves by Class')
    plt.legend(loc="lower left")
    plt.show()