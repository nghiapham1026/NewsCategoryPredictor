import preprocess
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.exceptions import UndefinedMetricWarning
import numpy as np
import warnings

def train_and_tune_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_scores = {}
    classes = np.unique(y)
    y_test_binarized = label_binarize(y_test, classes=classes)

    # Models and their hyperparameters for tuning
    models = {
        'Logistic Regression': {
            'pipeline': Pipeline([
                ('vect', TfidfVectorizer()),
                ('scaler', StandardScaler(with_mean=False)),  # Using StandardScaler to scale data
                ('clf', OneVsRestClassifier(LogisticRegression(max_iter=1000)))
            ]),
            'params': {
                'clf__estimator__C': [0.1, 1, 10, 100],
                'clf__estimator__solver': ['liblinear', 'lbfgs']  # Limiting to solvers that support OvR
            }
        },
        'Decision Tree': {
            'pipeline': Pipeline([
                ('vect', TfidfVectorizer()),
                ('clf', OneVsRestClassifier(DecisionTreeClassifier()))
            ]),
            'params': {
                'clf__estimator__max_depth': [None, 10, 20, 30],
                'clf__estimator__min_samples_leaf': [1, 2, 4]
            }
        },
        'Support Vector Machine': {
            'pipeline': Pipeline([
                ('vect', TfidfVectorizer()),
                ('scaler', StandardScaler(with_mean=False)),  # Scaling for SVM
                ('clf', OneVsRestClassifier(SVC(kernel='linear', probability=True)))
            ]),
            'params': {
                'clf__estimator__C': [0.1, 1, 10, 100]
            }
        }
    }

    # Handling warning for undefined metric in some edge cases
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    # Training and hyperparameter tuning
    for name, info in models.items():
        grid_search = GridSearchCV(info['pipeline'], info['params'], cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        y_pred = grid_search.predict(X_test)
        y_pred_proba = grid_search.predict_proba(X_test)
        model_scores[name] = grid_search.best_score_

        # ROC and Precision-Recall
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        precision = dict()
        recall = dict()
        pr_auc = dict()

        for i, label in enumerate(classes):
            fpr[label], tpr[label], _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
            roc_auc[label] = auc(fpr[label], tpr[label])
            precision[label], recall[label], _ = precision_recall_curve(y_test_binarized[:, i], y_pred_proba[:, i])
            pr_auc[label] = average_precision_score(y_test_binarized[:, i], y_pred_proba[:, i])

        # Plot ROC and Precision-Recall for each class
        plt.figure(figsize=(12, 6))
        for label in classes:
            plt.plot(fpr[label], tpr[label], linestyle='--', label=f'{name} ROC curve (area = {roc_auc[label]:.2f}) for class {label}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves for {name}')
        plt.legend(loc="lower right")
        plt.show()

        plt.figure(figsize=(12, 6))
        for label in classes:
            plt.plot(recall[label], precision[label], linestyle='--', label=f'{name} Precision-Recall curve (area = {pr_auc[label]:.2f}) for class {label}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curves for {name}')
        plt.legend(loc="lower left")
        plt.show()

        print(f"{name} Model Performance (Grid Search):")
        print("Best Parameters:", grid_search.best_params_)
        print(classification_report(y_test, y_pred))
        print("--------------------------------------------------\n")

def main():
    file_path = './uci-news-aggregator_small.csv'
    data = preprocess.load_data(file_path)
    data = preprocess.clean_missing_values(data)
    data = preprocess.normalize_text(data)
    data = preprocess.extract_url_features(data)
    
    X = data['TITLE']  # Using TITLE as the input feature
    y = data['CATEGORY']  # CATEGORY as the target
    train_and_tune_models(X, y)

if __name__ == '__main__':
    main()