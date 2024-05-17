import preprocess
import plot_metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
import numpy as np

def train_and_tune_models(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classes = np.unique(y)
    y_test_binarized = label_binarize(y_test, classes=classes)

    # Hyperparameter grids for each model
    param_grid_lr = {
        'clf__C': [0.01, 0.1, 1, 10],
        'clf__solver': ['liblinear', 'saga']
    }
    param_grid_dt = {
        'clf__max_depth': [None, 5, 10, 20],
        'clf__min_samples_leaf': [1, 2, 4]
    }
    param_grid_svc = {
        'clf__C': [0.1, 1, 10],
        'clf__kernel': ['linear', 'rbf']
    }

    # Define models with their pipelines and hyperparameter grids
    models = {
        'Logistic Regression': {
            'pipeline': Pipeline([
                ('vect', TfidfVectorizer()),  # Text vectorization
                ('scaler', StandardScaler(with_mean=False)),  # Standardize features
                ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))  # Logistic Regression classifier
            ]),
            'params': param_grid_lr
        },
        'Decision Tree': {
            'pipeline': Pipeline([
                ('vect', TfidfVectorizer()),  # Text vectorization
                ('clf', DecisionTreeClassifier(class_weight='balanced'))  # Decision Tree classifier
            ]),
            'params': param_grid_dt
        },
        'Support Vector Machine': {
            'pipeline': Pipeline([
                ('vect', TfidfVectorizer()),  # Text vectorization
                ('scaler', StandardScaler(with_mean=False)),  # Standardize features
                ('clf', SVC(class_weight='balanced', probability=True))  # SVM classifier
            ]),
            'params': param_grid_svc
        }
    }

    # Train and tune each model using GridSearchCV
    for name, info in models.items():
        grid_search = GridSearchCV(info['pipeline'], info['params'], cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test) if hasattr(best_model.named_steps['clf'], 'predict_proba') else None

        # Plot ROC and Precision-Recall curves if probability predictions are available
        if y_pred_proba is not None:
            plot_metrics.plot_roc_curves(name, classes, y_test_binarized, y_pred_proba)
            plot_metrics.plot_precision_recall_curves(name, classes, y_test_binarized, y_pred_proba)
        
        # Print model performance and best parameters
        print(f"{name} Model Performance (Grid Search Tuning):")
        print(classification_report(y_test, y_pred))
        print("Best Parameters:", grid_search.best_params_)
        print("--------------------------------------------------\n")

def main():
    file_path = './uci-news-aggregator_very_small.csv'
    data = preprocess.load_data(file_path)  # Load the dataset
    data = preprocess.clean_missing_values(data)  # Clean missing values
    data['TITLE'] = data['TITLE'].apply(preprocess.normalize_text)  # Normalize text data
    X = data['TITLE']  # Use TITLE as the input feature
    y = data['CATEGORY']  # Use CATEGORY as the target
    train_and_tune_models(X, y)  # Train and tune models

if __name__ == '__main__':
    main()