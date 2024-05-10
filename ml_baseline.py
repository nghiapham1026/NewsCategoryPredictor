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

    models = {
        'Logistic Regression': {
            'pipeline': Pipeline([
                ('vect', TfidfVectorizer()),
                ('scaler', StandardScaler(with_mean=False)),
                ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
            ]),
            'params': param_grid_lr
        },
        'Decision Tree': {
            'pipeline': Pipeline([
                ('vect', TfidfVectorizer()),
                ('clf', DecisionTreeClassifier(class_weight='balanced'))
            ]),
            'params': param_grid_dt
        },
        'Support Vector Machine': {
            'pipeline': Pipeline([
                ('vect', TfidfVectorizer()),
                ('scaler', StandardScaler(with_mean=False)),
                ('clf', SVC(class_weight='balanced', probability=True))
            ]),
            'params': param_grid_svc
        }
    }

    for name, info in models.items():
        grid_search = GridSearchCV(info['pipeline'], info['params'], cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test) if hasattr(best_model.named_steps['clf'], 'predict_proba') else None

        # Call to plot_metrics with model name
        if y_pred_proba is not None:
            plot_metrics.plot_roc_curves(name, classes, y_test_binarized, y_pred_proba)
            plot_metrics.plot_precision_recall_curves(name, classes, y_test_binarized, y_pred_proba)
        
        print(f"{name} Model Performance (Grid Search Tuning):")
        print(classification_report(y_test, y_pred))
        print("Best Parameters:", grid_search.best_params_)
        print("--------------------------------------------------\n")

def main():
    file_path = './uci-news-aggregator_very_small.csv'
    data = preprocess.load_data(file_path)
    data = preprocess.clean_missing_values(data)
    data['TITLE'] = data['TITLE'].apply(preprocess.normalize_text)  # Apply text normalization
    X = data['TITLE']  # Using TITLE as the input feature
    y = data['CATEGORY']  # CATEGORY as the target
    train_and_tune_models(X, y)

if __name__ == '__main__':
    main()