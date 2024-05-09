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
    model_scores = {}

    # Models and their hyperparameters for tuning
    models = {
        'Logistic Regression': {
            'pipeline': Pipeline([
                ('vect', TfidfVectorizer()),
                ('scaler', StandardScaler(with_mean=False)),  # Using StandardScaler to scale data
                ('clf', LogisticRegression(max_iter=1000))    # Increased max_iter
            ]),
            'params': {
                'clf__C': [0.1, 1, 10, 100],
                'clf__solver': ['liblinear', 'lbfgs', 'sag', 'saga']  # Exploring different solvers
            }
        },
        'Decision Tree': {
            'pipeline': Pipeline([
                ('vect', TfidfVectorizer()),
                ('clf', DecisionTreeClassifier())
            ]),
            'params': {
                'clf__max_depth': [None, 10, 20, 30],
                'clf__min_samples_leaf': [1, 2, 4]
            }
        },
        'Support Vector Machine': {
            'pipeline': Pipeline([
                ('vect', TfidfVectorizer()),
                ('scaler', StandardScaler(with_mean=False)),  # Scaling for SVM
                ('clf', SVC(kernel='linear', probability=True))
            ]),
            'params': {
                'clf__C': [0.1, 1, 10, 100]
            }
        }
    }

    # Training and hyperparameter tuning
    for name, info in models.items():
        grid_search = GridSearchCV(info['pipeline'], info['params'], cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        y_pred = grid_search.predict(X_test)
        y_pred_proba = grid_search.predict_proba(X_test)

        # Call to plot_metrics with model name
        plot_metrics.plot_roc_curves(name, classes, y_test_binarized, y_pred_proba)
        plot_metrics.plot_precision_recall_curves(name, classes, y_test_binarized, y_pred_proba)

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

'''
Logistic Regression Model Performance (Grid Search):
Best Parameters: {'C': 10, 'solver': 'liblinear'}
              precision    recall  f1-score   support

           b       0.93      0.93      0.93     23414
           e       0.98      0.98      0.98     30353
           m       0.96      0.94      0.95      9024
           t       0.93      0.93      0.93     21693

    accuracy                           0.95     84484
   macro avg       0.95      0.95      0.95     84484
weighted avg       0.95      0.95      0.95     84484


Decision Tree Model Performance (Grid Search):
Best Parameters: {'max_depth': None, 'min_samples_leaf': 1}
              precision    recall  f1-score   support

           b       0.86      0.87      0.86     23414
           e       0.91      0.93      0.92     30353
           m       0.86      0.82      0.84      9024
           t       0.88      0.86      0.87     21693

    accuracy                           0.88     84484
   macro avg       0.88      0.87      0.87     84484
weighted avg       0.88      0.88      0.88     84484


Support Vector Machine Model Performance (Grid Search):
Best Parameters: {'clf__C': 0.1}
              precision    recall  f1-score   support

           b       0.76      0.83      0.79       222
           e       0.87      0.91      0.89       307
           m       0.91      0.66      0.76        90
           t       0.78      0.75      0.76       226

    accuracy                           0.82       845
   macro avg       0.83      0.78      0.80       845
weighted avg       0.82      0.82      0.82       845
'''