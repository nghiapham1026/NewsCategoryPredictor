import preprocess
import plot_metrics
from sklearn.model_selection import train_test_split
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

    # Manually define the parameter grid for Decision Tree
    param_grid_dt = {
        'clf__max_depth': None,  # Specify only two values: None (unlimited depth) and 5
        'clf__min_samples_leaf': 2
    }

    # Models with manually tuned hyperparameters and class weighting
    models = {
        'Logistic Regression': {
            'pipeline': Pipeline([
                ('vect', TfidfVectorizer()),
                ('scaler', StandardScaler(with_mean=False)),
                ('clf', LogisticRegression(C=1, solver='liblinear', max_iter=1000, class_weight='balanced'))
            ])
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
                ('clf', SVC(kernel='linear', C=1, class_weight='balanced', probability=True))
            ])
        }
    }

    # Training and evaluating models
    for name, info in models.items():
        model = info['pipeline']
        
        if name == 'Decision Tree':
            model.set_params(**info['params'])  # Set Decision Tree hyperparameters
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # Call to plot_metrics with model name
        plot_metrics.plot_roc_curves(name, classes, y_test_binarized, y_pred_proba)
        plot_metrics.plot_precision_recall_curves(name, classes, y_test_binarized, y_pred_proba)

        print(f"{name} Model Performance (Manual Tuning with Class Weighting):")
        print(classification_report(y_test, y_pred))
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
