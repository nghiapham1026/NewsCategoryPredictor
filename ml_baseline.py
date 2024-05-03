import preprocess
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

def train_and_tune_models(X, y):
    vectorizer = TfidfVectorizer()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # Models and their hyperparameters for tuning
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(),
            'params': {
                'C': [0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs']
            }
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(),
            'params': {
                'max_depth': [None, 10, 20, 30],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'Support Vector Machine': {
            'model': SVC(),
            'params': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf']
            }
        }
    }

    # Training and hyperparameter tuning
    for name, info in models.items():
        grid_search = GridSearchCV(info['model'], info['params'], cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        y_pred = grid_search.predict(X_test)
        print(f"{name} Model Performance (Grid Search):")
        print("Best Parameters:", grid_search.best_params_)
        print(classification_report(y_test, y_pred))
        print("--------------------------------------------------\n")

def main():
    file_path = './uci-news-aggregator.csv'
    data = preprocess.load_data(file_path)
    data = preprocess.clean_missing_values(data)
    data = preprocess.normalize_text(data)
    data = preprocess.extract_url_features(data)
    
    X = data['TITLE']  # Using TITLE as the input feature
    y = data['CATEGORY']  # CATEGORY as the target
    train_and_tune_models(X, y)

if __name__ == '__main__':
    main()