import preprocess
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def train_baseline_model(X, y):
    vectorizer = TfidfVectorizer()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Baseline Model Performance:")
    print(classification_report(y_test, y_pred))

def main():
    file_path = './uci-news-aggregator.csv'
    data = preprocess.load_data(file_path)
    data = preprocess.clean_missing_values(data)
    data = preprocess.normalize_text(data)
    data = preprocess.extract_url_features(data)
    
    X = data['TITLE']  # using TITLE as the input feature
    y = data['CATEGORY']  # CATEGORY as the target
    train_baseline_model(X, y)

if __name__ == '__main__':
    main()

'''Baseline Model Performance:
              precision    recall  f1-score   support

           b       0.92      0.93      0.92     23414
           e       0.96      0.98      0.97     30353
           m       0.96      0.91      0.94      9024
           t       0.93      0.93      0.93     21693

    accuracy                           0.94     84484
   macro avg       0.94      0.94      0.94     84484
weighted avg       0.94      0.94      0.94     84484'''