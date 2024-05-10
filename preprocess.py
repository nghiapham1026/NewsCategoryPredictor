import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (run once)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def load_data(file_path):
    return pd.read_csv(file_path)

def clean_missing_values(data):
    data = data.dropna(subset=['TITLE', 'CATEGORY'])
    return data

def normalize_text(text):
    text = str(text).lower()  # Ensure text is converted to string
    text = re.sub(r'\W+', ' ', text) 
    text = text.strip()
    return text

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

def extract_url_features(data):
    def extract_keywords(url):
        url = re.sub(r'https?://', '', url)
        url = re.sub(r'www\.', '', url)
        url = re.sub(r'\.[a-z]{2,3}/.*', '', url)
        keywords = re.split(r'\W+', url)
        return " ".join(keywords)
    
    data['URL_KEYWORDS'] = data['URL'].apply(extract_keywords)
    return data

def preprocess_data(input_file_path, output_file_path):
    data = load_data(input_file_path)
    data = clean_missing_values(data)
    data['TITLE'] = data['TITLE'].apply(normalize_text)
    
    custom_stopwords = set(['new', 'u', 'to', 'the', 'in', 'and', 'of', 'a', 'for', 'on', 'with', 'at', 'is', 'that', 'it', 'this'])
    all_stopwords = set(stopwords.words('english')) | custom_stopwords
    
    def preprocess_title(title):
        words = nltk.word_tokenize(title)
        filtered_words = [word for word in words if word.lower() not in all_stopwords]
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
        return ' '.join(lemmatized_words)
    
    data['TITLE'] = data['TITLE'].apply(preprocess_title)
    
    data = extract_url_features(data)
    
    data.to_csv(output_file_path, index=False)
    print(f"Preprocessing completed. Preprocessed data saved to {output_file_path}")

if __name__ == '__main__':
    input_file_path = './uci-news-aggregator_small.csv'
    output_file_path = './preprocessed_data.csv'
    
    preprocess_data(input_file_path, output_file_path)
