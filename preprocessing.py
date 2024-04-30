import pandas as pd
import re

def load_data(file_path):
    return pd.read_csv(file_path)

def clean_missing_values(data):
    # Since the initial exploration showed no missing values, this is just a precautionary step.
    # Remove rows where TITLE or CATEGORY is missing (if any)
    data = data.dropna(subset=['TITLE', 'CATEGORY'])
    return data

def normalize_text(data):
    # Normalize text data in TITLE and other relevant fields
    def normalize(text):
        # Convert text to lowercase, remove punctuation, and strip whitespaces
        text = text.lower()
        text = re.sub(r'\W+', ' ', text)
        text = text.strip()
        return text
    
    data['TITLE'] = data['TITLE'].apply(normalize)
    return data

def extract_url_features(data):
    # Extract and preprocess URL components to use as features
    def extract_keywords(url):
        # Remove protocol, get the main content of the URL, and extract potential keywords
        url = re.sub(r'https?://', '', url)  # Remove the protocol
        url = re.sub(r'www\.', '', url)      # Remove 'www.'
        url = re.sub(r'\.[a-z]{2,3}/.*', '', url)  # Remove everything after the domain extension
        keywords = re.split(r'\W+', url)
        return " ".join(keywords)
    
    data['URL_KEYWORDS'] = data['URL'].apply(extract_keywords)
    return data

def main():
    file_path = './uci-news-aggregator_small.csv'  # Update the path to your dataset
    data = load_data(file_path)
    data = clean_missing_values(data)
    data = normalize_text(data)
    data = extract_url_features(data)
    print(data.head())  # Display the first few rows of the cleaned and processed dataset

if __name__ == "__main__":
    main()